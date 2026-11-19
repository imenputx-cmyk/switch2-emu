//! Runs single instructions for `tools/difftest.py`.
//!
//! Reads one case per line on stdin:
//!     insn pc nzcv fpcr x0 .. x30 sp v0 .. v31          (all hex)
//! and writes one result per line:
//!     status pc nzcv qc x0 .. x30 sp v0 .. v31 | addr=byte addr=byte ...
//! where `qc` is FPSR.QC (the other FPSR flags are not modelled yet).
//! where the memory part lists bytes that differ from the initial pattern.
//! `status` is ok, fault, align, unimpl, svc, brk or skip (the instruction
//! is architecturally UNPREDICTABLE for these operands).

use emu_cpu::decoder::{decode, AddrMode, ExclusiveOp, Instr, PairMode};
use emu_cpu::{Cpu, Exit, Memory, Nzcv, Perm};
use std::io::{self, BufRead, Write};

const CODE_BASE: u64 = 0x1000_0000;
const CODE_SIZE: u64 = 0x1000;
const DATA_BASE: u64 = 0x2000_0000;
const DATA_SIZE: u64 = 0x10000;

/// Must match `pattern()` in tools/difftest.py.
fn pattern(addr: u64) -> u8 {
    ((addr.wrapping_mul(0x9E37_79B1) >> 7) ^ (addr >> 3)) as u8
}

fn fresh_memory() -> Memory {
    let mut mem = Memory::new();
    mem.map(CODE_BASE, CODE_SIZE, Perm::RWX).unwrap();
    mem.map(DATA_BASE, DATA_SIZE, Perm::RW).unwrap();
    for (base, size) in [(CODE_BASE, CODE_SIZE), (DATA_BASE, DATA_SIZE)] {
        let bytes: Vec<u8> = (base..base + size).map(pattern).collect();
        mem.write_privileged(base, &bytes).unwrap();
    }
    mem
}

/// Classifies an alignment fault so the harness can tell CPU-model
/// differences apart. "align-lse2" means FEAT_LSE2 would allow the access:
/// an atomic inside one 16-byte block, or any LDAR/STLR/LDAPR. Unicorn's
/// "max" model has LSE2; the Switch 2's Cortex-A78C does not.
/// "align-stx" marks a store-exclusive.
fn align_status(instr: Instr, addr: u64) -> &'static str {
    let bytes: u64 = match instr {
        Instr::Exclusive {
            op: ExclusiveOp::StoreExclusive { .. },
            ..
        } => return "align-stx",
        Instr::ExclusivePair { store: Some(_), .. } => return "align-stx",
        Instr::Exclusive {
            op: ExclusiveOp::LoadAcquire | ExclusiveOp::StoreRelease,
            ..
        } => return "align-lse2",
        Instr::Exclusive { size, .. }
        | Instr::Atomic { size, .. }
        | Instr::CompareSwap { size, .. } => 1 << size,
        Instr::ExclusivePair { sf, .. } | Instr::CompareSwapPair { sf, .. } => {
            if sf {
                16
            } else {
                8
            }
        }
        _ => return "align",
    };
    if addr % 16 + bytes <= 16 {
        "align-lse2"
    } else {
        "align"
    }
}

/// Operand combinations the architecture leaves UNPREDICTABLE, where two
/// correct implementations may legitimately disagree.
fn unpredictable(instr: Instr) -> bool {
    match instr {
        Instr::LoadStore {
            mode: AddrMode::PreIndex { rn, .. } | AddrMode::PostIndex { rn, .. },
            rt,
            ..
        } => rn == rt && rn != 31,
        Instr::LoadStorePair {
            load,
            mode,
            rn,
            rt,
            rt2,
            ..
        } => {
            (load && rt == rt2) || (mode != PairMode::Offset && rn != 31 && (rn == rt || rn == rt2))
        }
        Instr::Exclusive {
            op: ExclusiveOp::StoreExclusive { rs },
            rn,
            rt,
            ..
        } => rs == rt || (rs == rn && rn != 31),
        _ => false,
    }
}

fn main() {
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    let initial = fresh_memory();
    let mut snapshot_code = vec![0u8; CODE_SIZE as usize];
    let mut snapshot_data = vec![0u8; DATA_SIZE as usize];

    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let v: Vec<u128> = line
            .split_whitespace()
            .map(|s| u128::from_str_radix(s, 16).unwrap())
            .collect();
        assert_eq!(v.len(), 4 + 31 + 1 + 32, "bad input line");
        let insn = v[0] as u32;

        if unpredictable(decode(insn)) {
            writeln!(out, "skip").unwrap();
            continue;
        }

        let mut mem = fresh_memory();
        let pc = v[1] as u64;
        mem.write_privileged(pc, &insn.to_le_bytes()).unwrap();
        let mut cpu = Cpu::new();
        cpu.set_pc(pc);
        cpu.nzcv = Nzcv::from_bits(v[2] as u64);
        cpu.fpcr = v[3] as u64;
        for r in 0..31u8 {
            cpu.write_xzr(r, v[4 + r as usize] as u64);
        }
        cpu.write_sp(31, v[35] as u64);
        for r in 0..32u8 {
            cpu.write_v(r, v[36 + r as usize]);
        }

        let status = match cpu.step(&mut mem) {
            Ok(()) => "ok",
            Err(Exit::Fault { .. }) => "fault",
            Err(Exit::Alignment { addr, .. }) => align_status(decode(insn), addr),
            Err(Exit::Unimplemented { .. }) => "unimpl",
            Err(Exit::Svc(_)) => "svc",
            Err(Exit::Brk(_)) => "brk",
        };

        let qc = cpu.fpsr & emu_cpu::simd::FPSR_QC != 0;
        write!(
            out,
            "{status} {:x} {:x} {:x}",
            cpu.pc(),
            cpu.nzcv.to_bits(),
            u8::from(qc)
        )
        .unwrap();
        for r in 0..31u8 {
            write!(out, " {:x}", cpu.read_xzr(r)).unwrap();
        }
        write!(out, " {:x}", cpu.sp()).unwrap();
        for r in 0..32u8 {
            write!(out, " {:x}", cpu.read_v(r)).unwrap();
        }
        write!(out, " |").unwrap();

        // Report bytes that differ from the initial pattern. The instruction
        // word itself is part of the initial state on both sides.
        mem.read(CODE_BASE, &mut snapshot_code).unwrap();
        mem.read(DATA_BASE, &mut snapshot_data).unwrap();
        let mut init_code = vec![0u8; CODE_SIZE as usize];
        initial.read(CODE_BASE, &mut init_code).unwrap();
        let off = (pc - CODE_BASE) as usize;
        init_code[off..off + 4].copy_from_slice(&insn.to_le_bytes());
        for (base, now, size) in [
            (CODE_BASE, &snapshot_code, CODE_SIZE),
            (DATA_BASE, &snapshot_data, DATA_SIZE),
        ] {
            for i in 0..size as usize {
                let addr = base + i as u64;
                let before = if base == CODE_BASE {
                    init_code[i]
                } else {
                    pattern(addr)
                };
                if now[i] != before {
                    write!(out, " {addr:x}={:x}", now[i]).unwrap();
                }
            }
        }
        writeln!(out).unwrap();
    }
}
