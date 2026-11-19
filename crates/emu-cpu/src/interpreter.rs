//! Reference interpreter: executes one decoded instruction at a time.
//!
//! Semantics follow the pseudocode in the ARM Architecture Reference
//! Manual. Speed is not a goal here; this is the ground truth that a later
//! JIT will be tested against.

use crate::decoder::{
    decode, ones, ror, AddrMode, AtomicOp, BitfieldOp, BranchRegOp, CondOperand, CondSelectOp,
    DataProc1Op, DataProc2Op, DataProc3Op, ElemSrc, ExclusiveOp, Extend, FpUnaryOp, Instr,
    LoadKind, LogicOp, MoveWideOp, PairMode, Shift, SysReg, VecImmOp, VecLogicOp,
};
use crate::fp::Fp;
use crate::{Cpu, Nzcv};
use emu_memory::{Fault, Memory};

/// Why execution stopped. Everything except `Svc` leaves the PC at the
/// instruction that caused it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// Supervisor call with its immediate. The PC already points to the
    /// next instruction, so the kernel can simply resume.
    Svc(u16),
    /// Breakpoint with its immediate.
    Brk(u16),
    /// A memory access failed.
    Fault { pc: u64, fault: Fault },
    /// An access that must be aligned was not.
    Alignment { pc: u64, addr: u64 },
    /// The instruction is unallocated or not implemented yet.
    Unimplemented { pc: u64, insn: u32 },
}

/// `AddWithCarry()` from the ARM pseudocode, for 32- or 64-bit operands.
fn add_with_carry(x: u64, y: u64, carry_in: bool, sf: bool) -> (u64, Nzcv) {
    let c = u64::from(carry_in);
    if sf {
        let unsigned_sum = u128::from(x) + u128::from(y) + u128::from(c);
        let signed_sum = i128::from(x as i64) + i128::from(y as i64) + i128::from(c);
        let result = unsigned_sum as u64;
        let flags = Nzcv {
            n: (result as i64) < 0,
            z: result == 0,
            c: u128::from(result) != unsigned_sum,
            v: i128::from(result as i64) != signed_sum,
        };
        (result, flags)
    } else {
        let (x, y) = (x as u32, y as u32);
        let unsigned_sum = u64::from(x) + u64::from(y) + c;
        let signed_sum = i64::from(x as i32) + i64::from(y as i32) + c as i64;
        let result = unsigned_sum as u32;
        let flags = Nzcv {
            n: (result as i32) < 0,
            z: result == 0,
            c: u64::from(result) != unsigned_sum,
            v: i64::from(result as i32) != signed_sum,
        };
        (u64::from(result), flags)
    }
}

fn datasize(sf: bool) -> u32 {
    if sf {
        64
    } else {
        32
    }
}

fn truncate(value: u64, sf: bool) -> u64 {
    value & ones(datasize(sf))
}

/// Sign-extends the low `width` bits of `value` to 64 bits.
fn sext(value: u64, width: u32) -> u64 {
    let shift = 64 - width;
    (((value << shift) as i64) >> shift) as u64
}

/// Flags of a logical operation: N and Z from the result, C and V cleared.
fn logic_flags(result: u64, sf: bool) -> Nzcv {
    Nzcv {
        n: (result >> (datasize(sf) - 1)) & 1 == 1,
        z: result == 0,
        c: false,
        v: false,
    }
}

/// `ShiftReg()`: shifts a register operand by an immediate amount.
fn shift_reg(value: u64, shift: Shift, amount: u32, sf: bool) -> u64 {
    let n = datasize(sf);
    let value = truncate(value, sf);
    let amount = amount % n;
    let result = match shift {
        Shift::Lsl => value << amount,
        Shift::Lsr => value >> amount,
        Shift::Asr => (sext(value, n) as i64 >> amount) as u64,
        Shift::Ror => ror(value, amount, n),
    };
    truncate(result, sf)
}

/// `ExtendReg()`: extends a register operand, then shifts it left.
fn extend_reg(value: u64, extend: Extend, shift: u32, datasize: u32) -> u64 {
    let v = match extend {
        Extend::Uxtb => value & 0xFF,
        Extend::Uxth => value & 0xFFFF,
        Extend::Uxtw => value & 0xFFFF_FFFF,
        Extend::Uxtx => value,
        Extend::Sxtb => sext(value, 8),
        Extend::Sxth => sext(value, 16),
        Extend::Sxtw => sext(value, 32),
        Extend::Sxtx => value,
    };
    (v << shift) & ones(datasize)
}

/// `ConditionHolds()`.
pub fn condition_holds(cond: u8, f: Nzcv) -> bool {
    let result = match cond >> 1 {
        0b000 => f.z,
        0b001 => f.c,
        0b010 => f.n,
        0b011 => f.v,
        0b100 => f.c && !f.z,
        0b101 => f.n == f.v,
        0b110 => f.n == f.v && !f.z,
        _ => true,
    };
    if cond & 1 == 1 && cond != 0b1111 {
        !result
    } else {
        result
    }
}

impl Cpu {
    /// Fetches the instruction at the PC and executes it.
    pub fn step(&mut self, mem: &mut Memory) -> Result<(), Exit> {
        let pc = self.pc();
        let insn = mem
            .fetch_u32(pc)
            .map_err(|fault| Exit::Fault { pc, fault })?;
        self.counter = self.counter.wrapping_add(1);
        self.execute(insn, mem)
    }

    /// Runs until something stops execution, or `max_steps` instructions
    /// have run. Returns `None` when the step limit was reached.
    pub fn run(&mut self, mem: &mut Memory, max_steps: u64) -> Option<Exit> {
        for _ in 0..max_steps {
            if let Err(exit) = self.step(mem) {
                return Some(exit);
            }
        }
        None
    }

    /// Executes one instruction as if it were located at the current PC.
    /// On success the PC points to the next instruction to run.
    pub fn execute(&mut self, insn: u32, mem: &mut Memory) -> Result<(), Exit> {
        let pc = self.pc();
        let mut next_pc = pc.wrapping_add(4);
        let fault = |fault| Exit::Fault { pc, fault };

        match decode(insn) {
            Instr::Nop => {}
            Instr::Clrex => self.exclusive = None,

            Instr::Svc(imm) => {
                // Taking an exception clears the local exclusive monitor.
                self.exclusive = None;
                self.set_pc(next_pc);
                return Err(Exit::Svc(imm));
            }
            Instr::Brk(imm) => return Err(Exit::Brk(imm)),

            Instr::Adr { page, imm, rd } => {
                let base = if page { pc & !0xFFF } else { pc };
                self.write_xzr(rd, base.wrapping_add_signed(imm));
            }

            Instr::AddSubImm {
                sf,
                sub,
                set_flags,
                imm,
                rn,
                rd,
            } => {
                let (result, flags) = self.add_sub(self.read_sp(rn), imm, sub, sf);
                if set_flags {
                    self.nzcv = flags;
                    self.write_xzr(rd, result);
                } else {
                    self.write_sp(rd, result);
                }
            }

            Instr::AddSubReg {
                sf,
                sub,
                set_flags,
                shift,
                amount,
                rm,
                rn,
                rd,
            } => {
                let operand2 = shift_reg(self.read_xzr(rm), shift, amount, sf);
                let (result, flags) = self.add_sub(self.read_xzr(rn), operand2, sub, sf);
                if set_flags {
                    self.nzcv = flags;
                }
                self.write_xzr(rd, result);
            }

            Instr::AddSubExt {
                sf,
                sub,
                set_flags,
                extend,
                amount,
                rm,
                rn,
                rd,
            } => {
                let operand2 = extend_reg(self.read_xzr(rm), extend, amount, datasize(sf));
                let (result, flags) = self.add_sub(self.read_sp(rn), operand2, sub, sf);
                if set_flags {
                    self.nzcv = flags;
                    self.write_xzr(rd, result);
                } else {
                    self.write_sp(rd, result);
                }
            }

            Instr::AddSubCarry {
                sf,
                sub,
                set_flags,
                rm,
                rn,
                rd,
            } => {
                let operand2 = self.read_xzr(rm);
                let operand2 = if sub { !operand2 } else { operand2 };
                let (result, flags) = add_with_carry(self.read_xzr(rn), operand2, self.nzcv.c, sf);
                if set_flags {
                    self.nzcv = flags;
                }
                self.write_xzr(rd, result);
            }

            Instr::LogicalImm {
                sf,
                op,
                imm,
                rn,
                rd,
            } => {
                let result = logic(op, truncate(self.read_xzr(rn), sf), imm);
                if op == LogicOp::Ands {
                    self.nzcv = logic_flags(result, sf);
                    self.write_xzr(rd, result);
                } else {
                    self.write_sp(rd, result);
                }
            }

            Instr::LogicalReg {
                sf,
                op,
                invert,
                shift,
                amount,
                rm,
                rn,
                rd,
            } => {
                let mut operand2 = shift_reg(self.read_xzr(rm), shift, amount, sf);
                if invert {
                    operand2 = truncate(!operand2, sf);
                }
                let result = logic(op, truncate(self.read_xzr(rn), sf), operand2);
                if op == LogicOp::Ands {
                    self.nzcv = logic_flags(result, sf);
                }
                self.write_xzr(rd, result);
            }

            Instr::MoveWide {
                sf,
                op,
                shift,
                imm16,
                rd,
            } => {
                let imm = u64::from(imm16) << shift;
                let result = match op {
                    MoveWideOp::Movz => imm,
                    MoveWideOp::Movn => !imm,
                    MoveWideOp::Movk => (self.read_xzr(rd) & !(0xFFFF_u64 << shift)) | imm,
                };
                self.write_xzr(rd, truncate(result, sf));
            }

            Instr::Bitfield {
                sf,
                op,
                immr,
                imms,
                wmask,
                tmask,
                rn,
                rd,
            } => {
                let n = datasize(sf);
                let src = truncate(self.read_xzr(rn), sf);
                let rotated = ror(src, immr, n);
                let result = match op {
                    BitfieldOp::Ubfm => rotated & wmask & tmask,
                    BitfieldOp::Sbfm => {
                        let bot = rotated & wmask;
                        let top = if (src >> imms) & 1 == 1 { ones(n) } else { 0 };
                        (top & !tmask) | (bot & tmask)
                    }
                    BitfieldOp::Bfm => {
                        let dst = truncate(self.read_xzr(rd), sf);
                        let bot = (dst & !wmask) | (rotated & wmask);
                        (dst & !tmask) | (bot & tmask)
                    }
                };
                self.write_xzr(rd, truncate(result, sf));
            }

            Instr::Extract {
                sf,
                lsb,
                rm,
                rn,
                rd,
            } => {
                let hi = truncate(self.read_xzr(rn), sf);
                let lo = truncate(self.read_xzr(rm), sf);
                let result = if sf {
                    if lsb == 0 {
                        lo
                    } else {
                        (lo >> lsb) | (hi << (64 - lsb))
                    }
                } else {
                    ((hi << 32) | lo) >> lsb
                };
                self.write_xzr(rd, truncate(result, sf));
            }

            Instr::CondCompare {
                sf,
                sub,
                cond,
                operand,
                nzcv,
                rn,
            } => {
                if condition_holds(cond, self.nzcv) {
                    let operand2 = match operand {
                        CondOperand::Reg(rm) => self.read_xzr(rm),
                        CondOperand::Imm(imm) => imm,
                    };
                    self.nzcv = self.add_sub(self.read_xzr(rn), operand2, sub, sf).1;
                } else {
                    self.nzcv = Nzcv::from_nibble(nzcv);
                }
            }

            Instr::CondSelect {
                sf,
                op,
                cond,
                rm,
                rn,
                rd,
            } => {
                let result = if condition_holds(cond, self.nzcv) {
                    self.read_xzr(rn)
                } else {
                    let m = self.read_xzr(rm);
                    match op {
                        CondSelectOp::Csel => m,
                        CondSelectOp::Csinc => m.wrapping_add(1),
                        CondSelectOp::Csinv => !m,
                        CondSelectOp::Csneg => m.wrapping_neg(),
                    }
                };
                self.write_xzr(rd, truncate(result, sf));
            }

            Instr::DataProc2 { sf, op, rm, rn, rd } => {
                let n_bits = datasize(sf);
                let a = truncate(self.read_xzr(rn), sf);
                let b = truncate(self.read_xzr(rm), sf);
                let result = match op {
                    // Division by zero yields zero on AArch64.
                    DataProc2Op::Udiv => a.checked_div(b).unwrap_or(0),
                    DataProc2Op::Sdiv => {
                        let (a, b) = (sext(a, n_bits) as i64, sext(b, n_bits) as i64);
                        if b == 0 {
                            0
                        } else {
                            a.wrapping_div(b) as u64
                        }
                    }
                    DataProc2Op::Lslv => shift_reg(a, Shift::Lsl, (b % 64) as u32, sf),
                    DataProc2Op::Lsrv => shift_reg(a, Shift::Lsr, (b % 64) as u32, sf),
                    DataProc2Op::Asrv => shift_reg(a, Shift::Asr, (b % 64) as u32, sf),
                    DataProc2Op::Rorv => shift_reg(a, Shift::Ror, (b % 64) as u32, sf),
                    DataProc2Op::Crc32 { castagnoli, size } => {
                        crc32(a as u32, b, 1 << size, castagnoli)
                    }
                };
                self.write_xzr(rd, truncate(result, sf));
            }

            Instr::DataProc1 { sf, op, rn, rd } => {
                let v = truncate(self.read_xzr(rn), sf);
                let result = match (op, sf) {
                    (DataProc1Op::Rbit, true) => v.reverse_bits(),
                    (DataProc1Op::Rbit, false) => u64::from((v as u32).reverse_bits()),
                    (DataProc1Op::Rev16, _) => {
                        ((v & 0x00FF_00FF_00FF_00FF) << 8) | ((v >> 8) & 0x00FF_00FF_00FF_00FF)
                    }
                    (DataProc1Op::Rev32, _) => {
                        u64::from((v as u32).swap_bytes())
                            | (u64::from(((v >> 32) as u32).swap_bytes()) << 32)
                    }
                    (DataProc1Op::Rev, true) => v.swap_bytes(),
                    (DataProc1Op::Rev, false) => u64::from((v as u32).swap_bytes()),
                    (DataProc1Op::Clz, true) => u64::from(v.leading_zeros()),
                    (DataProc1Op::Clz, false) => u64::from((v as u32).leading_zeros()),
                    (DataProc1Op::Cls, true) => {
                        let lead = if (v as i64) < 0 {
                            v.leading_ones()
                        } else {
                            v.leading_zeros()
                        };
                        u64::from(lead - 1)
                    }
                    (DataProc1Op::Cls, false) => {
                        let w = v as u32;
                        let lead = if (w as i32) < 0 {
                            w.leading_ones()
                        } else {
                            w.leading_zeros()
                        };
                        u64::from(lead - 1)
                    }
                };
                self.write_xzr(rd, truncate(result, sf));
            }

            Instr::DataProc3 {
                sf,
                op,
                rm,
                ra,
                rn,
                rd,
            } => {
                let (n, m, a) = (self.read_xzr(rn), self.read_xzr(rm), self.read_xzr(ra));
                let sw = |v: u64| sext(v, 32);
                let uw = |v: u64| v & 0xFFFF_FFFF;
                let result = match op {
                    DataProc3Op::Madd => a.wrapping_add(n.wrapping_mul(m)),
                    DataProc3Op::Msub => a.wrapping_sub(n.wrapping_mul(m)),
                    DataProc3Op::Smaddl => a.wrapping_add(sw(n).wrapping_mul(sw(m))),
                    DataProc3Op::Smsubl => a.wrapping_sub(sw(n).wrapping_mul(sw(m))),
                    DataProc3Op::Umaddl => a.wrapping_add(uw(n).wrapping_mul(uw(m))),
                    DataProc3Op::Umsubl => a.wrapping_sub(uw(n).wrapping_mul(uw(m))),
                    DataProc3Op::Smulh => {
                        ((i128::from(n as i64) * i128::from(m as i64)) >> 64) as u64
                    }
                    DataProc3Op::Umulh => ((u128::from(n) * u128::from(m)) >> 64) as u64,
                };
                self.write_xzr(rd, truncate(result, sf));
            }

            Instr::Branch { link, offset } => {
                if link {
                    self.write_xzr(30, pc.wrapping_add(4));
                }
                next_pc = pc.wrapping_add_signed(offset);
            }

            Instr::BranchCond { cond, offset } => {
                if condition_holds(cond, self.nzcv) {
                    next_pc = pc.wrapping_add_signed(offset);
                }
            }

            Instr::BranchReg { op, rn } => {
                // Read the target before writing X30, so `blr x30` works.
                let target = self.read_xzr(rn);
                if op == BranchRegOp::Blr {
                    self.write_xzr(30, pc.wrapping_add(4));
                }
                next_pc = target;
            }

            Instr::CompareBranch {
                sf,
                nonzero,
                offset,
                rt,
            } => {
                let is_zero = truncate(self.read_xzr(rt), sf) == 0;
                if is_zero != nonzero {
                    next_pc = pc.wrapping_add_signed(offset);
                }
            }

            Instr::TestBranch {
                bit,
                nonzero,
                offset,
                rt,
            } => {
                let set = (self.read_xzr(rt) >> bit) & 1 == 1;
                if set == nonzero {
                    next_pc = pc.wrapping_add_signed(offset);
                }
            }

            Instr::Mrs { reg, rt } => {
                let value = match reg {
                    SysReg::Nzcv => self.nzcv.to_bits(),
                    SysReg::Fpcr => self.fpcr,
                    SysReg::Fpsr => self.fpsr,
                    SysReg::TpidrEl0 => self.tpidr_el0,
                    SysReg::TpidrroEl0 => self.tpidrro_el0,
                    SysReg::CntvctEl0 => self.counter,
                    SysReg::CntfrqEl0 => self.counter_freq,
                    // 64-byte cache lines.
                    SysReg::CtrEl0 => 0x8444_C004,
                    // DC ZVA is not implemented yet, so report it prohibited.
                    SysReg::DczidEl0 => 0x10,
                };
                self.write_xzr(rt, value);
            }

            Instr::Msr { reg, rt } => {
                let value = self.read_xzr(rt);
                match reg {
                    SysReg::Nzcv => self.nzcv = Nzcv::from_bits(value),
                    SysReg::Fpcr => self.fpcr = value,
                    SysReg::Fpsr => self.fpsr = value,
                    SysReg::TpidrEl0 => self.tpidr_el0 = value,
                    // Read-only at EL0.
                    _ => return Err(Exit::Unimplemented { pc, insn }),
                }
            }

            Instr::LoadStore {
                simd,
                size,
                kind,
                mode,
                rt,
            } => {
                let (addr, writeback) = self.address(mode, pc);
                let bytes = 1usize << size;
                if simd {
                    let mut buf = [0u8; 16];
                    if kind == LoadKind::Store {
                        buf.copy_from_slice(&self.read_v(rt).to_le_bytes());
                        mem.write(addr, &buf[..bytes]).map_err(fault)?;
                    } else {
                        mem.read(addr, &mut buf[..bytes]).map_err(fault)?;
                        self.write_v(rt, u128::from_le_bytes(buf));
                    }
                } else {
                    match kind {
                        LoadKind::Prefetch => {}
                        LoadKind::Store => {
                            mem.write_uint(addr, bytes, self.read_xzr(rt))
                                .map_err(fault)?;
                        }
                        _ => {
                            let raw = mem.read_uint(addr, bytes).map_err(fault)?;
                            self.write_xzr(rt, extend_load(raw, size, kind));
                        }
                    }
                }
                if let Some((rn, value)) = writeback {
                    self.write_sp(rn, value);
                }
            }

            Instr::LoadStorePair {
                simd,
                size,
                load,
                signed,
                mode,
                rn,
                imm,
                rt,
                rt2,
            } => {
                let base = self.read_sp(rn);
                let addr = match mode {
                    PairMode::PostIndex => base,
                    _ => base.wrapping_add_signed(imm),
                };
                let bytes = 1usize << size;
                let mut buf = [0u8; 32];
                if load {
                    mem.read(addr, &mut buf[..bytes * 2]).map_err(fault)?;
                    let mut word = [0u8; 16];
                    word[..bytes].copy_from_slice(&buf[..bytes]);
                    let first = u128::from_le_bytes(word);
                    word = [0; 16];
                    word[..bytes].copy_from_slice(&buf[bytes..bytes * 2]);
                    let second = u128::from_le_bytes(word);
                    if simd {
                        self.write_v(rt, first);
                        self.write_v(rt2, second);
                    } else {
                        let kind = if signed {
                            LoadKind::LoadSigned64
                        } else {
                            LoadKind::Load
                        };
                        self.write_xzr(rt, extend_load(first as u64, size, kind));
                        self.write_xzr(rt2, extend_load(second as u64, size, kind));
                    }
                } else {
                    let (a, b) = if simd {
                        (self.read_v(rt), self.read_v(rt2))
                    } else {
                        (
                            u128::from(self.read_xzr(rt)),
                            u128::from(self.read_xzr(rt2)),
                        )
                    };
                    buf[..bytes].copy_from_slice(&a.to_le_bytes()[..bytes]);
                    buf[bytes..bytes * 2].copy_from_slice(&b.to_le_bytes()[..bytes]);
                    mem.write(addr, &buf[..bytes * 2]).map_err(fault)?;
                }
                if mode != PairMode::Offset {
                    self.write_sp(rn, base.wrapping_add_signed(imm));
                }
            }

            Instr::Exclusive { size, op, rn, rt } => {
                let addr = self.read_sp(rn);
                let bytes = 1usize << size;
                if !addr.is_multiple_of(bytes as u64) {
                    return Err(Exit::Alignment { pc, addr });
                }
                match op {
                    ExclusiveOp::LoadExclusive | ExclusiveOp::LoadAcquire => {
                        let value = mem.read_uint(addr, bytes).map_err(fault)?;
                        self.write_xzr(rt, value);
                        if op == ExclusiveOp::LoadExclusive {
                            self.exclusive = Some(addr);
                        }
                    }
                    ExclusiveOp::StoreRelease => {
                        mem.write_uint(addr, bytes, self.read_xzr(rt))
                            .map_err(fault)?;
                    }
                    ExclusiveOp::StoreExclusive { rs } => {
                        let status = if self.exclusive == Some(addr) {
                            mem.write_uint(addr, bytes, self.read_xzr(rt))
                                .map_err(fault)?;
                            0
                        } else {
                            1
                        };
                        self.exclusive = None;
                        self.write_xzr(rs, status);
                    }
                }
            }

            Instr::FpUnary { ty, op, rn, rd } => {
                let fp = Fp::new(ty, self.fpcr);
                let x = self.read_d(rn);
                let (result, out_ty) = match op {
                    FpUnaryOp::Mov => (x, ty),
                    FpUnaryOp::Abs => (fp.abs(x), ty),
                    FpUnaryOp::Neg => (fp.neg(x), ty),
                    FpUnaryOp::Sqrt => (fp.sqrt(x), ty),
                    FpUnaryOp::Round(mode) => (fp.round_int(x, mode), ty),
                    FpUnaryOp::Convert(to) => (fp.convert(x, to), to),
                };
                self.write_scalar(rd, result, out_ty.bytes());
            }

            Instr::FpBinary { ty, op, rm, rn, rd } => {
                let fp = Fp::new(ty, self.fpcr);
                let result = fp.binop(op, self.read_d(rn), self.read_d(rm));
                self.write_scalar(rd, result, ty.bytes());
            }

            Instr::FpMulAdd {
                ty,
                neg_addend,
                neg_product,
                rm,
                ra,
                rn,
                rd,
            } => {
                let fp = Fp::new(ty, self.fpcr);
                let mut a = self.read_d(ra) & ty.mask();
                let mut n = self.read_d(rn) & ty.mask();
                if neg_addend {
                    a = fp.neg(a);
                }
                if neg_product {
                    n = fp.neg(n);
                }
                let result = fp.mul_add(a, n, self.read_d(rm));
                self.write_scalar(rd, result, ty.bytes());
            }

            Instr::FpCompare { ty, rn, rm } => {
                let fp = Fp::new(ty, self.fpcr);
                let b = rm.map_or(0, |rm| self.read_d(rm));
                self.nzcv = fp.compare_flags(self.read_d(rn), b);
            }

            Instr::FpCondCompare {
                ty,
                rn,
                rm,
                cond,
                nzcv,
            } => {
                self.nzcv = if condition_holds(cond, self.nzcv) {
                    Fp::new(ty, self.fpcr).compare_flags(self.read_d(rn), self.read_d(rm))
                } else {
                    Nzcv::from_nibble(nzcv)
                };
            }

            Instr::FpCondSelect {
                ty,
                rm,
                rn,
                cond,
                rd,
            } => {
                let src = if condition_holds(cond, self.nzcv) {
                    rn
                } else {
                    rm
                };
                self.write_scalar(rd, self.read_d(src), ty.bytes());
            }

            Instr::FpMovImm { ty, imm, rd } => self.write_scalar(rd, imm, ty.bytes()),

            Instr::FpToInt {
                sf,
                ty,
                signed,
                rounding,
                fbits,
                rn,
                rd,
            } => {
                let fp = Fp::new(ty, self.fpcr);
                let result = fp.to_int(self.read_d(rn), signed, datasize(sf), fbits, rounding);
                self.write_xzr(rd, result);
            }

            Instr::IntToFp {
                sf,
                ty,
                signed,
                fbits,
                rn,
                rd,
            } => {
                let fp = Fp::new(ty, self.fpcr);
                let value = truncate(self.read_xzr(rn), sf);
                let result = fp.from_int(value, signed, datasize(sf), fbits);
                self.write_scalar(rd, result, ty.bytes());
            }

            Instr::FpIntInVec {
                ty,
                to_int,
                signed,
                rounding,
                rn,
                rd,
            } => {
                let fp = Fp::new(ty, self.fpcr);
                let width = ty.bytes() as u32 * 8;
                let x = self.read_d(rn) & ty.mask();
                let result = if to_int {
                    fp.to_int(x, signed, width, 0, rounding)
                } else {
                    fp.from_int(x, signed, width, 0)
                };
                self.write_scalar(rd, result, ty.bytes());
            }

            Instr::FmovToGp { sf, top, rn, rd } => {
                let v = self.read_v(rn);
                let value = if top { (v >> 64) as u64 } else { v as u64 };
                self.write_xzr(rd, truncate(value, sf));
            }

            Instr::FmovFromGp { sf, top, rn, rd } => {
                let value = truncate(self.read_xzr(rn), sf);
                if top {
                    let low = self.read_v(rd) & u128::from(u64::MAX);
                    self.write_v(rd, low | (u128::from(value) << 64));
                } else {
                    self.write_scalar(rd, value, if sf { 8 } else { 4 });
                }
            }

            Instr::VecMoveImm { q, op, imm, rd } => {
                let imm = u128::from(imm) | (u128::from(imm) << 64);
                let old = self.read_v(rd);
                let result = match op {
                    VecImmOp::Movi => imm,
                    VecImmOp::Mvni => !imm,
                    VecImmOp::Orr => old | imm,
                    VecImmOp::Bic => old & !imm,
                };
                self.write_v(rd, vec_width(result, q));
            }

            Instr::VecDup { q, esize, src, rd } => {
                let elem = self.element_source(src, esize);
                let lanes = (if q { 16 } else { 8 }) >> esize;
                let mut result = 0u128;
                for i in 0..lanes {
                    result |= u128::from(elem) << (i * (8 << esize));
                }
                self.write_v(rd, result);
            }

            Instr::VecIns {
                esize,
                index,
                src,
                rd,
            } => {
                let elem = self.element_source(src, esize);
                let v = set_element(self.read_v(rd), esize, index, elem);
                self.write_v(rd, v);
            }

            Instr::VecToGp {
                signed,
                sf,
                esize,
                index,
                rn,
                rd,
            } => {
                let elem = get_element(self.read_v(rn), esize, index);
                let value = if signed { sext(elem, 8 << esize) } else { elem };
                self.write_xzr(rd, truncate(value, sf));
            }

            Instr::ScalarDup {
                esize,
                index,
                rn,
                rd,
            } => {
                let elem = get_element(self.read_v(rn), esize, index);
                self.write_scalar(rd, elem, 1 << esize);
            }

            Instr::VecLogical { q, op, rm, rn, rd } => {
                let (n, m, d) = (self.read_v(rn), self.read_v(rm), self.read_v(rd));
                let result = match op {
                    VecLogicOp::And => n & m,
                    VecLogicOp::Bic => n & !m,
                    VecLogicOp::Orr => n | m,
                    VecLogicOp::Orn => n | !m,
                    VecLogicOp::Eor => n ^ m,
                    VecLogicOp::Bsl => (n & d) | (m & !d),
                    VecLogicOp::Bit => (d & !m) | (n & m),
                    VecLogicOp::Bif => (d & m) | (n & !m),
                };
                self.write_v(rd, vec_width(result, q));
            }

            Instr::VecAddSub {
                q,
                sub,
                esize,
                rm,
                rn,
                rd,
            } => {
                let (n, m) = (self.read_v(rn), self.read_v(rm));
                let lanes = 16 >> esize;
                let mut result = 0u128;
                for i in 0..lanes {
                    let (a, b) = (get_element(n, esize, i), get_element(m, esize, i));
                    let r = if sub {
                        a.wrapping_sub(b)
                    } else {
                        a.wrapping_add(b)
                    };
                    result = set_element(result, esize, i, r);
                }
                self.write_v(rd, vec_width(result, q));
            }

            Instr::Atomic {
                size,
                op,
                rs,
                rn,
                rt,
            } => {
                let addr = self.read_sp(rn);
                let bytes = 1usize << size;
                if !addr.is_multiple_of(bytes as u64) {
                    return Err(Exit::Alignment { pc, addr });
                }
                let width = 8u32 << size;
                let old = mem.read_uint(addr, bytes).map_err(fault)?;
                let value = self.read_xzr(rs) & ones(width);
                let new = match op {
                    AtomicOp::Add => old.wrapping_add(value),
                    AtomicOp::Clr => old & !value,
                    AtomicOp::Eor => old ^ value,
                    AtomicOp::Set => old | value,
                    AtomicOp::Smax => {
                        if (sext(old, width) as i64) >= (sext(value, width) as i64) {
                            old
                        } else {
                            value
                        }
                    }
                    AtomicOp::Smin => {
                        if (sext(old, width) as i64) <= (sext(value, width) as i64) {
                            old
                        } else {
                            value
                        }
                    }
                    AtomicOp::Umax => old.max(value),
                    AtomicOp::Umin => old.min(value),
                    AtomicOp::Swp => value,
                };
                mem.write_uint(addr, bytes, new).map_err(fault)?;
                self.write_xzr(rt, old);
            }

            Instr::CompareSwap { size, rs, rn, rt } => {
                let addr = self.read_sp(rn);
                let bytes = 1usize << size;
                if !addr.is_multiple_of(bytes as u64) {
                    return Err(Exit::Alignment { pc, addr });
                }
                let old = mem.read_uint(addr, bytes).map_err(fault)?;
                if old == self.read_xzr(rs) & ones(8 << size) {
                    mem.write_uint(addr, bytes, self.read_xzr(rt))
                        .map_err(fault)?;
                }
                self.write_xzr(rs, old);
            }

            Instr::CompareSwapPair { sf, rs, rn, rt } => {
                let addr = self.read_sp(rn);
                let each = if sf { 8 } else { 4 };
                if !addr.is_multiple_of(each as u64 * 2) {
                    return Err(Exit::Alignment { pc, addr });
                }
                let mut buf = [0u8; 16];
                mem.read(addr, &mut buf[..each * 2]).map_err(fault)?;
                let word = |b: &[u8]| {
                    let mut w = [0u8; 8];
                    w[..each].copy_from_slice(b);
                    u64::from_le_bytes(w)
                };
                let (old0, old1) = (word(&buf[..each]), word(&buf[each..each * 2]));
                let mask = ones(each as u32 * 8);
                let (c0, c1) = (self.read_xzr(rs) & mask, self.read_xzr(rs + 1) & mask);
                if old0 == c0 && old1 == c1 {
                    buf[..each].copy_from_slice(&self.read_xzr(rt).to_le_bytes()[..each]);
                    buf[each..each * 2]
                        .copy_from_slice(&self.read_xzr(rt + 1).to_le_bytes()[..each]);
                    mem.write(addr, &buf[..each * 2]).map_err(fault)?;
                }
                self.write_xzr(rs, old0);
                self.write_xzr(rs + 1, old1);
            }

            Instr::ExclusivePair {
                sf,
                store,
                rn,
                rt,
                rt2,
            } => {
                let addr = self.read_sp(rn);
                let each = if sf { 8 } else { 4 };
                if !addr.is_multiple_of(each as u64 * 2) {
                    return Err(Exit::Alignment { pc, addr });
                }
                match store {
                    None => {
                        let mut buf = [0u8; 16];
                        mem.read(addr, &mut buf[..each * 2]).map_err(fault)?;
                        let mut w = [0u8; 8];
                        w[..each].copy_from_slice(&buf[..each]);
                        self.write_xzr(rt, u64::from_le_bytes(w));
                        w = [0; 8];
                        w[..each].copy_from_slice(&buf[each..each * 2]);
                        self.write_xzr(rt2, u64::from_le_bytes(w));
                        self.exclusive = Some(addr);
                    }
                    Some(rs) => {
                        let status = if self.exclusive == Some(addr) {
                            let mut buf = [0u8; 16];
                            buf[..each].copy_from_slice(&self.read_xzr(rt).to_le_bytes()[..each]);
                            buf[each..each * 2]
                                .copy_from_slice(&self.read_xzr(rt2).to_le_bytes()[..each]);
                            mem.write(addr, &buf[..each * 2]).map_err(fault)?;
                            0
                        } else {
                            1
                        };
                        self.exclusive = None;
                        self.write_xzr(rs, status);
                    }
                }
            }

            Instr::Unknown(raw) => return Err(Exit::Unimplemented { pc, insn: raw }),
        }

        self.set_pc(next_pc);
        Ok(())
    }

    /// Reads the element feeding DUP or INS.
    fn element_source(&self, src: ElemSrc, esize: u8) -> u64 {
        match src {
            ElemSrc::Gp(rn) => self.read_xzr(rn) & ones(8 << esize),
            ElemSrc::Element { rn, index } => get_element(self.read_v(rn), esize, index),
        }
    }

    /// Shared ADD/SUB core: `x + y` or `x - y`, returning result and flags.
    fn add_sub(&self, x: u64, y: u64, sub: bool, sf: bool) -> (u64, Nzcv) {
        if sub {
            add_with_carry(x, !y, true, sf)
        } else {
            add_with_carry(x, y, false, sf)
        }
    }

    /// Computes the address of a single-register load/store, and the
    /// base-register writeback if the mode has one.
    fn address(&self, mode: AddrMode, pc: u64) -> (u64, Option<(u8, u64)>) {
        match mode {
            AddrMode::Offset { rn, imm } => (self.read_sp(rn).wrapping_add_signed(imm), None),
            AddrMode::PreIndex { rn, imm } => {
                let addr = self.read_sp(rn).wrapping_add_signed(imm);
                (addr, Some((rn, addr)))
            }
            AddrMode::PostIndex { rn, imm } => {
                let base = self.read_sp(rn);
                (base, Some((rn, base.wrapping_add_signed(imm))))
            }
            AddrMode::RegOffset {
                rn,
                rm,
                extend,
                shift,
            } => {
                let offset = extend_reg(self.read_xzr(rm), extend, shift, 64);
                (self.read_sp(rn).wrapping_add(offset), None)
            }
            AddrMode::Literal { offset } => (pc.wrapping_add_signed(offset), None),
        }
    }
}

/// CRC32/CRC32C step over the low `bytes` bytes of `value`, as the ARM
/// instructions define it: reflected polynomial, no pre/post inversion.
fn crc32(acc: u32, value: u64, bytes: usize, castagnoli: bool) -> u64 {
    let poly: u32 = if castagnoli { 0x82F6_3B78 } else { 0xEDB8_8320 };
    let mut crc = acc;
    for i in 0..bytes {
        crc ^= u32::from((value >> (8 * i)) as u8);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ poly
            } else {
                crc >> 1
            };
        }
    }
    u64::from(crc)
}

/// Clears the upper 64 bits of a vector result when Q = 0.
fn vec_width(v: u128, q: bool) -> u128 {
    if q {
        v
    } else {
        v & u128::from(u64::MAX)
    }
}

/// Element `index` of a vector, each `1 << esize` bytes wide.
fn get_element(v: u128, esize: u8, index: u32) -> u64 {
    let bits = 8u32 << esize;
    ((v >> (index * bits)) as u64) & ones(bits)
}

fn set_element(v: u128, esize: u8, index: u32, value: u64) -> u128 {
    let bits = 8u32 << esize;
    let mask = u128::from(ones(bits)) << (index * bits);
    (v & !mask) | (u128::from(value & ones(bits)) << (index * bits))
}

fn logic(op: LogicOp, a: u64, b: u64) -> u64 {
    match op {
        LogicOp::And | LogicOp::Ands => a & b,
        LogicOp::Orr => a | b,
        LogicOp::Eor => a ^ b,
    }
}

/// Applies the zero/sign extension of a load to the raw loaded value.
fn extend_load(raw: u64, size: u8, kind: LoadKind) -> u64 {
    let width = 8u32 << size;
    match kind {
        LoadKind::LoadSigned64 => sext(raw, width),
        LoadKind::LoadSigned32 => sext(raw, width) & 0xFFFF_FFFF,
        _ => raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emu_memory::Perm;

    /// Assembles nothing at runtime: every encoding below was produced by
    /// `llvm-mc -triple=aarch64 -show-encoding`.
    const ADD_X0_X1_1: u32 = 0x9100_0420;
    const ADDS_W3_W4_4095: u32 = 0x313F_FC83;
    const SUB_SP_SP_32: u32 = 0xD100_83FF;
    const CMP_X5_7: u32 = 0xF100_1CBF;
    const MOVZ_X1_BEEF_LSL48: u32 = 0xD2F7_DDE1;
    const MOVN_W2_0: u32 = 0x1280_0002;
    const MOVK_X3_FFFF_LSL16: u32 = 0xF2BF_FFE3;
    const B_MINUS_4: u32 = 0x17FF_FFFF;
    const BL_256: u32 = 0x9400_0040;
    const RET: u32 = 0xD65F_03C0;
    const BLR_X9: u32 = 0xD63F_0120;
    const CBZ_X0_12: u32 = 0xB400_0060;
    const CBNZ_W1_MINUS_8: u32 = 0x35FF_FFC1;
    const SVC_0X21: u32 = 0xD400_0421;
    const STR_X1_X0_8: u32 = 0xF900_0401; // str x1, [x0, #8]
    const LDRSB_W2_X0_8: u32 = 0x39C0_2002; // ldrsb w2, [x0, #8]
    const STP_X1_X2_SP_PRE: u32 = 0xA9BF_0BE1; // stp x1, x2, [sp, #-16]!
    const LDP_X3_X4_SP_POST: u32 = 0xA8C1_13E3; // ldp x3, x4, [sp], #16

    fn exec(cpu: &mut Cpu, insn: u32) -> Result<(), Exit> {
        let mut mem = Memory::new();
        cpu.execute(insn, &mut mem)
    }

    fn cpu_at(pc: u64) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.set_pc(pc);
        cpu
    }

    #[test]
    fn add_immediate_advances_pc() {
        let mut cpu = cpu_at(0x1000);
        cpu.write_xzr(1, 41);
        exec(&mut cpu, ADD_X0_X1_1).unwrap();
        assert_eq!(cpu.read_xzr(0), 42);
        assert_eq!(cpu.pc(), 0x1004);
        assert_eq!(cpu.nzcv, Nzcv::default(), "ADD must not touch flags");
    }

    #[test]
    fn adds_32bit_carry_and_overflow() {
        let mut cpu = Cpu::new();
        cpu.write_xzr(4, 0xFFFF_F001);
        exec(&mut cpu, ADDS_W3_W4_4095).unwrap();
        assert_eq!(cpu.read_xzr(3), 0);
        assert_eq!(
            cpu.nzcv,
            Nzcv {
                n: false,
                z: true,
                c: true,
                v: false
            }
        );

        cpu.write_xzr(4, 0x7FFF_FFFF);
        exec(&mut cpu, ADDS_W3_W4_4095).unwrap();
        assert_eq!(cpu.read_xzr(3), 0x8000_0FFE);
        assert_eq!(
            cpu.nzcv,
            Nzcv {
                n: true,
                z: false,
                c: false,
                v: true
            }
        );
    }

    #[test]
    fn sub_and_cmp() {
        let mut cpu = Cpu::new();
        cpu.write_sp(31, 0x8000);
        exec(&mut cpu, SUB_SP_SP_32).unwrap();
        assert_eq!(cpu.sp(), 0x8000 - 32);

        cpu.write_xzr(5, 3);
        exec(&mut cpu, CMP_X5_7).unwrap();
        assert_eq!(
            cpu.nzcv,
            Nzcv {
                n: true,
                z: false,
                c: false,
                v: false
            }
        );
        assert_eq!(cpu.sp(), 0x8000 - 32, "CMP writes XZR, not SP");
    }

    #[test]
    fn move_wide_variants() {
        let mut cpu = Cpu::new();
        exec(&mut cpu, MOVZ_X1_BEEF_LSL48).unwrap();
        assert_eq!(cpu.read_xzr(1), 0xBEEF_0000_0000_0000);
        cpu.write_xzr(2, u64::MAX);
        exec(&mut cpu, MOVN_W2_0).unwrap();
        assert_eq!(cpu.read_xzr(2), 0xFFFF_FFFF, "W writes zero the top half");
        cpu.write_xzr(3, 0x1111_2222_3333_4444);
        exec(&mut cpu, MOVK_X3_FFFF_LSL16).unwrap();
        assert_eq!(cpu.read_xzr(3), 0x1111_2222_FFFF_4444);
    }

    #[test]
    fn calls_and_branches() {
        let mut cpu = cpu_at(0x2000);
        exec(&mut cpu, BL_256).unwrap();
        assert_eq!((cpu.pc(), cpu.read_xzr(30)), (0x2100, 0x2004));
        exec(&mut cpu, RET).unwrap();
        assert_eq!(cpu.pc(), 0x2004);
        exec(&mut cpu, B_MINUS_4).unwrap();
        assert_eq!(cpu.pc(), 0x2000);

        cpu.write_xzr(9, 0x4000);
        exec(&mut cpu, BLR_X9).unwrap();
        assert_eq!((cpu.pc(), cpu.read_xzr(30)), (0x4000, 0x2004));
    }

    #[test]
    fn compare_and_branch() {
        let mut cpu = cpu_at(0x100);
        exec(&mut cpu, CBZ_X0_12).unwrap();
        assert_eq!(cpu.pc(), 0x10C);
        cpu.set_pc(0x100);
        cpu.write_xzr(1, 0x1_0000_0000);
        exec(&mut cpu, CBNZ_W1_MINUS_8).unwrap();
        assert_eq!(cpu.pc(), 0x104, "W1 is zero, fall through");
    }

    #[test]
    fn svc_advances_pc_and_reports() {
        let mut cpu = cpu_at(0x100);
        assert_eq!(exec(&mut cpu, SVC_0X21), Err(Exit::Svc(0x21)));
        assert_eq!(cpu.pc(), 0x104);
    }

    #[test]
    fn unknown_instruction_keeps_pc() {
        let mut cpu = cpu_at(0x100);
        assert_eq!(
            exec(&mut cpu, 0),
            Err(Exit::Unimplemented { pc: 0x100, insn: 0 })
        );
        assert_eq!(cpu.pc(), 0x100);
    }

    #[test]
    fn loads_and_stores() {
        let mut mem = Memory::new();
        mem.map(0x10000, 0x1000, Perm::RW).unwrap();
        let mut cpu = Cpu::new();
        cpu.write_xzr(0, 0x10000);
        cpu.write_xzr(1, 0x0000_0000_0000_0080);
        cpu.execute(STR_X1_X0_8, &mut mem).unwrap();
        assert_eq!(mem.read_u64(0x10008).unwrap(), 0x80);
        cpu.execute(LDRSB_W2_X0_8, &mut mem).unwrap();
        assert_eq!(
            cpu.read_xzr(2),
            0xFFFF_FF80,
            "sign-extended to 32 bits only"
        );

        cpu.write_sp(31, 0x10100);
        cpu.write_xzr(2, 0x2222);
        cpu.execute(STP_X1_X2_SP_PRE, &mut mem).unwrap();
        assert_eq!(cpu.sp(), 0x100F0);
        cpu.execute(LDP_X3_X4_SP_POST, &mut mem).unwrap();
        assert_eq!((cpu.read_xzr(3), cpu.read_xzr(4)), (0x80, 0x2222));
        assert_eq!(cpu.sp(), 0x10100);
    }

    #[test]
    fn faulting_store_has_no_effect() {
        let mut mem = Memory::new();
        let mut cpu = cpu_at(0x40);
        cpu.write_xzr(0, 0x5000);
        let exit = cpu.execute(STR_X1_X0_8, &mut mem).unwrap_err();
        assert!(matches!(exit, Exit::Fault { pc: 0x40, .. }));
        assert_eq!(cpu.pc(), 0x40);
    }

    #[test]
    fn crc32_matches_standard_check_values() {
        // The standard check value of CRC-32 over "123456789" is 0xCBF43926
        // and of CRC-32C 0xE3069283, with the usual ~0 pre/post inversion.
        let run = |castagnoli| {
            let mut crc = u32::MAX;
            for b in b"123456789" {
                crc = crc32(crc, u64::from(*b), 1, castagnoli) as u32;
            }
            !crc
        };
        assert_eq!(run(false), 0xCBF4_3926);
        assert_eq!(run(true), 0xE306_9283);
    }

    #[test]
    fn condition_codes() {
        let f = |n, z, c, v| Nzcv { n, z, c, v };
        assert!(condition_holds(0b0000, f(false, true, false, false))); // EQ
        assert!(condition_holds(0b0001, f(false, false, false, false))); // NE
        assert!(condition_holds(0b1000, f(false, false, true, false))); // HI
        assert!(!condition_holds(0b1000, f(false, true, true, false))); // HI with Z
        assert!(condition_holds(0b1011, f(true, false, false, false))); // LT
        assert!(condition_holds(0b1110, f(true, true, true, true))); // AL
        assert!(condition_holds(0b1111, f(false, false, false, false))); // NV acts as AL
    }
}
