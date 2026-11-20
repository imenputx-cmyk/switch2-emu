//! Switch 2 (Ampere, SM86) shader instructions, decoded into the same IR
//! as Maxwell ones so the interpreter and the SPIR-V translation serve both.
//!
//! A graphics program is a 0x80-byte header (the SPH, as on Maxwell for
//! the fields used here) followed by 128-bit instructions; compute
//! programs have no header. Scheduling information lives inside each
//! instruction, so there are no separate control words.
//!
//! Fields (bit offsets into the 128-bit instruction):
//!
//! | bits    | field |
//! |---------|-------|
//! | 0..12   | opcode; bits 9..12 are the operand form |
//! | 12..16  | guard predicate (bit 15 negates; 7 is PT) |
//! | 16..24  | destination register |
//! | 24..32  | source A (register) |
//! | 32..40 / 32..64 | source B register / 32-bit immediate |
//! | 38..54, 54..59 | constant buffer offset (bytes) and bank |
//! | 64..72  | source C register |
//!
//! Forms: 1 B and C registers; 2 C immediate (B moves to 64); 3 C
//! constant (B at 64); 4 B immediate; 5 B constant; 6/7 uniform registers.
//!
//! Encodings follow Mesa NAK's `sm70_encode.rs` (MIT) and Lichii/Eden's
//! `lower_to_maxwell.h` (GPL-3.0-or-later), which lowers these same
//! instructions to Maxwell ones.

use super::decode::{
    Bop, DecodeError, Inst, IpaMode, Mods, MufuOp, Op, Operand, Pred, Round, PT, RZ,
};

/// Bytes of a graphics program header before the first instruction.
pub const HEADER_SIZE: usize = 0x80;

/// Uniform registers UR0-UR62 live in R191-R253 (URZ is RZ): the IR has
/// one register file. A program whose own registers reach R191 cannot use
/// uniforms this way and is reported.
pub const UNIFORM_BASE: u8 = 191;

/// The IR register of uniform register `n`.
pub fn ureg(n: u32) -> u8 {
    match n {
        63.. => RZ,
        n => UNIFORM_BASE + n as u8,
    }
}

/// NVN2's root tables: constant banks 24-31, 256 bytes each, which the IR
/// reads as bank 17 (unused by games, as Lichii notes) at (bank - 24) * 256.
pub const ROOT_TABLE_BANK: u8 = 17;

/// Uniform-datapath opcodes (registers are URs): UMOV, UIADD3, ULOP3,
/// UIMAD, USHF, ULDC.
fn is_uniform_op(op: u32) -> bool {
    matches!(
        op,
        0x882
            | 0xC82
            | 0x290
            | 0x890
            | 0x292
            | 0x892
            | 0x2A4
            | 0x4A4
            | 0x8A4
            | 0x299
            | 0x899
            | 0xAB9
            | 0xABB
    )
}

/// Whether an instruction reads or writes uniform registers.
fn uses_uniforms(op: u32) -> bool {
    is_uniform_op(op) || matches!(op >> 9, 6 | 7) || op == 0x3C2 || op == 0xC02
}

/// The highest ordinary register a program names (fields 16, 24, 64, and
/// 32 in register forms), ignoring uniform-datapath instructions.
fn highest_register(insts: &[Instruction]) -> u8 {
    let mut top = 0u8;
    for i in insts {
        let op = i.opcode();
        if is_uniform_op(op) {
            continue;
        }
        let mut fields = vec![16, 24, 64];
        if op >> 9 == 1 {
            fields.push(32);
        }
        for f in fields {
            let r = i.reg(f);
            if r != RZ {
                top = top.max(r);
            }
        }
    }
    top
}

/// One 128-bit instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction(pub [u32; 4]);

impl Instruction {
    pub fn bits(&self, offset: u32, count: u32) -> u32 {
        let w = (offset / 32) as usize;
        let mut v = u64::from(self.0[w]);
        if w + 1 < 4 {
            v |= u64::from(self.0[w + 1]) << 32;
        }
        ((v >> (offset % 32)) & ((1u64 << count) - 1)) as u32
    }

    fn on(&self, bit: u32) -> bool {
        self.bits(bit, 1) != 0
    }

    fn opcode(&self) -> u32 {
        self.bits(0, 12)
    }

    fn reg(&self, offset: u32) -> u8 {
        self.bits(offset, 8) as u8
    }

    fn pred_at(&self, offset: u32) -> Pred {
        Pred {
            index: self.bits(offset, 3) as u8,
            neg: self.on(offset + 3),
        }
    }

    /// A branch-style target: instruction index (byte offset relative to
    /// the next instruction, in words 1 and 2).
    fn target(&self, index: usize) -> Option<usize> {
        let raw = u64::from(self.0[1] & !3) | (u64::from(self.0[2] & 0x3_FFFF) << 32);
        let relative = ((raw << 14) as i64) >> 14;
        let byte = index as i64 * 16 + 16 + relative;
        (byte >= 0 && byte % 16 == 0).then_some((byte / 16) as usize)
    }
}

/// An instruction that is not decoded (yet).
fn native(index: usize, i: &Instruction, what: &'static str) -> DecodeError {
    DecodeError::Native {
        index,
        opcode: i.opcode(),
        what,
    }
}

/// Decodes 128-bit instructions (after the header) until the program's
/// end. Branch targets in the result are IR indices.
pub fn decode_program(code: &[u8]) -> Result<Vec<Inst>, DecodeError> {
    // Each native instruction becomes one or more IR instructions; targets
    // are native indices until all are decoded.
    let mut lowered: Vec<Vec<Inst>> = Vec::new();
    let mut furthest = 0usize;
    let all: Vec<Instruction> = code
        .chunks_exact(16)
        .map(|chunk| {
            Instruction(std::array::from_fn(|i| {
                u32::from_le_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap())
            }))
        })
        .collect();
    for (index, &inst) in all.iter().enumerate() {
        let insts = decode(inst, index)?;
        for inst in &insts {
            if let Op::Bra { target } | Op::Ssy { target } = inst.op {
                furthest = furthest.max(target);
            }
        }
        let terminal = insts.last().is_some_and(Inst::is_terminal);
        lowered.push(insts);
        if terminal && index >= furthest {
            break;
        }
    }
    // Registers past the program's own hold the uniforms.
    let program = &all[..lowered.len()];
    if highest_register(program) >= UNIFORM_BASE {
        if let Some(index) = program.iter().position(|i| {
            let op = i.opcode();
            uses_uniforms(op)
        }) {
            return Err(native(
                index,
                &program[index],
                "uniform registers with R191 and up in use",
            ));
        }
    }
    let mut first = Vec::with_capacity(lowered.len() + 1);
    let mut n = 0;
    for insts in &lowered {
        first.push(n);
        n += insts.len();
    }
    first.push(n);
    let mut out = Vec::with_capacity(n);
    for (index, insts) in lowered.into_iter().enumerate() {
        for mut inst in insts {
            if let Op::Bra { target } | Op::Ssy { target } = &mut inst.op {
                *target = *first.get(*target).ok_or(DecodeError::Native {
                    index,
                    opcode: 0,
                    what: "branch past the program",
                })?;
            }
            out.push(inst);
        }
    }
    Ok(out)
}

/// Decodes one instruction (at native index `index`).
pub fn decode(i: Instruction, index: usize) -> Result<Vec<Inst>, DecodeError> {
    let op = i.opcode();
    if is_uniform_op(op) {
        return decode_uniform(i, index);
    }
    let form = op >> 9;
    let pred = i.pred_at(12);
    let dst = i.reg(16);
    let a = i.reg(24);
    let err = |what| Err(native(index, &i, what));
    let one = |op: Op| Ok(vec![Inst { pred, op }]);

    // Source A modifiers (float ops).
    let a_mods = Mods {
        abs: i.on(73),
        neg: i.on(72),
    };
    let cbuf = || cbuf_operand(&i, index);
    // Source B and its modifiers by form.
    let src_b = || -> Result<(Operand, Mods), DecodeError> {
        let m = |abs, neg| Mods {
            abs: i.on(abs),
            neg: i.on(neg),
        };
        Ok(match form {
            1 => (Operand::Reg(i.reg(32)), m(62, 63)),
            2 | 3 => (Operand::Reg(i.reg(64)), m(74, 75)),
            4 => (Operand::Imm(i.0[1]), Mods::default()),
            5 => (cbuf()?, m(62, 63)),
            6 => (Operand::Reg(ureg(i.bits(32, 8))), m(62, 63)),
            _ => (Operand::Reg(i.reg(64)), m(74, 75)), // 7: C is the uniform
        })
    };
    // Source C and its modifiers by form.
    let src_c = || -> Result<(Operand, Mods), DecodeError> {
        let m = |abs, neg| Mods {
            abs: i.on(abs),
            neg: i.on(neg),
        };
        Ok(match form {
            1 | 4 | 5 => (Operand::Reg(i.reg(64)), m(74, 75)),
            2 => (Operand::Imm(i.0[1]), Mods::default()),
            3 => (cbuf()?, m(62, 63)),
            6 => (Operand::Reg(i.reg(64)), m(74, 75)),
            _ => (Operand::Reg(ureg(i.bits(32, 8))), m(62, 63)), // 7
        })
    };
    let bop = |at| match i.bits(at, 2) {
        0 => Ok(Bop::And),
        1 => Ok(Bop::Or),
        2 => Ok(Bop::Xor),
        _ => Err(native(index, &i, "predicate combination")),
    };

    match op & 0x1FF {
        // MOV (lane mask 72..76 must be all lanes).
        0x002 => {
            if i.bits(72, 4) != 15 {
                return err("MOV lane mask");
            }
            one(Op::Mov {
                dst,
                src: src_b()?.0,
            })
        }
        // FADD: B sits in C's place when C is the wide operand.
        0x021 => {
            let (b, b_mods) = if matches!(form, 2 | 3 | 7) {
                src_c()?
            } else {
                src_b()?
            };
            one(Op::Fadd {
                dst,
                a,
                a_mods,
                b,
                b_mods,
                sat: i.on(77),
            })
        }
        // FMUL (bits 84..87: result scale, 4 = none).
        0x020 => {
            let (b, b_mods) = src_b()?;
            if i.bits(84, 3) != 4 || a_mods.abs || b_mods.abs {
                return err("FMUL scale or absolute value");
            }
            one(Op::Fmul {
                dst,
                a,
                b,
                neg_b: a_mods.neg != b_mods.neg,
                sat: i.on(77),
            })
        }
        // FFMA: the product's sign folds into B.
        0x023 => {
            let (b, b_mods) = src_b()?;
            let (c, c_mods) = src_c()?;
            if a_mods.abs || b_mods.abs || c_mods.abs {
                return err("FFMA absolute value");
            }
            one(Op::Ffma {
                dst,
                a,
                b,
                neg_b: a_mods.neg != b_mods.neg,
                c,
                neg_c: c_mods.neg,
                sat: i.on(77),
            })
        }
        // FMNMX: min when the predicate at 87 holds.
        0x009 => {
            let (b, b_mods) = src_b()?;
            one(Op::Fmnmx {
                dst,
                a,
                a_mods,
                b,
                b_mods,
                min: i.pred_at(87),
            })
        }
        // MUFU on source B; SIN/COS take turns, the IR takes radians.
        0x108 => {
            if i.on(73) {
                return err("MUFU half source");
            }
            let (src, mods) = src_b()?;
            let func = match i.bits(74, 6) {
                0 => MufuOp::Cos,
                1 => MufuOp::Sin,
                2 => MufuOp::Ex2,
                3 => MufuOp::Lg2,
                4 => MufuOp::Rcp,
                5 => MufuOp::Rsq,
                8 => MufuOp::Sqrt,
                _ => return err("MUFU function"),
            };
            let mut out = vec![Inst {
                pred,
                op: Op::Mov { dst, src },
            }];
            if matches!(func, MufuOp::Cos | MufuOp::Sin) {
                out.push(Inst {
                    pred,
                    op: Op::Fmul {
                        dst,
                        a: dst,
                        b: Operand::Imm(std::f32::consts::TAU.to_bits()),
                        neg_b: false,
                        sat: false,
                    },
                });
            }
            out.push(Inst {
                pred,
                op: Op::Mufu {
                    dst,
                    func,
                    a: dst,
                    mods,
                    sat: false,
                },
            });
            Ok(out)
        }
        // FSETP / ISETP: P at 81, Q at 84, combined with the predicate at
        // 87 by bits 74..76.
        0x00B => {
            let (b, b_mods) = src_b()?;
            one(Op::Fsetp {
                dst: i.bits(81, 3) as u8,
                dst2: i.bits(84, 3) as u8,
                cmp: i.bits(76, 4) as u8,
                a,
                a_mods,
                b,
                b_mods,
                bop: bop(74)?,
                c: i.pred_at(87),
            })
        }
        0x00C => {
            if i.on(72) {
                return err("ISETP.EX");
            }
            one(Op::Isetp {
                dst: i.bits(81, 3) as u8,
                dst2: i.bits(84, 3) as u8,
                cmp: i.bits(76, 3) as u8,
                signed: i.on(73),
                a,
                b: src_b()?.0,
                bop: bop(74)?,
                c: i.pred_at(87),
            })
        }
        // FSET.BF: 1.0 or 0.0 into a register.
        0x00A => {
            let (b, b_mods) = src_b()?;
            one(Op::Fset {
                dst,
                cmp: i.bits(76, 4) as u8,
                a,
                a_mods,
                b,
                b_mods,
                bop: bop(74)?,
                c: i.pred_at(87),
                bf: true,
            })
        }
        // SEL / FSEL: A when the predicate at 87 holds, else B.
        0x007 | 0x008 => {
            let (b, b_mods) = src_b()?;
            if op & 0x1FF == 0x008 && (a_mods != Mods::default() || b_mods != Mods::default()) {
                return err("FSEL modifiers");
            }
            one(Op::Sel {
                dst,
                a,
                b,
                cond: i.pred_at(87),
            })
        }
        // LOP3.LUT, optionally P = result != 0.
        0x012 => {
            if i.on(80) || i.bits(87, 4) != 15 {
                return err("LOP3 predicate input");
            }
            one(Op::Lop3 {
                dst,
                a,
                b: src_b()?.0,
                c: Operand::Reg(i.reg(64)),
                lut: i.bits(72, 8) as u8,
                pred: i.bits(81, 3) as u8,
            })
        }
        // IADD3 without carries.
        0x010 => {
            if i.on(74) || i.bits(81, 3) != 7 || i.bits(84, 3) != 7 {
                return err("IADD3 carry");
            }
            let (b, b_mods) = src_b()?;
            let (c, c_mods) = src_c()?;
            one(Op::Iadd3 {
                dst,
                a,
                neg_a: i.on(72),
                b,
                neg_b: b_mods.neg,
                c,
                neg_c: c_mods.neg,
            })
        }
        // IMAD (low word). Bit 75 negates the addend.
        0x024 => {
            if i.on(74) {
                return err("IMAD.X");
            }
            let c = src_c()?.0;
            one(Op::Imad {
                dst,
                a,
                b: src_b()?.0,
                c,
                neg_c: i.on(75),
            })
        }
        // SHF: type at 73..75 (0 S64, 1 U64, 2 S32, 3 U32).
        0x019 => {
            let kind = i.bits(73, 2);
            one(Op::Shf {
                dst,
                a,
                b: src_b()?.0,
                c: Operand::Reg(i.reg(64)),
                right: i.on(76),
                hi: i.on(80),
                wrap: i.on(75),
                wide: kind < 2,
                signed: kind.is_multiple_of(2),
            })
        }
        // I2FP.F32.{U32,S32}.
        0x045 => one(Op::I2f {
            dst,
            src: src_b()?.0,
            signed: i.on(74),
            neg: false,
            abs: false,
        }),
        // F2I to a 32-bit integer from F32 (bits 72 signed, 78..80 rounding).
        0x105 => {
            if i.0[2] & !0x1_C100 != 0x0020_3000 {
                return err("F2I types");
            }
            let (src, mods) = src_b()?;
            let round = match i.bits(78, 2) {
                0 => Round::Nearest,
                1 => Round::Floor,
                2 => Round::Ceil,
                _ => Round::Trunc,
            };
            one(Op::F2i {
                dst,
                src,
                mods,
                signed: i.on(72),
                round,
            })
        }
        // IPA: attribute word at 64..72, flat (constant) at 78.
        0x126 if form == 1 => {
            if i.bits(81, 3) != 7 || i.bits(76, 2) > 1 {
                return err("IPA mode");
            }
            let mode = if i.on(78) {
                IpaMode::Constant
            } else if i.reg(32) != RZ {
                IpaMode::Multiply
            } else {
                IpaMode::Pass
            };
            one(Op::Ipa {
                dst,
                addr: i.bits(64, 8) * 4,
                mode,
                sat: false,
            })
        }
        // ALD / AST of 1-4 words; patch and output forms are not decoded.
        0x121 | 0x122 if form == 1 => {
            if i.on(76) || i.on(77) || i.on(79) {
                return err("ALD/AST patch or output");
            }
            let (addr, count) = (i.bits(40, 10), i.bits(74, 2) + 1);
            one(if op & 0x1FF == 0x121 {
                Op::Ald {
                    dst,
                    base: a,
                    addr,
                    count,
                }
            } else {
                Op::Ast {
                    src: i.reg(32),
                    base: a,
                    addr,
                    count,
                }
            })
        }
        _ => match op {
            // MOV R, UR (the source in bits 32..38).
            0xC02 => one(Op::Mov {
                dst,
                src: Operand::Reg(ureg(i.bits(32, 6))),
            }),
            // R2UR: one invocation at a time, the value itself.
            0x3C2 => one(Op::Mov {
                dst: ureg(u32::from(dst)),
                src: Operand::Reg(a),
            }),
            0x94D => one(Op::Exit),
            0x95B => one(Op::Kil),
            // NOP, WARPSYNC, DEPBAR, MEMBAR: nothing to do one invocation at a time.
            0x918 | 0x948 | 0x91A | 0x992 => one(Op::Nop),
            // BRA (its own predicate at 87 must be PT).
            0x947 => {
                if i.pred_at(87) != PT {
                    return err("BRA condition");
                }
                let target = i.target(index).ok_or(native(index, &i, "branch target"))?;
                one(Op::Bra { target })
            }
            // BSSY / BSYNC: reconvergence like SSY / SYNC.
            0x945 => {
                let target = i.target(index).ok_or(native(index, &i, "BSSY target"))?;
                one(Op::Ssy { target })
            }
            0x941 => one(Op::Sync),
            _ => err("opcode not decoded yet"),
        },
    }
}

/// A constant buffer operand: banks 0-17, or NVN2 root tables (24-31,
/// first 256 bytes) read from [`ROOT_TABLE_BANK`].
fn cbuf_operand(i: &Instruction, index: usize) -> Result<Operand, DecodeError> {
    let bank = i.bits(54, 5);
    let offset = i.bits(38, 16);
    if i.on(91) {
        return Err(native(index, i, "indexed constant buffer"));
    }
    match bank {
        0..=17 => Ok(Operand::Cbuf {
            index: bank as u8,
            offset,
        }),
        24..=31 if offset < 256 => Ok(Operand::Cbuf {
            index: ROOT_TABLE_BANK,
            offset: (bank - 24) * 256 + offset,
        }),
        _ => Err(native(index, i, "constant bank")),
    }
}

/// Uniform-datapath instructions: their registers are URs. The ALU ones
/// are the ordinary instruction with uniform registers (Lichii rewrites
/// them the same way); the guard must be UPT.
fn decode_uniform(i: Instruction, index: usize) -> Result<Vec<Inst>, DecodeError> {
    let op = i.opcode();
    if i.bits(12, 4) != 7 {
        return Err(native(index, &i, "uniform predicate guard"));
    }
    let dst = ureg(i.bits(16, 8));
    let one = |op: Op| Ok(vec![Inst::always(op)]);
    match op {
        0x882 => one(Op::Mov {
            dst,
            src: Operand::Imm(i.0[1]),
        }),
        0xC82 => one(Op::Mov {
            dst,
            src: Operand::Reg(ureg(i.bits(32, 6))),
        }),
        // ULDC: 32 bits (width 4) or a 64-bit pair (5), direct.
        0xAB9 => {
            let width = i.bits(73, 3);
            let mut plain = i;
            plain.0[2] &= !(1 << 27); // bit 91: the uniform datapath
            let src = cbuf_operand(&plain, index)?;
            let Operand::Cbuf {
                index: bank,
                offset,
            } = src
            else {
                unreachable!()
            };
            match width {
                4 => one(Op::Mov { dst, src }),
                5 if dst != RZ => Ok(vec![
                    Inst::always(Op::Mov { dst, src }),
                    Inst::always(Op::Mov {
                        dst: dst + 1,
                        src: Operand::Cbuf {
                            index: bank,
                            offset: offset + 4,
                        },
                    }),
                ]),
                _ => Err(native(index, &i, "ULDC width")),
            }
        }
        0xABB => Err(native(index, &i, "indexed ULDC")),
        // UIADD3, ULOP3, UIMAD, USHF: the ordinary opcode with the
        // registers rewritten to their IR homes.
        _ => {
            let mut r = i;
            let set = |r: &mut Instruction, offset: u32, value: u8| {
                let w = (offset / 32) as usize;
                let shift = offset % 32;
                r.0[w] = (r.0[w] & !(0xFF << shift)) | (u32::from(value) << shift);
            };
            set(&mut r, 16, ureg(i.bits(16, 8)));
            set(&mut r, 24, ureg(i.bits(24, 8)));
            set(&mut r, 64, ureg(i.bits(64, 8)));
            if op >> 9 == 1 {
                set(&mut r, 32, ureg(i.bits(32, 8)));
            }
            // ULOP3's predicate output is a uniform predicate.
            if op & 0x1FF == 0x092 {
                if i.bits(81, 3) != 7 {
                    return Err(native(index, &i, "ULOP3 predicate output"));
                }
                r.0[2] |= 0xF << 23; // input PT (bits 87..91)
            }
            r.0[0] &= !0x80; // the ordinary opcode
            r.0[2] &= !(1 << 27); // bit 91
            decode(r, index)
        }
    }
}

/// Builds SM86 instructions for tests.
pub mod asm {
    use super::Instruction;

    /// An instruction from its opcode, guard (7 = always), destination and
    /// fields given as (offset, count, value).
    pub fn inst(opcode: u32, dst: u8, fields: &[(u32, u32, u32)]) -> Instruction {
        let mut v: u128 = u128::from(opcode) | (7 << 12) | (u128::from(dst) << 16);
        for &(offset, count, value) in fields {
            let mask = (1u128 << count) - 1;
            v = (v & !(mask << offset)) | ((u128::from(value) & mask) << offset);
        }
        Instruction(std::array::from_fn(|i| (v >> (32 * i)) as u32))
    }

    /// Serializes instructions after an empty header.
    pub fn program(insts: &[Instruction]) -> Vec<u8> {
        let mut out = vec![0u8; super::HEADER_SIZE];
        for i in insts {
            for w in i.0 {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    pub fn exit() -> Instruction {
        inst(0x94D, 0, &[])
    }

    /// A branch from instruction `at` to `to` (indices).
    pub fn bra(opcode: u32, at: usize, to: usize) -> Instruction {
        let rel = (to as i64 - at as i64 - 1) * 16;
        let raw = (rel as u64) & ((1 << 50) - 1);
        inst(
            opcode,
            0,
            &[
                (32, 32, raw as u32),
                (64, 18, (raw >> 32) as u32),
                (87, 3, 7),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::asm::*;
    use super::*;
    use crate::shader::interp::{run, ShaderIo};

    struct Io;
    impl ShaderIo for Io {
        fn load_attr(&self, _: u32) -> u32 {
            0
        }
        fn store_attr(&mut self, _: u32, _: u32) {}
        fn interpolate(&self, _: u32, _: IpaMode) -> f32 {
            0.0
        }
        fn cbuf(&self, _: u8, _: u32) -> u32 {
            0
        }
    }

    struct Out([u32; 256]);
    impl Out {
        fn get(&self, r: u8) -> u32 {
            self.0[r as usize]
        }
    }

    /// Runs the program with registers preset by MOVs put in front.
    fn exec(insts: &[Instruction], init: &[(u8, u32)]) -> Out {
        let code = program(insts);
        let decoded = decode_program(&code[HEADER_SIZE..]).expect("decodes");
        let k = init.len();
        let mut program: Vec<Inst> = init
            .iter()
            .map(|&(dst, v)| {
                Inst::always(Op::Mov {
                    dst,
                    src: Operand::Imm(v),
                })
            })
            .collect();
        for mut inst in decoded {
            if let Op::Bra { target } | Op::Ssy { target } = &mut inst.op {
                *target += k;
            }
            program.push(inst);
        }
        Out(run(&program, &mut Io).expect("runs").0)
    }

    fn f(v: f32) -> u32 {
        v.to_bits()
    }

    #[test]
    fn float_alu_forms() {
        let r = exec(
            &[
                // FFMA R2 = R0 * R1 + 1.5 (form 2: C immediate, B at 64)
                inst(0x423, 2, &[(24, 8, 0), (64, 8, 1), (32, 32, f(1.5))]),
                // FADD R3 = -R2 + R1
                inst(0x221, 3, &[(24, 8, 2), (32, 8, 1), (72, 1, 1)]),
                // FMUL R4 = R0 * 4.0 (scale none)
                inst(0x820, 4, &[(24, 8, 0), (32, 32, f(4.0)), (84, 3, 4)]),
                // MUFU.RCP R5 = 1 / R1
                inst(0x308, 5, &[(32, 8, 1), (74, 6, 4)]),
                exit(),
            ],
            &[(0, f(2.0)), (1, f(3.0))],
        );
        assert_eq!(f32::from_bits(r.get(2)), 7.5);
        assert_eq!(f32::from_bits(r.get(3)), -4.5);
        assert_eq!(f32::from_bits(r.get(4)), 8.0);
        assert!((f32::from_bits(r.get(5)) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn integer_alu_and_shifts() {
        let r = exec(
            &[
                // IADD3 R2 = R0 + 0x10 + R1 (form 4)
                inst(
                    0x810,
                    2,
                    &[(24, 8, 0), (32, 32, 0x10), (64, 8, 1), (81, 6, 0x3F)],
                ),
                // IMAD R3 = R0 * R1 + R2
                inst(0x224, 3, &[(24, 8, 0), (32, 8, 1), (64, 8, 2)]),
                // LOP3 R4 = R0 & R1 | ~R0 & RZ  (lut 0xC0 = a & b), P0 = R4 != 0
                inst(
                    0x212,
                    4,
                    &[
                        (24, 8, 0),
                        (32, 8, 1),
                        (64, 8, 255),
                        (72, 8, 0xC0),
                        (81, 3, 0),
                        (87, 4, 15),
                    ],
                ),
                // SHF.L.U32.HI R5 = high word of (R1:R0) << 4 (form 4, type U32)
                inst(
                    0x819,
                    5,
                    &[(24, 8, 0), (32, 32, 4), (64, 8, 1), (73, 2, 3), (80, 1, 1)],
                ),
                // SHF.R.U32 R6 = (RZ:R1) >> 1
                inst(
                    0x819,
                    6,
                    &[
                        (24, 8, 1),
                        (32, 32, 1),
                        (64, 8, 255),
                        (73, 2, 3),
                        (76, 1, 1),
                    ],
                ),
                // ISETP.GT.U32 P1 = R1 > R0 (cmp 4), combined AND with PT
                inst(
                    0x20C,
                    0,
                    &[
                        (24, 8, 1),
                        (32, 8, 0),
                        (76, 3, 4),
                        (81, 3, 1),
                        (84, 3, 7),
                        (87, 3, 7),
                    ],
                ),
                // @P1 MOV R7 = 0x55
                inst(0x802, 7, &[(12, 4, 1), (32, 32, 0x55), (72, 4, 15)]),
                exit(),
            ],
            &[(0, 0xF000_0006), (1, 7)],
        );
        assert_eq!(r.get(2), 0xF000_001D);
        assert_eq!(
            r.get(3),
            0xF000_0006u32.wrapping_mul(7).wrapping_add(0xF000_001D)
        );
        assert_eq!(r.get(4), 6);
        assert_eq!(r.get(5), (7 << 4) | 0xF);
        assert_eq!(r.get(6), 3);
        assert_eq!(r.get(7), 0); // 7 > 0xF0000006 is false
    }

    #[test]
    fn branches_and_reconvergence() {
        // 0: BSSY -> 4
        // 1: ISETP.EQ P0 = R0 == 0
        // 2: @P0 BRA 4... keep simple: @!P0 MOV R1 = 1
        // 3: BSYNC
        // 4: MOV R2 = 2 ; 5: EXIT
        let mut mov1 = inst(0x802, 1, &[(32, 32, 1), (72, 4, 15)]);
        mov1.0[0] = (mov1.0[0] & !0xF000) | (8 << 12); // @!P0
        let code = [
            bra(0x945, 0, 4),
            inst(
                0x80C,
                0,
                &[
                    (24, 8, 0),
                    (32, 32, 0),
                    (76, 3, 2),
                    (81, 3, 0),
                    (84, 3, 7),
                    (87, 3, 7),
                ],
            ),
            mov1,
            inst(0x941, 0, &[]),
            inst(0x802, 2, &[(32, 32, 2), (72, 4, 15)]),
            exit(),
        ];
        let r = exec(&code, &[(0, 0)]);
        assert_eq!((r.get(1), r.get(2)), (0, 2));
        let r = exec(&code, &[(0, 5)]);
        assert_eq!((r.get(1), r.get(2)), (1, 2));
        // A forward BRA skips an instruction.
        let r = exec(
            &[
                bra(0x947, 0, 2),
                inst(0x802, 1, &[(32, 32, 9), (72, 4, 15)]),
                exit(),
            ],
            &[],
        );
        assert_eq!(r.get(1), 0);
    }

    #[test]
    fn unknown_opcodes_are_reported() {
        let code = program(&[inst(0x3A1, 0, &[]), exit()]);
        match decode_program(&code[HEADER_SIZE..]) {
            Err(DecodeError::Native {
                index: 0,
                opcode: 0x3A1,
                ..
            }) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn uniform_registers_and_root_tables() {
        let r = exec(
            &[
                // UMOV UR2 = 40
                inst(0x882, 2, &[(32, 32, 40)]),
                // R2UR UR3 = R0
                inst(0x3C2, 3, &[(24, 8, 0)]),
                // UIADD3 UR4 = UR2 + UR3 + URZ
                inst(
                    0x290,
                    4,
                    &[
                        (24, 8, 2),
                        (32, 8, 3),
                        (64, 8, 63),
                        (81, 6, 0x3F),
                        (91, 1, 1),
                    ],
                ),
                // IADD3 R1 = R0 + UR4 + RZ (form 6: B uniform)
                inst(
                    0xC10,
                    1,
                    &[
                        (24, 8, 0),
                        (32, 8, 4),
                        (64, 8, 255),
                        (81, 6, 0x3F),
                        (91, 1, 1),
                    ],
                ),
                // MOV R2 = UR4
                inst(0xC02, 2, &[(32, 6, 4), (72, 4, 15), (91, 1, 1)]),
                exit(),
            ],
            &[(0, 2)],
        );
        assert_eq!(r.get(ureg(4)), 42);
        assert_eq!(r.get(1), 44);
        assert_eq!(r.get(2), 42);

        // ULDC UR6 = c[0x19][0x10] (root table 1) and a 64-bit pair from bank 3.
        let code = program(&[
            inst(
                0xAB9,
                6,
                &[(38, 16, 0x10), (54, 5, 25), (73, 3, 4), (91, 1, 1)],
            ),
            inst(
                0xAB9,
                8,
                &[(38, 16, 0x20), (54, 5, 3), (73, 3, 5), (91, 1, 1)],
            ),
            exit(),
        ]);
        let p = decode_program(&code[HEADER_SIZE..]).unwrap();
        assert_eq!(
            p[0].op,
            Op::Mov {
                dst: ureg(6),
                src: Operand::Cbuf {
                    index: ROOT_TABLE_BANK,
                    offset: 0x110
                }
            }
        );
        assert_eq!(
            p[2].op,
            Op::Mov {
                dst: ureg(9),
                src: Operand::Cbuf {
                    index: 3,
                    offset: 0x24
                }
            }
        );
    }
}
