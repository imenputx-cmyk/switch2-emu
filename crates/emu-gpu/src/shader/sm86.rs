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
    AtomOp, Bop, DecodeError, HalfOp, HalfSrc, Inst, IntFunc, IpaMode, Mods, MufuOp, Op, Operand,
    Pred, Round, Space, HALF_ZERO, PT, PT_FALSE, RZ,
};

/// Uniform predicates UP0-UP6 are IR predicates 8-14 (UPT is PT).
pub fn upred(n: u32) -> u8 {
    match n & 7 {
        7 => 7,
        n => 8 + n as u8,
    }
}

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
            | 0x28C
            | 0x88C
            | 0x287
            | 0x887
            | 0x89C
    )
}

/// Whether an instruction reads or writes uniform registers.
fn uses_uniforms(op: u32) -> bool {
    is_uniform_op(op)
        || matches!(op >> 9, 6 | 7)
        || matches!(
            op,
            0x3C2 | 0xC02 | 0x9C3 | 0x981 | 0x986 | 0x9A8 | 0x98E | 0x987 | 0x988 | 0x984 | 0x983
        )
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

    /// A uniform predicate operand (UPn, negated by the fourth bit).
    fn upred_at(&self, offset: u32) -> Pred {
        Pred {
            index: upred(self.bits(offset, 3)),
            neg: self.on(offset + 3),
        }
    }

    /// The truth table of PLOP3/UPLOP3: bits 64..67 and 72..77.
    fn plop3_lut(&self) -> u8 {
        (self.bits(64, 3) | self.bits(72, 5) << 3) as u8
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
        words: i.0,
    }
}

/// Decodes 128-bit instructions (after the header) until the program's
/// end. Branch targets in the result are IR indices.
pub fn decode_program(code: &[u8]) -> Result<Vec<Inst>, DecodeError> {
    decode_program_ext(code, false)
}

/// As `decode_program`; `at_end` says no more code can be read after
/// `code` (a program with a computed branch may then end with it).
pub fn decode_program_ext(code: &[u8], at_end: bool) -> Result<Vec<Inst>, DecodeError> {
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
    let mut ended = false;
    // After a computed branch the end is not known from the branches:
    // its cases follow, and may end in EXIT. Decode on until the
    // padding (a branch to itself), an undecodable word or the end.
    let mut computed = false;
    let ends_here = |l: &[Vec<Inst>]| {
        l.last()
            .and_then(|i| i.last())
            .is_some_and(Inst::is_terminal)
    };
    for (index, &inst) in all.iter().enumerate() {
        let insts = match decode(inst, index) {
            Ok(i) => i,
            Err(_) if computed && index > furthest && ends_here(&lowered) => {
                ended = true;
                break;
            }
            Err(e) => return Err(e),
        };
        if computed
            && index > furthest
            && ends_here(&lowered)
            && matches!(insts.as_slice(), [Inst { pred: PT, op: Op::Bra { target } }] if *target == index)
        {
            ended = true;
            break;
        }
        for inst in &insts {
            match inst.op {
                Op::Bra { target } | Op::Ssy { target } => furthest = furthest.max(target),
                Op::Brx { .. } => computed = true,
                _ => {}
            }
        }
        let terminal = insts.last().is_some_and(Inst::is_terminal);
        lowered.push(insts);
        if terminal && index >= furthest && !computed {
            ended = true;
            break;
        }
    }
    if !ended && computed && at_end && ends_here(&lowered) {
        ended = true;
    }
    if !ended {
        return Err(DecodeError::Truncated);
    }
    lower_computed_branches(&mut lowered);
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
                    words: [0; 4],
                    what: "branch past the program",
                })?;
            }
            out.push(inst);
        }
    }
    Ok(out)
}

/// The predicate BRX lowering compares into (not a guest predicate).
const BRX_PRED: u8 = 15;

/// Lowers each BRX into `reg == target` compares and branches over the
/// likely jump-table targets: the existing branch targets and every
/// instruction after an unconditional transfer (switch cases end with
/// one). An unmatched value exits.
fn lower_computed_branches(lowered: &mut [Vec<Inst>]) {
    if !lowered
        .iter()
        .flatten()
        .any(|i| matches!(i.op, Op::Brx { .. }))
    {
        return;
    }
    let mut targets = std::collections::BTreeSet::new();
    for (n, insts) in lowered.iter().enumerate() {
        for i in insts {
            if let Op::Bra { target } | Op::Ssy { target } = i.op {
                targets.insert(target);
            }
        }
        if insts.last().is_some_and(Inst::is_terminal) {
            targets.insert(n + 1);
        }
    }
    targets.retain(|&t| t < lowered.len());
    for insts in lowered.iter_mut() {
        let Some(k) = insts.iter().position(|i| matches!(i.op, Op::Brx { .. })) else {
            continue;
        };
        let Inst {
            pred,
            op: Op::Brx { reg, bias },
        } = insts[k]
        else {
            unreachable!()
        };
        let compare = |cmp, value: u32| Op::Isetp {
            dst: BRX_PRED,
            dst2: 7,
            cmp,
            signed: false,
            a: reg,
            b: Operand::Imm(value),
            bop: Bop::And,
            c: PT,
        };
        let mut chain = Vec::with_capacity(2 * targets.len() + 2);
        if pred != PT {
            chain.push(Inst::always(compare(0, 0))); // false
        }
        let on = Pred {
            index: BRX_PRED,
            neg: false,
        };
        for &t in &targets {
            chain.push(Inst {
                pred,
                op: compare(2, (t as u32 * 16).wrapping_sub(bias)),
            });
            chain.push(Inst {
                pred: on,
                op: Op::Bra { target: t },
            });
        }
        chain.push(Inst { pred, op: Op::Exit });
        insts.splice(k..=k, chain);
    }
}

/// Decodes one instruction (at native index `index`).
pub fn decode(i: Instruction, index: usize) -> Result<Vec<Inst>, DecodeError> {
    let op = i.opcode();
    // BRX / BRXU: to (next PC + relative + source); the source holds a
    // native code offset (Lichii lower_to_maxwell.h).
    if op == 0x949 || op == 0x958 {
        let raw = u64::from(i.0[1]) | u64::from(i.0[2] & 0x3FFFF) << 32;
        let relative = ((raw << 14) as i64 >> 14) as i32;
        let reg = if op == 0x958 {
            ureg(i.bits(24, 8))
        } else {
            i.reg(24)
        };
        let bias = ((index as i64 + 1) * 16 + i64::from(relative)) as u32;
        return Ok(vec![Inst {
            pred: i.pred_at(12),
            op: Op::Brx { reg, bias },
        }]);
    }
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
            // Scale 1..7 is 2^(s-4); 0 is invalid (Lichii).
            let s = i.bits(84, 3);
            if s == 0 {
                return err("FMUL scale or absolute value");
            }
            one(Op::Fmul {
                dst,
                a,
                b,
                neg_b: a_mods.neg != b_mods.neg,
                a_abs: a_mods.abs,
                b_abs: b_mods.abs,
                scale: s as i8 - 4,
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
                        a_abs: false,
                        b_abs: false,
                        scale: 0,
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
        // IADD3: carry outs at 81 and 84; .X (74) adds the carry-in
        // predicates at 87 and 77.
        0x010 => {
            let (b, b_mods) = src_b()?;
            let (c, c_mods) = src_c()?;
            let x = i.on(74);
            one(Op::Iadd3 {
                dst,
                a,
                neg_a: i.on(72),
                b,
                neg_b: b_mods.neg,
                c,
                neg_c: c_mods.neg,
                x,
                cin: if x {
                    [i.pred_at(87), i.pred_at(77)]
                } else {
                    [PT_FALSE; 2]
                },
                co: [i.bits(81, 3) as u8, i.bits(84, 3) as u8],
            })
        }
        // IMAD (low word). Bit 75 negates the addend.
        0x024 => {
            let x = i.on(74);
            let c = src_c()?.0;
            one(Op::Imad {
                dst,
                a,
                b: src_b()?.0,
                c,
                neg_c: i.on(75),
                x,
                cin: if x { i.pred_at(87) } else { PT_FALSE },
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
            let small = i.0[2] & !0xC000 == 0x0021_2800;
            if i.0[2] & !0x1_C100 != 0x0020_3000 && !small {
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
                bits: if small { 16 } else { 32 },
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
        // Packed half arithmetic and compares.
        0x030 | 0x031 | 0x032 | 0x033 | 0x034 | 0x040 => {
            if form >= 6 {
                return err("half operation with a uniform operand");
            }
            let a = half_source(&i, index, 0)?;
            let low = op & 0x1FF;
            // HADD2/HSET2/HSETP2/HMNMX2 take their second operand in C's
            // place in the forms where C is wide.
            let second = if matches!(form, 2 | 3) && low != 0x031 && low != 0x032 {
                half_source(&i, index, 2)?
            } else {
                half_source(&i, index, 1)?
            };
            match low {
                0x030 | 0x032 => {
                    if i.on(76) || i.on(78) || i.on(79) {
                        return err("half add/multiply mode");
                    }
                    one(Op::Half2 {
                        dst,
                        op: if low == 0x030 {
                            HalfOp::Add
                        } else {
                            HalfOp::Mul
                        },
                        a,
                        b: second,
                        c: HALF_ZERO,
                        sat: i.on(77),
                    })
                }
                0x031 => {
                    if i.on(78) || i.on(79) || i.on(85) || i.on(86) {
                        return err("HFMA2 mode");
                    }
                    one(Op::Half2 {
                        dst,
                        op: HalfOp::Fma,
                        a,
                        b: second,
                        c: half_source(&i, index, 2)?,
                        sat: i.on(77),
                    })
                }
                0x040 => one(Op::Half2 {
                    dst,
                    op: HalfOp::Mnmx(i.pred_at(87)),
                    a,
                    b: second,
                    c: HALF_ZERO,
                    sat: false,
                }),
                0x033 => one(Op::Hset2 {
                    dst,
                    cmp: i.bits(76, 4) as u8,
                    a,
                    b: second,
                    bop: bop(69)?,
                    c: i.pred_at(87),
                    bf: i.on(71),
                }),
                _ => one(Op::Hsetp2 {
                    dst: i.bits(81, 3) as u8,
                    dst2: i.bits(84, 3) as u8,
                    cmp: i.bits(76, 4) as u8,
                    a,
                    b: second,
                    bop: bop(69)?,
                    c: i.pred_at(87),
                    h_and: i.on(71),
                }),
            }
        }
        // F2FP.PACK_AB (MERGE_C with bit 78: the high half from C as is).
        0x03E => {
            if i.bits(72, 6) != 0 || i.bits(79, 12) != 0 || i.on(91) {
                return err("F2FP mode");
            }
            let merge = i.on(78);
            let (b, _) = src_b()?;
            let a = if merge { src_c()?.0 } else { Operand::Reg(a) };
            one(Op::F2fp { dst, a, b, merge })
        }
        // F2F.F16.F32 (round to nearest).
        0x104 => {
            if i.0[2] != 0x0020_0800 {
                return err("F2F types");
            }
            let (src, mods) = src_b()?;
            one(Op::F2h { dst, src, mods })
        }
        // P2R / R2P with an immediate mask; byte select at 76.
        0x003 | 0x004 if form == 4 => {
            let shift = 8 * i.bits(76, 2);
            let mask = i.0[1] & 0x7F;
            one(if op & 0x1FF == 0x003 {
                Op::P2r {
                    dst,
                    a,
                    mask,
                    shift,
                }
            } else {
                Op::R2p { a, mask, shift }
            })
        }
        // POPC, FLO, BREV, IABS on source B.
        0x109 | 0x100 | 0x101 | 0x013 => {
            let func = match op & 0x1FF {
                0x109 => IntFunc::Popc,
                0x100 => IntFunc::Flo {
                    signed: i.on(73),
                    shift: i.on(74),
                },
                0x101 => IntFunc::Brev,
                _ => IntFunc::Iabs,
            };
            if op & 0x1FF == 0x100 && i.bits(81, 3) != 7 {
                return err("FLO predicate output");
            }
            one(Op::IntUnary {
                dst,
                func,
                src: src_b()?.0,
            })
        }
        // IMNMX: min when the predicate at 87 holds; bit 73 signed.
        0x017 => one(Op::Imnmx {
            dst,
            a,
            b: src_b()?.0,
            signed: i.on(73),
            min: i.pred_at(87),
        }),
        // PRMT (default mode only).
        0x016 => {
            if i.bits(72, 3) != 0 {
                return err("PRMT mode");
            }
            one(Op::Prmt {
                dst,
                a,
                sel: src_b()?.0,
                c: src_c()?.0,
            })
        }
        // LEA: (a << shift) + b; .HI (80) shifts the c:a pair and keeps
        // the high word; .X (74) adds the carry-in at 87; carry out at 81.
        0x011 if i.on(80) || i.on(74) || i.bits(81, 3) != 7 => {
            let (b, b_mods) = src_b()?;
            if b_mods.neg {
                return err("LEA with carries and a negated addend");
            }
            let x = i.on(74);
            one(Op::Lea {
                dst,
                a,
                neg_a: i.on(72),
                b,
                c: if i.on(80) { i.reg(64) } else { RZ },
                shift: i.bits(75, 5),
                hi: i.on(80),
                x,
                cin: if x { i.pred_at(87) } else { PT_FALSE },
                co: i.bits(81, 3) as u8,
            })
        }
        0x011 => {
            let (b, b_mods) = src_b()?;
            one(Op::Iscadd {
                dst,
                a,
                neg_a: i.on(72),
                b,
                neg_b: b_mods.neg,
                shift: i.bits(75, 5),
            })
        }
        // FRND.F32: rounding mode at 78..80.
        0x107 => {
            if i.bits(75, 2) != 2 || i.bits(84, 2) != 2 {
                return err("FRND types");
            }
            let (src, mods) = src_b()?;
            let round = match i.bits(78, 2) {
                0 => Round::Nearest,
                1 => Round::Floor,
                2 => Round::Ceil,
                _ => Round::Trunc,
            };
            one(Op::F2f {
                dst,
                src,
                mods,
                round,
                sat: false,
            })
        }
        // IMAD.WIDE (c and the result are register pairs) and IMAD.HI.
        0x025 | 0x027 => {
            if i.on(74) || i.bits(81, 3) != 7 {
                return err("IMAD.WIDE/HI carry");
            }
            let b = src_b()?.0;
            let (c, c_mods) = src_c()?;
            if c_mods.neg {
                return err("IMAD.WIDE/HI negated addend");
            }
            if op & 0x1FF == 0x025 {
                let Operand::Reg(c) = c else {
                    return err("IMAD.WIDE addend");
                };
                one(Op::ImadWide {
                    dst,
                    a,
                    b,
                    c,
                    signed: i.on(73),
                })
            } else {
                one(Op::ImadHi {
                    dst,
                    a,
                    b,
                    c,
                    signed: i.on(73),
                })
            }
        }
        _ => match op {
            // TEX / TLD on a 2D texture whose handle is word 40..54 of
            // constant bank 54..59: x in Ra, y in Rb; components (mask
            // 72..76) to Rd, Rd + 1, then the second destination (64..72)
            // or Rd + 2.
            0xB60 | 0xB66 => {
                let mask = i.bits(72, 4) as u8;
                let lod = i.bits(87, 3);
                let fetch = op == 0xB66;
                let dim = i.bits(61, 3);
                if dim == 6 {
                    return err("texture dimension 6");
                }
                if i.bits(76, 3) != 0 || i.bits(81, 3) != 7 || i.on(59) || mask == 0 {
                    return err("texture offsets, depth compare, FP16 or predicate");
                }
                // Level modes: 0 auto, 1 zero (LZ), 2 bias (LB), 3 level
                // (LL); 6/7 the same with a clamp. Only level 0 is kept,
                // so a bias or level operand is read past and ignored.
                let has_lod = matches!(lod, 2 | 3 | 6 | 7);
                if (fetch && !matches!(lod, 1 | 3)) || (!fetch && lod > 3 && lod < 6) {
                    return err("texture level mode");
                }
                let bank = i.bits(54, 5);
                if bank >= 18 {
                    return err("texture handle bank");
                }
                let handle = super::decode::bank_handle(bank, i.bits(40, 14));
                let second = i.reg(64);
                let dst1 = match (second, dst) {
                    (RZ, RZ) => RZ,
                    (RZ, d) => d.wrapping_add(2),
                    (s, _) => s,
                };
                let (ra, rb) = (i.reg(24), i.reg(32));
                // Bit 79: FP16 results. The components land in Rd.. as f32
                // and are then packed in pairs: (c0, c1) into Rd, (c2, c3)
                // into the second destination or Rd + 1 (Lichii).
                let fp16 = i.on(79);
                let finish =
                    |res: Result<Vec<Inst>, DecodeError>| -> Result<Vec<Inst>, DecodeError> {
                        let mut out = res?;
                        if fp16 && dst != RZ {
                            let n = mask.count_ones();
                            let at = |r: u8, k: u8| if r == RZ { RZ } else { r.wrapping_add(k) };
                            let pack = |target: u8, low: u8, high: u8| Inst {
                                pred,
                                op: Op::F2fp {
                                    dst: target,
                                    a: Operand::Reg(high),
                                    b: Operand::Reg(low),
                                    merge: false,
                                },
                            };
                            out.push(pack(dst, dst, if n > 1 { at(dst, 1) } else { RZ }));
                            if n > 2 {
                                let target = if second == RZ { at(dst, 1) } else { second };
                                out.push(pack(target, dst1, if n > 3 { at(dst1, 1) } else { RZ }));
                            }
                        }
                        Ok(out)
                    };
                if dim == 1 && !has_lod {
                    return finish(one(if fetch {
                        Op::Tld {
                            dst0: dst,
                            dst1,
                            a: ra,
                            b: rb,
                            handle,
                            comps: mask,
                        }
                    } else {
                        Op::Texs {
                            dst0: dst,
                            dst1,
                            a: ra,
                            b: rb,
                            handle,
                            comps: mask,
                            lz: lod == 1,
                        }
                    }));
                }
                // The operand list (array layer first, coordinates, then
                // the level) packs into Ra.. and Rb.. (Lichii general_tex).
                let array = matches!(dim, 4 | 5 | 7);
                let coords = [1u32, 2, 3, 3, 1, 2, 0, 3][dim as usize];
                let count = coords + u32::from(array) + u32::from(has_lod);
                let in_a = if count <= 4 { count.div_ceil(2) } else { 4 };
                let value = |k: u32| -> u8 {
                    let (base, n) = if k < in_a { (ra, k) } else { (rb, k - in_a) };
                    if base == RZ {
                        RZ
                    } else {
                        base.wrapping_add(n as u8)
                    }
                };
                let first = u32::from(array);
                let mut c = [RZ; 3];
                for (k, slot) in c.iter_mut().enumerate().take(coords as usize) {
                    *slot = value(first + k as u32);
                }
                finish(one(Op::TexN {
                    dst0: dst,
                    dst1,
                    dim: dim as u8,
                    coords: c,
                    layer: if array { value(0) } else { RZ },
                    handle,
                    comps: mask,
                    fetch,
                    auto_lod: !fetch && lod == 0,
                }))
            }
            // S2R / S2UR: the special register at 72..80.
            0x919 | 0x9C3 => {
                let target = if op == 0x9C3 {
                    ureg(u32::from(dst))
                } else {
                    dst
                };
                one(Op::S2r {
                    dst: target,
                    sr: i.bits(72, 8) as u8,
                })
            }
            // CS2R: SRZ (zero) or the clock, 32 or 64 bits (bit 80).
            0x805 => {
                let sr = i.bits(72, 8) as u8;
                let mut out = vec![Inst {
                    pred,
                    op: Op::S2r {
                        dst,
                        sr: if sr == 255 {
                            super::decode::sr::LANE_ID
                        } else {
                            sr
                        },
                    },
                }];
                if i.on(80) && dst != RZ {
                    out.push(Inst {
                        pred,
                        op: Op::S2r {
                            dst: dst + 1,
                            sr: if sr == 255 { 0 } else { sr.wrapping_add(1) },
                        },
                    });
                }
                Ok(out)
            }
            // LDC: bank 54..59, offset 38..54 from Ra; 32 or 64 bits.
            0xB82 => {
                let width = i.bits(73, 3);
                if i.bits(78, 2) != 0 || !matches!(width, 4 | 5) {
                    return err("LDC mode or width");
                }
                let bank = i.bits(54, 5);
                if bank >= 18 {
                    return err("LDC bank");
                }
                one(Op::Ldc {
                    dst,
                    index: bank as u8,
                    base: a,
                    offset: i.bits(38, 16) as i16 as i32 as u32,
                    wide: width == 5,
                })
            }
            // LDG / STG: 24-bit signed offset at 40, size at 73.
            0x381 | 0x981 | 0x386 | 0x986 => {
                let size = i.bits(73, 3) as u8;
                if size > 6 {
                    return err("global access size");
                }
                let offset = ((i.bits(40, 24) << 8) as i32) >> 8;
                let load = op & 0xFF == 0x81;
                let uaddr = match op {
                    0x981 => {
                        if i.bits(38, 2) != 0 {
                            return err("LDG uniform address");
                        }
                        ureg(i.bits(32, 6))
                    }
                    0x986 => ureg(i.bits(64, 6)),
                    _ => RZ,
                };
                if load && op == 0x381 && i.bits(64, 4) != 0 {
                    return err("LDG load predicate");
                }
                let wide = if load { i.on(90) } else { i.on(72) };
                one(if load {
                    Op::Ldg {
                        dst,
                        addr: a,
                        wide,
                        uaddr,
                        offset,
                        size,
                    }
                } else {
                    Op::Stg {
                        src: i.reg(32),
                        addr: a,
                        wide,
                        uaddr,
                        offset,
                        size,
                    }
                })
            }
            // PLOP3.LUT: a at 87, b at 77, c at 68 (a uniform predicate if
            // bit 67); dst at 81 by the table, dst2 at 84 by bits 16..24.
            0x81C => {
                let c = if i.on(67) {
                    i.upred_at(68)
                } else {
                    i.pred_at(68)
                };
                one(Op::Plop3 {
                    dst: i.bits(81, 3) as u8,
                    dst2: i.bits(84, 3) as u8,
                    a: i.pred_at(87),
                    b: i.pred_at(77),
                    c,
                    lut: i.plop3_lut(),
                    lut2: i.bits(16, 8) as u8,
                })
            }
            // VOTE.{ALL,ANY,EQ}, one invocation at a time.
            0x806 => {
                let mode = i.bits(72, 2);
                if mode > 2 {
                    return err("VOTE mode");
                }
                one(Op::Vote {
                    dst,
                    pdst: i.bits(81, 3) as u8,
                    src: i.pred_at(87),
                    eq: mode == 2,
                })
            }
            // SHFL: one invocation reads its own value; in range.
            0xF89 | 0x989 | 0x589 | 0x389 => Ok(vec![
                Inst {
                    pred,
                    op: Op::Mov {
                        dst,
                        src: Operand::Reg(a),
                    },
                },
                Inst {
                    pred,
                    op: Op::Plop3 {
                        dst: i.bits(81, 3) as u8,
                        dst2: 7,
                        a: PT,
                        b: PT,
                        c: PT,
                        lut: 0xFF,
                        lut2: 0,
                    },
                },
            ]),
            // TXQ: dimensions (query 0 in bits 59..64) of a bound texture.
            0xB6F => {
                let mask = i.bits(72, 4) as u8;
                if i.bits(59, 5) != 0 || mask == 0 || i.bits(76, 14) != 0 {
                    return err("TXQ query");
                }
                let bank = i.bits(54, 5);
                if bank >= 18 {
                    return err("texture handle bank");
                }
                let second = i.reg(64);
                one(Op::Txq {
                    dst,
                    dst1: match (second, dst) {
                        (RZ, RZ) => RZ,
                        (RZ, d) => d.wrapping_add(2),
                        (s, _) => s,
                    },
                    lod: a,
                    handle: super::decode::bank_handle(bank, i.bits(40, 14)),
                    comps: mask,
                })
            }
            // TLD4 on a 2D texture: component at 87..89.
            0xB63 => {
                let mask = i.bits(72, 4) as u8;
                if i.bits(61, 3) != 1
                    || i.bits(76, 4) != 0
                    || i.bits(81, 3) != 7
                    || i.on(59)
                    || i.on(89)
                    || mask == 0
                {
                    return err("TLD4 mode");
                }
                let bank = i.bits(54, 5);
                if bank >= 18 {
                    return err("texture handle bank");
                }
                let second = i.reg(64);
                one(Op::Tld4 {
                    dst0: dst,
                    dst1: match (second, dst) {
                        (RZ, RZ) => RZ,
                        (RZ, d) => d.wrapping_add(2),
                        (s, _) => s,
                    },
                    a: i.reg(24),
                    b: i.reg(32),
                    handle: super::decode::bank_handle(bank, i.bits(40, 14)),
                    comp: i.bits(87, 2) as u8,
                    comps: mask,
                })
            }
            // BAR.SYNC 0 (the whole workgroup).
            0xB1D => {
                if i.0[1] != 0 || i.0[2] & !0x10000 != 0 {
                    return err("barrier mode");
                }
                one(Op::Bar)
            }
            // LDS / STS: size at 73, address scale at 78..80, 24-bit
            // offset at 40; with bit 91 a uniform register adds in (LDS:
            // at 32; STS 0x988: at 64).
            0x984 | 0x988 | 0x388 => {
                let size = i.bits(73, 3) as u8;
                if size > 6 || i.bits(78, 2) != 0 {
                    return err("shared memory size or scaled address");
                }
                let load = op == 0x984;
                let uaddr = match (op, i.on(91)) {
                    (0x984, true) => ureg(i.bits(32, 8)),
                    (0x988, true) => ureg(i.bits(64, 8)),
                    (0x988, false) => return err("STS form"),
                    _ => RZ,
                };
                let offset = i.bits(40, 24) as i32;
                one(if load {
                    Op::Lds {
                        space: Space::Shared,
                        dst,
                        addr: a,
                        uaddr,
                        offset,
                        size,
                    }
                } else {
                    Op::Sts {
                        space: Space::Shared,
                        src: i.reg(32),
                        addr: a,
                        uaddr,
                        offset,
                        size,
                    }
                })
            }
            // LDL / STL (local memory): a uniform base with bit 91 (LDL at
            // 32, STL at 64).
            0x983 | 0x387 | 0x987 => {
                let size = i.bits(73, 3) as u8;
                if size > 6 {
                    return err("local memory size");
                }
                let load = op == 0x983;
                let uaddr = match (load, i.on(91)) {
                    (true, true) => ureg(i.bits(32, 8)),
                    (false, true) => ureg(i.bits(64, 8)),
                    _ => RZ,
                };
                let offset = i.bits(40, 24) as i32;
                one(if load {
                    Op::Lds {
                        space: Space::Local,
                        dst,
                        addr: a,
                        uaddr,
                        offset,
                        size,
                    }
                } else {
                    Op::Sts {
                        space: Space::Local,
                        src: i.reg(32),
                        addr: a,
                        uaddr,
                        offset,
                        size,
                    }
                })
            }
            // ATOMS: operation at 87 (Maxwell numbering), S32 at 73.
            0x38C => {
                let Some(op) = AtomOp::from_bits(i.bits(87, 4)) else {
                    return err("shared atomic operation");
                };
                one(Op::Atom {
                    space: Space::Shared,
                    dst,
                    addr: a,
                    wide: false,
                    uaddr: RZ,
                    offset: i.bits(40, 24) as i32,
                    src: i.reg(32),
                    op,
                    signed: i.on(73),
                })
            }
            // ATOMG (register pair address, or 0x9A8: uniform, ADD) and
            // RED (no result; type at 73: 0 U32, 1 S32).
            0x3A8 | 0x9A8 | 0x98E => {
                let red = op == 0x98E;
                let bits = if op == 0x9A8 { 0 } else { i.bits(87, 4) };
                let bits = if red { bits & 7 } else { bits };
                let Some(atom) = AtomOp::from_bits(bits) else {
                    return err("global atomic operation");
                };
                if red && i.bits(73, 3) > 1 {
                    return err("RED type");
                }
                let uaddr = if op == 0x9A8 || (red && a == RZ) {
                    ureg(i.bits(64, 6))
                } else {
                    RZ
                };
                one(Op::Atom {
                    space: Space::Global,
                    dst: if red { RZ } else { dst },
                    addr: a,
                    wide: true,
                    uaddr,
                    offset: ((i.bits(40, 24) << 8) as i32) >> 8,
                    src: i.reg(32),
                    op: atom,
                    signed: i.on(73),
                })
            }
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

/// A packed half operand (Lichii's `half_source`): 0 is A, 1 is B and 2
/// is C, each in the place the instruction's form puts it.
fn half_source(i: &Instruction, index: usize, which: u32) -> Result<HalfSrc, DecodeError> {
    let form = i.opcode() >> 9;
    let at = |src, sw, abs, neg| HalfSrc {
        src,
        swizzle: i.bits(sw, 2) as u8,
        abs: i.on(abs),
        neg: i.on(neg),
    };
    if which == 0 {
        return Ok(at(Operand::Reg(i.reg(24)), 74, 73, 72));
    }
    let wide = if which == 1 {
        matches!(form, 4 | 5)
    } else {
        matches!(form, 2 | 3)
    };
    if wide {
        return Ok(if matches!(form, 2 | 4) {
            HalfSrc {
                src: Operand::Imm(i.0[1]),
                swizzle: 0,
                abs: false,
                neg: false,
            }
        } else {
            at(cbuf_operand(i, index)?, 60, 62, 63)
        });
    }
    let upper = which == 2 || matches!(form, 2 | 3);
    Ok(if upper {
        at(Operand::Reg(i.reg(64)), 81, 83, 84)
    } else {
        at(Operand::Reg(i.reg(32)), 60, 62, 63)
    })
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
    let guard = i.upred_at(12);
    let insts = decode_uniform_op(i, index)?;
    Ok(insts
        .into_iter()
        .map(|mut inst| {
            if inst.pred == PT {
                inst.pred = guard;
            }
            inst
        })
        .collect())
}

fn decode_uniform_op(i: Instruction, index: usize) -> Result<Vec<Inst>, DecodeError> {
    let op = i.opcode();
    let dst = ureg(i.bits(16, 8));
    let one = |op: Op| Ok(vec![Inst::always(op)]);
    let ub = |i: &Instruction| {
        if op & 0x800 != 0 {
            Operand::Imm(i.0[1])
        } else {
            Operand::Reg(ureg(i.bits(32, 8)))
        }
    };
    match op {
        // UISETP: UP outputs at 81 and 84, combined with the UP at 87.
        0x28C | 0x88C => {
            if i.on(72) {
                return Err(native(index, &i, "UISETP.EX"));
            }
            let bop = match i.bits(74, 2) {
                0 => Bop::And,
                1 => Bop::Or,
                2 => Bop::Xor,
                _ => return Err(native(index, &i, "predicate combination")),
            };
            one(Op::Isetp {
                dst: upred(i.bits(81, 3)),
                dst2: upred(i.bits(84, 3)),
                cmp: i.bits(76, 3) as u8,
                signed: i.on(73),
                a: ureg(i.bits(24, 8)),
                b: ub(&i),
                bop,
                c: i.upred_at(87),
            })
        }
        // USEL: A when the UP at 87 holds.
        0x287 | 0x887 => one(Op::Sel {
            dst,
            a: ureg(i.bits(24, 8)),
            b: ub(&i),
            cond: i.upred_at(87),
        }),
        // UPLOP3.LUT on uniform predicates.
        0x89C => one(Op::Plop3 {
            dst: upred(i.bits(81, 3)),
            dst2: upred(i.bits(84, 3)),
            a: i.upred_at(87),
            b: i.upred_at(77),
            c: i.upred_at(68),
            lut: i.plop3_lut(),
            lut2: i.bits(16, 8) as u8,
        }),
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
        // ULDC with a uniform index register (24..32).
        0xABB => {
            let width = i.bits(73, 3);
            let bank = i.bits(54, 5);
            if !matches!(width, 4 | 5) || bank >= 18 {
                return Err(native(index, &i, "indexed ULDC width or bank"));
            }
            one(Op::Ldc {
                dst,
                index: bank as u8,
                base: ureg(i.bits(24, 8)),
                offset: i.bits(38, 16) as i16 as i32 as u32,
                wide: width == 5,
            })
        }
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
            let lop3_pred = (op & 0x1FF == 0x092).then(|| upred(i.bits(81, 3)));
            if lop3_pred.is_some() {
                r.0[2] |= 0xF << 23; // input PT (bits 87..91)
            }
            r.0[0] &= !0x80; // the ordinary opcode
            r.0[0] = (r.0[0] & !(0xF << 12)) | (7 << 12); // guard: applied by the caller
            r.0[2] &= !(1 << 27); // bit 91
            let mut out = decode(r, index)?;
            if let Some(up) = lop3_pred {
                for inst in &mut out {
                    if let Op::Lop3 { pred, .. } = &mut inst.op {
                        *pred = up;
                    }
                }
            }
            // UIADD3/ULEA carries are uniform predicates.
            let remap = |p: &mut Pred| {
                if *p != PT_FALSE {
                    p.index = upred(u32::from(p.index));
                }
            };
            for inst in &mut out {
                match &mut inst.op {
                    Op::Iadd3 { cin, co, .. } => {
                        cin.iter_mut().for_each(remap);
                        co.iter_mut().for_each(|c| *c = upred(u32::from(*c)));
                    }
                    Op::Lea { cin, co, .. } => {
                        remap(cin);
                        *co = upred(u32::from(*co));
                    }
                    _ => {}
                }
            }
            Ok(out)
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
        let decoded = decode_program_ext(&code[HEADER_SIZE..], true).expect("decodes");
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
    fn textures() {
        // TEX R4 (RGBA, contiguous), handle c[2][0x30 words]; TLD.LZ R8 (RG).
        let code = program(&[
            inst(
                0xB60,
                4,
                &[
                    (24, 8, 0),
                    (32, 8, 1),
                    (40, 14, 0x30),
                    (54, 5, 2),
                    (61, 3, 1),
                    (64, 8, 255),
                    (72, 4, 15),
                    (81, 3, 7),
                ],
            ),
            inst(
                0xB66,
                8,
                &[
                    (24, 8, 2),
                    (32, 8, 3),
                    (40, 14, 0x31),
                    (54, 5, 2),
                    (61, 3, 1),
                    (64, 8, 255),
                    (72, 4, 3),
                    (81, 3, 7),
                    (87, 3, 1),
                ],
            ),
            exit(),
        ]);
        let p = decode_program(&code[HEADER_SIZE..]).unwrap();
        assert_eq!(
            p[0].op,
            Op::Texs {
                dst0: 4,
                dst1: 6,
                a: 0,
                b: 1,
                handle: super::super::decode::bank_handle(2, 0x30),
                comps: 15,
                lz: false
            }
        );
        assert!(matches!(
            p[1].op,
            Op::Tld {
                dst0: 8,
                a: 2,
                b: 3,
                comps: 3,
                ..
            }
        ));
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

#[cfg(test)]
mod more_tests {
    use super::asm::*;
    use super::*;
    use crate::shader::interp::{run, ShaderIo};
    use std::collections::HashMap;

    #[derive(Default)]
    struct Io {
        global: HashMap<u64, u8>,
    }
    impl ShaderIo for Io {
        fn load_attr(&self, _: u32) -> u32 {
            0
        }
        fn store_attr(&mut self, _: u32, _: u32) {}
        fn interpolate(&self, _: u32, _: IpaMode) -> f32 {
            0.0
        }
        /// Word at `offset` of bank `index` holds index * 0x1000 + offset.
        fn cbuf(&self, index: u8, offset: u32) -> u32 {
            u32::from(index) * 0x1000 + offset
        }
        fn load_global(&mut self, va: u64, buf: &mut [u8]) {
            for (k, b) in buf.iter_mut().enumerate() {
                *b = self.global.get(&(va + k as u64)).copied().unwrap_or(0);
            }
        }
        fn store_global(&mut self, va: u64, data: &[u8]) {
            for (k, &b) in data.iter().enumerate() {
                self.global.insert(va + k as u64, b);
            }
        }
    }

    fn exec(insts: &[Instruction], init: &[(u8, u32)], io: &mut Io) -> [u32; 256] {
        let code = program(insts);
        let decoded = decode_program_ext(&code[HEADER_SIZE..], true).expect("decodes");
        let mut program: Vec<Inst> = init
            .iter()
            .map(|&(dst, v)| {
                Inst::always(Op::Mov {
                    dst,
                    src: Operand::Imm(v),
                })
            })
            .collect();
        // Branch targets move past the prepended moves.
        let shift = program.len();
        program.extend(decoded.into_iter().map(|mut i| {
            if let Op::Bra { target } | Op::Ssy { target } = &mut i.op {
                *target += shift;
            }
            i
        }));
        run(&program, io).expect("runs").0
    }

    #[test]
    fn computed_branch() {
        let set = |v: u32| {
            inst(
                0x810,
                1,
                &[
                    (24, 8, 255),
                    (32, 32, v),
                    (64, 8, 255),
                    (81, 3, 7),
                    (84, 3, 7),
                ],
            )
        };
        // BRX R0 with relative -16: to the code offset in R0.
        let brx = inst(
            0x949,
            0,
            &[(24, 8, 0), (32, 32, (-16i32) as u32), (64, 18, 0x3FFFF)],
        );
        let code = [brx, set(1), exit(), set(2), exit()];
        for (offset, want) in [(0x10, 1), (0x30, 2)] {
            let r = exec(&code, &[(0, offset)], &mut Io::default());
            assert_eq!(r[1], want, "offset {offset:#x}");
        }
    }

    #[test]
    fn carry_chains() {
        let r = exec(
            &[
                // IADD3 R4, P0, PT = R0 + R2 + RZ
                inst(
                    0x210,
                    4,
                    &[(24, 8, 0), (32, 8, 2), (64, 8, 255), (81, 3, 0), (84, 3, 7)],
                ),
                // IADD3.X R5 = R1 + R3 + RZ + P0 + !PT
                inst(
                    0x210,
                    5,
                    &[
                        (24, 8, 1),
                        (32, 8, 3),
                        (64, 8, 255),
                        (74, 1, 1),
                        (87, 4, 0),
                        (77, 4, 0xF),
                    ],
                ),
                // IADD3 R6, P1 = -R2 + R0
                inst(
                    0x210,
                    6,
                    &[
                        (24, 8, 2),
                        (72, 1, 1),
                        (32, 8, 0),
                        (64, 8, 255),
                        (81, 3, 1),
                        (84, 3, 7),
                    ],
                ),
                // IADD3.X R7 = ~R3 + R1 + P1
                inst(
                    0x210,
                    7,
                    &[
                        (24, 8, 3),
                        (72, 1, 1),
                        (32, 8, 1),
                        (64, 8, 255),
                        (74, 1, 1),
                        (87, 4, 1),
                        (77, 4, 0xF),
                    ],
                ),
                // LEA R8, P2 = (R0 << 4) + R9
                inst(0x211, 8, &[(24, 8, 0), (32, 8, 9), (75, 5, 4), (81, 3, 2)]),
                // LEA.HI.X R10 = (R1:R0 << 4 >> 32) + R3 + P2
                inst(
                    0x211,
                    10,
                    &[
                        (24, 8, 0),
                        (32, 8, 3),
                        (64, 8, 1),
                        (75, 5, 4),
                        (80, 1, 1),
                        (74, 1, 1),
                        (81, 3, 7),
                        (87, 4, 2),
                    ],
                ),
                exit(),
            ],
            &[(0, 0xFFFF_FFFF), (1, 1), (2, 1), (3, 2), (9, 0x20)],
            &mut Io::default(),
        );
        assert_eq!((r[4], r[5]), (0, 4));
        assert_eq!((r[6], r[7]), (0xFFFF_FFFE, 0xFFFF_FFFF));
        assert_eq!((r[8], r[10]), (0x10, 0x22));
    }

    #[test]
    fn bit_ops_and_wide_multiply() {
        let r = exec(
            &[
                // POPC R2 = popc(R0)
                inst(0x309, 2, &[(32, 8, 0)]),
                // BREV R3 = brev(R0)
                inst(0x301, 3, &[(32, 8, 0)]),
                // FLO.U32 R4 = msb(R0)
                inst(0x300, 4, &[(32, 8, 0), (81, 3, 7)]),
                // PRMT R5 = R0, 0x7610 (imm), R1
                inst(0x816, 5, &[(24, 8, 0), (32, 32, 0x7610), (64, 8, 1)]),
                // IMAD.WIDE.U32 R6:R7 = R0 * R1 + R8:R9
                inst(0x225, 6, &[(24, 8, 0), (32, 8, 1), (64, 8, 8), (81, 3, 7)]),
                // IABS R10 = |R11|
                inst(0x213, 10, &[(32, 8, 11)]),
                // LEA R12 = (R0 << 4) + R1
                inst(0x211, 12, &[(24, 8, 0), (32, 8, 1), (75, 5, 4), (81, 3, 7)]),
                exit(),
            ],
            &[
                (0, 0x0000_F00F),
                (1, 0x1234_5678),
                (8, 0xFFFF_FFFF),
                (9, 1),
                (11, (-5i32) as u32),
            ],
            &mut Io::default(),
        );
        assert_eq!(r[2], 8);
        assert_eq!(r[3], 0xF00F_0000);
        assert_eq!(r[4], 15);
        assert_eq!(r[5], 0x1234_F00F);
        let wide = 0xF00Fu64 * 0x1234_5678 + 0x1_FFFF_FFFF;
        assert_eq!((r[6], r[7]), (wide as u32, (wide >> 32) as u32));
        assert_eq!(r[10], 5);
        assert_eq!(r[12], (0xF00F << 4) + 0x1234_5678);
    }

    #[test]
    fn uniform_predicates_and_indexed_constants() {
        let r = exec(
            &[
                // UISETP.GE.U32 UP1 = UR4 >= 7 (combined with UPT)
                inst(
                    0x88C,
                    0,
                    &[
                        (24, 8, 4),
                        (32, 32, 7),
                        (76, 3, 6),
                        (81, 3, 1),
                        (84, 3, 7),
                        (87, 4, 7),
                    ],
                ),
                // USEL UR5 = UP1 ? UR4 : 99
                inst(0x887, 5, &[(24, 8, 4), (32, 32, 99), (87, 4, 1)]),
                // UPLOP3 UP2 = !UP1 (lut of a only: 0x0F)
                inst(
                    0x89C,
                    0,
                    &[
                        (87, 4, 1),
                        (77, 4, 7),
                        (68, 4, 7),
                        (64, 3, 7),
                        (72, 5, 1),
                        (81, 3, 2),
                        (84, 3, 7),
                    ],
                ),
                // @UP2 UMOV UR6, 1 / @!UP2 UMOV UR6, 2
                inst(0x882, 6, &[(12, 4, 2), (32, 32, 1)]),
                inst(0x882, 6, &[(12, 4, 0b1010), (32, 32, 2)]),
                // LDC.64 R2:R3 = c[3][R0 + 0x10]
                inst(
                    0xB82,
                    2,
                    &[(24, 8, 0), (38, 16, 0x10), (54, 5, 3), (73, 3, 5)],
                ),
                // S2R R4 = SR_EQMASK
                inst(0x919, 4, &[(72, 8, 56)]),
                exit(),
            ],
            &[(ureg(4), 9), (0, 0x20)],
            &mut Io::default(),
        );
        assert_eq!(r[ureg(5) as usize], 9);
        assert_eq!(r[ureg(6) as usize], 2);
        assert_eq!((r[2], r[3]), (0x3030, 0x3034));
        assert_eq!(r[4], 1);
    }

    #[test]
    fn global_memory() {
        let mut io = Io::default();
        let r = exec(
            &[
                // STG.E.64 [R0.64 + 8], R2:R3
                inst(
                    0x386,
                    0,
                    &[(24, 8, 0), (32, 8, 2), (40, 24, 8), (72, 1, 1), (73, 3, 5)],
                ),
                // LDG.E R4 = [R0.64 + 12]
                inst(
                    0x381,
                    4,
                    &[(24, 8, 0), (40, 24, 12), (73, 3, 4), (90, 1, 1)],
                ),
                // LDG.E.U8 R5 = [R0.64 + 8]
                inst(0x381, 5, &[(24, 8, 0), (40, 24, 8), (73, 3, 0), (90, 1, 1)]),
                exit(),
            ],
            &[(0, 0x1000), (1, 0x1), (2, 0xAABB_CCDD), (3, 0x1122_3344)],
            &mut io,
        );
        assert_eq!(r[4], 0x1122_3344);
        assert_eq!(r[5], 0xDD);
        assert_eq!(io.global.get(&0x1_0000_1008), Some(&0xDD));
    }

    #[test]
    fn half_precision_ops() {
        // 1.5h | 2.0h << 16 and 0.5h | -1.0h << 16.
        let (x, y) = (0x3E00 | 0x4000 << 16, 0x3800 | 0xBC00 << 16);
        let r = exec(
            &[
                // HADD2 R2 = R0 + R1
                inst(0x230, 2, &[(24, 8, 0), (32, 8, 1)]),
                // HMUL2 R3 = R0 * R1.H1_H1 (swizzle 3)
                inst(0x232, 3, &[(24, 8, 0), (32, 8, 1), (60, 2, 3)]),
                // HFMA2 R4 = R0 * R1 + R0
                inst(0x231, 4, &[(24, 8, 0), (32, 8, 1), (64, 8, 0)]),
                // HSETP2.GT.AND P0, P1 = R0 > R1 (per lane), PT
                inst(
                    0x234,
                    0,
                    &[
                        (24, 8, 0),
                        (32, 8, 1),
                        (76, 4, 4),
                        (81, 3, 0),
                        (84, 3, 1),
                        (87, 4, 7),
                    ],
                ),
                // F2FP.PACK_AB R5 = half(R6) << 16 | half(R7)
                inst(0x23E, 5, &[(24, 8, 6), (32, 8, 7)]),
                // P2R R8 = P0..P1 (mask 3) into R9's low bits
                inst(0x803, 8, &[(24, 8, 9), (32, 32, 3)]),
                exit(),
            ],
            &[
                (0, x),
                (1, y),
                (6, 3.0f32.to_bits()),
                (7, 0.25f32.to_bits()),
                (9, 0xF0),
            ],
            &mut Io::default(),
        );
        assert_eq!(r[2], 0x4000 | 0x3C00 << 16); // 2.0, 1.0
        assert_eq!(r[3], 0xBE00 | 0xC000 << 16); // -1.5, -2.0
        assert_eq!(r[4], 0x4080); // 1.5*0.5+1.5=2.25, 2*-1+2=0
        assert_eq!(r[5], 0x4200 << 16 | 0x3400); // 3.0, 0.25
        assert_eq!(r[8], 0xF3); // P0 and P1 both true
    }

    #[test]
    fn texture_operand_lists() {
        // TEX.ARRAY_2D R0, R4 (layer, x), R6 (y), mask RGBA.
        let tex = inst(
            0xB60,
            0,
            &[(24, 8, 4), (32, 8, 6), (61, 3, 5), (72, 4, 15), (81, 3, 7)],
        );
        let d = decode(tex, 0).unwrap();
        assert!(matches!(
            d[0].op,
            Op::TexN {
                coords: [5, 6, RZ],
                layer: 4,
                fetch: false,
                ..
            }
        ));
        // TEX.3D.LL: x, y in Ra; z, level in Rb.
        let tex = inst(
            0xB60,
            0,
            &[
                (24, 8, 4),
                (32, 8, 6),
                (61, 3, 2),
                (72, 4, 1),
                (81, 3, 7),
                (87, 3, 3),
            ],
        );
        let d = decode(tex, 0).unwrap();
        assert!(matches!(
            d[0].op,
            Op::TexN {
                coords: [4, 5, 6],
                layer: RZ,
                ..
            }
        ));
    }
}
