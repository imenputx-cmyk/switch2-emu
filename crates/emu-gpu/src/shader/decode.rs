//! Decoding Maxwell instructions (the subset emulated so far).
//!
//! Common fields: destination register in bits 7:0, guard predicate in
//! 19:16 (bit 19 negates; predicate 7 is PT, always true). Register 255
//! is RZ (reads zero, writes discarded). Compare instructions write
//! predicates in bits 5:3 and 2:0 and combine with a predicate in 41:39
//! (bit 42 negates).
//!
//! Instructions are numbered in program order without the scheduling
//! words; branch targets are such indices.

pub const RZ: u8 = 255;

/// A predicate operand: P0-P6, or PT (7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pred {
    pub index: u8,
    pub neg: bool,
}

pub const PT: Pred = Pred {
    index: 7,
    neg: false,
};

impl Pred {
    fn at(v: u64, lo: u32) -> Pred {
        Pred {
            index: bits(v, lo, 3) as u8,
            neg: bits(v, lo + 3, 1) != 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(u8),
    /// 32-bit immediate (already expanded to its final bit pattern).
    Imm(u32),
    /// Constant buffer `index`, byte `offset`.
    Cbuf {
        index: u8,
        offset: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpaMode {
    /// Perspective-correct interpolation.
    Pass,
    /// Perspective-correct, multiplied by a register (drivers pass the
    /// interpolated w there; the result is the attribute itself).
    Multiply,
    /// The provoking vertex's value (flat).
    Constant,
    /// Per sample (treated like Pass).
    Sc,
}

/// Floating point operand modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Mods {
    pub neg: bool,
    pub abs: bool,
}

/// Boolean combination of a comparison with a predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bop {
    And,
    Or,
    Xor,
}

/// Float comparison (FSETP/FSET): bit 0 less, bit 1 equal, bit 2
/// greater, bit 3 "or unordered" (true if either input is NaN). So 7 is
/// "num" (ordered), 8 "nan", 0 false and 15 true.
pub type FCmp = u8;

/// Integer comparison (ISETP/ISET): bit 0 less, bit 1 equal, bit 2
/// greater.
pub type ICmp = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LopOp {
    And,
    Or,
    Xor,
    PassB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MufuOp {
    Cos,
    Sin,
    Ex2,
    Lg2,
    Rcp,
    Rsq,
    Sqrt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Round {
    Nearest,
    Floor,
    Ceil,
    Trunc,
    /// F2F without rounding to an integer.
    Pass,
}

/// How XMAD's third operand is formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XmadMode {
    C,
    /// Low 16 bits of C.
    Clo,
    /// High 16 bits of C.
    Chi,
    /// C + (B << 16).
    Cbcc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Mov {
        dst: u8,
        src: Operand,
    },
    Fadd {
        dst: u8,
        a: u8,
        a_mods: Mods,
        b: Operand,
        b_mods: Mods,
        sat: bool,
    },
    Fmul {
        dst: u8,
        a: u8,
        b: Operand,
        neg_b: bool,
        sat: bool,
    },
    Ffma {
        dst: u8,
        a: u8,
        b: Operand,
        neg_b: bool,
        c: Operand,
        neg_c: bool,
        sat: bool,
    },
    /// Minimum if `min` holds, else maximum.
    Fmnmx {
        dst: u8,
        a: u8,
        a_mods: Mods,
        b: Operand,
        b_mods: Mods,
        min: Pred,
    },
    Mufu {
        dst: u8,
        func: MufuOp,
        a: u8,
        mods: Mods,
        sat: bool,
    },
    /// Range reduction before MUFU sin/cos/ex2: emulated as a move.
    Rro {
        dst: u8,
        src: Operand,
        mods: Mods,
    },
    /// dst = cmp(a, b) bop c; dst2 = !cmp(a, b) bop c.
    Fsetp {
        dst: u8,
        dst2: u8,
        cmp: FCmp,
        a: u8,
        a_mods: Mods,
        b: Operand,
        b_mods: Mods,
        bop: Bop,
        c: Pred,
    },
    /// Writes 1.0 (`bf`) or 0xFFFFFFFF if cmp(a, b) bop c, else 0.
    Fset {
        dst: u8,
        cmp: FCmp,
        a: u8,
        a_mods: Mods,
        b: Operand,
        b_mods: Mods,
        bop: Bop,
        c: Pred,
        bf: bool,
    },
    Isetp {
        dst: u8,
        dst2: u8,
        cmp: ICmp,
        signed: bool,
        a: u8,
        b: Operand,
        bop: Bop,
        c: Pred,
    },
    Iset {
        dst: u8,
        cmp: ICmp,
        signed: bool,
        a: u8,
        b: Operand,
        bop: Bop,
        c: Pred,
        bf: bool,
    },
    /// dst = (a op1 b) op2 c; dst2 = !(a op1 b) op2 c.
    Psetp {
        dst: u8,
        dst2: u8,
        op1: Bop,
        op2: Bop,
        a: Pred,
        b: Pred,
        c: Pred,
    },
    Iadd {
        dst: u8,
        a: u8,
        neg_a: bool,
        b: Operand,
        neg_b: bool,
    },
    /// dst = (a << shift) + b.
    Iscadd {
        dst: u8,
        a: u8,
        neg_a: bool,
        b: Operand,
        neg_b: bool,
        shift: u32,
    },
    Shl {
        dst: u8,
        a: u8,
        b: Operand,
    },
    Shr {
        dst: u8,
        a: u8,
        b: Operand,
        signed: bool,
    },
    Lop {
        dst: u8,
        op: LopOp,
        a: u8,
        inv_a: bool,
        b: Operand,
        inv_b: bool,
    },
    /// Minimum if `min` holds, else maximum.
    Imnmx {
        dst: u8,
        a: u8,
        b: Operand,
        signed: bool,
        min: Pred,
    },
    /// dst = cond ? a : b.
    Sel {
        dst: u8,
        a: u8,
        b: Operand,
        cond: Pred,
    },
    /// 16x16-bit multiply-add.
    Xmad {
        dst: u8,
        a: u8,
        a_hi: bool,
        a_signed: bool,
        b: Operand,
        b_hi: bool,
        b_signed: bool,
        c: Operand,
        mode: XmadMode,
        /// Shift the product left by 16.
        psl: bool,
        /// Merge: the result's high half is B's low half.
        mrg: bool,
    },
    /// f32 to a 32-bit integer (saturating; NaN gives 0).
    F2i {
        dst: u8,
        src: Operand,
        mods: Mods,
        signed: bool,
        round: Round,
    },
    /// 32-bit integer to f32.
    I2f {
        dst: u8,
        src: Operand,
        signed: bool,
        neg: bool,
        abs: bool,
    },
    /// f32 to f32, optionally rounded to an integral value.
    F2f {
        dst: u8,
        src: Operand,
        mods: Mods,
        round: Round,
        sat: bool,
    },
    /// Load `count` attribute words starting at `addr` (+ register `base`).
    Ald {
        dst: u8,
        base: u8,
        addr: u32,
        count: u32,
    },
    /// Store `count` attribute words from `src` up.
    Ast {
        src: u8,
        base: u8,
        addr: u32,
        count: u32,
    },
    Ipa {
        dst: u8,
        addr: u32,
        mode: IpaMode,
        sat: bool,
    },
    Bra {
        target: usize,
    },
    /// Sets the reconvergence point SYNC jumps to.
    Ssy {
        target: usize,
    },
    /// Jumps to the innermost SSY target.
    Sync,
    /// Discards the pixel.
    Kil,
    Exit,
    Nop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inst {
    /// Guard: the instruction runs only if this predicate holds.
    pub pred: Pred,
    pub op: Op,
}

impl Inst {
    /// An unpredicated instruction.
    pub const fn always(op: Op) -> Inst {
        Inst { pred: PT, op }
    }

    /// Does execution never continue to the next instruction?
    pub fn is_terminal(&self) -> bool {
        self.pred == PT && matches!(self.op, Op::Exit | Op::Bra { .. } | Op::Sync)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// An instruction this decoder does not know (raw bits).
    Unknown(u64),
    /// A branch to a scheduling word or before the program.
    BadTarget(u64),
}

fn bits(v: u64, lo: u32, len: u32) -> u64 {
    (v >> lo) & ((1u64 << len) - 1)
}

fn reg(v: u64, lo: u32) -> u8 {
    bits(v, lo, 8) as u8
}

fn on(v: u64, bit: u32) -> bool {
    v >> bit & 1 != 0
}

/// Constant buffer operand: index in bits 38:34, offset (words) 33:20.
fn cbuf(v: u64) -> Operand {
    Operand::Cbuf {
        index: bits(v, 34, 5) as u8,
        offset: (bits(v, 20, 14) as u32) << 2,
    }
}

/// A 20-bit float immediate: bits 38:20 are the top bits of the value
/// below the sign, bit 56 the sign.
fn f20(v: u64) -> u32 {
    ((bits(v, 20, 19) | bits(v, 56, 1) << 19) as u32) << 12
}

/// A 20-bit signed integer immediate (bits 38:20, sign in bit 56).
fn s20(v: u64) -> u32 {
    let raw = (bits(v, 20, 19) | bits(v, 56, 1) << 19) as u32;
    ((raw << 12) as i32 >> 12) as u32
}

/// The second source of the register (0x5..), constant buffer (0x4..)
/// and immediate (0x3..) forms of an ALU instruction.
fn alu_b(v: u64, float: bool) -> Operand {
    match v >> 60 {
        0x5 => Operand::Reg(reg(v, 20)),
        0x4 => cbuf(v),
        _ => Operand::Imm(if float { f20(v) } else { s20(v) }),
    }
}

/// Is `v` the 0x5cXX (register), 0x4cXX (constant buffer) or
/// 0x38XX/0x39XX (immediate) form of the ALU opcode with low byte `low`?
fn alu(v: u64, low: u64) -> bool {
    let t = v >> 48 & 0xfff8;
    t == 0x5c00 | low || t == 0x4c00 | low || t & 0xfef8 == 0x3800 | low
}

/// Like `alu` for the compare family (0x5bXX / 0x4bXX / 0x36XX-0x37XX).
fn cmp_form(v: u64, low: u64) -> bool {
    let t = v >> 48 & 0xfff0;
    t == 0x5b00 | low || t == 0x4b00 | low || t & 0xfef0 == 0x3600 | low
}

fn bop(v: u64, lo: u32) -> Bop {
    match bits(v, lo, 2) {
        0 => Bop::And,
        1 => Bop::Or,
        _ => Bop::Xor,
    }
}

fn round(b: u64) -> Round {
    match b {
        0 => Round::Nearest,
        1 => Round::Floor,
        2 => Round::Ceil,
        _ => Round::Trunc,
    }
}

/// Instruction index of a byte offset from the first scheduling word.
pub fn index_of(byte: i64) -> Option<usize> {
    if byte < 0 || byte % 32 < 8 || byte % 8 != 0 {
        return None;
    }
    Some((byte / 32 * 3 + (byte % 32 - 8) / 8) as usize)
}

/// Byte offset of instruction `index` from the first scheduling word.
pub fn byte_of(index: usize) -> i64 {
    (index / 3 * 32 + 8 + index % 3 * 8) as i64
}

/// Decodes the instruction with index `at` (branch targets are relative).
pub fn decode(v: u64, at: usize) -> Result<Inst, DecodeError> {
    let top = |mask: u64, value: u64| v >> 48 & mask == value;
    let dst = reg(v, 0);
    let a = reg(v, 8);
    let always = bits(v, 0, 5) == 0xF; // flow condition code "T"
    let target = || -> Result<usize, DecodeError> {
        let off = ((bits(v, 20, 24) << 40) as i64) >> 40;
        index_of(byte_of(at) + 8 + off).ok_or(DecodeError::BadTarget(v))
    };
    let unknown = Err(DecodeError::Unknown(v));
    let op = if top(0xfff8, 0x50b0) {
        Op::Nop
    } else if top(0xfff0, 0xe300) && always {
        Op::Exit
    } else if top(0xfff0, 0xe330) && always {
        Op::Kil
    } else if top(0xfff8, 0xf0f8) && always {
        Op::Sync
    } else if top(0xfff0, 0xe240) && always && !on(v, 5) {
        Op::Bra { target: target()? }
    } else if top(0xfff0, 0xe290) && !on(v, 5) {
        Op::Ssy { target: target()? }
    } else if top(0xfff8, 0x5c98) {
        Op::Mov {
            dst,
            src: Operand::Reg(reg(v, 20)),
        }
    } else if top(0xfff8, 0x4c98) {
        Op::Mov { dst, src: cbuf(v) }
    } else if top(0xfef8, 0x3898) {
        Op::Mov {
            dst,
            src: Operand::Imm(s20(v)),
        }
    } else if top(0xfff0, 0x0100) {
        Op::Mov {
            dst,
            src: Operand::Imm(bits(v, 20, 32) as u32),
        }
    } else if alu(v, 0x58) {
        Op::Fadd {
            dst,
            a,
            a_mods: Mods {
                neg: on(v, 48),
                abs: on(v, 46),
            },
            b: alu_b(v, true),
            b_mods: Mods {
                neg: on(v, 45),
                abs: on(v, 49),
            },
            sat: on(v, 50),
        }
    } else if top(0xfc00, 0x0800) {
        // FADD32I
        Op::Fadd {
            dst,
            a,
            a_mods: Mods {
                neg: on(v, 56),
                abs: on(v, 54),
            },
            b: Operand::Imm(bits(v, 20, 32) as u32),
            b_mods: Mods {
                neg: on(v, 53),
                abs: on(v, 57),
            },
            sat: false,
        }
    } else if alu(v, 0x68) {
        Op::Fmul {
            dst,
            a,
            b: alu_b(v, true),
            neg_b: on(v, 48),
            sat: on(v, 50),
        }
    } else if top(0xff00, 0x1e00) {
        // FMUL32I
        Op::Fmul {
            dst,
            a,
            b: Operand::Imm(bits(v, 20, 32) as u32),
            neg_b: false,
            sat: on(v, 55),
        }
    } else if top(0xff80, 0x5980)
        || top(0xff80, 0x4980)
        || top(0xff80, 0x5180)
        || top(0xfe80, 0x3280)
    {
        let r20 = Operand::Reg(reg(v, 20));
        let r39 = Operand::Reg(reg(v, 39));
        let (b, c) = match v >> 55 {
            0xb3 => (r20, r39),     // 0x5980
            0x93 => (cbuf(v), r39), // 0x4980
            0xa3 => (r39, cbuf(v)), // 0x5180
            _ => (Operand::Imm(f20(v)), r39),
        };
        Op::Ffma {
            dst,
            a,
            b,
            neg_b: on(v, 48),
            c,
            neg_c: on(v, 49),
            sat: on(v, 50),
        }
    } else if alu(v, 0x60) {
        Op::Fmnmx {
            dst,
            a,
            a_mods: Mods {
                neg: on(v, 48),
                abs: on(v, 46),
            },
            b: alu_b(v, true),
            b_mods: Mods {
                neg: on(v, 45),
                abs: on(v, 49),
            },
            min: Pred::at(v, 39),
        }
    } else if top(0xfff8, 0x5080) {
        let func = match bits(v, 20, 4) {
            0 => MufuOp::Cos,
            1 => MufuOp::Sin,
            2 => MufuOp::Ex2,
            3 => MufuOp::Lg2,
            4 => MufuOp::Rcp,
            5 => MufuOp::Rsq,
            8 => MufuOp::Sqrt,
            _ => return unknown,
        };
        Op::Mufu {
            dst,
            func,
            a,
            mods: Mods {
                neg: on(v, 48),
                abs: on(v, 46),
            },
            sat: on(v, 50),
        }
    } else if alu(v, 0x90) {
        Op::Rro {
            dst,
            src: alu_b(v, true),
            mods: Mods {
                neg: on(v, 45),
                abs: on(v, 49),
            },
        }
    } else if cmp_form(v, 0xb0) {
        Op::Fsetp {
            dst: bits(v, 3, 3) as u8,
            dst2: bits(v, 0, 3) as u8,
            cmp: bits(v, 48, 4) as u8,
            a,
            a_mods: Mods {
                neg: on(v, 43),
                abs: on(v, 7),
            },
            b: alu_b(v, true),
            b_mods: Mods {
                neg: on(v, 6),
                abs: on(v, 44),
            },
            bop: bop(v, 45),
            c: Pred::at(v, 39),
        }
    } else if top(0xff80, 0x5800) || top(0xff80, 0x4800) || top(0xfe80, 0x3000) {
        Op::Fset {
            dst,
            cmp: bits(v, 48, 4) as u8,
            a,
            a_mods: Mods {
                neg: on(v, 43),
                abs: on(v, 54),
            },
            b: alu_b(v, true),
            b_mods: Mods {
                neg: on(v, 53),
                abs: on(v, 44),
            },
            bop: bop(v, 45),
            c: Pred::at(v, 39),
            bf: on(v, 52),
        }
    } else if cmp_form(v, 0x60) {
        Op::Isetp {
            dst: bits(v, 3, 3) as u8,
            dst2: bits(v, 0, 3) as u8,
            cmp: bits(v, 49, 3) as u8,
            signed: on(v, 48),
            a,
            b: alu_b(v, false),
            bop: bop(v, 45),
            c: Pred::at(v, 39),
        }
    } else if cmp_form(v, 0x50) {
        Op::Iset {
            dst,
            cmp: bits(v, 49, 3) as u8,
            signed: on(v, 48),
            a,
            b: alu_b(v, false),
            bop: bop(v, 45),
            c: Pred::at(v, 39),
            bf: on(v, 44),
        }
    } else if top(0xfff8, 0x5090) {
        Op::Psetp {
            dst: bits(v, 3, 3) as u8,
            dst2: bits(v, 0, 3) as u8,
            op1: bop(v, 24),
            op2: bop(v, 45),
            a: Pred::at(v, 12),
            b: Pred::at(v, 29),
            c: Pred::at(v, 39),
        }
    } else if alu(v, 0x10) {
        if on(v, 43) || on(v, 47) || on(v, 50) {
            return unknown; // carry and saturating forms
        }
        Op::Iadd {
            dst,
            a,
            neg_a: on(v, 49),
            b: alu_b(v, false),
            neg_b: on(v, 48),
        }
    } else if top(0xfe80, 0x1c00) {
        // IADD32I
        if on(v, 52) || on(v, 53) || on(v, 54) {
            return unknown;
        }
        Op::Iadd {
            dst,
            a,
            neg_a: on(v, 56),
            b: Operand::Imm(bits(v, 20, 32) as u32),
            neg_b: false,
        }
    } else if alu(v, 0x18) {
        Op::Iscadd {
            dst,
            a,
            neg_a: on(v, 49),
            b: alu_b(v, false),
            neg_b: on(v, 48),
            shift: bits(v, 39, 5) as u32,
        }
    } else if alu(v, 0x48) {
        Op::Shl {
            dst,
            a,
            b: alu_b(v, false),
        }
    } else if alu(v, 0x28) {
        Op::Shr {
            dst,
            a,
            b: alu_b(v, false),
            signed: on(v, 48),
        }
    } else if alu(v, 0x40) {
        Op::Lop {
            dst,
            op: [LopOp::And, LopOp::Or, LopOp::Xor, LopOp::PassB][bits(v, 41, 2) as usize],
            a,
            inv_a: on(v, 39),
            b: alu_b(v, false),
            inv_b: on(v, 40),
        }
    } else if top(0xfc00, 0x0400) {
        // LOP32I
        Op::Lop {
            dst,
            op: [LopOp::And, LopOp::Or, LopOp::Xor, LopOp::PassB][bits(v, 53, 2) as usize],
            a,
            inv_a: on(v, 55),
            b: Operand::Imm(bits(v, 20, 32) as u32),
            inv_b: on(v, 56),
        }
    } else if alu(v, 0x20) {
        Op::Imnmx {
            dst,
            a,
            b: alu_b(v, false),
            signed: on(v, 48),
            min: Pred::at(v, 39),
        }
    } else if alu(v, 0xa0) {
        Op::Sel {
            dst,
            a,
            b: alu_b(v, false),
            cond: Pred::at(v, 39),
        }
    } else if top(0xffc0, 0x5b00)
        || top(0xff80, 0x5100)
        || top(0xfe00, 0x4e00)
        || top(0xfec0, 0x3600)
    {
        // XMAD: register, register/cbuf, cbuf and immediate forms.
        let (b, c, b_hi, mode, psl, mrg, x) = match v >> 56 {
            0x5b => (
                Operand::Reg(reg(v, 20)),
                Operand::Reg(reg(v, 39)),
                on(v, 35),
                bits(v, 50, 3),
                on(v, 36),
                on(v, 37),
                on(v, 38),
            ),
            0x51 => (
                Operand::Reg(reg(v, 39)),
                cbuf(v),
                on(v, 52),
                bits(v, 50, 2),
                false,
                false,
                on(v, 54),
            ),
            0x4e | 0x4f => (
                cbuf(v),
                Operand::Reg(reg(v, 39)),
                on(v, 52),
                bits(v, 50, 2),
                on(v, 55),
                on(v, 56),
                on(v, 54),
            ),
            _ => (
                Operand::Imm(bits(v, 20, 16) as u32),
                Operand::Reg(reg(v, 39)),
                false,
                bits(v, 50, 3),
                on(v, 36),
                on(v, 37),
                on(v, 38),
            ),
        };
        if x || on(v, 47) {
            return unknown; // carry forms
        }
        let mode = match mode {
            0 => XmadMode::C,
            1 => XmadMode::Clo,
            2 => XmadMode::Chi,
            4 => XmadMode::Cbcc,
            _ => return unknown, // CSFU
        };
        Op::Xmad {
            dst,
            a,
            a_hi: on(v, 53),
            a_signed: on(v, 48),
            b,
            b_hi,
            b_signed: on(v, 49),
            c,
            mode,
            psl,
            mrg,
        }
    } else if alu(v, 0xb0) {
        // F2I: f32 source (bits 11:10 = 2), 32-bit destination (9:8 = 2).
        if bits(v, 10, 2) != 2 || bits(v, 8, 2) != 2 {
            return unknown;
        }
        Op::F2i {
            dst,
            src: alu_b(v, true),
            mods: Mods {
                neg: on(v, 45),
                abs: on(v, 49),
            },
            signed: on(v, 12),
            round: round(bits(v, 39, 2)),
        }
    } else if alu(v, 0xb8) {
        // I2F: 32-bit source, f32 destination, byte selector 0.
        if bits(v, 8, 2) != 2 || bits(v, 10, 2) != 2 || bits(v, 41, 2) != 0 {
            return unknown;
        }
        Op::I2f {
            dst,
            src: alu_b(v, false),
            signed: on(v, 13),
            neg: on(v, 45),
            abs: on(v, 49),
        }
    } else if alu(v, 0xa8) {
        // F2F: f32 to f32; bit 42 selects rounding to an integer.
        if bits(v, 8, 2) != 2 || bits(v, 10, 2) != 2 {
            return unknown;
        }
        Op::F2f {
            dst,
            src: alu_b(v, true),
            mods: Mods {
                neg: on(v, 45),
                abs: on(v, 49),
            },
            round: if on(v, 42) {
                round(bits(v, 39, 2))
            } else {
                Round::Pass
            },
            sat: on(v, 50),
        }
    } else if top(0xfff8, 0xefd8) {
        Op::Ald {
            dst,
            base: a,
            addr: bits(v, 20, 10) as u32,
            count: bits(v, 47, 2) as u32 + 1,
        }
    } else if top(0xfff8, 0xeff0) {
        Op::Ast {
            src: dst,
            base: a,
            addr: bits(v, 20, 10) as u32,
            count: bits(v, 47, 2) as u32 + 1,
        }
    } else if v & 0xff00_0040_0000_ff00 == 0xe000_0000_0000_ff00 {
        let mode = match bits(v, 54, 2) {
            0 => IpaMode::Pass,
            1 => IpaMode::Multiply,
            2 => IpaMode::Constant,
            _ => IpaMode::Sc,
        };
        Op::Ipa {
            dst,
            addr: bits(v, 28, 10) as u32,
            mode,
            sat: on(v, 51),
        }
    } else {
        return unknown;
    };
    Ok(Inst {
        pred: Pred::at(v, 16),
        op,
    })
}

/// Decodes a program (the instructions after the header), skipping the
/// scheduling words, through its last reachable instruction: the first
/// unconditional EXIT, BRA or SYNC with no branch target after it. The
/// result ends early (without such an instruction) if `code` does.
pub fn decode_program(code: &[u8]) -> Result<Vec<Inst>, DecodeError> {
    let mut out = Vec::new();
    let mut furthest = 0usize;
    for (i, q) in code.chunks_exact(8).enumerate() {
        if i % 4 == 0 {
            continue; // scheduling control
        }
        let at = out.len();
        let inst = decode(u64::from_le_bytes(q.try_into().unwrap()), at)?;
        if let Op::Bra { target } | Op::Ssy { target } = inst.op {
            furthest = furthest.max(target);
        }
        out.push(inst);
        if inst.is_terminal() && at >= furthest {
            break;
        }
    }
    Ok(out)
}

/// Does a decoded program end where execution cannot continue?
pub fn is_complete(program: &[Inst]) -> bool {
    program.last().is_some_and(Inst::is_terminal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_indices_skip_scheduling_words() {
        for i in 0..20 {
            assert_eq!(index_of(byte_of(i)), Some(i));
        }
        assert_eq!(index_of(0), None);
        assert_eq!(index_of(32), None);
        assert_eq!(index_of(-8), None);
    }
}
