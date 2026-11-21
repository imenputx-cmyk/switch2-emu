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

/// A texture handle naming its own constant bank (SM86): set in bit 31,
/// the bank in bits 16..21 and the word offset below. Other handles are
/// word offsets in the bindless constant buffer (register 0x2608).
pub const HANDLE_IN_BANK: u32 = 1 << 31;

pub const fn bank_handle(bank: u32, word: u32) -> u32 {
    HANDLE_IN_BANK | (bank & 0x1F) << 16 | (word & 0xFFFF)
}

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

/// !PT: always false (an absent carry-in).
pub const PT_FALSE: Pred = Pred {
    index: 7,
    neg: true,
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
    /// Fetches the texel at integer (a, b) of level 0 of a 2D texture
    /// (TLD.LZ); components go out as for TEXS.
    Tld {
        dst0: u8,
        dst1: u8,
        a: u8,
        b: u8,
        handle: u32,
        comps: u8,
    },
    /// Samples a 2D texture at (a, b); the enabled components (`comps`,
    /// RGBA bits) go to dst0, dst0 + 1, dst1, dst1 + 1 in order.
    Texs {
        dst0: u8,
        dst1: u8,
        a: u8,
        b: u8,
        /// Word offset of the handle in the bindless constant buffer.
        handle: u32,
        comps: u8,
        /// Level 0 explicitly ("LZ").
        lz: bool,
    },
    /// Samples a 2D texture at (a, a + 1); the enabled components go to
    /// consecutive registers from dst.
    Tex {
        dst: u8,
        a: u8,
        handle: u32,
        comps: u8,
        lz: bool,
    },
    /// Three-input integer add (SM70+ IADD3): a + b + c, each optionally
    /// negated.
    Iadd3 {
        dst: u8,
        a: u8,
        neg_a: bool,
        b: Operand,
        neg_b: bool,
        c: Operand,
        neg_c: bool,
        /// .X: add the carry-in predicates; negation is then a bitwise
        /// NOT (the +1 comes from the carry).
        x: bool,
        cin: [Pred; 2],
        /// Predicates receiving bits 32 and 33 of the full sum (PT: none).
        co: [u8; 2],
    },
    /// LEA with carries or the high half: lo form `(a << shift) + b`,
    /// .HI form `((c:a) << shift >> 32) + b`.
    Lea {
        dst: u8,
        a: u8,
        neg_a: bool,
        b: Operand,
        c: u8,
        shift: u32,
        hi: bool,
        x: bool,
        cin: Pred,
        co: u8,
    },
    /// Three-input logic by truth table (LOP3.LUT): bit i of `lut` is the
    /// result for (a, b, c) = (i >> 2 & 1, i >> 1 & 1, i & 1). `pred`
    /// (PT for none) is set to result != 0.
    Lop3 {
        dst: u8,
        a: u8,
        b: Operand,
        c: Operand,
        lut: u8,
        pred: u8,
    },
    /// Integer multiply-add, low 32 bits (IMAD): a * b + c.
    Imad {
        dst: u8,
        a: u8,
        b: Operand,
        c: Operand,
        neg_c: bool,
    },
    /// Funnel shift (SHF) of the 64-bit value c:a (high:low) by b.
    Shf {
        dst: u8,
        a: u8,
        b: Operand,
        c: Operand,
        right: bool,
        /// Return the high 32 bits of the shifted value.
        hi: bool,
        /// Shift amount modulo the width instead of clamped to it.
        wrap: bool,
        /// 64-bit type (shifts up to 63) rather than 32-bit.
        wide: bool,
        signed: bool,
    },
    /// Special register (S2R): see [`sr`].
    S2r {
        dst: u8,
        sr: u8,
    },
    /// Constant load at register `base` + `offset` (LDC); `wide` loads
    /// a register pair.
    Ldc {
        dst: u8,
        index: u8,
        base: u8,
        offset: u32,
        wide: bool,
    },
    /// Global memory load (LDG): `size` as Maxwell's (0 U8, 1 S8, 2 U16,
    /// 3 S16, 4 32, 5 64, 6 128 bits) at the address `addr` (a register
    /// pair if `wide`, else zero-extended) + `uaddr` (a pair, RZ for
    /// none) + `offset`.
    Ldg {
        dst: u8,
        addr: u8,
        wide: bool,
        uaddr: u8,
        offset: i32,
        size: u8,
    },
    /// Global memory store (STG), addressed like LDG.
    Stg {
        src: u8,
        addr: u8,
        wide: bool,
        uaddr: u8,
        offset: i32,
        size: u8,
    },
    /// Predicate truth tables (PLOP3.LUT): dst = lut(a, b, c),
    /// dst2 = lut2(a, b, c), bits as for LOP3.
    Plop3 {
        dst: u8,
        dst2: u8,
        a: Pred,
        b: Pred,
        c: Pred,
        lut: u8,
        lut2: u8,
    },
    /// One-source integer functions.
    IntUnary {
        dst: u8,
        func: IntFunc,
        src: Operand,
    },
    /// Byte permute (PRMT, default mode): byte i of the result is byte
    /// (selector nibble i & 7) of c:a, sign-replicated if nibble bit 3.
    Prmt {
        dst: u8,
        a: u8,
        sel: Operand,
        c: Operand,
    },
    /// 32x32 to 64-bit multiply-add (IMAD.WIDE): dst:dst+1 = a * b + c:c+1.
    ImadWide {
        dst: u8,
        a: u8,
        b: Operand,
        c: u8,
        signed: bool,
    },
    /// High 32 bits of a 32x32 multiply plus c (IMAD.HI).
    ImadHi {
        dst: u8,
        a: u8,
        b: Operand,
        c: Operand,
        signed: bool,
    },
    /// Warp vote, one invocation at a time: `pdst` = src (ALL/ANY) or
    /// true (EQ); `dst` = the ballot (1 if src).
    Vote {
        dst: u8,
        pdst: u8,
        src: Pred,
        eq: bool,
    },
    /// Texture query (TXQ, dimensions): width, height, depth, levels of
    /// level `lod` (a register), the enabled ones to dst on.
    Txq {
        dst: u8,
        dst1: u8,
        lod: u8,
        handle: u32,
        comps: u8,
    },
    /// Gather (TLD4) of component `comp` at (a, b) of a 2D texture;
    /// outputs like TEXS.
    Tld4 {
        dst0: u8,
        dst1: u8,
        a: u8,
        b: u8,
        handle: u32,
        comp: u8,
        comps: u8,
    },
    /// Shared or local memory load (LDS/LDL), `size` as for LDG, at
    /// register `addr` + `uaddr` + `offset`.
    Lds {
        space: Space,
        dst: u8,
        addr: u8,
        uaddr: u8,
        offset: i32,
        size: u8,
    },
    /// Shared or local memory store (STS/STL).
    Sts {
        space: Space,
        src: u8,
        addr: u8,
        uaddr: u8,
        offset: i32,
        size: u8,
    },
    /// Atomic on a 32-bit word (ATOMS/ATOMG/RED): `dst` gets the old
    /// value; global addresses as for LDG (`wide` register pair).
    Atom {
        space: Space,
        dst: u8,
        addr: u8,
        wide: bool,
        uaddr: u8,
        offset: i32,
        src: u8,
        op: AtomOp,
        signed: bool,
    },
    /// Workgroup barrier (BAR.SYNC).
    Bar,
    /// Packed half-precision arithmetic (HADD2/HMUL2/HFMA2/HMNMX2): each
    /// 16-bit lane computed in f32 and rounded back.
    Half2 {
        dst: u8,
        op: HalfOp,
        a: HalfSrc,
        b: HalfSrc,
        c: HalfSrc,
        sat: bool,
    },
    /// Packed half compare to predicates (HSETP2): dst from lane 0 and
    /// dst2 from lane 1 (each combined with c), or with `h_and` dst from
    /// both lanes and dst2 its negation.
    Hsetp2 {
        dst: u8,
        dst2: u8,
        cmp: FCmp,
        a: HalfSrc,
        b: HalfSrc,
        bop: Bop,
        c: Pred,
        h_and: bool,
    },
    /// Packed half compare to a register (HSET2): per lane 1.0 (bf) or
    /// 0xFFFF, else 0.
    Hset2 {
        dst: u8,
        cmp: FCmp,
        a: HalfSrc,
        b: HalfSrc,
        bop: Bop,
        c: Pred,
        bf: bool,
    },
    /// F2FP.PACK_AB: dst = half(a) << 16 | half(b); with `merge` the high
    /// half is a's low 16 bits as they are.
    F2fp {
        dst: u8,
        a: Operand,
        b: Operand,
        merge: bool,
    },
    /// F2F.F16.F32: the half of an f32 in the low 16 bits.
    F2h {
        dst: u8,
        src: Operand,
        mods: Mods,
    },
    /// P2R: dst = a with the bits of `mask << shift` replaced by
    /// predicates P0-P6 (bit i = Pi) masked and shifted.
    P2r {
        dst: u8,
        a: u8,
        mask: u32,
        shift: u32,
    },
    /// R2P: Pi = bit (i + shift) of a, for the bits of `mask`.
    R2p {
        a: u8,
        mask: u32,
        shift: u32,
    },
    /// Texture sample or fetch of any type (1D/2D/3D/cube, arrays):
    /// coordinates in `coords` (RZ for none), the array layer in `layer`
    /// (an integer; RZ for none). Level 0 only. Outputs like TEXS.
    TexN {
        dst0: u8,
        dst1: u8,
        coords: [u8; 3],
        layer: u8,
        handle: u32,
        comps: u8,
        fetch: bool,
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

/// A packed half operand: `swizzle` 0 takes (low, high) as lanes, 1 the
/// value as an f32 for both, 2 the low half for both, 3 the high one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalfSrc {
    pub src: Operand,
    pub swizzle: u8,
    pub neg: bool,
    pub abs: bool,
}

pub const HALF_ZERO: HalfSrc = HalfSrc {
    src: Operand::Reg(RZ),
    swizzle: 0,
    neg: false,
    abs: false,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfOp {
    Add,
    Mul,
    /// a * b + c.
    Fma,
    /// Minimum where the predicate holds, else maximum.
    Mnmx(Pred),
}

/// Memory spaces of LDS/STS-style accesses and atomics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    Shared,
    Local,
    Global,
}

/// Atomic operations, numbered as Maxwell's ATOM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomOp {
    Add,
    Min,
    Max,
    Inc,
    Dec,
    And,
    Or,
    Xor,
    Exch,
}

impl AtomOp {
    pub fn from_bits(v: u32) -> Option<AtomOp> {
        Some(match v {
            0 => AtomOp::Add,
            1 => AtomOp::Min,
            2 => AtomOp::Max,
            3 => AtomOp::Inc,
            4 => AtomOp::Dec,
            5 => AtomOp::And,
            6 => AtomOp::Or,
            7 => AtomOp::Xor,
            8 => AtomOp::Exch,
            _ => return None,
        })
    }

    /// The new value of a word holding `old` after the operation.
    pub fn apply(self, old: u32, v: u32, signed: bool) -> u32 {
        match self {
            AtomOp::Add => old.wrapping_add(v),
            AtomOp::Min if signed => (old as i32).min(v as i32) as u32,
            AtomOp::Min => old.min(v),
            AtomOp::Max if signed => (old as i32).max(v as i32) as u32,
            AtomOp::Max => old.max(v),
            AtomOp::Inc => {
                if old >= v {
                    0
                } else {
                    old + 1
                }
            }
            AtomOp::Dec => {
                if old == 0 || old > v {
                    v
                } else {
                    old - 1
                }
            }
            AtomOp::And => old & v,
            AtomOp::Or => old | v,
            AtomOp::Xor => old ^ v,
            AtomOp::Exch => v,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntFunc {
    /// Population count.
    Popc,
    /// Index of the most significant set bit (of the magnitude for
    /// signed), 0xFFFFFFFF if none; `shift` gives 31 minus it.
    Flo { signed: bool, shift: bool },
    /// Bit reverse.
    Brev,
    /// Absolute value.
    Iabs,
}

/// Special registers (S2R), numbered as on Maxwell.
pub mod sr {
    pub const LANE_ID: u8 = 0;
    pub const Y_DIRECTION: u8 = 18;
    pub const TID_X: u8 = 33;
    pub const TID_Y: u8 = 34;
    pub const TID_Z: u8 = 35;
    pub const CTAID_X: u8 = 37;
    pub const CTAID_Y: u8 = 38;
    pub const CTAID_Z: u8 = 39;
    pub const EQ_MASK: u8 = 56;
    pub const LT_MASK: u8 = 57;
    pub const LE_MASK: u8 = 58;
    pub const GT_MASK: u8 = 59;
    pub const GE_MASK: u8 = 60;
    pub const CLOCK_LO: u8 = 80;
    pub const CLOCK_HI: u8 = 81;

    /// The value of a special register that does not depend on the
    /// invocation (one lane of one warp), if it is one.
    pub fn constant(sr: u8) -> Option<u32> {
        Some(match sr {
            LANE_ID | LT_MASK | CLOCK_LO | CLOCK_HI => 0,
            EQ_MASK | LE_MASK => 1,
            GT_MASK => 0xFFFF_FFFE,
            GE_MASK => 0xFFFF_FFFF,
            Y_DIRECTION => 1.0f32.to_bits(),
            _ => return None,
        })
    }
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
    /// A Switch 2 (SM86) instruction that is not decoded: its index,
    /// 12-bit opcode and what is missing.
    Native {
        index: usize,
        opcode: u32,
        what: &'static str,
    },
    /// The code ran out before the program's end: read more.
    Truncated,
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
    } else if top(0xf600, 0xd000) {
        // TEXS (bit 59: f16 results, not supported).
        let lz = match bits(v, 53, 4) {
            1 => false,
            2 => true,
            _ => return unknown, // other targets and LOD modes
        };
        let dst1 = reg(v, 28);
        let sel = bits(v, 50, 3) as usize;
        let comps = if dst1 == RZ {
            [1, 2, 4, 8, 3, 9, 10, 12][sel]
        } else {
            *[7, 11, 13, 14, 15]
                .get(sel)
                .ok_or(DecodeError::Unknown(v))?
        };
        if on(v, 59) {
            return unknown;
        }
        Op::Texs {
            dst0: dst,
            dst1,
            a,
            b: reg(v, 20),
            handle: bits(v, 36, 13) as u32,
            comps,
            lz,
        }
    } else if top(0xfc38, 0xc038) {
        // TEX: 2D only, no offsets, depth compare or arrays.
        let lz = match bits(v, 55, 3) {
            0 => false,
            1 => true,
            _ => return unknown,
        };
        if on(v, 54) || on(v, 50) || on(v, 28) || bits(v, 29, 2) != 1 {
            return unknown;
        }
        Op::Tex {
            dst,
            a,
            handle: bits(v, 36, 13) as u32,
            comps: bits(v, 31, 4) as u8,
            lz,
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
