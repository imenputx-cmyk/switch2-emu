//! Advanced SIMD (NEON) data processing: decoding and execution.
//!
//! Every instruction is decoded into a [`Simd`]: an operation, element
//! size and registers. Execution works lane by lane on 128-bit values,
//! following the ARM pseudocode. "By element" forms are executed as their
//! vector forms with the selected element broadcast into the second
//! operand, and scalar forms as one-lane vector forms.
//!
//! Saturating instructions set FPSR.QC. Floating-point lanes may be half,
//! single or double precision.

use crate::fp::{self, BinOp, Fp, FpType, Rounding};
use crate::Cpu;

/// FPSR.QC: cumulative saturation flag.
pub const FPSR_QC: u64 = 1 << 27;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    /// Signed greater than.
    Gt,
    /// Signed greater or equal.
    Ge,
    /// Unsigned higher.
    Hi,
    /// Unsigned higher or same.
    Hs,
    Eq,
    /// CMTST: `(a & b) != 0`.
    Tst,
    /// Signed less than (zero forms only).
    Lt,
    /// Signed less or equal (zero forms only).
    Le,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FCmp {
    Eq,
    Ge,
    Gt,
    Lt,
    Le,
    /// FACGE: `|a| >= |b|`.
    AbsGe,
    /// FACGT: `|a| > |b|`.
    AbsGt,
}

/// Accumulation of a widening multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acc {
    None,
    Add,
    Sub,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    // ---- three same (integer) ----
    Hadd {
        unsigned: bool,
        round: bool,
    },
    Hsub {
        unsigned: bool,
    },
    QAdd {
        unsigned: bool,
    },
    QSub {
        unsigned: bool,
    },
    Compare(Cmp),
    /// SSHL, USHL, SQSHL, UQSHL, SRSHL, URSHL, SQRSHL, UQRSHL (register).
    ShlReg {
        unsigned: bool,
        round: bool,
        sat: bool,
    },
    MaxMin {
        unsigned: bool,
        max: bool,
        pairwise: bool,
    },
    Abd {
        unsigned: bool,
        acc: bool,
    },
    AddSub {
        sub: bool,
    },
    Mla {
        sub: bool,
    },
    Mul {
        poly: bool,
    },
    /// SQDMULH, SQRDMULH.
    Qdmulh {
        round: bool,
    },
    /// SQRDMLAH, SQRDMLSH.
    Qrdmlah {
        sub: bool,
    },
    Addp,
    /// SDOT, UDOT.
    Dot {
        unsigned: bool,
    },
    // ---- three different ----
    /// SADDL, SADDW, SSUBL, SSUBW (and unsigned).
    AddSubLong {
        unsigned: bool,
        sub: bool,
        wide: bool,
    },
    /// ADDHN, RADDHN, SUBHN, RSUBHN.
    AddSubHn {
        sub: bool,
        round: bool,
    },
    AbdLong {
        unsigned: bool,
        acc: bool,
    },
    MulLong {
        unsigned: bool,
        acc: Acc,
    },
    QdmulLong {
        acc: Acc,
    },
    Pmull,
    // ---- two-register miscellaneous ----
    /// REV16/REV32/REV64: reverse elements in containers of `1 << c` bytes.
    Rev {
        container: u8,
    },
    AddLongPairwise {
        unsigned: bool,
        acc: bool,
    },
    /// SUQADD (signed accumulator) and USQADD (unsigned accumulator).
    QAccumulate {
        unsigned: bool,
    },
    Cls,
    Clz,
    Cnt,
    Not,
    Rbit,
    QAbs,
    QNeg,
    Abs,
    Neg,
    CompareZero(Cmp),
    Xtn,
    /// SQXTN (s->s), UQXTN (u->u), SQXTUN (s->u).
    Qxtn {
        src_unsigned: bool,
        dst_unsigned: bool,
    },
    Shll,
    // ---- across lanes ----
    AddLongV {
        unsigned: bool,
    },
    MaxMinV {
        unsigned: bool,
        max: bool,
    },
    AddV,
    // ---- permute / extract / table ----
    Uzp {
        second: bool,
    },
    Trn {
        second: bool,
    },
    Zip {
        second: bool,
    },
    Ext,
    /// TBL / TBX over `imm` + 1 table registers.
    Tbl {
        tbx: bool,
    },
    // ---- shift by immediate (`imm` is the shift) ----
    ShrImm {
        unsigned: bool,
        round: bool,
        acc: bool,
    },
    ShlImm,
    Sri,
    Sli,
    QShlImm {
        src_unsigned: bool,
        dst_unsigned: bool,
    },
    /// SHRN, RSHRN.
    ShrNarrow {
        round: bool,
    },
    /// SQSHRN, SQRSHRN, UQSHRN, UQRSHRN, SQSHRUN, SQRSHRUN.
    QShrNarrow {
        round: bool,
        src_unsigned: bool,
        dst_unsigned: bool,
    },
    /// SSHLL, USHLL.
    ShlLong {
        unsigned: bool,
    },
    /// SCVTF/UCVTF (vector, fixed-point): `imm` is fbits.
    FixedToFp {
        signed: bool,
    },
    /// FCVTZS/FCVTZU (vector, fixed-point): `imm` is fbits.
    FpToFixed {
        signed: bool,
    },
    // ---- floating point ----
    FArith(BinOp),
    FPairwise(BinOp),
    FMla {
        sub: bool,
    },
    FMulx,
    FRecps,
    FRsqrts,
    FAbd,
    FCompare(FCmp),
    FCompareZero(FCmp),
    FAbs,
    FNeg,
    FSqrt,
    FRound(Rounding),
    FToInt {
        signed: bool,
        rounding: Rounding,
    },
    IntToF {
        signed: bool,
    },
    FRecpe,
    FRsqrte,
    FRecpx,
    Urecpe,
    Ursqrte,
    /// FCVTN/FCVTXN: narrow to the next smaller precision.
    FCvtNarrow {
        round_odd: bool,
    },
    /// FCVTL: widen to the next larger precision.
    FCvtLong,
    FMaxMinV {
        max: bool,
        num: bool,
    },
    // ---- Cryptographic Extension ----
    Aese,
    Aesd,
    Aesmc,
    Aesimc,
    Sha1Rounds(crate::crypto::Sha1Fn),
    Sha1H,
    Sha1Su0,
    Sha1Su1,
    Sha256H,
    Sha256H2,
    Sha256Su0,
    Sha256Su1,
    /// PMULL/PMULL2 with 64-bit elements.
    Pmull64,
}

/// A decoded Advanced SIMD instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Simd {
    pub op: Op,
    /// 128-bit vector when true, 64-bit when false.
    pub q: bool,
    /// log2 of the element size in bytes (source size for widening and
    /// narrowing operations: the wider of the two).
    pub esize: u8,
    /// Scalar form: one lane, rest of Vd cleared.
    pub scalar: bool,
    pub rd: u8,
    pub rn: u8,
    pub rm: u8,
    /// "By element" forms: the element of Vm used for every lane.
    pub index: Option<u32>,
    /// Shift amount, EXT byte offset, fbits, or TBL length - 1.
    pub imm: u32,
}

fn bits(insn: u32, hi: u32, lo: u32) -> u32 {
    (insn >> lo) & (((1u64 << (hi - lo + 1)) - 1) as u32)
}

fn bit(insn: u32, n: u32) -> bool {
    (insn >> n) & 1 == 1
}

impl Simd {
    fn new(op: Op, insn: u32, esize: u8) -> Simd {
        Simd {
            op,
            q: bit(insn, 30),
            esize,
            scalar: false,
            rd: bits(insn, 4, 0) as u8,
            rn: bits(insn, 9, 5) as u8,
            rm: bits(insn, 20, 16) as u8,
            index: None,
            imm: 0,
        }
    }

    fn scalar(mut self) -> Simd {
        self.scalar = true;
        self
    }

    fn imm(mut self, imm: u32) -> Simd {
        self.imm = imm;
        self
    }
}

/// Decodes the Advanced SIMD data-processing groups not handled in
/// `decoder.rs`.
pub fn decode(insn: u32) -> Option<Simd> {
    // Cryptographic AES, SHA three-register and SHA two-register.
    if insn & 0xFFFE_0C00 == 0x4E28_0800 {
        let op = match bits(insn, 16, 12) {
            0b00100 => Op::Aese,
            0b00101 => Op::Aesd,
            0b00110 => Op::Aesmc,
            0b00111 => Op::Aesimc,
            _ => return None,
        };
        return Some(Simd::new(op, insn, 0));
    }
    if insn & 0xFFE0_8C00 == 0x5E00_0000 {
        use crate::crypto::Sha1Fn;
        let op = match bits(insn, 14, 12) {
            0b000 => Op::Sha1Rounds(Sha1Fn::Choose),
            0b001 => Op::Sha1Rounds(Sha1Fn::Parity),
            0b010 => Op::Sha1Rounds(Sha1Fn::Majority),
            0b011 => Op::Sha1Su0,
            0b100 => Op::Sha256H,
            0b101 => Op::Sha256H2,
            0b110 => Op::Sha256Su1,
            _ => return None,
        };
        return Some(Simd::new(op, insn, 2));
    }
    if insn & 0xFFFE_0C00 == 0x5E28_0800 {
        let op = match bits(insn, 16, 12) {
            0b00000 => Op::Sha1H,
            0b00001 => Op::Sha1Su1,
            0b00010 => Op::Sha256Su0,
            _ => return None,
        };
        return Some(Simd::new(op, insn, 2));
    }

    // Half-precision groups.
    if insn & 0x9F60_C400 == 0x0E40_0400 {
        return fp16_three_same(insn, false);
    }
    if insn & 0xDF60_C400 == 0x5E40_0400 {
        return fp16_three_same(insn, true);
    }
    if insn & 0x9F7E_0C00 == 0x0E78_0800 {
        return fp16_two_reg_misc(insn, false);
    }
    if insn & 0xDF7E_0C00 == 0x5E78_0800 {
        return fp16_two_reg_misc(insn, true);
    }
    if insn & 0x9F20_0400 == 0x0E20_0400 {
        return three_same(insn, false);
    }
    if insn & 0xDF20_0400 == 0x5E20_0400 {
        return three_same(insn, true);
    }
    if insn & 0x9F20_8400 == 0x0E00_8400 {
        return three_same_extra(insn, false);
    }
    if insn & 0xDF20_8400 == 0x5E00_8400 {
        return three_same_extra(insn, true);
    }
    if insn & 0x9F20_0C00 == 0x0E20_0000 {
        return three_different(insn, false);
    }
    if insn & 0xDF20_0C00 == 0x5E20_0000 {
        return three_different(insn, true);
    }
    if insn & 0x9F3E_0C00 == 0x0E20_0800 {
        return two_reg_misc(insn, false);
    }
    if insn & 0xDF3E_0C00 == 0x5E20_0800 {
        return two_reg_misc(insn, true);
    }
    if insn & 0x9F3E_0C00 == 0x0E30_0800 {
        return across_lanes(insn);
    }
    if insn & 0xDF3E_0C00 == 0x5E30_0800 {
        return scalar_pairwise(insn);
    }
    if insn & 0xBF20_8C00 == 0x0E00_0800 {
        return permute(insn);
    }
    if insn & 0xBFE0_8400 == 0x2E00_0000 {
        let imm4 = bits(insn, 14, 11);
        if !bit(insn, 30) && imm4 >= 8 {
            return None;
        }
        return Some(Simd::new(Op::Ext, insn, 0).imm(imm4));
    }
    if insn & 0xBFE0_8C00 == 0x0E00_0000 {
        return Some(Simd::new(Op::Tbl { tbx: bit(insn, 12) }, insn, 0).imm(bits(insn, 14, 13)));
    }
    if insn & 0x9F80_0400 == 0x0F00_0400 && bits(insn, 22, 19) != 0 {
        return shift_imm(insn, false);
    }
    if insn & 0xDF80_0400 == 0x5F00_0400 && bits(insn, 22, 19) != 0 {
        return shift_imm(insn, true);
    }
    if insn & 0x9F00_0400 == 0x0F00_0000 {
        return by_element(insn, false);
    }
    if insn & 0xDF00_0400 == 0x5F00_0000 {
        return by_element(insn, true);
    }
    None
}

/// Floating-point three-same and two-reg-misc ops use size<1> as an extra
/// opcode bit and size<0> as the precision.
fn fp_esize(size: u32) -> u8 {
    2 + (size & 1) as u8
}

fn three_same(insn: u32, scalar: bool) -> Option<Simd> {
    let u = bit(insn, 29);
    let q = bit(insn, 30);
    let size = bits(insn, 23, 22);
    let opcode = bits(insn, 15, 11);

    if opcode >= 0b11000 {
        let a = size >> 1 == 1;
        let esize = fp_esize(size);
        if !scalar && esize == 3 && !q {
            return None;
        }
        let op = fp_three_same_op(u, a, opcode, scalar)?;
        let s = Simd::new(op, insn, esize);
        return Some(if scalar { s.scalar() } else { s });
    }
    int_three_same(insn, scalar, u, q, size, opcode)
}

fn fp16_three_same(insn: u32, scalar: bool) -> Option<Simd> {
    let op = fp_three_same_op(
        bit(insn, 29),
        bit(insn, 23),
        0b11000 | bits(insn, 13, 11),
        scalar,
    )?;
    let s = Simd::new(op, insn, 1);
    Some(if scalar { s.scalar() } else { s })
}

fn fp16_two_reg_misc(insn: u32, scalar: bool) -> Option<Simd> {
    let op = fp_two_reg_op(bit(insn, 29), bit(insn, 23), bits(insn, 16, 12), scalar)?;
    let s = Simd::new(op, insn, 1);
    Some(if scalar { s.scalar() } else { s })
}

/// Floating-point three-same operations, shared by every precision.
fn fp_three_same_op(u: bool, a: bool, opcode: u32, scalar: bool) -> Option<Op> {
    {
        let op = match (u, a, opcode) {
            (false, false, 0b11000) if !scalar => Op::FArith(BinOp::MaxNum),
            (false, false, 0b11001) if !scalar => Op::FMla { sub: false },
            (false, false, 0b11010) if !scalar => Op::FArith(BinOp::Add),
            (false, false, 0b11011) => Op::FMulx,
            (false, false, 0b11100) => Op::FCompare(FCmp::Eq),
            (false, false, 0b11110) if !scalar => Op::FArith(BinOp::Max),
            (false, false, 0b11111) => Op::FRecps,
            (false, true, 0b11000) if !scalar => Op::FArith(BinOp::MinNum),
            (false, true, 0b11001) if !scalar => Op::FMla { sub: true },
            (false, true, 0b11010) if !scalar => Op::FArith(BinOp::Sub),
            (false, true, 0b11110) if !scalar => Op::FArith(BinOp::Min),
            (false, true, 0b11111) => Op::FRsqrts,
            (true, false, 0b11000) if !scalar => Op::FPairwise(BinOp::MaxNum),
            (true, false, 0b11010) if !scalar => Op::FPairwise(BinOp::Add),
            (true, false, 0b11011) if !scalar => Op::FArith(BinOp::Mul),
            (true, false, 0b11100) => Op::FCompare(FCmp::Ge),
            (true, false, 0b11101) => Op::FCompare(FCmp::AbsGe),
            (true, false, 0b11110) if !scalar => Op::FPairwise(BinOp::Max),
            (true, false, 0b11111) if !scalar => Op::FArith(BinOp::Div),
            (true, true, 0b11000) if !scalar => Op::FPairwise(BinOp::MinNum),
            (true, true, 0b11010) => Op::FAbd,
            (true, true, 0b11100) => Op::FCompare(FCmp::Gt),
            (true, true, 0b11101) => Op::FCompare(FCmp::AbsGt),
            (true, true, 0b11110) if !scalar => Op::FPairwise(BinOp::Min),
            _ => return None,
        };
        Some(op)
    }
}

fn int_three_same(
    insn: u32,
    scalar: bool,
    u: bool,
    q: bool,
    size: u32,
    opcode: u32,
) -> Option<Simd> {
    // Integer. `wide_ok`: the operation exists for 64-bit elements.
    let (op, wide_ok) = match (u, opcode) {
        (_, 0b00000) if !scalar => (
            Op::Hadd {
                unsigned: u,
                round: false,
            },
            false,
        ),
        (_, 0b00001) => (Op::QAdd { unsigned: u }, true),
        (_, 0b00010) if !scalar => (
            Op::Hadd {
                unsigned: u,
                round: true,
            },
            false,
        ),
        (_, 0b00100) if !scalar => (Op::Hsub { unsigned: u }, false),
        (_, 0b00101) => (Op::QSub { unsigned: u }, true),
        (false, 0b00110) => (Op::Compare(Cmp::Gt), true),
        (true, 0b00110) => (Op::Compare(Cmp::Hi), true),
        (false, 0b00111) => (Op::Compare(Cmp::Ge), true),
        (true, 0b00111) => (Op::Compare(Cmp::Hs), true),
        (_, 0b01000) => (
            Op::ShlReg {
                unsigned: u,
                round: false,
                sat: false,
            },
            true,
        ),
        (_, 0b01001) => (
            Op::ShlReg {
                unsigned: u,
                round: false,
                sat: true,
            },
            true,
        ),
        (_, 0b01010) => (
            Op::ShlReg {
                unsigned: u,
                round: true,
                sat: false,
            },
            true,
        ),
        (_, 0b01011) => (
            Op::ShlReg {
                unsigned: u,
                round: true,
                sat: true,
            },
            true,
        ),
        (_, 0b01100) if !scalar => (
            Op::MaxMin {
                unsigned: u,
                max: true,
                pairwise: false,
            },
            false,
        ),
        (_, 0b01101) if !scalar => (
            Op::MaxMin {
                unsigned: u,
                max: false,
                pairwise: false,
            },
            false,
        ),
        (_, 0b01110) if !scalar => (
            Op::Abd {
                unsigned: u,
                acc: false,
            },
            false,
        ),
        (_, 0b01111) if !scalar => (
            Op::Abd {
                unsigned: u,
                acc: true,
            },
            false,
        ),
        (_, 0b10000) if scalar => (Op::AddSub { sub: u }, true),
        (false, 0b10001) => (Op::Compare(Cmp::Tst), true),
        (true, 0b10001) => (Op::Compare(Cmp::Eq), true),
        (_, 0b10010) if !scalar => (Op::Mla { sub: u }, false),
        (false, 0b10011) if !scalar => (Op::Mul { poly: false }, false),
        (true, 0b10011) if !scalar && size == 0 => (Op::Mul { poly: true }, false),
        (_, 0b10100) if !scalar => (
            Op::MaxMin {
                unsigned: u,
                max: true,
                pairwise: true,
            },
            false,
        ),
        (_, 0b10101) if !scalar => (
            Op::MaxMin {
                unsigned: u,
                max: false,
                pairwise: true,
            },
            false,
        ),
        (_, 0b10110) if size == 1 || size == 2 => (Op::Qdmulh { round: u }, false),
        (false, 0b10111) if !scalar => (Op::Addp, true),
        _ => return None,
    };
    if size == 3 && (!wide_ok || (!scalar && !q)) {
        return None;
    }
    if scalar {
        // Scalar compares, shifts by register (non-saturating), ADD/SUB
        // exist for 64-bit elements only.
        let d_only = matches!(
            op,
            Op::Compare(_) | Op::AddSub { .. } | Op::ShlReg { sat: false, .. }
        );
        if d_only && size != 3 {
            return None;
        }
        return Some(Simd::new(op, insn, size as u8).scalar());
    }
    Some(Simd::new(op, insn, size as u8))
}

fn three_same_extra(insn: u32, scalar: bool) -> Option<Simd> {
    let u = bit(insn, 29);
    let size = bits(insn, 23, 22);
    let op = match (u, bits(insn, 14, 11)) {
        (true, 0b0000) if size == 1 || size == 2 => Op::Qrdmlah { sub: false },
        (true, 0b0001) if size == 1 || size == 2 => Op::Qrdmlah { sub: true },
        (_, 0b0010) if size == 2 && !scalar => Op::Dot { unsigned: u },
        _ => return None,
    };
    let s = Simd::new(op, insn, size as u8);
    Some(if scalar { s.scalar() } else { s })
}

fn three_different(insn: u32, scalar: bool) -> Option<Simd> {
    let u = bit(insn, 29);
    let size = bits(insn, 23, 22);
    if size == 3 {
        // Only PMULL/PMULL2 with 64-bit elements (Cryptographic Extension).
        return if !scalar && !u && bits(insn, 15, 12) == 0b1110 {
            Some(Simd::new(Op::Pmull64, insn, 3))
        } else {
            None
        };
    }
    let op = match (u, bits(insn, 15, 12)) {
        (_, 0b0000) => Op::AddSubLong {
            unsigned: u,
            sub: false,
            wide: false,
        },
        (_, 0b0001) => Op::AddSubLong {
            unsigned: u,
            sub: false,
            wide: true,
        },
        (_, 0b0010) => Op::AddSubLong {
            unsigned: u,
            sub: true,
            wide: false,
        },
        (_, 0b0011) => Op::AddSubLong {
            unsigned: u,
            sub: true,
            wide: true,
        },
        (_, 0b0100) => Op::AddSubHn {
            sub: false,
            round: u,
        },
        (_, 0b0101) => Op::AbdLong {
            unsigned: u,
            acc: true,
        },
        (_, 0b0110) => Op::AddSubHn {
            sub: true,
            round: u,
        },
        (_, 0b0111) => Op::AbdLong {
            unsigned: u,
            acc: false,
        },
        (_, 0b1000) => Op::MulLong {
            unsigned: u,
            acc: Acc::Add,
        },
        (false, 0b1001) if size != 0 => Op::QdmulLong { acc: Acc::Add },
        (_, 0b1010) => Op::MulLong {
            unsigned: u,
            acc: Acc::Sub,
        },
        (false, 0b1011) if size != 0 => Op::QdmulLong { acc: Acc::Sub },
        (_, 0b1100) => Op::MulLong {
            unsigned: u,
            acc: Acc::None,
        },
        (false, 0b1101) if size != 0 => Op::QdmulLong { acc: Acc::None },
        (false, 0b1110) if size == 0 => Op::Pmull,
        _ => return None,
    };
    if scalar {
        return match op {
            Op::QdmulLong { .. } => Some(Simd::new(op, insn, size as u8).scalar()),
            _ => None,
        };
    }
    Some(Simd::new(op, insn, size as u8))
}

fn two_reg_misc(insn: u32, scalar: bool) -> Option<Simd> {
    let u = bit(insn, 29);
    let q = bit(insn, 30);
    let size = bits(insn, 23, 22);
    let opcode = bits(insn, 16, 12);
    let a = size >> 1 == 1;
    let fe = fp_esize(size);

    // Floating-point forms: opcode 0b01100.. with a = 1, or 0b10110.. .
    let fp_op = fp_two_reg_op(u, a, opcode, scalar);
    if let Some(op) = fp_op {
        if !scalar && fe == 3 && !q {
            return None;
        }
        let s = Simd::new(op, insn, fe);
        return Some(if scalar { s.scalar() } else { s });
    }
    int_two_reg_misc(insn, scalar, u, q, size, opcode, a, fe)
}

/// Floating-point two-register misc operations, shared by every precision.
fn fp_two_reg_op(u: bool, a: bool, opcode: u32, scalar: bool) -> Option<Op> {
    match (u, a, opcode) {
        (false, true, 0b01100) => Some(Op::FCompareZero(FCmp::Gt)),
        (false, true, 0b01101) => Some(Op::FCompareZero(FCmp::Eq)),
        (false, true, 0b01110) => Some(Op::FCompareZero(FCmp::Lt)),
        (false, true, 0b01111) if !scalar => Some(Op::FAbs),
        (true, true, 0b01100) => Some(Op::FCompareZero(FCmp::Ge)),
        (true, true, 0b01101) => Some(Op::FCompareZero(FCmp::Le)),
        (true, true, 0b01111) if !scalar => Some(Op::FNeg),
        (true, true, 0b11111) if !scalar => Some(Op::FSqrt),
        (false, false, 0b11000) if !scalar => Some(Op::FRound(Rounding::TieEven)),
        (false, false, 0b11001) if !scalar => Some(Op::FRound(Rounding::NegInf)),
        (false, true, 0b11000) if !scalar => Some(Op::FRound(Rounding::PosInf)),
        (false, true, 0b11001) if !scalar => Some(Op::FRound(Rounding::Zero)),
        (true, false, 0b11000) if !scalar => Some(Op::FRound(Rounding::TieAway)),
        (true, false, 0b11001) if !scalar => Some(Op::FRound(Rounding::TieEven)),
        (true, true, 0b11001) if !scalar => Some(Op::FRound(Rounding::TieEven)),
        (_, false, 0b11010) => Some(Op::FToInt {
            signed: !u,
            rounding: Rounding::TieEven,
        }),
        (_, false, 0b11011) => Some(Op::FToInt {
            signed: !u,
            rounding: Rounding::NegInf,
        }),
        (_, false, 0b11100) => Some(Op::FToInt {
            signed: !u,
            rounding: Rounding::TieAway,
        }),
        (_, false, 0b11101) => Some(Op::IntToF { signed: !u }),
        (_, true, 0b11010) => Some(Op::FToInt {
            signed: !u,
            rounding: Rounding::PosInf,
        }),
        (_, true, 0b11011) => Some(Op::FToInt {
            signed: !u,
            rounding: Rounding::Zero,
        }),
        (false, true, 0b11101) => Some(Op::FRecpe),
        (true, true, 0b11101) => Some(Op::FRsqrte),
        (false, true, 0b11111) if scalar => Some(Op::FRecpx),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn int_two_reg_misc(
    insn: u32,
    scalar: bool,
    u: bool,
    q: bool,
    size: u32,
    opcode: u32,
    a: bool,
    fe: u8,
) -> Option<Simd> {
    // Narrowing/widening FP conversions.
    match (u, opcode) {
        (false, 0b10110) if !scalar && !a => {
            return Some(Simd::new(Op::FCvtNarrow { round_odd: false }, insn, fe));
        }
        (false, 0b10111) if !scalar && !a => return Some(Simd::new(Op::FCvtLong, insn, fe - 1)),
        (true, 0b10110) if size == 0b01 => {
            let s = Simd::new(Op::FCvtNarrow { round_odd: true }, insn, 3);
            return Some(if scalar { s.scalar() } else { s });
        }
        _ => {}
    }

    // URECPE / URSQRTE (vector, 32-bit only).
    if !scalar && a && size & 1 == 0 && opcode == 0b11100 {
        let op = if u { Op::Ursqrte } else { Op::Urecpe };
        return Some(Simd::new(op, insn, 2));
    }

    let op = match (u, opcode) {
        (false, 0b00000) if !scalar && size < 3 => Op::Rev { container: 3 },
        (false, 0b00001) if !scalar && size == 0 => Op::Rev { container: 1 },
        (true, 0b00000) if !scalar && size < 2 => Op::Rev { container: 2 },
        (_, 0b00010) if !scalar && size < 3 => Op::AddLongPairwise {
            unsigned: u,
            acc: false,
        },
        (_, 0b00110) if !scalar && size < 3 => Op::AddLongPairwise {
            unsigned: u,
            acc: true,
        },
        (_, 0b00011) => Op::QAccumulate { unsigned: u },
        (false, 0b00100) if !scalar && size < 3 => Op::Cls,
        (true, 0b00100) if !scalar && size < 3 => Op::Clz,
        (false, 0b00101) if !scalar && size == 0 => Op::Cnt,
        (true, 0b00101) if !scalar && size == 0 => Op::Not,
        (true, 0b00101) if !scalar && size == 1 => Op::Rbit,
        (false, 0b00111) => Op::QAbs,
        (true, 0b00111) => Op::QNeg,
        (false, 0b01000) => Op::CompareZero(Cmp::Gt),
        (true, 0b01000) => Op::CompareZero(Cmp::Ge),
        (false, 0b01001) => Op::CompareZero(Cmp::Eq),
        (true, 0b01001) => Op::CompareZero(Cmp::Le),
        (false, 0b01010) => Op::CompareZero(Cmp::Lt),
        (false, 0b01011) => Op::Abs,
        (true, 0b01011) => Op::Neg,
        (false, 0b10010) if !scalar && size < 3 => Op::Xtn,
        (true, 0b10010) if size < 3 => Op::Qxtn {
            src_unsigned: false,
            dst_unsigned: true,
        },
        (false, 0b10100) if size < 3 => Op::Qxtn {
            src_unsigned: false,
            dst_unsigned: false,
        },
        (true, 0b10100) if size < 3 => Op::Qxtn {
            src_unsigned: true,
            dst_unsigned: true,
        },
        (true, 0b10011) if !scalar && size < 3 => Op::Shll,
        _ => return None,
    };
    if matches!(op, Op::Not | Op::Rbit | Op::Cnt) {
        return Some(Simd::new(op, insn, 0));
    }
    let narrowing = matches!(op, Op::Xtn | Op::Qxtn { .. });
    if narrowing {
        // `esize` is the source (double-width) element.
        let s = Simd::new(op, insn, size as u8 + 1);
        return Some(if scalar { s.scalar() } else { s });
    }
    if scalar {
        // Scalar compares, ABS and NEG exist for 64-bit elements only.
        let d_only = matches!(op, Op::CompareZero(_) | Op::Abs | Op::Neg);
        if d_only && size != 3 {
            return None;
        }
        return Some(Simd::new(op, insn, size as u8).scalar());
    }
    if size == 3 && !q {
        return None;
    }
    Some(Simd::new(op, insn, size as u8))
}

fn across_lanes(insn: u32) -> Option<Simd> {
    let u = bit(insn, 29);
    let q = bit(insn, 30);
    let size = bits(insn, 23, 22);
    let opcode = bits(insn, 16, 12);
    if opcode == 0b01100 || opcode == 0b01111 {
        // FP reductions: U = 1 over 4 singles, U = 0 over halves.
        if size & 1 != 0 || (u && !q) {
            return None;
        }
        let max = size >> 1 == 0;
        let op = Op::FMaxMinV {
            max,
            num: opcode == 0b01100,
        };
        return Some(Simd::new(op, insn, if u { 2 } else { 1 }));
    }
    let op = match (u, opcode) {
        (_, 0b00011) => Op::AddLongV { unsigned: u },
        (_, 0b01010) => Op::MaxMinV {
            unsigned: u,
            max: true,
        },
        (_, 0b11010) => Op::MaxMinV {
            unsigned: u,
            max: false,
        },
        (false, 0b11011) => Op::AddV,
        _ => return None,
    };
    if size == 3 || (size == 2 && !q) {
        return None;
    }
    Some(Simd::new(op, insn, size as u8))
}

fn scalar_pairwise(insn: u32) -> Option<Simd> {
    let u = bit(insn, 29);
    let size = bits(insn, 23, 22);
    let opcode = bits(insn, 16, 12);
    if !u && opcode == 0b11011 {
        return if size == 3 {
            Some(Simd::new(Op::Addp, insn, 3).scalar())
        } else {
            None
        };
    }
    if !u && size & 1 != 0 {
        return None;
    }
    let a = size >> 1 == 1;
    let op = match (a, opcode) {
        (false, 0b01100) => BinOp::MaxNum,
        (true, 0b01100) => BinOp::MinNum,
        (false, 0b01101) => BinOp::Add,
        (false, 0b01111) => BinOp::Max,
        (true, 0b01111) => BinOp::Min,
        _ => return None,
    };
    // U = 0 selects half precision.
    let esize = if u { fp_esize(size) } else { 1 };
    Some(Simd::new(Op::FPairwise(op), insn, esize).scalar())
}

fn permute(insn: u32) -> Option<Simd> {
    let size = bits(insn, 23, 22);
    if size == 3 && !bit(insn, 30) {
        return None;
    }
    let op = match bits(insn, 14, 12) {
        0b001 => Op::Uzp { second: false },
        0b010 => Op::Trn { second: false },
        0b011 => Op::Zip { second: false },
        0b101 => Op::Uzp { second: true },
        0b110 => Op::Trn { second: true },
        0b111 => Op::Zip { second: true },
        _ => return None,
    };
    Some(Simd::new(op, insn, size as u8))
}

fn shift_imm(insn: u32, scalar: bool) -> Option<Simd> {
    let u = bit(insn, 29);
    let q = bit(insn, 30);
    let immh = bits(insn, 22, 19);
    let imm7 = bits(insn, 22, 16);
    let esize = (31 - immh.leading_zeros()) as u8; // highest set bit of immh
    let ebits = 8u32 << esize;
    let right = 2 * ebits - imm7;
    let left = imm7 - ebits;
    let opcode = bits(insn, 15, 11);

    let narrow = |op: Op| -> Option<Simd> {
        if esize == 3 {
            return None;
        }
        // The source element is twice the size encoded by immh.
        let s = Simd::new(op, insn, esize + 1).imm(right);
        Some(if scalar { s.scalar() } else { s })
    };
    match (u, opcode) {
        (_, 0b10000) if !u && !scalar => return narrow(Op::ShrNarrow { round: false }),
        (_, 0b10001) if !u && !scalar => return narrow(Op::ShrNarrow { round: true }),
        (true, 0b10000) => {
            return narrow(Op::QShrNarrow {
                round: false,
                src_unsigned: false,
                dst_unsigned: true,
            })
        }
        (true, 0b10001) => {
            return narrow(Op::QShrNarrow {
                round: true,
                src_unsigned: false,
                dst_unsigned: true,
            })
        }
        (_, 0b10010) => {
            return narrow(Op::QShrNarrow {
                round: false,
                src_unsigned: u,
                dst_unsigned: u,
            })
        }
        (_, 0b10011) => {
            return narrow(Op::QShrNarrow {
                round: true,
                src_unsigned: u,
                dst_unsigned: u,
            })
        }
        (_, 0b10100) if !scalar => {
            if esize == 3 {
                return None;
            }
            return Some(Simd::new(Op::ShlLong { unsigned: u }, insn, esize).imm(left));
        }
        _ => {}
    }

    let (op, amount, d_only) = match (u, opcode) {
        (_, 0b00000) => (
            Op::ShrImm {
                unsigned: u,
                round: false,
                acc: false,
            },
            right,
            true,
        ),
        (_, 0b00010) => (
            Op::ShrImm {
                unsigned: u,
                round: false,
                acc: true,
            },
            right,
            true,
        ),
        (_, 0b00100) => (
            Op::ShrImm {
                unsigned: u,
                round: true,
                acc: false,
            },
            right,
            true,
        ),
        (_, 0b00110) => (
            Op::ShrImm {
                unsigned: u,
                round: true,
                acc: true,
            },
            right,
            true,
        ),
        (false, 0b01010) => (Op::ShlImm, left, true),
        (true, 0b01000) => (Op::Sri, right, true),
        (true, 0b01010) => (Op::Sli, left, true),
        (true, 0b01100) => (
            Op::QShlImm {
                src_unsigned: false,
                dst_unsigned: true,
            },
            left,
            false,
        ),
        (_, 0b01110) => (
            Op::QShlImm {
                src_unsigned: u,
                dst_unsigned: u,
            },
            left,
            false,
        ),
        (_, 0b11100) | (_, 0b11111) => {
            // Fixed-point conversions: half, single or double precision.
            if esize < 1 {
                return None;
            }
            let op = if opcode == 0b11100 {
                Op::FixedToFp { signed: !u }
            } else {
                Op::FpToFixed { signed: !u }
            };
            (op, right, false)
        }
        _ => return None,
    };
    if scalar {
        if d_only && esize != 3 {
            return None;
        }
        return Some(Simd::new(op, insn, esize).scalar().imm(amount));
    }
    if esize == 3 && !q {
        return None;
    }
    Some(Simd::new(op, insn, esize).imm(amount))
}

fn by_element(insn: u32, scalar: bool) -> Option<Simd> {
    let u = bit(insn, 29);
    let q = bit(insn, 30);
    let size = bits(insn, 23, 22);
    let (l, m, h) = (bits(insn, 21, 21), bits(insn, 20, 20), bits(insn, 11, 11));
    let rm4 = bits(insn, 19, 16);
    let opcode = bits(insn, 15, 12);

    let is_fp = matches!((u, opcode), (false, 0b0001) | (false, 0b0101) | (_, 0b1001));
    if is_fp {
        if size == 0b01 {
            return None;
        }
        let double = size == 0b11;
        let (index, esize) = if size == 0b00 {
            // Half precision: H:L:M index, Rm limited to V0-V15.
            let mut s = Simd::new(
                match (u, opcode) {
                    (false, 0b0001) => Op::FMla { sub: false },
                    (false, 0b0101) => Op::FMla { sub: true },
                    (false, 0b1001) => Op::FArith(BinOp::Mul),
                    _ => Op::FMulx,
                },
                insn,
                1,
            );
            s.rm = rm4 as u8;
            s.index = Some((h << 2) | (l << 1) | m);
            return Some(if scalar { s.scalar() } else { s });
        } else if double {
            if l == 1 || (!scalar && !q) {
                return None;
            }
            (h, 3)
        } else {
            ((h << 1) | l, 2)
        };
        let op = match (u, opcode) {
            (false, 0b0001) => Op::FMla { sub: false },
            (false, 0b0101) => Op::FMla { sub: true },
            (false, 0b1001) => Op::FArith(BinOp::Mul),
            _ => Op::FMulx,
        };
        let mut s = Simd::new(op, insn, esize);
        s.rm = ((m << 4) | rm4) as u8;
        s.index = Some(index);
        return Some(if scalar { s.scalar() } else { s });
    }

    let (index, rm) = match size {
        1 => ((h << 2) | (l << 1) | m, rm4),
        2 => ((h << 1) | l, (m << 4) | rm4),
        _ => return None,
    };
    let op = match (u, opcode) {
        (true, 0b0000) if !scalar => Op::Mla { sub: false },
        (true, 0b0100) if !scalar => Op::Mla { sub: true },
        (false, 0b1000) if !scalar => Op::Mul { poly: false },
        (_, 0b0010) if !scalar => Op::MulLong {
            unsigned: u,
            acc: Acc::Add,
        },
        (_, 0b0110) if !scalar => Op::MulLong {
            unsigned: u,
            acc: Acc::Sub,
        },
        (_, 0b1010) if !scalar => Op::MulLong {
            unsigned: u,
            acc: Acc::None,
        },
        (false, 0b0011) => Op::QdmulLong { acc: Acc::Add },
        (false, 0b0111) => Op::QdmulLong { acc: Acc::Sub },
        (false, 0b1011) => Op::QdmulLong { acc: Acc::None },
        (false, 0b1100) => Op::Qdmulh { round: false },
        (false, 0b1101) => Op::Qdmulh { round: true },
        (true, 0b1101) => Op::Qrdmlah { sub: false },
        (true, 0b1111) => Op::Qrdmlah { sub: true },
        (_, 0b1110) if !scalar && size == 2 => Op::Dot { unsigned: u },
        _ => return None,
    };
    let mut s = Simd::new(op, insn, size as u8);
    s.rm = rm as u8;
    s.index = Some(index);
    Some(if scalar { s.scalar() } else { s })
}

// ===========================================================================
// Execution
// ===========================================================================

fn ones(n: u32) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

fn get(v: u128, esize: u8, i: u32) -> u64 {
    let b = 8u32 << esize;
    ((v >> (i * b)) as u64) & ones(b)
}

fn set(v: u128, esize: u8, i: u32, value: u64) -> u128 {
    let b = 8u32 << esize;
    let mask = u128::from(ones(b)) << (i * b);
    (v & !mask) | (u128::from(value & ones(b)) << (i * b))
}

/// Element as a signed or unsigned wide integer.
fn int(v: u64, esize: u8, unsigned: bool) -> i128 {
    let b = 8u32 << esize;
    if unsigned {
        i128::from(v & ones(b))
    } else {
        let shift = 64 - b;
        i128::from(((v << shift) as i64) >> shift)
    }
}

/// `SatQ()`: saturates to an element of `esize`, returning (value, saturated).
fn sat(v: i128, esize: u8, unsigned: bool) -> (u64, bool) {
    let b = 8u32 << esize;
    let (min, max) = if unsigned {
        (0i128, (1i128 << b) - 1)
    } else {
        (-(1i128 << (b - 1)), (1i128 << (b - 1)) - 1)
    };
    if v < min {
        (min as u64, true)
    } else if v > max {
        (max as u64, true)
    } else {
        (v as u64, false)
    }
}

/// Shifts left (positive) or arithmetically right (negative), with the
/// result kept wide enough to detect saturation.
fn shift_wide(v: i128, shift: i64) -> i128 {
    if shift >= 0 {
        if v == 0 {
            return 0;
        }
        let shift = shift.min(100) as u32;
        // Clamp far out of range values; saturation only needs the sign.
        v.checked_mul(1i128 << shift)
            .unwrap_or(if v < 0 { i128::MIN / 2 } else { i128::MAX / 2 })
    } else {
        v >> (-shift).min(127) as u32
    }
}

fn lanes(s: &Simd) -> u32 {
    if s.scalar {
        1
    } else {
        (if s.q { 16 } else { 8 }) >> s.esize
    }
}

fn broadcast(elem: u64, esize: u8) -> u128 {
    (0..(16 >> esize)).fold(0, |acc, i| set(acc, esize, i, elem))
}

fn fp_type(esize: u8) -> FpType {
    match esize {
        1 => FpType::Half,
        2 => FpType::Single,
        _ => FpType::Double,
    }
}

impl Cpu {
    fn finish_simd(&mut self, s: &Simd, result: u128) {
        let value = if s.scalar {
            result & u128::from(ones(8 << s.esize))
        } else if s.q {
            result
        } else {
            result & u128::from(u64::MAX)
        };
        self.write_v(s.rd, value);
    }

    /// Writes a narrowed result: the low half of Vd, or the high half
    /// (keeping the low half) for the "2" forms.
    fn finish_narrow(&mut self, s: &Simd, half: u64) {
        let value = if s.scalar {
            u128::from(half & ones(8 << (s.esize - 1)))
        } else if s.q {
            (self.read_v(s.rd) & u128::from(u64::MAX)) | (u128::from(half) << 64)
        } else {
            u128::from(half)
        };
        self.write_v(s.rd, value);
    }

    fn saturated(&mut self, flag: bool) {
        if flag {
            self.fpsr |= FPSR_QC;
        }
    }

    pub(crate) fn exec_simd(&mut self, s: Simd) {
        let n = self.read_v(s.rn);
        let d = self.read_v(s.rd);
        let m = match s.index {
            None => self.read_v(s.rm),
            Some(index) => {
                // SDOT/UDOT index 32-bit groups; everything else elements.
                let es = if matches!(s.op, Op::Dot { .. }) {
                    2
                } else {
                    s.esize
                };
                broadcast(get(self.read_v(s.rm), es, index), es)
            }
        };
        let e = s.esize;
        let ebits = 8u32 << e;
        let count = lanes(&s);
        let mut qc = false;

        // Applies `f` to every lane: (a, b, d) -> result.
        let map3 = |f: &mut dyn FnMut(u64, u64, u64) -> u64| -> u128 {
            (0..count).fold(0u128, |acc, i| {
                set(acc, e, i, f(get(n, e, i), get(m, e, i), get(d, e, i)))
            })
        };

        let result: u128 = match s.op {
            Op::Hadd { unsigned, round } => map3(&mut |a, b, _| {
                let r = int(a, e, unsigned) + int(b, e, unsigned) + i128::from(round);
                (r >> 1) as u64
            }),
            Op::Hsub { unsigned } => {
                map3(&mut |a, b, _| ((int(a, e, unsigned) - int(b, e, unsigned)) >> 1) as u64)
            }
            Op::QAdd { unsigned } => map3(&mut |a, b, _| {
                let (r, f) = sat(int(a, e, unsigned) + int(b, e, unsigned), e, unsigned);
                qc |= f;
                r
            }),
            Op::QSub { unsigned } => map3(&mut |a, b, _| {
                let (r, f) = sat(int(a, e, unsigned) - int(b, e, unsigned), e, unsigned);
                qc |= f;
                r
            }),
            Op::Compare(c) => map3(&mut |a, b, _| {
                let (sa, sb) = (int(a, e, false), int(b, e, false));
                let (ua, ub) = (int(a, e, true), int(b, e, true));
                let t = match c {
                    Cmp::Gt => sa > sb,
                    Cmp::Ge => sa >= sb,
                    Cmp::Hi => ua > ub,
                    Cmp::Hs => ua >= ub,
                    Cmp::Eq => ua == ub,
                    Cmp::Tst => a & b != 0,
                    Cmp::Lt => sa < sb,
                    Cmp::Le => sa <= sb,
                };
                if t {
                    u64::MAX
                } else {
                    0
                }
            }),
            Op::CompareZero(c) => {
                let mut s2 = s;
                s2.op = Op::Compare(c);
                return self.exec_compare_zero(s2, n);
            }
            Op::ShlReg {
                unsigned,
                round,
                sat: saturating,
            } => map3(&mut |a, b, _| {
                let shift = i64::from(b as u8 as i8);
                let v = int(a, e, unsigned);
                let round_const = if round && shift < 0 {
                    1i128 << (-shift - 1).min(126)
                } else {
                    0
                };
                if saturating {
                    let (r, f) = sat(shift_wide(v + round_const, shift), e, unsigned);
                    qc |= f;
                    r
                } else if shift >= i64::from(ebits) {
                    // Every bit is shifted out of the element.
                    0
                } else {
                    shift_wide(v + round_const, shift) as u64
                }
            }),
            Op::MaxMin {
                unsigned,
                max,
                pairwise,
            } => {
                let pick = |a: u64, b: u64| {
                    let (x, y) = (int(a, e, unsigned), int(b, e, unsigned));
                    if (x >= y) == max {
                        a
                    } else {
                        b
                    }
                };
                if pairwise {
                    self.pairwise(&s, n, m, &mut |a, b| pick(a, b))
                } else {
                    map3(&mut |a, b, _| pick(a, b))
                }
            }
            Op::Abd { unsigned, acc } => map3(&mut |a, b, dv| {
                let diff = (int(a, e, unsigned) - int(b, e, unsigned)).unsigned_abs() as u64;
                if acc {
                    dv.wrapping_add(diff)
                } else {
                    diff
                }
            }),
            Op::AddSub { sub } => map3(&mut |a, b, _| {
                if sub {
                    a.wrapping_sub(b)
                } else {
                    a.wrapping_add(b)
                }
            }),
            Op::Mla { sub } => map3(&mut |a, b, dv| {
                let p = a.wrapping_mul(b);
                if sub {
                    dv.wrapping_sub(p)
                } else {
                    dv.wrapping_add(p)
                }
            }),
            Op::Mul { poly } => map3(&mut |a, b, _| {
                if poly {
                    clmul(a, b) as u64
                } else {
                    a.wrapping_mul(b)
                }
            }),
            Op::Qdmulh { round } => map3(&mut |a, b, _| {
                let p = 2 * int(a, e, false) * int(b, e, false);
                let p = p + if round { 1i128 << (ebits - 1) } else { 0 };
                let (r, f) = sat(p >> ebits, e, false);
                qc |= f;
                r
            }),
            Op::Qrdmlah { sub } => map3(&mut |a, b, dv| {
                let p = 2 * int(a, e, false) * int(b, e, false);
                let acc = int(dv, e, false) << ebits;
                let total = if sub { acc - p } else { acc + p } + (1i128 << (ebits - 1));
                let (r, f) = sat(total >> ebits, e, false);
                qc |= f;
                r
            }),
            Op::Addp => {
                if s.scalar {
                    u128::from(get(n, 3, 0).wrapping_add(get(n, 3, 1)))
                } else {
                    self.pairwise(&s, n, m, &mut |a, b| a.wrapping_add(b))
                }
            }
            Op::Dot { unsigned } => (0..count).fold(0u128, |acc, i| {
                let mut sum = get(d, 2, i) as u32;
                for k in 0..4 {
                    let a = int(get(n, 0, i * 4 + k), 0, unsigned);
                    let b = int(get(m, 0, i * 4 + k), 0, unsigned);
                    sum = sum.wrapping_add((a * b) as u32);
                }
                set(acc, 2, i, u64::from(sum))
            }),

            // ---- widening and narrowing ----
            Op::AddSubLong {
                unsigned,
                sub,
                wide,
            } => {
                let part = u32::from(s.q) * (8 >> e);
                (0..(8 >> e)).fold(0u128, |acc, i| {
                    let a = if wide {
                        int(get(n, e + 1, i), e + 1, unsigned)
                    } else {
                        int(get(n, e, part + i), e, unsigned)
                    };
                    let b = int(get(m, e, part + i), e, unsigned);
                    let r = if sub { a - b } else { a + b };
                    set(acc, e + 1, i, r as u64)
                })
            }
            Op::AddSubHn { sub, round } => {
                // `e` is the narrow (destination) element here.
                let half = (0..(8 >> e)).fold(0u64, |acc, i| {
                    let (a, b) = (get(n, e + 1, i), get(m, e + 1, i));
                    let mut r = if sub {
                        a.wrapping_sub(b)
                    } else {
                        a.wrapping_add(b)
                    };
                    if round {
                        r = r.wrapping_add(1u64 << (ebits - 1));
                    }
                    let hi = (r >> ebits) & ones(ebits);
                    acc | (hi << (i * ebits))
                });
                self.finish_narrow(&Simd { esize: e + 1, ..s }, half);
                return self.saturated(qc);
            }
            Op::AbdLong { unsigned, acc } => {
                let part = u32::from(s.q) * (8 >> e);
                (0..(8 >> e)).fold(0u128, |res, i| {
                    let a = int(get(n, e, part + i), e, unsigned);
                    let b = int(get(m, e, part + i), e, unsigned);
                    let diff = (a - b).unsigned_abs() as u64;
                    let r = if acc {
                        get(d, e + 1, i).wrapping_add(diff)
                    } else {
                        diff
                    };
                    set(res, e + 1, i, r)
                })
            }
            Op::MulLong { unsigned, acc } => {
                // The "2" forms (Q = 1) use the upper halves of the sources.
                let part = if s.q { 8 >> e } else { 0 };
                (0..(8 >> e)).fold(0u128, |res, i| {
                    let p = int(get(n, e, part + i), e, unsigned)
                        * int(get(m, e, part + i), e, unsigned);
                    let dv = get(d, e + 1, i);
                    let r = match acc {
                        Acc::None => p as u64,
                        Acc::Add => dv.wrapping_add(p as u64),
                        Acc::Sub => dv.wrapping_sub(p as u64),
                    };
                    set(res, e + 1, i, r)
                })
            }
            Op::QdmulLong { acc } => {
                let count = if s.scalar { 1 } else { 8 >> e };
                let part = if s.q && !s.scalar { 8 >> e } else { 0 };
                let r = (0..count).fold(0u128, |res, i| {
                    let p =
                        2 * int(get(n, e, part + i), e, false) * int(get(m, e, part + i), e, false);
                    let (p, f1) = sat(p, e + 1, false);
                    qc |= f1;
                    let p = int(p, e + 1, false);
                    let r = match acc {
                        Acc::None => p,
                        Acc::Add => int(get(d, e + 1, i), e + 1, false) + p,
                        Acc::Sub => int(get(d, e + 1, i), e + 1, false) - p,
                    };
                    let (r, f2) = sat(r, e + 1, false);
                    qc |= f2 && acc != Acc::None;
                    set(res, e + 1, i, r)
                });
                let out = if s.scalar {
                    r & u128::from(ones(16 << e))
                } else {
                    r
                };
                self.write_v(s.rd, out);
                return self.saturated(qc);
            }
            Op::Pmull => {
                let part = u32::from(s.q) * 8;
                (0..8).fold(0u128, |res, i| {
                    set(
                        res,
                        1,
                        i,
                        clmul(get(n, 0, part + i), get(m, 0, part + i)) as u64,
                    )
                })
            }
            Op::Shll => {
                let part = u32::from(s.q) * (8 >> e);
                (0..(8 >> e)).fold(0u128, |res, i| {
                    set(res, e + 1, i, get(n, e, part + i) << ebits)
                })
            }
            Op::ShlLong { unsigned } => {
                let part = u32::from(s.q) * (8 >> e);
                (0..(8 >> e)).fold(0u128, |res, i| {
                    let v = int(get(n, e, part + i), e, unsigned) << s.imm;
                    set(res, e + 1, i, v as u64)
                })
            }
            Op::Xtn | Op::Qxtn { .. } | Op::ShrNarrow { .. } | Op::QShrNarrow { .. } => {
                // `e` is the wide source element; results have `e - 1`.
                let ne = e - 1;
                let count = if s.scalar { 1 } else { 8 >> ne };
                let half = (0..count).fold(0u64, |acc, i| {
                    let v = get(n, e, i);
                    let r = match s.op {
                        Op::Xtn => v,
                        Op::Qxtn {
                            src_unsigned,
                            dst_unsigned,
                        } => {
                            let (r, f) = sat(int(v, e, src_unsigned), ne, dst_unsigned);
                            qc |= f;
                            r
                        }
                        Op::ShrNarrow { round } => {
                            let rc = if round { 1u128 << (s.imm - 1) } else { 0 };
                            ((u128::from(v) + rc) >> s.imm) as u64
                        }
                        Op::QShrNarrow {
                            round,
                            src_unsigned,
                            dst_unsigned,
                        } => {
                            let rc = if round { 1i128 << (s.imm - 1) } else { 0 };
                            let (r, f) =
                                sat((int(v, e, src_unsigned) + rc) >> s.imm, ne, dst_unsigned);
                            qc |= f;
                            r
                        }
                        _ => unreachable!(),
                    };
                    acc | ((r & ones(8 << ne)) << (i * (8 << ne)))
                });
                self.finish_narrow(&s, half);
                return self.saturated(qc);
            }

            // ---- two-register misc ----
            Op::Rev { container } => {
                let per = 1u32 << (container - e);
                map_index(count, e, |i| {
                    get(n, e, (i / per) * per + (per - 1 - i % per))
                })
            }
            Op::AddLongPairwise { unsigned, acc } => {
                let out = (if s.q { 16 } else { 8 }) >> (e + 1);
                (0..out).fold(0u128, |res, i| {
                    let sum =
                        int(get(n, e, 2 * i), e, unsigned) + int(get(n, e, 2 * i + 1), e, unsigned);
                    let r = if acc {
                        (sum as u64).wrapping_add(get(d, e + 1, i))
                    } else {
                        sum as u64
                    };
                    set(res, e + 1, i, r)
                })
            }
            Op::QAccumulate { unsigned } => map3(&mut |a, _, dv| {
                // SUQADD: signed acc += unsigned a. USQADD: unsigned acc += signed a.
                let (r, f) = sat(int(dv, e, unsigned) + int(a, e, !unsigned), e, unsigned);
                qc |= f;
                r
            }),
            Op::Cls => map3(&mut |a, _, _| {
                let v = a << (64 - ebits);
                let lead = if (v as i64) < 0 {
                    v.leading_ones()
                } else {
                    v.leading_zeros()
                };
                u64::from(lead.min(ebits) - 1)
            }),
            Op::Clz => {
                map3(&mut |a, _, _| u64::from((a << (64 - ebits)).leading_zeros().min(ebits)))
            }
            Op::Cnt => map3(&mut |a, _, _| u64::from(a.count_ones())),
            Op::Not => map3(&mut |a, _, _| !a),
            Op::Rbit => map3(&mut |a, _, _| u64::from((a as u8).reverse_bits())),
            Op::QAbs | Op::QNeg | Op::Abs | Op::Neg => map3(&mut |a, _, _| {
                let v = int(a, e, false);
                let r = match s.op {
                    Op::QAbs | Op::Abs => v.abs(),
                    _ => -v,
                };
                if matches!(s.op, Op::QAbs | Op::QNeg) {
                    let (r, f) = sat(r, e, false);
                    qc |= f;
                    r
                } else {
                    r as u64
                }
            }),

            // ---- across lanes ----
            Op::AddV => u128::from(
                (0..count).fold(0u64, |acc, i| acc.wrapping_add(get(n, e, i))) & ones(ebits),
            ),
            Op::AddLongV { unsigned } => {
                let sum = (0..count).fold(0i128, |acc, i| acc + int(get(n, e, i), e, unsigned));
                u128::from(sum as u64 & ones(ebits * 2))
            }
            Op::MaxMinV { unsigned, max } => {
                let mut best = get(n, e, 0);
                for i in 1..count {
                    let v = get(n, e, i);
                    let better = if max {
                        int(v, e, unsigned) > int(best, e, unsigned)
                    } else {
                        int(v, e, unsigned) < int(best, e, unsigned)
                    };
                    if better {
                        best = v;
                    }
                }
                u128::from(best)
            }
            Op::FMaxMinV { max, num } => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                let op = match (max, num) {
                    (true, false) => BinOp::Max,
                    (false, false) => BinOp::Min,
                    (true, true) => BinOp::MaxNum,
                    (false, true) => BinOp::MinNum,
                };
                // `Reduce()`: a pairwise tree, lower half first.
                let mut vals: Vec<u64> = (0..count).map(|i| get(n, e, i)).collect();
                while vals.len() > 1 {
                    vals = vals.chunks(2).map(|p| fp.binop(op, p[0], p[1])).collect();
                }
                u128::from(vals[0])
            }

            // ---- permute, extract, table ----
            Op::Uzp { second } => {
                let half = count / 2;
                map_index(count, e, |i| {
                    let src = if i < half { n } else { m };
                    get(src, e, 2 * (i % half) + u32::from(second))
                })
            }
            Op::Trn { second } => map_index(count, e, |i| {
                let base = (i & !1) + u32::from(second);
                if i % 2 == 0 {
                    get(n, e, base)
                } else {
                    get(m, e, base)
                }
            }),
            Op::Zip { second } => {
                let base = u32::from(second) * count / 2;
                map_index(count, e, |i| {
                    if i % 2 == 0 {
                        get(n, e, base + i / 2)
                    } else {
                        get(m, e, base + i / 2)
                    }
                })
            }
            Op::Ext => {
                let width = if s.q { 16 } else { 8 };
                let pos = s.imm;
                (0..width).fold(0u128, |res, i| {
                    let k = pos + i;
                    let byte = if k < width {
                        get(n, 0, k)
                    } else {
                        get(m, 0, k - width)
                    };
                    set(res, 0, i, byte)
                })
            }
            Op::Tbl { tbx } => {
                let regs = s.imm + 1;
                (0..count).fold(0u128, |res, i| {
                    let idx = get(m, 0, i) as u32;
                    let byte = if idx < 16 * regs {
                        get(
                            self.read_v(((u32::from(s.rn) + idx / 16) % 32) as u8),
                            0,
                            idx % 16,
                        )
                    } else if tbx {
                        get(d, 0, i)
                    } else {
                        0
                    };
                    set(res, 0, i, byte)
                })
            }

            // ---- shift by immediate ----
            Op::ShrImm {
                unsigned,
                round,
                acc,
            } => map3(&mut |a, _, dv| {
                let rc = if round { 1i128 << (s.imm - 1) } else { 0 };
                let r = ((int(a, e, unsigned) + rc) >> s.imm) as u64;
                if acc {
                    dv.wrapping_add(r)
                } else {
                    r
                }
            }),
            Op::ShlImm => map3(&mut |a, _, _| a << s.imm),
            Op::Sri => map3(&mut |a, _, dv| {
                let mask = if s.imm >= 64 { 0 } else { ones(ebits) >> s.imm };
                let shifted = if s.imm >= 64 {
                    0
                } else {
                    (a & ones(ebits)) >> s.imm
                };
                (dv & !mask) | (shifted & mask)
            }),
            Op::Sli => map3(&mut |a, _, dv| {
                let mask = ones(ebits) << s.imm;
                (dv & !mask) | ((a << s.imm) & mask)
            }),
            Op::QShlImm {
                src_unsigned,
                dst_unsigned,
            } => map3(&mut |a, _, _| {
                let (r, f) = sat(
                    shift_wide(int(a, e, src_unsigned), i64::from(s.imm)),
                    e,
                    dst_unsigned,
                );
                qc |= f;
                r
            }),
            Op::FixedToFp { signed } => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.from_int(a, signed, ebits, s.imm))
            }
            Op::FpToFixed { signed } => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.to_int(a, signed, ebits, s.imm, Rounding::Zero))
            }

            // ---- floating point ----
            Op::FArith(op) => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, b, _| fp.binop(op, a, b))
            }
            Op::FPairwise(op) => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                if s.scalar {
                    u128::from(fp.binop(op, get(n, e, 0), get(n, e, 1)))
                } else {
                    self.pairwise(&s, n, m, &mut |a, b| fp.binop(op, a, b))
                }
            }
            Op::FMla { sub } => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, b, dv| {
                    let a = if sub { fp.neg(a) } else { a };
                    fp.mul_add(dv, a, b)
                })
            }
            Op::FMulx => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, b, _| fp.mulx(a, b))
            }
            Op::FRecps => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, b, _| fp.recip_step(a, b))
            }
            Op::FRsqrts => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, b, _| fp.rsqrt_step(a, b))
            }
            Op::FAbd => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                // FPAbs clears the sign of a NaN result too.
                map3(&mut |a, b, _| fp.abs(fp.binop(BinOp::Sub, a, b)))
            }
            Op::FCompare(c) => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, b, _| fcompare(fp, c, a, b))
            }
            Op::FCompareZero(c) => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fcompare(fp, c, a, 0))
            }
            Op::FAbs => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.abs(a))
            }
            Op::FNeg => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.neg(a))
            }
            Op::FSqrt => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.sqrt(a))
            }
            Op::FRound(mode) => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.round_int(a, mode))
            }
            Op::FToInt { signed, rounding } => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.to_int(a, signed, ebits, 0, rounding))
            }
            Op::IntToF { signed } => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.from_int(a, signed, ebits, 0))
            }
            Op::FRecpe => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.recip_estimate(a))
            }
            Op::FRsqrte => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.rsqrt_estimate(a))
            }
            Op::FRecpx => {
                let fp = Fp::new(fp_type(e), self.fpcr);
                map3(&mut |a, _, _| fp.recpx(a))
            }
            Op::Urecpe => map3(&mut |a, _, _| {
                if a >> 31 & 1 == 0 {
                    0xFFFF_FFFF
                } else {
                    fp::recip_estimate_int((a >> 23) & 0x1FF) << 23
                }
            }),
            Op::Ursqrte => map3(&mut |a, _, _| {
                if a >> 30 & 3 == 0 {
                    0xFFFF_FFFF
                } else {
                    fp::rsqrt_estimate_int((a >> 23) & 0x1FF) << 23
                }
            }),
            Op::FCvtNarrow { round_odd } => {
                // `e` is the source precision (3 = double, 2 = single).
                let dn = self.fpcr & fp::FPCR_DN != 0;
                let count = if s.scalar { 1 } else { 8 >> (e - 1) };
                let half = (0..count).fold(0u64, |acc, i| {
                    let v = get(n, e, i);
                    let r = if e == 3 {
                        let fp = Fp::new(FpType::Double, self.fpcr);
                        if round_odd {
                            fp.convert_round_odd(v)
                        } else {
                            fp.convert(v, FpType::Single)
                        }
                    } else {
                        u64::from(fp::to_half(v, FpType::Single, dn))
                    };
                    acc | ((r & ones(8 << (e - 1))) << (i * (8 << (e - 1))))
                });
                self.finish_narrow(&s, half);
                return;
            }
            Op::Aese | Op::Aesd | Op::Aesmc | Op::Aesimc => {
                use crate::crypto;
                let v = match s.op {
                    Op::Aese => crypto::aese(d, n),
                    Op::Aesd => crypto::aesd(d, n),
                    Op::Aesmc => crypto::aesmc(n),
                    _ => crypto::aesimc(n),
                };
                self.write_v(s.rd, v);
                return;
            }
            Op::Sha1Rounds(f) => {
                let v = crate::crypto::sha1_rounds(d, n as u32, m, f);
                self.write_v(s.rd, v);
                return;
            }
            Op::Sha1H => {
                self.write_v(s.rd, u128::from(crate::crypto::sha1h(n as u32)));
                return;
            }
            Op::Sha1Su0 => {
                self.write_v(s.rd, crate::crypto::sha1su0(d, n, m));
                return;
            }
            Op::Sha1Su1 => {
                self.write_v(s.rd, crate::crypto::sha1su1(d, n));
                return;
            }
            Op::Sha256H => {
                self.write_v(s.rd, crate::crypto::sha256_hash(d, n, m, true));
                return;
            }
            Op::Sha256H2 => {
                self.write_v(s.rd, crate::crypto::sha256_hash(n, d, m, false));
                return;
            }
            Op::Sha256Su0 => {
                self.write_v(s.rd, crate::crypto::sha256su0(d, n));
                return;
            }
            Op::Sha256Su1 => {
                self.write_v(s.rd, crate::crypto::sha256su1(d, n, m));
                return;
            }
            Op::Pmull64 => {
                let half = u32::from(s.q);
                let v = crate::crypto::clmul64(get(n, 3, half), get(m, 3, half));
                self.write_v(s.rd, v);
                return;
            }
            Op::FCvtLong => {
                // `e` is the source precision (2 = single, 1 = half).
                let dn = self.fpcr & fp::FPCR_DN != 0;
                let part = u32::from(s.q) * (8 >> e);
                (0..(8 >> e)).fold(0u128, |res, i| {
                    let v = get(n, e, part + i);
                    let r = if e == 2 {
                        Fp::new(FpType::Single, self.fpcr).convert(v, FpType::Double)
                    } else {
                        fp::half_to(v as u16, FpType::Single, dn)
                    };
                    set(res, e + 1, i, r)
                })
            }
        };

        let widening = matches!(
            s.op,
            Op::AddSubLong { .. }
                | Op::AbdLong { .. }
                | Op::MulLong { .. }
                | Op::Pmull
                | Op::Shll
                | Op::ShlLong { .. }
                | Op::FCvtLong
        );
        // Widening results and reductions fill or clear the whole register.
        let whole = widening
            || matches!(
                s.op,
                Op::AddLongV { .. } | Op::AddV | Op::MaxMinV { .. } | Op::FMaxMinV { .. }
            );
        if whole {
            self.write_v(s.rd, result);
        } else if matches!(s.op, Op::AddLongPairwise { .. }) {
            self.write_v(
                s.rd,
                if s.q {
                    result
                } else {
                    result & u128::from(u64::MAX)
                },
            );
        } else {
            self.finish_simd(&s, result);
        }
        self.saturated(qc);
    }

    fn exec_compare_zero(&mut self, s: Simd, n: u128) {
        let e = s.esize;
        let Op::Compare(c) = s.op else { unreachable!() };
        let result = (0..lanes(&s)).fold(0u128, |acc, i| {
            let v = int(get(n, e, i), e, false);
            let t = match c {
                Cmp::Gt => v > 0,
                Cmp::Ge => v >= 0,
                Cmp::Eq => v == 0,
                Cmp::Le => v <= 0,
                Cmp::Lt => v < 0,
                _ => unreachable!(),
            };
            set(acc, e, i, if t { u64::MAX } else { 0 })
        });
        self.finish_simd(&s, result);
    }

    /// Pairwise operation over the concatenation `m:n` (n in the low half).
    fn pairwise(&self, s: &Simd, n: u128, m: u128, f: &mut dyn FnMut(u64, u64) -> u64) -> u128 {
        let e = s.esize;
        let count = lanes(s);
        let half = count / 2;
        (0..count).fold(0u128, |acc, i| {
            let src = if i < half { n } else { m };
            let j = 2 * (i % half);
            set(acc, e, i, f(get(src, e, j), get(src, e, j + 1)))
        })
    }
}

fn map_index(count: u32, e: u8, f: impl Fn(u32) -> u64) -> u128 {
    (0..count).fold(0u128, |acc, i| set(acc, e, i, f(i)))
}

/// Carry-less (polynomial) multiplication of two bytes.
fn clmul(a: u64, b: u64) -> u128 {
    let (a, b) = (a & 0xFF, b & 0xFF);
    (0..8).fold(0u128, |acc, i| {
        if (b >> i) & 1 == 1 {
            acc ^ (u128::from(a) << i)
        } else {
            acc
        }
    })
}

fn fcompare(fp: Fp, c: FCmp, a: u64, b: u64) -> u64 {
    use std::cmp::Ordering::*;
    let (x, y) = match c {
        FCmp::AbsGe | FCmp::AbsGt => (fp.abs(a), fp.abs(b)),
        _ => (a, b),
    };
    let t = match (c, fp.compare(x, y)) {
        (_, None) => false,
        (FCmp::Eq, Some(o)) => o == Equal,
        (FCmp::Ge | FCmp::AbsGe, Some(o)) => o != Less,
        (FCmp::Gt | FCmp::AbsGt, Some(o)) => o == Greater,
        (FCmp::Lt, Some(o)) => o == Less,
        (FCmp::Le, Some(o)) => o != Greater,
    };
    if t {
        u64::MAX
    } else {
        0
    }
}
