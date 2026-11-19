//! A64 instruction decoder.
//!
//! Each instruction class is decoded from the bit layouts in the ARM
//! Architecture Reference Manual (section C4, "A64 Instruction Set
//! Encoding"). Test vectors are produced with `llvm-mc -triple=aarch64
//! -show-encoding`, so the expected values come from a real assembler.

/// Operation of a move-wide instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveWideOp {
    /// MOVN: move the inverted immediate.
    Movn,
    /// MOVZ: move the immediate, zeroing other bits.
    Movz,
    /// MOVK: insert the immediate, keeping other bits.
    Movk,
}

/// Kind of unconditional branch to a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchRegOp {
    Br,
    Blr,
    Ret,
}

/// A decoded A64 instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instr {
    /// ADD, ADDS, SUB, SUBS (immediate).
    AddSubImm {
        /// 64-bit operation when true, 32-bit when false.
        sf: bool,
        sub: bool,
        set_flags: bool,
        /// Immediate, already shifted left by 12 when `sh` was set.
        imm: u64,
        rn: u8,
        rd: u8,
    },
    /// MOVN, MOVZ, MOVK.
    MoveWide {
        sf: bool,
        op: MoveWideOp,
        /// Bit position of the 16-bit immediate: 0, 16, 32 or 48.
        shift: u32,
        imm16: u16,
        rd: u8,
    },
    /// B and BL. `offset` is in bytes, relative to this instruction.
    Branch {
        link: bool,
        offset: i64,
    },
    /// BR, BLR, RET.
    BranchReg {
        op: BranchRegOp,
        rn: u8,
    },
    /// CBZ and CBNZ. `offset` is in bytes, relative to this instruction.
    CompareBranch {
        sf: bool,
        nonzero: bool,
        offset: i64,
        rt: u8,
    },
    Nop,
    /// Not yet implemented or unallocated.
    Unknown(u32),
}

fn bits(insn: u32, hi: u32, lo: u32) -> u32 {
    (insn >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}

fn bit(insn: u32, n: u32) -> bool {
    (insn >> n) & 1 == 1
}

fn reg(insn: u32, lo: u32) -> u8 {
    bits(insn, lo + 4, lo) as u8
}

/// Sign-extends the low `width` bits of `value`.
fn sign_extend(value: u32, width: u32) -> i64 {
    let shift = 64 - width;
    ((value as i64) << shift) >> shift
}

pub fn decode(insn: u32) -> Instr {
    // NOP is a hint with a fixed encoding.
    if insn == 0xD503_201F {
        return Instr::Nop;
    }

    // Add/subtract (immediate): bits[28:23] = 100010.
    if insn & 0x1F80_0000 == 0x1100_0000 {
        let imm12 = u64::from(bits(insn, 21, 10));
        let sh = bit(insn, 22);
        return Instr::AddSubImm {
            sf: bit(insn, 31),
            sub: bit(insn, 30),
            set_flags: bit(insn, 29),
            imm: if sh { imm12 << 12 } else { imm12 },
            rn: reg(insn, 5),
            rd: reg(insn, 0),
        };
    }

    // Move wide (immediate): bits[28:23] = 100101.
    if insn & 0x1F80_0000 == 0x1280_0000 {
        let sf = bit(insn, 31);
        let hw = bits(insn, 22, 21);
        let op = match bits(insn, 30, 29) {
            0b00 => MoveWideOp::Movn,
            0b10 => MoveWideOp::Movz,
            0b11 => MoveWideOp::Movk,
            _ => return Instr::Unknown(insn),
        };
        if !sf && hw >= 2 {
            return Instr::Unknown(insn);
        }
        return Instr::MoveWide {
            sf,
            op,
            shift: hw * 16,
            imm16: bits(insn, 20, 5) as u16,
            rd: reg(insn, 0),
        };
    }

    // Unconditional branch (immediate): bits[30:26] = 00101.
    if insn & 0x7C00_0000 == 0x1400_0000 {
        return Instr::Branch {
            link: bit(insn, 31),
            offset: sign_extend(bits(insn, 25, 0), 26) * 4,
        };
    }

    // Unconditional branch (register), the plain forms only.
    let op = match insn & 0xFFFF_FC1F {
        0xD61F_0000 => Some(BranchRegOp::Br),
        0xD63F_0000 => Some(BranchRegOp::Blr),
        0xD65F_0000 => Some(BranchRegOp::Ret),
        _ => None,
    };
    if let Some(op) = op {
        return Instr::BranchReg {
            op,
            rn: reg(insn, 5),
        };
    }

    // Compare and branch: bits[30:25] = 011010.
    if insn & 0x7E00_0000 == 0x3400_0000 {
        return Instr::CompareBranch {
            sf: bit(insn, 31),
            nonzero: bit(insn, 24),
            offset: sign_extend(bits(insn, 23, 5), 19) * 4,
            rt: reg(insn, 0),
        };
    }

    Instr::Unknown(insn)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Turns llvm-mc's little-endian byte list into an instruction word.
    fn le(b: [u8; 4]) -> u32 {
        u32::from_le_bytes(b)
    }

    #[test]
    fn add_sub_immediate() {
        // add x0, x1, #1
        assert_eq!(
            decode(le([0x20, 0x04, 0x00, 0x91])),
            Instr::AddSubImm {
                sf: true,
                sub: false,
                set_flags: false,
                imm: 1,
                rn: 1,
                rd: 0
            }
        );
        // add x2, sp, #16, lsl #12
        assert_eq!(
            decode(le([0xe2, 0x43, 0x40, 0x91])),
            Instr::AddSubImm {
                sf: true,
                sub: false,
                set_flags: false,
                imm: 16 << 12,
                rn: 31,
                rd: 2
            }
        );
        // adds w3, w4, #4095
        assert_eq!(
            decode(le([0x83, 0xfc, 0x3f, 0x31])),
            Instr::AddSubImm {
                sf: false,
                sub: false,
                set_flags: true,
                imm: 4095,
                rn: 4,
                rd: 3
            }
        );
        // sub sp, sp, #32
        assert_eq!(
            decode(le([0xff, 0x83, 0x00, 0xd1])),
            Instr::AddSubImm {
                sf: true,
                sub: true,
                set_flags: false,
                imm: 32,
                rn: 31,
                rd: 31
            }
        );
        // cmp x5, #7 (subs xzr, x5, #7)
        assert_eq!(
            decode(le([0xbf, 0x1c, 0x00, 0xf1])),
            Instr::AddSubImm {
                sf: true,
                sub: true,
                set_flags: true,
                imm: 7,
                rn: 5,
                rd: 31
            }
        );
    }

    #[test]
    fn move_wide() {
        // movz x0, #0x1234
        assert_eq!(
            decode(le([0x80, 0x46, 0x82, 0xd2])),
            Instr::MoveWide {
                sf: true,
                op: MoveWideOp::Movz,
                shift: 0,
                imm16: 0x1234,
                rd: 0
            }
        );
        // movz x1, #0xbeef, lsl #48
        assert_eq!(
            decode(le([0xe1, 0xdd, 0xf7, 0xd2])),
            Instr::MoveWide {
                sf: true,
                op: MoveWideOp::Movz,
                shift: 48,
                imm16: 0xbeef,
                rd: 1
            }
        );
        // movn w2, #0
        assert_eq!(
            decode(le([0x02, 0x00, 0x80, 0x12])),
            Instr::MoveWide {
                sf: false,
                op: MoveWideOp::Movn,
                shift: 0,
                imm16: 0,
                rd: 2
            }
        );
        // movk x3, #0xffff, lsl #16
        assert_eq!(
            decode(le([0xe3, 0xff, 0xbf, 0xf2])),
            Instr::MoveWide {
                sf: true,
                op: MoveWideOp::Movk,
                shift: 16,
                imm16: 0xffff,
                rd: 3
            }
        );
    }

    #[test]
    fn branches() {
        assert_eq!(
            decode(le([0x02, 0x00, 0x00, 0x14])),
            Instr::Branch {
                link: false,
                offset: 8
            }
        );
        assert_eq!(
            decode(le([0xff, 0xff, 0xff, 0x17])),
            Instr::Branch {
                link: false,
                offset: -4
            }
        );
        assert_eq!(
            decode(le([0x40, 0x00, 0x00, 0x94])),
            Instr::Branch {
                link: true,
                offset: 0x100
            }
        );
        assert_eq!(
            decode(le([0xc0, 0x03, 0x5f, 0xd6])),
            Instr::BranchReg {
                op: BranchRegOp::Ret,
                rn: 30
            }
        );
        assert_eq!(
            decode(le([0xa0, 0x00, 0x5f, 0xd6])),
            Instr::BranchReg {
                op: BranchRegOp::Ret,
                rn: 5
            }
        );
        assert_eq!(
            decode(le([0xe0, 0x00, 0x1f, 0xd6])),
            Instr::BranchReg {
                op: BranchRegOp::Br,
                rn: 7
            }
        );
        assert_eq!(
            decode(le([0x20, 0x01, 0x3f, 0xd6])),
            Instr::BranchReg {
                op: BranchRegOp::Blr,
                rn: 9
            }
        );
        // cbz x0, #12
        assert_eq!(
            decode(le([0x60, 0x00, 0x00, 0xb4])),
            Instr::CompareBranch {
                sf: true,
                nonzero: false,
                offset: 12,
                rt: 0
            }
        );
        // cbnz w1, #-8
        assert_eq!(
            decode(le([0xc1, 0xff, 0xff, 0x35])),
            Instr::CompareBranch {
                sf: false,
                nonzero: true,
                offset: -8,
                rt: 1
            }
        );
    }

    #[test]
    fn nop_and_unknown() {
        assert_eq!(decode(le([0x1f, 0x20, 0x03, 0xd5])), Instr::Nop);
        assert_eq!(decode(0), Instr::Unknown(0));
    }
}
