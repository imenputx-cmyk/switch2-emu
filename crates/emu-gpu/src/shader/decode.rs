//! Decoding Maxwell instructions (the subset emulated so far).
//!
//! Common fields: destination register in bits 7:0, guard predicate in
//! 19:16 (bit 19 negates; predicate 7 is PT, always true). Register 255
//! is RZ (reads zero, writes discarded).

pub const RZ: u8 = 255;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inst {
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
    Exit,
    Nop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// An instruction this decoder does not know (raw bits).
    Unknown(u64),
    /// A guard predicate other than PT.
    Predicated(u64),
}

fn bits(v: u64, lo: u32, len: u32) -> u64 {
    (v >> lo) & ((1u64 << len) - 1)
}

fn reg(v: u64, lo: u32) -> u8 {
    bits(v, lo, 8) as u8
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

/// Decodes one instruction.
pub fn decode(v: u64) -> Result<Inst, DecodeError> {
    let top = |mask: u64, value: u64| v & mask == value;
    let dst = reg(v, 0);
    let a = reg(v, 8);
    // NOP and EXIT may be predicated in practice; everything decoded here
    // requires PT.
    let inst = if top(0xfff8 << 48, 0x50b0 << 48) {
        Inst::Nop
    } else if top(0xfff8 << 48, 0xe300 << 48) {
        // Condition code in bits 4:0; 0xF is "always".
        if bits(v, 0, 5) != 0xF {
            return Err(DecodeError::Unknown(v));
        }
        Inst::Exit
    } else if top(0xfff8 << 48, 0x5c98 << 48) {
        Inst::Mov {
            dst,
            src: Operand::Reg(reg(v, 20)),
        }
    } else if top(0xfff8 << 48, 0x4c98 << 48) {
        Inst::Mov { dst, src: cbuf(v) }
    } else if top(0xfef8 << 48, 0x3898 << 48) {
        Inst::Mov {
            dst,
            src: Operand::Imm(s20(v)),
        }
    } else if top(0xfff0 << 48, 0x0100 << 48) {
        Inst::Mov {
            dst,
            src: Operand::Imm(bits(v, 20, 32) as u32),
        }
    } else if top(0xfff8 << 48, 0x5c58 << 48)
        || top(0xfff8 << 48, 0x4c58 << 48)
        || top(0xfef8 << 48, 0x3858 << 48)
    {
        let b = match v >> 56 {
            0x5c => Operand::Reg(reg(v, 20)),
            0x4c => cbuf(v),
            _ => Operand::Imm(f20(v)),
        };
        Inst::Fadd {
            dst,
            a,
            a_mods: Mods {
                neg: v >> 48 & 1 != 0,
                abs: v >> 46 & 1 != 0,
            },
            b,
            b_mods: Mods {
                neg: v >> 45 & 1 != 0,
                abs: v >> 49 & 1 != 0,
            },
            sat: v >> 50 & 1 != 0,
        }
    } else if top(0xfc00 << 48, 0x0800 << 48) {
        // FADD32I
        Inst::Fadd {
            dst,
            a,
            a_mods: Mods {
                neg: v >> 56 & 1 != 0,
                abs: v >> 54 & 1 != 0,
            },
            b: Operand::Imm(bits(v, 20, 32) as u32),
            b_mods: Mods {
                neg: v >> 53 & 1 != 0,
                abs: v >> 57 & 1 != 0,
            },
            sat: false,
        }
    } else if top(0xfff8 << 48, 0x5c68 << 48)
        || top(0xfff8 << 48, 0x4c68 << 48)
        || top(0xfef8 << 48, 0x3868 << 48)
    {
        let b = match v >> 56 {
            0x5c => Operand::Reg(reg(v, 20)),
            0x4c => cbuf(v),
            _ => Operand::Imm(f20(v)),
        };
        Inst::Fmul {
            dst,
            a,
            b,
            neg_b: v >> 48 & 1 != 0,
            sat: v >> 50 & 1 != 0,
        }
    } else if top(0xff00 << 48, 0x1e00 << 48) {
        // FMUL32I
        Inst::Fmul {
            dst,
            a,
            b: Operand::Imm(bits(v, 20, 32) as u32),
            neg_b: false,
            sat: v >> 55 & 1 != 0,
        }
    } else if top(0xff80 << 48, 0x5980 << 48)
        || top(0xff80 << 48, 0x4980 << 48)
        || top(0xff80 << 48, 0x5180 << 48)
        || top(0xfe80 << 48, 0x3280 << 48)
    {
        let r20 = Operand::Reg(reg(v, 20));
        let r39 = Operand::Reg(reg(v, 39));
        let (b, c) = match v >> 55 {
            0xb3 => (r20, r39),     // 0x5980
            0x93 => (cbuf(v), r39), // 0x4980
            0xa3 => (r39, cbuf(v)), // 0x5180
            _ => (Operand::Imm(f20(v)), r39),
        };
        Inst::Ffma {
            dst,
            a,
            b,
            neg_b: v >> 48 & 1 != 0,
            c,
            neg_c: v >> 49 & 1 != 0,
            sat: v >> 50 & 1 != 0,
        }
    } else if top(0xfff8 << 48, 0xefd8 << 48) {
        Inst::Ald {
            dst,
            base: a,
            addr: bits(v, 20, 10) as u32,
            count: bits(v, 47, 2) as u32 + 1,
        }
    } else if top(0xfff8 << 48, 0xeff0 << 48) {
        Inst::Ast {
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
        Inst::Ipa {
            dst,
            addr: bits(v, 28, 10) as u32,
            mode,
            sat: v >> 51 & 1 != 0,
        }
    } else {
        return Err(DecodeError::Unknown(v));
    };
    if bits(v, 16, 4) != 7 {
        return Err(DecodeError::Predicated(v));
    }
    Ok(inst)
}

/// Decodes a program (instructions after the header) up to the first
/// EXIT, skipping the scheduling words.
pub fn decode_program(code: &[u8]) -> Result<Vec<Inst>, DecodeError> {
    let mut out = Vec::new();
    for (i, q) in code.chunks_exact(8).enumerate() {
        if i % 4 == 0 {
            continue; // scheduling control
        }
        let inst = decode(u64::from_le_bytes(q.try_into().unwrap()))?;
        out.push(inst);
        if inst == Inst::Exit {
            return Ok(out);
        }
    }
    Ok(out)
}
