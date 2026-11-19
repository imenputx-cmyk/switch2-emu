//! A64 instruction decoder.
//!
//! Each instruction class is decoded from the bit layouts in the ARM
//! Architecture Reference Manual (section C4, "A64 Instruction Set
//! Encoding"). Test vectors are produced with `llvm-mc -triple=aarch64
//! -show-encoding`, and semantics are cross-checked against Unicorn by
//! `tools/difftest.py`.
//!
//! Only general-purpose (integer) instructions are covered so far. SIMD and
//! floating point decode as [`Instr::Unknown`].

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Orr,
    Eor,
    /// AND that also sets flags.
    Ands,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitfieldOp {
    Sbfm,
    Bfm,
    Ubfm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shift {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

/// Register extension used by extended-register forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extend {
    Uxtb,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}

impl Extend {
    fn from_option(option: u32) -> Extend {
        match option & 7 {
            0 => Extend::Uxtb,
            1 => Extend::Uxth,
            2 => Extend::Uxtw,
            3 => Extend::Uxtx,
            4 => Extend::Sxtb,
            5 => Extend::Sxth,
            6 => Extend::Sxtw,
            _ => Extend::Sxtx,
        }
    }
}

/// What a single-register load/store does with the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadKind {
    Store,
    /// Load and zero-extend.
    Load,
    /// Load and sign-extend to 64 bits.
    LoadSigned64,
    /// Load and sign-extend to 32 bits (upper half zeroed).
    LoadSigned32,
    /// PRFM: a hint, no architectural effect.
    Prefetch,
}

/// How a load/store computes its address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrMode {
    /// `[Xn|SP, #imm]`
    Offset { rn: u8, imm: i64 },
    /// `[Xn|SP, #imm]!`
    PreIndex { rn: u8, imm: i64 },
    /// `[Xn|SP], #imm`
    PostIndex { rn: u8, imm: i64 },
    /// `[Xn|SP, Rm, extend #shift]`
    RegOffset {
        rn: u8,
        rm: u8,
        extend: Extend,
        shift: u32,
    },
    /// PC-relative literal.
    Literal { offset: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairMode {
    Offset,
    PreIndex,
    PostIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExclusiveOp {
    /// LDXR / LDAXR.
    LoadExclusive,
    /// STXR / STLXR. `rs` receives 0 on success, 1 on failure.
    StoreExclusive { rs: u8 },
    /// LDAR / LDLAR.
    LoadAcquire,
    /// STLR / STLLR.
    StoreRelease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondSelectOp {
    Csel,
    Csinc,
    Csinv,
    Csneg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondOperand {
    Reg(u8),
    Imm(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataProc2Op {
    Udiv,
    Sdiv,
    Lslv,
    Lsrv,
    Asrv,
    Rorv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataProc1Op {
    Rbit,
    Rev16,
    /// REV32 (64-bit only): reverse bytes in each word.
    Rev32,
    /// REV: reverse all bytes of the register.
    Rev,
    Clz,
    Cls,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataProc3Op {
    Madd,
    Msub,
    Smaddl,
    Smsubl,
    Umaddl,
    Umsubl,
    Smulh,
    Umulh,
}

/// System registers reachable from EL0 that are implemented so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SysReg {
    Nzcv,
    Fpcr,
    Fpsr,
    TpidrEl0,
    TpidrroEl0,
    CntvctEl0,
    CntfrqEl0,
    CtrEl0,
    DczidEl0,
}

impl SysReg {
    /// Decodes `o0:op1:CRn:CRm:op2` (instruction bits [19:5]).
    fn from_key(key: u32) -> Option<SysReg> {
        Some(match key {
            0x5a10 => SysReg::Nzcv,
            0x5a20 => SysReg::Fpcr,
            0x5a21 => SysReg::Fpsr,
            0x5e82 => SysReg::TpidrEl0,
            0x5e83 => SysReg::TpidrroEl0,
            0x5f02 => SysReg::CntvctEl0,
            0x5f00 => SysReg::CntfrqEl0,
            0x5801 => SysReg::CtrEl0,
            0x5807 => SysReg::DczidEl0,
            _ => return None,
        })
    }
}

/// A decoded A64 instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instr {
    /// ADR and ADRP. `imm` is the final byte offset (already shifted by
    /// 12 for ADRP).
    Adr {
        page: bool,
        imm: i64,
        rd: u8,
    },
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
    /// AND, ORR, EOR, ANDS (immediate). `imm` is the decoded bitmask.
    LogicalImm {
        sf: bool,
        op: LogicOp,
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
    /// SBFM, BFM, UBFM (and their aliases LSL, LSR, ASR, SXTB, UBFX, ...).
    Bitfield {
        sf: bool,
        op: BitfieldOp,
        immr: u32,
        imms: u32,
        wmask: u64,
        tmask: u64,
        rn: u8,
        rd: u8,
    },
    /// EXTR (and its alias ROR immediate).
    Extract {
        sf: bool,
        lsb: u32,
        rm: u8,
        rn: u8,
        rd: u8,
    },

    /// B and BL. `offset` is in bytes, relative to this instruction.
    Branch {
        link: bool,
        offset: i64,
    },
    /// B.cond.
    BranchCond {
        cond: u8,
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
    /// TBZ and TBNZ.
    TestBranch {
        bit: u32,
        nonzero: bool,
        offset: i64,
        rt: u8,
    },

    /// Supervisor call. This is how guest code talks to the OS kernel.
    Svc(u16),
    /// Breakpoint.
    Brk(u16),
    /// Hints (NOP, YIELD, ...) and barriers (DMB, DSB, ISB). On a single
    /// in-order interpreter these have no effect.
    Nop,
    /// CLREX: clear the exclusive monitor.
    Clrex,
    Mrs {
        reg: SysReg,
        rt: u8,
    },
    Msr {
        reg: SysReg,
        rt: u8,
    },

    /// Single-register loads and stores. `size` is log2 of the access
    /// width in bytes (0 = byte ... 3 = doubleword).
    LoadStore {
        size: u8,
        kind: LoadKind,
        mode: AddrMode,
        rt: u8,
    },
    /// LDP, STP, LDPSW, LDNP, STNP. `size` is log2 bytes per register.
    LoadStorePair {
        size: u8,
        load: bool,
        signed: bool,
        mode: PairMode,
        rn: u8,
        imm: i64,
        rt: u8,
        rt2: u8,
    },
    /// Exclusive and acquire/release single-register accesses.
    Exclusive {
        size: u8,
        op: ExclusiveOp,
        rn: u8,
        rt: u8,
    },

    /// AND, BIC, ORR, ORN, EOR, EON, ANDS, BICS (shifted register).
    LogicalReg {
        sf: bool,
        op: LogicOp,
        invert: bool,
        shift: Shift,
        amount: u32,
        rm: u8,
        rn: u8,
        rd: u8,
    },
    /// ADD, ADDS, SUB, SUBS (shifted register).
    AddSubReg {
        sf: bool,
        sub: bool,
        set_flags: bool,
        shift: Shift,
        amount: u32,
        rm: u8,
        rn: u8,
        rd: u8,
    },
    /// ADD, ADDS, SUB, SUBS (extended register).
    AddSubExt {
        sf: bool,
        sub: bool,
        set_flags: bool,
        extend: Extend,
        amount: u32,
        rm: u8,
        rn: u8,
        rd: u8,
    },
    /// ADC, ADCS, SBC, SBCS.
    AddSubCarry {
        sf: bool,
        sub: bool,
        set_flags: bool,
        rm: u8,
        rn: u8,
        rd: u8,
    },
    /// CCMN, CCMP (register or immediate).
    CondCompare {
        sf: bool,
        sub: bool,
        cond: u8,
        operand: CondOperand,
        nzcv: u8,
        rn: u8,
    },
    /// CSEL, CSINC, CSINV, CSNEG (and aliases CSET, CINC, CNEG, ...).
    CondSelect {
        sf: bool,
        op: CondSelectOp,
        cond: u8,
        rm: u8,
        rn: u8,
        rd: u8,
    },
    DataProc2 {
        sf: bool,
        op: DataProc2Op,
        rm: u8,
        rn: u8,
        rd: u8,
    },
    DataProc1 {
        sf: bool,
        op: DataProc1Op,
        rn: u8,
        rd: u8,
    },
    /// MADD, MSUB (MUL, MNEG), the long multiplies and SMULH/UMULH.
    DataProc3 {
        sf: bool,
        op: DataProc3Op,
        rm: u8,
        ra: u8,
        rn: u8,
        rd: u8,
    },

    /// Not yet implemented or unallocated.
    Unknown(u32),
}

fn bits(insn: u32, hi: u32, lo: u32) -> u32 {
    (insn >> lo) & (((1u64 << (hi - lo + 1)) - 1) as u32)
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

/// A value with the low `n` bits set (`n` may be 64).
pub(crate) fn ones(n: u32) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

/// Rotates the low `width` bits of `x` right by `r`.
pub(crate) fn ror(x: u64, r: u32, width: u32) -> u64 {
    let x = x & ones(width);
    let r = r % width;
    if r == 0 {
        x
    } else {
        ((x >> r) | (x << (width - r))) & ones(width)
    }
}

fn replicate(elem: u64, esize: u32, datasize: u32) -> u64 {
    let mut out = 0;
    let mut pos = 0;
    while pos < datasize {
        out |= elem << pos;
        pos += esize;
    }
    out & ones(datasize)
}

/// `DecodeBitMasks()` from the ARM pseudocode. Returns `(wmask, tmask)`,
/// or `None` for reserved encodings.
pub fn decode_bit_masks(
    n: bool,
    imms: u32,
    immr: u32,
    immediate: bool,
    datasize: u32,
) -> Option<(u64, u64)> {
    let combined = (u32::from(n) << 6) | (!imms & 0x3F);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 {
        return None;
    }
    let esize = 1u32 << len;
    if esize > datasize {
        return None;
    }
    let levels = esize - 1;
    if immediate && (imms & levels) == levels {
        return None;
    }
    let s = imms & levels;
    let r = immr & levels;
    let d = s.wrapping_sub(r) & levels;
    let welem = ones(s + 1);
    let telem = ones(d + 1);
    let wmask = replicate(ror(welem, r, esize), esize, datasize);
    let tmask = replicate(telem, esize, datasize);
    Some((wmask, tmask))
}

fn shift_type(insn: u32) -> Shift {
    match bits(insn, 23, 22) {
        0 => Shift::Lsl,
        1 => Shift::Lsr,
        2 => Shift::Asr,
        _ => Shift::Ror,
    }
}

pub fn decode(insn: u32) -> Instr {
    let op0 = bits(insn, 28, 25);
    let result = match op0 {
        0b1000 | 0b1001 => decode_dp_imm(insn),
        0b1010 | 0b1011 => decode_branch_sys(insn),
        0b0100 | 0b0110 | 0b1100 | 0b1110 => decode_load_store(insn),
        0b0101 | 0b1101 => decode_dp_reg(insn),
        _ => None,
    };
    result.unwrap_or(Instr::Unknown(insn))
}

fn decode_dp_imm(insn: u32) -> Option<Instr> {
    let sf = bit(insn, 31);
    let datasize = if sf { 64 } else { 32 };
    let rn = reg(insn, 5);
    let rd = reg(insn, 0);
    match bits(insn, 25, 23) {
        0b000 | 0b001 => {
            let page = bit(insn, 31);
            let raw = (bits(insn, 23, 5) << 2) | bits(insn, 30, 29);
            let imm = sign_extend(raw, 21);
            Some(Instr::Adr {
                page,
                imm: if page { imm << 12 } else { imm },
                rd,
            })
        }
        0b010 => {
            let imm12 = u64::from(bits(insn, 21, 10));
            Some(Instr::AddSubImm {
                sf,
                sub: bit(insn, 30),
                set_flags: bit(insn, 29),
                imm: if bit(insn, 22) { imm12 << 12 } else { imm12 },
                rn,
                rd,
            })
        }
        0b100 => {
            let n = bit(insn, 22);
            if !sf && n {
                return None;
            }
            let (imm, _) =
                decode_bit_masks(n, bits(insn, 15, 10), bits(insn, 21, 16), true, datasize)?;
            let op = match bits(insn, 30, 29) {
                0 => LogicOp::And,
                1 => LogicOp::Orr,
                2 => LogicOp::Eor,
                _ => LogicOp::Ands,
            };
            Some(Instr::LogicalImm {
                sf,
                op,
                imm,
                rn,
                rd,
            })
        }
        0b101 => {
            let hw = bits(insn, 22, 21);
            let op = match bits(insn, 30, 29) {
                0b00 => MoveWideOp::Movn,
                0b10 => MoveWideOp::Movz,
                0b11 => MoveWideOp::Movk,
                _ => return None,
            };
            if !sf && hw >= 2 {
                return None;
            }
            Some(Instr::MoveWide {
                sf,
                op,
                shift: hw * 16,
                imm16: bits(insn, 20, 5) as u16,
                rd,
            })
        }
        0b110 => {
            let n = bit(insn, 22);
            let immr = bits(insn, 21, 16);
            let imms = bits(insn, 15, 10);
            if sf != n || (!sf && (immr >= 32 || imms >= 32)) {
                return None;
            }
            let op = match bits(insn, 30, 29) {
                0 => BitfieldOp::Sbfm,
                1 => BitfieldOp::Bfm,
                2 => BitfieldOp::Ubfm,
                _ => return None,
            };
            let (wmask, tmask) = decode_bit_masks(n, imms, immr, false, datasize)?;
            Some(Instr::Bitfield {
                sf,
                op,
                immr,
                imms,
                wmask,
                tmask,
                rn,
                rd,
            })
        }
        0b111 => {
            let lsb = bits(insn, 15, 10);
            if bits(insn, 30, 29) != 0 || bit(insn, 21) || sf != bit(insn, 22) || (!sf && lsb >= 32)
            {
                return None;
            }
            Some(Instr::Extract {
                sf,
                lsb,
                rm: reg(insn, 16),
                rn,
                rd,
            })
        }
        _ => None,
    }
}

fn decode_branch_sys(insn: u32) -> Option<Instr> {
    // Unconditional branch (immediate).
    if insn & 0x7C00_0000 == 0x1400_0000 {
        return Some(Instr::Branch {
            link: bit(insn, 31),
            offset: sign_extend(bits(insn, 25, 0), 26) * 4,
        });
    }
    // Conditional branch.
    if insn & 0xFF00_0010 == 0x5400_0000 {
        return Some(Instr::BranchCond {
            cond: bits(insn, 3, 0) as u8,
            offset: sign_extend(bits(insn, 23, 5), 19) * 4,
        });
    }
    // Compare and branch.
    if insn & 0x7E00_0000 == 0x3400_0000 {
        return Some(Instr::CompareBranch {
            sf: bit(insn, 31),
            nonzero: bit(insn, 24),
            offset: sign_extend(bits(insn, 23, 5), 19) * 4,
            rt: reg(insn, 0),
        });
    }
    // Test and branch.
    if insn & 0x7E00_0000 == 0x3600_0000 {
        return Some(Instr::TestBranch {
            bit: (bits(insn, 31, 31) << 5) | bits(insn, 23, 19),
            nonzero: bit(insn, 24),
            offset: sign_extend(bits(insn, 18, 5), 14) * 4,
            rt: reg(insn, 0),
        });
    }
    // Unconditional branch (register), plain forms only.
    let op = match insn & 0xFFFF_FC1F {
        0xD61F_0000 => Some(BranchRegOp::Br),
        0xD63F_0000 => Some(BranchRegOp::Blr),
        0xD65F_0000 => Some(BranchRegOp::Ret),
        _ => None,
    };
    if let Some(op) = op {
        return Some(Instr::BranchReg {
            op,
            rn: reg(insn, 5),
        });
    }
    // Exception generation.
    let imm16 = bits(insn, 20, 5) as u16;
    match insn & 0xFFE0_001F {
        0xD400_0001 => return Some(Instr::Svc(imm16)),
        0xD420_0000 => return Some(Instr::Brk(imm16)),
        _ => {}
    }
    // Hints.
    if insn & 0xFFFF_F01F == 0xD503_201F {
        return Some(Instr::Nop);
    }
    // Barriers and CLREX.
    if insn & 0xFFFF_F01F == 0xD503_301F {
        return match bits(insn, 7, 5) {
            0b010 => Some(Instr::Clrex),
            0b100..=0b110 => Some(Instr::Nop),
            _ => None,
        };
    }
    // MRS / MSR (register).
    let key = bits(insn, 19, 5);
    match insn & 0xFFF0_0000 {
        0xD530_0000 => {
            return SysReg::from_key(key).map(|reg_| Instr::Mrs {
                reg: reg_,
                rt: reg(insn, 0),
            })
        }
        0xD510_0000 => {
            return SysReg::from_key(key).map(|reg_| Instr::Msr {
                reg: reg_,
                rt: reg(insn, 0),
            })
        }
        _ => {}
    }
    None
}

/// Maps `size:opc` of an integer load/store to what it does.
fn load_kind(size: u32, opc: u32) -> Option<LoadKind> {
    Some(match (opc, size) {
        (0, _) => LoadKind::Store,
        (1, _) => LoadKind::Load,
        (2, 3) => LoadKind::Prefetch,
        (2, _) => LoadKind::LoadSigned64,
        (3, 0..=1) => LoadKind::LoadSigned32,
        _ => return None,
    })
}

fn decode_load_store(insn: u32) -> Option<Instr> {
    // SIMD&FP loads and stores are not implemented yet.
    if bit(insn, 26) {
        return None;
    }
    let size = bits(insn, 31, 30);
    let rn = reg(insn, 5);
    let rt = reg(insn, 0);

    // Load/store exclusive and acquire/release.
    if insn & 0x3F00_0000 == 0x0800_0000 {
        let o2 = bit(insn, 23);
        let l = bit(insn, 22);
        let o1 = bit(insn, 21);
        if o1 {
            return None; // Pair exclusives and CAS: later.
        }
        // Rt2 must be all ones, and Rs too unless it is the status register
        // of a store-exclusive.
        let store_exclusive = !o2 && !l;
        if reg(insn, 10) != 31 || (!store_exclusive && reg(insn, 16) != 31) {
            return None;
        }
        let op = match (o2, l) {
            (false, false) => ExclusiveOp::StoreExclusive { rs: reg(insn, 16) },
            (false, true) => ExclusiveOp::LoadExclusive,
            (true, false) => ExclusiveOp::StoreRelease,
            (true, true) => ExclusiveOp::LoadAcquire,
        };
        return Some(Instr::Exclusive {
            size: size as u8,
            op,
            rn,
            rt,
        });
    }

    // Load register (literal).
    if insn & 0x3B00_0000 == 0x1800_0000 {
        let offset = sign_extend(bits(insn, 23, 5), 19) * 4;
        let (size, kind) = match bits(insn, 31, 30) {
            0 => (2, LoadKind::Load),
            1 => (3, LoadKind::Load),
            2 => (2, LoadKind::LoadSigned64),
            _ => (3, LoadKind::Prefetch),
        };
        return Some(Instr::LoadStore {
            size,
            kind,
            mode: AddrMode::Literal { offset },
            rt,
        });
    }

    // Load/store pair.
    if insn & 0x3A00_0000 == 0x2800_0000 {
        let opc = bits(insn, 31, 30);
        let load = bit(insn, 22);
        let mode = match bits(insn, 24, 23) {
            0 | 2 => PairMode::Offset,
            1 => PairMode::PostIndex,
            _ => PairMode::PreIndex,
        };
        let (size, signed) = match opc {
            0 => (2, false),
            1 if load && bits(insn, 24, 23) != 0 => (2, true), // LDPSW
            2 => (3, false),
            _ => return None,
        };
        let imm = sign_extend(bits(insn, 21, 15), 7) << size;
        return Some(Instr::LoadStorePair {
            size,
            load,
            signed,
            mode,
            rn,
            imm,
            rt,
            rt2: reg(insn, 10),
        });
    }

    let opc = bits(insn, 23, 22);

    // Load/store register (unsigned immediate).
    if insn & 0x3B00_0000 == 0x3900_0000 {
        let kind = load_kind(size, opc)?;
        let imm = i64::from(bits(insn, 21, 10)) << size;
        return Some(Instr::LoadStore {
            size: size as u8,
            kind,
            mode: AddrMode::Offset { rn, imm },
            rt,
        });
    }

    // Load/store register (unscaled, post-index, unprivileged, pre-index).
    if insn & 0x3B20_0000 == 0x3800_0000 {
        let kind = load_kind(size, opc)?;
        let imm = sign_extend(bits(insn, 20, 12), 9);
        let form = bits(insn, 11, 10);
        // Only the unscaled form (PRFUM) has a prefetch variant.
        if kind == LoadKind::Prefetch && form != 0b00 {
            return None;
        }
        let mode = match form {
            0b01 => AddrMode::PostIndex { rn, imm },
            0b11 => AddrMode::PreIndex { rn, imm },
            // Unscaled and unprivileged both behave as a plain offset at EL0.
            _ => AddrMode::Offset { rn, imm },
        };
        return Some(Instr::LoadStore {
            size: size as u8,
            kind,
            mode,
            rt,
        });
    }

    // Load/store register (register offset).
    if insn & 0x3B20_0C00 == 0x3820_0800 {
        let kind = load_kind(size, opc)?;
        let option = bits(insn, 15, 13);
        if option & 0b010 == 0 {
            return None;
        }
        let shift = if bit(insn, 12) { size } else { 0 };
        return Some(Instr::LoadStore {
            size: size as u8,
            kind,
            mode: AddrMode::RegOffset {
                rn,
                rm: reg(insn, 16),
                extend: Extend::from_option(option),
                shift,
            },
            rt,
        });
    }

    None
}

fn decode_dp_reg(insn: u32) -> Option<Instr> {
    let sf = bit(insn, 31);
    let rm = reg(insn, 16);
    let rn = reg(insn, 5);
    let rd = reg(insn, 0);
    let amount = bits(insn, 15, 10);

    // Logical (shifted register).
    if insn & 0x1F00_0000 == 0x0A00_0000 {
        if !sf && amount >= 32 {
            return None;
        }
        let op = match bits(insn, 30, 29) {
            0 => LogicOp::And,
            1 => LogicOp::Orr,
            2 => LogicOp::Eor,
            _ => LogicOp::Ands,
        };
        return Some(Instr::LogicalReg {
            sf,
            op,
            invert: bit(insn, 21),
            shift: shift_type(insn),
            amount,
            rm,
            rn,
            rd,
        });
    }

    // Add/subtract (shifted register).
    if insn & 0x1F20_0000 == 0x0B00_0000 {
        let shift = shift_type(insn);
        if shift == Shift::Ror || (!sf && amount >= 32) {
            return None;
        }
        return Some(Instr::AddSubReg {
            sf,
            sub: bit(insn, 30),
            set_flags: bit(insn, 29),
            shift,
            amount,
            rm,
            rn,
            rd,
        });
    }

    // Add/subtract (extended register).
    if insn & 0x1F20_0000 == 0x0B20_0000 {
        let imm3 = bits(insn, 12, 10);
        if bits(insn, 23, 22) != 0 || imm3 > 4 {
            return None;
        }
        return Some(Instr::AddSubExt {
            sf,
            sub: bit(insn, 30),
            set_flags: bit(insn, 29),
            extend: Extend::from_option(bits(insn, 15, 13)),
            amount: imm3,
            rm,
            rn,
            rd,
        });
    }

    // Add/subtract with carry.
    if insn & 0x1FE0_FC00 == 0x1A00_0000 {
        return Some(Instr::AddSubCarry {
            sf,
            sub: bit(insn, 30),
            set_flags: bit(insn, 29),
            rm,
            rn,
            rd,
        });
    }

    // Conditional compare.
    if insn & 0x1FE0_0000 == 0x1A40_0000 {
        if !bit(insn, 29) || bit(insn, 10) || bit(insn, 4) {
            return None;
        }
        let operand = if bit(insn, 11) {
            CondOperand::Imm(u64::from(bits(insn, 20, 16)))
        } else {
            CondOperand::Reg(rm)
        };
        return Some(Instr::CondCompare {
            sf,
            sub: bit(insn, 30),
            cond: bits(insn, 15, 12) as u8,
            operand,
            nzcv: bits(insn, 3, 0) as u8,
            rn,
        });
    }

    // Conditional select.
    if insn & 0x1FE0_0000 == 0x1A80_0000 {
        if bit(insn, 29) {
            return None;
        }
        let op = match (bit(insn, 30), bits(insn, 11, 10)) {
            (false, 0) => CondSelectOp::Csel,
            (false, 1) => CondSelectOp::Csinc,
            (true, 0) => CondSelectOp::Csinv,
            (true, 1) => CondSelectOp::Csneg,
            _ => return None,
        };
        return Some(Instr::CondSelect {
            sf,
            op,
            cond: bits(insn, 15, 12) as u8,
            rm,
            rn,
            rd,
        });
    }

    // Data-processing (2 source).
    if insn & 0x7FE0_0000 == 0x1AC0_0000 {
        let op = match amount {
            0b000010 => DataProc2Op::Udiv,
            0b000011 => DataProc2Op::Sdiv,
            0b001000 => DataProc2Op::Lslv,
            0b001001 => DataProc2Op::Lsrv,
            0b001010 => DataProc2Op::Asrv,
            0b001011 => DataProc2Op::Rorv,
            _ => return None,
        };
        return Some(Instr::DataProc2 { sf, op, rm, rn, rd });
    }

    // Data-processing (1 source).
    if insn & 0x7FFF_0000 == 0x5AC0_0000 {
        let op = match (amount, sf) {
            (0, _) => DataProc1Op::Rbit,
            (1, _) => DataProc1Op::Rev16,
            (2, false) => DataProc1Op::Rev,
            (2, true) => DataProc1Op::Rev32,
            (3, true) => DataProc1Op::Rev,
            (4, _) => DataProc1Op::Clz,
            (5, _) => DataProc1Op::Cls,
            _ => return None,
        };
        return Some(Instr::DataProc1 { sf, op, rn, rd });
    }

    // Data-processing (3 source).
    if insn & 0x1F00_0000 == 0x1B00_0000 {
        if bits(insn, 30, 29) != 0 {
            return None;
        }
        let o0 = bit(insn, 15);
        let op = match (bits(insn, 23, 21), o0, sf) {
            (0, false, _) => DataProc3Op::Madd,
            (0, true, _) => DataProc3Op::Msub,
            (1, false, true) => DataProc3Op::Smaddl,
            (1, true, true) => DataProc3Op::Smsubl,
            (2, false, true) => DataProc3Op::Smulh,
            (5, false, true) => DataProc3Op::Umaddl,
            (5, true, true) => DataProc3Op::Umsubl,
            (6, false, true) => DataProc3Op::Umulh,
            _ => return None,
        };
        return Some(Instr::DataProc3 {
            sf,
            op,
            rm,
            ra: reg(insn, 10),
            rn,
            rd,
        });
    }

    None
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
    fn bit_masks_match_known_immediates() {
        // and x0, x1, #0xff  -> N=1 immr=0 imms=7
        assert_eq!(decode_bit_masks(true, 7, 0, true, 64).unwrap().0, 0xff);
        // orr w0, w1, #0x55555555 -> N=0 immr=0 imms=0b111100
        assert_eq!(
            decode_bit_masks(false, 0b111100, 0, true, 32).unwrap().0,
            0x5555_5555
        );
        // All-ones is not encodable as a logical immediate.
        assert_eq!(decode_bit_masks(true, 0b111111, 0, true, 64), None);
    }

    #[test]
    fn nop_and_unknown() {
        assert_eq!(decode(le([0x1f, 0x20, 0x03, 0xd5])), Instr::Nop);
        assert_eq!(decode(0), Instr::Unknown(0));
    }
}
