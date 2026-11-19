//! A64 instruction decoder.
//!
//! Each instruction class is decoded from the bit layouts in the ARM
//! Architecture Reference Manual (section C4, "A64 Instruction Set
//! Encoding"). Test vectors are produced with `llvm-mc -triple=aarch64
//! -show-encoding`, and semantics are cross-checked against Unicorn by
//! `tools/difftest.py`.
//!
//! Covered: the general-purpose (integer) instructions, scalar single and
//! double precision floating point, SIMD&FP loads and stores, and the most
//! common Advanced SIMD moves and lane-wise operations. Everything else
//! decodes as [`Instr::Unknown`].

pub use crate::fp::{BinOp as FpBinOp, FpType, Rounding};

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

/// Read-modify-write operation of an LSE atomic (LDADD, LDCLR, ..., SWP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    /// Bit clear: `old & !value`.
    Clr,
    Eor,
    /// Bit set: `old | value`.
    Set,
    Smax,
    Smin,
    Umax,
    Umin,
    Swp,
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
    /// CRC32 or CRC32C (`castagnoli`) over `1 << size` bytes of Rm.
    Crc32 {
        castagnoli: bool,
        size: u8,
    },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpUnaryOp {
    Mov,
    Abs,
    Neg,
    Sqrt,
    /// FCVT to the given precision.
    Convert(FpType),
    /// FRINTx. FRINTX and FRINTI use the FPCR mode, which is always
    /// round-to-nearest here.
    Round(Rounding),
}

/// Operation of an Advanced SIMD modified-immediate instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecImmOp {
    /// MOVI and FMOV (vector, immediate).
    Movi,
    Mvni,
    Orr,
    Bic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecLogicOp {
    And,
    Bic,
    Orr,
    Orn,
    Eor,
    Bsl,
    Bit,
    Bif,
}

/// Base-register writeback of a SIMD structure load/store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostIndex {
    /// Add the number of bytes transferred.
    Imm,
    /// Add the value of Xm.
    Reg(u8),
}

/// Source of DUP and INS: a general register or a vector element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemSrc {
    Gp(u8),
    Element { rn: u8, index: u32 },
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
    /// width in bytes (0 = byte ... 3 = doubleword, 4 = quadword for SIMD).
    /// With `simd`, `rt` names a SIMD&FP register.
    LoadStore {
        simd: bool,
        size: u8,
        kind: LoadKind,
        mode: AddrMode,
        rt: u8,
    },
    /// LDP, STP, LDPSW, LDNP, STNP. `size` is log2 bytes per register.
    LoadStorePair {
        simd: bool,
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

    /// LD1-LD4 / ST1-ST4 (multiple structures): `rpt` consecutive groups
    /// of `selem` interleaved registers starting at Vt.
    SimdLoadStoreMulti {
        load: bool,
        q: bool,
        size: u8,
        rpt: u8,
        selem: u8,
        rn: u8,
        rt: u8,
        post: Option<PostIndex>,
    },
    /// LD1-LD4 / ST1-ST4 (single structure) and LD1R-LD4R (`replicate`).
    /// `size` is log2 bytes per element.
    SimdLoadStoreSingle {
        load: bool,
        replicate: bool,
        q: bool,
        size: u8,
        index: u32,
        selem: u8,
        rn: u8,
        rt: u8,
        post: Option<PostIndex>,
    },
    /// LDXP, LDAXP, STXP, STLXP. `sf` selects 64-bit registers. For
    /// stores, `store` holds the status register.
    ExclusivePair {
        sf: bool,
        store: Option<u8>,
        rn: u8,
        rt: u8,
        rt2: u8,
    },
    /// LSE atomic read-modify-write: Rt receives the old value.
    Atomic {
        size: u8,
        op: AtomicOp,
        rs: u8,
        rn: u8,
        rt: u8,
    },
    /// CAS family: if `[Xn] == Rs` then `[Xn] = Rt`; Rs receives the old
    /// value either way.
    CompareSwap {
        size: u8,
        rs: u8,
        rn: u8,
        rt: u8,
    },
    /// CASP family, on the register pairs (Rs, Rs+1) and (Rt, Rt+1).
    CompareSwapPair {
        sf: bool,
        rs: u8,
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

    /// FMOV (register), FABS, FNEG, FSQRT, FCVT, FRINTx.
    FpUnary {
        ty: FpType,
        op: FpUnaryOp,
        rn: u8,
        rd: u8,
    },
    /// FADD, FSUB, FMUL, FDIV, FMAX, FMIN, FMAXNM, FMINNM, FNMUL.
    FpBinary {
        ty: FpType,
        op: FpBinOp,
        rm: u8,
        rn: u8,
        rd: u8,
    },
    /// FMADD, FMSUB, FNMADD, FNMSUB: `(+/-)ra + (+/-)(rn * rm)`.
    FpMulAdd {
        ty: FpType,
        neg_addend: bool,
        neg_product: bool,
        rm: u8,
        ra: u8,
        rn: u8,
        rd: u8,
    },
    /// FCMP, FCMPE. `rm` is `None` for the compare-with-zero form.
    FpCompare {
        ty: FpType,
        rn: u8,
        rm: Option<u8>,
    },
    /// FCCMP, FCCMPE.
    FpCondCompare {
        ty: FpType,
        rn: u8,
        rm: u8,
        cond: u8,
        nzcv: u8,
    },
    FpCondSelect {
        ty: FpType,
        rm: u8,
        rn: u8,
        cond: u8,
        rd: u8,
    },
    /// FMOV (scalar, immediate). `imm` holds the expanded bits.
    FpMovImm {
        ty: FpType,
        imm: u64,
        rd: u8,
    },
    /// FCVTxS, FCVTxU, including fixed-point forms (`fbits` > 0).
    FpToInt {
        sf: bool,
        ty: FpType,
        signed: bool,
        rounding: Rounding,
        fbits: u32,
        rn: u8,
        rd: u8,
    },
    /// SCVTF, UCVTF, including fixed-point forms.
    IntToFp {
        sf: bool,
        ty: FpType,
        signed: bool,
        fbits: u32,
        rn: u8,
        rd: u8,
    },
    /// FCVTxS/FCVTxU/SCVTF/UCVTF (Advanced SIMD scalar): the integer
    /// stays in a SIMD&FP register of the same width as `ty`.
    FpIntInVec {
        ty: FpType,
        to_int: bool,
        signed: bool,
        rounding: Rounding,
        rn: u8,
        rd: u8,
    },
    /// FMOV from a SIMD&FP register to a general register. `top` reads
    /// V.D[1].
    FmovToGp {
        sf: bool,
        top: bool,
        half: bool,
        rn: u8,
        rd: u8,
    },
    /// FMOV from a general register. `top` writes V.D[1] only.
    FmovFromGp {
        sf: bool,
        top: bool,
        half: bool,
        rn: u8,
        rd: u8,
    },

    /// MOVI, MVNI, ORR/BIC (vector, immediate), FMOV (vector, immediate).
    /// `imm` is the 64-bit pattern, replicated to 128 bits when `q`.
    VecMoveImm {
        q: bool,
        op: VecImmOp,
        imm: u64,
        rd: u8,
    },
    /// DUP. `esize` is log2 of the element size in bytes.
    VecDup {
        q: bool,
        esize: u8,
        src: ElemSrc,
        rd: u8,
    },
    /// INS (general or element), alias MOV.
    VecIns {
        esize: u8,
        index: u32,
        src: ElemSrc,
        rd: u8,
    },
    /// UMOV, SMOV. `sf` selects a 64-bit destination.
    VecToGp {
        signed: bool,
        sf: bool,
        esize: u8,
        index: u32,
        rn: u8,
        rd: u8,
    },
    /// DUP (element, scalar), alias MOV: copies one element to a scalar.
    ScalarDup {
        esize: u8,
        index: u32,
        rn: u8,
        rd: u8,
    },
    /// AND, BIC, ORR, ORN, EOR, BSL, BIT, BIF (vector).
    VecLogical {
        q: bool,
        op: VecLogicOp,
        rm: u8,
        rn: u8,
        rd: u8,
    },
    /// ADD, SUB (vector).
    VecAddSub {
        q: bool,
        sub: bool,
        esize: u8,
        rm: u8,
        rn: u8,
        rd: u8,
    },

    /// Other Advanced SIMD data processing, see [`crate::simd`].
    Simd(crate::simd::Simd),

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
        0b0111 | 0b1111 => decode_simd_fp(insn),
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

/// Size (log2 bytes) and direction of a SIMD&FP single-register access:
/// the opc high bit extends `size` up to 128 bits.
fn simd_size_kind(size: u32, opc: u32) -> Option<(u32, LoadKind)> {
    let scale = ((opc >> 1) << 2) | size;
    if scale > 4 {
        return None;
    }
    let kind = if opc & 1 == 1 {
        LoadKind::Load
    } else {
        LoadKind::Store
    };
    Some((scale, kind))
}

fn decode_simd_structure(insn: u32) -> Option<Instr> {
    let q = bit(insn, 30);
    let load = bit(insn, 22);
    let rn = reg(insn, 5);
    let rt = reg(insn, 0);
    let post = if bit(insn, 23) {
        let rm = reg(insn, 16);
        Some(if rm == 31 {
            PostIndex::Imm
        } else {
            PostIndex::Reg(rm)
        })
    } else {
        if bits(insn, 20, 16) != 0 {
            return None;
        }
        None
    };
    let size = bits(insn, 11, 10) as u8;

    if !bit(insn, 24) {
        // Multiple structures.
        if bit(insn, 21) {
            return None;
        }
        let (rpt, selem) = match bits(insn, 15, 12) {
            0b0000 => (1, 4),
            0b0010 => (4, 1),
            0b0100 => (1, 3),
            0b0110 => (3, 1),
            0b0111 => (1, 1),
            0b1000 => (1, 2),
            0b1010 => (2, 1),
            _ => return None,
        };
        if size == 3 && !q && selem > 1 {
            return None;
        }
        return Some(Instr::SimdLoadStoreMulti {
            load,
            q,
            size,
            rpt,
            selem,
            rn,
            rt,
            post,
        });
    }

    // Single structure.
    let opcode = bits(insn, 15, 13);
    let s = bits(insn, 12, 12);
    let selem = (((opcode & 1) << 1) | bits(insn, 21, 21)) as u8 + 1;
    let qb = u32::from(q);
    let (scale, index, replicate) = match opcode >> 1 {
        0b00 => (0, (qb << 3) | (s << 2) | u32::from(size), false),
        0b01 => {
            if size & 1 != 0 {
                return None;
            }
            (1, (qb << 2) | (s << 1) | u32::from(size >> 1), false)
        }
        0b10 => match (size, s) {
            (0b00, _) => (2, (qb << 1) | s, false),
            (0b01, 0) => (3, qb, false),
            _ => return None,
        },
        _ => {
            if !load || s != 0 {
                return None;
            }
            (size, 0, true)
        }
    };
    Some(Instr::SimdLoadStoreSingle {
        load,
        replicate,
        q,
        size: scale,
        index,
        selem,
        rn,
        rt,
        post,
    })
}

fn decode_load_store(insn: u32) -> Option<Instr> {
    // Advanced SIMD load/store multiple and single structures.
    if insn & 0xBE00_0000 == 0x0C00_0000 {
        return decode_simd_structure(insn);
    }
    let simd = bit(insn, 26);
    let size = bits(insn, 31, 30);
    let rn = reg(insn, 5);
    let rt = reg(insn, 0);

    // Load/store exclusive and acquire/release (integer only).
    if insn & 0x3F00_0000 == 0x0800_0000 {
        let o2 = bit(insn, 23);
        let l = bit(insn, 22);
        let o1 = bit(insn, 21);
        let rs = reg(insn, 16);
        let rt2 = reg(insn, 10);
        if o1 {
            if o2 {
                // CAS, CASA, CASL, CASAL (all sizes). Unlike the other
                // exclusives, QEMU and LLVM require Rt2 = 11111 here.
                if rt2 != 31 {
                    return None;
                }
                return Some(Instr::CompareSwap {
                    size: size as u8,
                    rs,
                    rn,
                    rt,
                });
            }
            if !bit(insn, 31) {
                // CASP family: register pairs must start at even numbers.
                if rt2 != 31 || rs % 2 == 1 || rt % 2 == 1 {
                    return None;
                }
                return Some(Instr::CompareSwapPair {
                    sf: bit(insn, 30),
                    rs,
                    rn,
                    rt,
                });
            }
            // LDXP, LDAXP, STXP, STLXP.
            return Some(Instr::ExclusivePair {
                sf: bit(insn, 30),
                store: if l { None } else { Some(rs) },
                rn,
                rt,
                rt2,
            });
        }
        // Rt2 (and Rs, unless it holds a store-exclusive status) are
        // should-be-one fields; like hardware and QEMU, ignore them.
        let _ = rt2;
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
        let (size, kind) = match (simd, bits(insn, 31, 30)) {
            (false, 0) => (2, LoadKind::Load),
            (false, 1) => (3, LoadKind::Load),
            (false, 2) => (2, LoadKind::LoadSigned64),
            (false, _) => (3, LoadKind::Prefetch),
            (true, 0) => (2, LoadKind::Load),
            (true, 1) => (3, LoadKind::Load),
            (true, 2) => (4, LoadKind::Load),
            (true, _) => return None,
        };
        return Some(Instr::LoadStore {
            simd,
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
        let (size, signed) = match (simd, opc) {
            (false, 0) => (2, false),
            (false, 1) if load && bits(insn, 24, 23) != 0 => (2, true), // LDPSW
            (false, 2) => (3, false),
            (true, 0) => (2, false),
            (true, 1) => (3, false),
            (true, 2) => (4, false),
            _ => return None,
        };
        let imm = sign_extend(bits(insn, 21, 15), 7) << size;
        return Some(Instr::LoadStorePair {
            simd,
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
    let size_kind = |size: u32, opc: u32| -> Option<(u32, LoadKind)> {
        if simd {
            simd_size_kind(size, opc)
        } else {
            load_kind(size, opc).map(|k| (size, k))
        }
    };

    // Load/store register (unsigned immediate).
    if insn & 0x3B00_0000 == 0x3900_0000 {
        let (size, kind) = size_kind(size, opc)?;
        let imm = i64::from(bits(insn, 21, 10)) << size;
        return Some(Instr::LoadStore {
            simd,
            size: size as u8,
            kind,
            mode: AddrMode::Offset { rn, imm },
            rt,
        });
    }

    // Load/store register (unscaled, post-index, unprivileged, pre-index).
    if insn & 0x3B20_0000 == 0x3800_0000 {
        let (size, kind) = size_kind(size, opc)?;
        let imm = sign_extend(bits(insn, 20, 12), 9);
        let form = bits(insn, 11, 10);
        // Only the unscaled form (PRFUM) has a prefetch variant, and there
        // are no unprivileged SIMD&FP accesses.
        if (kind == LoadKind::Prefetch || simd) && form == 0b10 {
            return None;
        }
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
            simd,
            size: size as u8,
            kind,
            mode,
            rt,
        });
    }

    // Atomic memory operations (LSE) and LDAPR.
    if !simd && insn & 0x3F20_0C00 == 0x3820_0000 {
        let rs = reg(insn, 16);
        let o3 = bit(insn, 15);
        let op = match (o3, bits(insn, 14, 12)) {
            (false, 0b000) => AtomicOp::Add,
            (false, 0b001) => AtomicOp::Clr,
            (false, 0b010) => AtomicOp::Eor,
            (false, 0b011) => AtomicOp::Set,
            (false, 0b100) => AtomicOp::Smax,
            (false, 0b101) => AtomicOp::Smin,
            (false, 0b110) => AtomicOp::Umax,
            (false, 0b111) => AtomicOp::Umin,
            (true, 0b000) => AtomicOp::Swp,
            (true, 0b100) => {
                // LDAPR: A = 1, R = 0, Rs = 11111.
                if bit(insn, 23) && !bit(insn, 22) && rs == 31 {
                    return Some(Instr::Exclusive {
                        size: size as u8,
                        op: ExclusiveOp::LoadAcquire,
                        rn,
                        rt,
                    });
                }
                return None;
            }
            _ => return None,
        };
        return Some(Instr::Atomic {
            size: size as u8,
            op,
            rs,
            rn,
            rt,
        });
    }

    // Load/store register (register offset).
    if insn & 0x3B20_0C00 == 0x3820_0800 {
        let (size, kind) = size_kind(size, opc)?;
        let option = bits(insn, 15, 13);
        if option & 0b010 == 0 {
            return None;
        }
        let shift = if bit(insn, 12) { size } else { 0 };
        return Some(Instr::LoadStore {
            simd,
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
            0b010000..=0b010111 => {
                let size = (amount & 3) as u8;
                // Only CRC32X/CRC32CX use a 64-bit Rm, and they need sf = 1.
                if (size == 3) != sf {
                    return None;
                }
                DataProc2Op::Crc32 {
                    castagnoli: amount & 0b100 != 0,
                    size,
                }
            }
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

fn fp_type(insn: u32) -> Option<FpType> {
    match bits(insn, 23, 22) {
        0b00 => Some(FpType::Single),
        0b01 => Some(FpType::Double),
        0b11 => Some(FpType::Half),
        _ => None,
    }
}

/// `AdvSIMDExpandImm`: returns the 64-bit pattern for the modified
/// immediate, or `None` for unallocated combinations.
fn simd_expand_imm(op: bool, cmode: u32, imm8: u64, q: bool) -> Option<u64> {
    let rep32 = |x: u64| x | (x << 32);
    let rep16 = |x: u64| x | (x << 16) | (x << 32) | (x << 48);
    Some(match cmode >> 1 {
        0b000 => rep32(imm8),
        0b001 => rep32(imm8 << 8),
        0b010 => rep32(imm8 << 16),
        0b011 => rep32(imm8 << 24),
        0b100 => rep16(imm8),
        0b101 => rep16(imm8 << 8),
        0b110 => {
            if cmode & 1 == 0 {
                rep32((imm8 << 8) | 0xFF)
            } else {
                rep32((imm8 << 16) | 0xFFFF)
            }
        }
        _ => match (cmode & 1, op) {
            (0, false) => imm8 * 0x0101_0101_0101_0101,
            (0, true) => (0..8).fold(0, |acc, i| {
                acc | if (imm8 >> i) & 1 == 1 {
                    0xFF << (8 * i)
                } else {
                    0
                }
            }),
            (_, false) => rep32(crate::fp::expand_imm(imm8 as u8, FpType::Single)),
            (_, true) => {
                if !q {
                    return None;
                }
                crate::fp::expand_imm(imm8 as u8, FpType::Double)
            }
        },
    })
}

/// Splits an `imm5` field into (log2 element size, element index).
fn element_of(imm5: u32) -> Option<(u8, u32)> {
    let esize = imm5.trailing_zeros();
    if esize > 3 {
        return None;
    }
    Some((esize as u8, imm5 >> (esize + 1)))
}

fn decode_simd_fp(insn: u32) -> Option<Instr> {
    let rn = reg(insn, 5);
    let rd = reg(insn, 0);
    let rm = reg(insn, 16);
    let q = bit(insn, 30);

    // Floating-point data-processing (3 source).
    if insn & 0xFF00_0000 == 0x1F00_0000 {
        let ty = fp_type(insn)?;
        let (o1, o0) = (bit(insn, 21), bit(insn, 15));
        return Some(Instr::FpMulAdd {
            ty,
            neg_addend: o1,
            neg_product: o1 != o0,
            rm,
            ra: reg(insn, 10),
            rn,
            rd,
        });
    }

    // Scalar floating point and conversions.
    if insn & 0x7F00_0000 == 0x1E00_0000 {
        return decode_fp_scalar(insn);
    }

    // Advanced SIMD modified immediate.
    if insn & 0x9FF8_0400 == 0x0F00_0400 {
        if bit(insn, 11) {
            // FMOV (vector, half precision): op = 0, cmode = 1111 only.
            if bit(insn, 29) || bits(insn, 15, 12) != 0b1111 {
                return None;
            }
            let imm8 = (bits(insn, 18, 16) << 5) | bits(insn, 9, 5);
            let h = crate::fp::expand_imm(imm8 as u8, FpType::Half);
            let imm = h | (h << 16) | (h << 32) | (h << 48);
            return Some(Instr::VecMoveImm {
                q,
                op: VecImmOp::Movi,
                imm,
                rd,
            });
        }
        let op = bit(insn, 29);
        let cmode = bits(insn, 15, 12);
        let imm8 = u64::from((bits(insn, 18, 16) << 5) | bits(insn, 9, 5));
        let imm = simd_expand_imm(op, cmode, imm8, q)?;
        let kind = match (cmode, op) {
            (0b1110 | 0b1111, _) => VecImmOp::Movi,
            (c, false) if c & 0b1001 == 0b0001 || c & 0b1101 == 0b1001 => VecImmOp::Orr,
            (c, true) if c & 0b1001 == 0b0001 || c & 0b1101 == 0b1001 => VecImmOp::Bic,
            (_, false) => VecImmOp::Movi,
            (_, true) => VecImmOp::Mvni,
        };
        return Some(Instr::VecMoveImm {
            q,
            op: kind,
            imm,
            rd,
        });
    }

    // Advanced SIMD scalar two-register miscellaneous: FP <-> integer
    // conversions that keep the integer in a SIMD&FP register. Other ops of
    // this group are handled in `simd.rs`.
    if insn & 0xDF3E_0C00 == 0x5E20_0800 {
        let unsigned = bit(insn, 29);
        let o2 = bit(insn, 23);
        let ty = if bit(insn, 22) {
            FpType::Double
        } else {
            FpType::Single
        };
        let conv = match (o2, bits(insn, 16, 12)) {
            (false, 0b11010) => Some((true, Rounding::TieEven)),
            (false, 0b11011) => Some((true, Rounding::NegInf)),
            (false, 0b11100) => Some((true, Rounding::TieAway)),
            (false, 0b11101) => Some((false, Rounding::TieEven)),
            (true, 0b11010) => Some((true, Rounding::PosInf)),
            (true, 0b11011) => Some((true, Rounding::Zero)),
            _ => None,
        };
        if let Some((to_int, rounding)) = conv {
            return Some(Instr::FpIntInVec {
                ty,
                to_int,
                signed: !unsigned,
                rounding,
                rn,
                rd,
            });
        }
    }

    // Advanced SIMD scalar copy: DUP (element), alias MOV.
    if insn & 0xFFE0_FC00 == 0x5E00_0400 {
        let (esize, index) = element_of(bits(insn, 20, 16))?;
        return Some(Instr::ScalarDup {
            esize,
            index,
            rn,
            rd,
        });
    }

    // Advanced SIMD copy.
    if insn & 0x9FE0_8400 == 0x0E00_0400 {
        let (esize, index) = element_of(bits(insn, 20, 16))?;
        let imm4 = bits(insn, 14, 11);
        if bit(insn, 29) {
            // INS (element).
            if !q {
                return None;
            }
            let src_index = imm4 >> esize;
            return Some(Instr::VecIns {
                esize,
                index,
                src: ElemSrc::Element {
                    rn,
                    index: src_index,
                },
                rd,
            });
        }
        return match imm4 {
            0b0000 if !(esize == 3 && !q) => Some(Instr::VecDup {
                q,
                esize,
                src: ElemSrc::Element { rn, index },
                rd,
            }),
            0b0001 if !(esize == 3 && !q) => Some(Instr::VecDup {
                q,
                esize,
                src: ElemSrc::Gp(rn),
                rd,
            }),
            0b0011 if q => Some(Instr::VecIns {
                esize,
                index,
                src: ElemSrc::Gp(rn),
                rd,
            }),
            0b0101 if esize < 2 || (esize == 2 && q) => Some(Instr::VecToGp {
                signed: true,
                sf: q,
                esize,
                index,
                rn,
                rd,
            }),
            0b0111 if (esize == 3) == q => Some(Instr::VecToGp {
                signed: false,
                sf: q,
                esize,
                index,
                rn,
                rd,
            }),
            _ => None,
        };
    }

    // Advanced SIMD three same: logical, ADD, SUB.
    if insn & 0x9F20_FC00 == 0x0E20_1C00 {
        let op = match (bit(insn, 29), bits(insn, 23, 22)) {
            (false, 0) => VecLogicOp::And,
            (false, 1) => VecLogicOp::Bic,
            (false, 2) => VecLogicOp::Orr,
            (false, _) => VecLogicOp::Orn,
            (true, 0) => VecLogicOp::Eor,
            (true, 1) => VecLogicOp::Bsl,
            (true, 2) => VecLogicOp::Bit,
            (true, _) => VecLogicOp::Bif,
        };
        return Some(Instr::VecLogical { q, op, rm, rn, rd });
    }
    if insn & 0x9F20_FC00 == 0x0E20_8400 {
        let esize = bits(insn, 23, 22) as u8;
        if esize == 3 && !q {
            return None;
        }
        return Some(Instr::VecAddSub {
            q,
            sub: bit(insn, 29),
            esize,
            rm,
            rn,
            rd,
        });
    }

    crate::simd::decode(insn).map(Instr::Simd)
}

fn decode_fp_scalar(insn: u32) -> Option<Instr> {
    let sf = bit(insn, 31);
    let rn = reg(insn, 5);
    let rd = reg(insn, 0);
    let rm = reg(insn, 16);
    if bit(insn, 29) {
        return None;
    }
    let rmode = bits(insn, 20, 19);
    let opcode = bits(insn, 18, 16);

    // Conversion between floating point and fixed point.
    if !bit(insn, 21) {
        let ty = fp_type(insn)?;
        let scale = bits(insn, 15, 10);
        if !sf && scale < 32 {
            return None;
        }
        let fbits = 64 - scale;
        return match (rmode, opcode) {
            (0b00, 0b010 | 0b011) => Some(Instr::IntToFp {
                sf,
                ty,
                signed: opcode == 0b010,
                fbits,
                rn,
                rd,
            }),
            (0b11, 0b000 | 0b001) => Some(Instr::FpToInt {
                sf,
                ty,
                signed: opcode == 0b000,
                rounding: Rounding::Zero,
                fbits,
                rn,
                rd,
            }),
            _ => None,
        };
    }

    // Conversion between floating point and integer.
    if bits(insn, 15, 10) == 0 {
        // FMOV Xd, Vn.D[1] / FMOV Vd.D[1], Xn use ftype = 0b10.
        if bits(insn, 23, 22) == 0b10 {
            return match (sf, rmode, opcode) {
                (true, 0b01, 0b110) => Some(Instr::FmovToGp {
                    sf,
                    top: true,
                    half: false,
                    rn,
                    rd,
                }),
                (true, 0b01, 0b111) => Some(Instr::FmovFromGp {
                    sf,
                    top: true,
                    half: false,
                    rn,
                    rd,
                }),
                _ => None,
            };
        }
        let ty = fp_type(insn)?;
        let rounding = match rmode {
            0b00 => Rounding::TieEven,
            0b01 => Rounding::PosInf,
            0b10 => Rounding::NegInf,
            _ => Rounding::Zero,
        };
        return match (rmode, opcode) {
            (_, 0b000 | 0b001) => Some(Instr::FpToInt {
                sf,
                ty,
                signed: opcode == 0b000,
                rounding,
                fbits: 0,
                rn,
                rd,
            }),
            (0b00, 0b100 | 0b101) => Some(Instr::FpToInt {
                sf,
                ty,
                signed: opcode == 0b100,
                rounding: Rounding::TieAway,
                fbits: 0,
                rn,
                rd,
            }),
            (0b00, 0b010 | 0b011) => Some(Instr::IntToFp {
                sf,
                ty,
                signed: opcode == 0b010,
                fbits: 0,
                rn,
                rd,
            }),
            (0b00, 0b110 | 0b111) => {
                // Valid: W<->S, X<->D, and W or X<->H.
                let half = ty == FpType::Half;
                if !half && sf != (ty == FpType::Double) {
                    return None;
                }
                if opcode == 0b110 {
                    Some(Instr::FmovToGp {
                        sf,
                        top: false,
                        half,
                        rn,
                        rd,
                    })
                } else {
                    Some(Instr::FmovFromGp {
                        sf,
                        top: false,
                        half,
                        rn,
                        rd,
                    })
                }
            }
            _ => None,
        };
    }

    // Everything below has M (bit 31) = 0.
    if sf {
        return None;
    }
    let ty = fp_type(insn)?;
    let cond = bits(insn, 15, 12) as u8;

    match bits(insn, 11, 10) {
        0b01 => {
            return Some(Instr::FpCondCompare {
                ty,
                rn,
                rm,
                cond,
                nzcv: bits(insn, 3, 0) as u8,
            })
        }
        0b10 => {
            let op = match bits(insn, 15, 12) {
                0b0000 => FpBinOp::Mul,
                0b0001 => FpBinOp::Div,
                0b0010 => FpBinOp::Add,
                0b0011 => FpBinOp::Sub,
                0b0100 => FpBinOp::Max,
                0b0101 => FpBinOp::Min,
                0b0110 => FpBinOp::MaxNum,
                0b0111 => FpBinOp::MinNum,
                0b1000 => FpBinOp::NMul,
                _ => return None,
            };
            return Some(Instr::FpBinary { ty, op, rm, rn, rd });
        }
        0b11 => {
            return Some(Instr::FpCondSelect {
                ty,
                rm,
                rn,
                cond,
                rd,
            })
        }
        _ => {}
    }

    // Data-processing (1 source).
    if bits(insn, 14, 10) == 0b10000 {
        let op = match bits(insn, 20, 15) {
            0b000000 => FpUnaryOp::Mov,
            0b000001 => FpUnaryOp::Abs,
            0b000010 => FpUnaryOp::Neg,
            0b000011 => FpUnaryOp::Sqrt,
            0b000100 if ty != FpType::Single => FpUnaryOp::Convert(FpType::Single),
            0b000101 if ty != FpType::Double => FpUnaryOp::Convert(FpType::Double),
            0b000111 if ty != FpType::Half => FpUnaryOp::Convert(FpType::Half),
            0b001000 => FpUnaryOp::Round(Rounding::TieEven),
            0b001001 => FpUnaryOp::Round(Rounding::PosInf),
            0b001010 => FpUnaryOp::Round(Rounding::NegInf),
            0b001011 => FpUnaryOp::Round(Rounding::Zero),
            0b001100 => FpUnaryOp::Round(Rounding::TieAway),
            0b001110 | 0b001111 => FpUnaryOp::Round(Rounding::TieEven),
            _ => return None,
        };
        return Some(Instr::FpUnary { ty, op, rn, rd });
    }

    // Compare.
    if bits(insn, 15, 10) == 0b001000 {
        if bits(insn, 2, 0) != 0 {
            return None;
        }
        // The compare-with-zero form ignores the Rm field, like hardware
        // and QEMU do (the ARM ARM marks it should-be-zero).
        let with_zero = bit(insn, 3);
        return Some(Instr::FpCompare {
            ty,
            rn,
            rm: if with_zero { None } else { Some(rm) },
        });
    }

    // Move immediate.
    if bits(insn, 12, 10) == 0b100 {
        if bits(insn, 9, 5) != 0 {
            return None;
        }
        let imm = crate::fp::expand_imm(bits(insn, 20, 13) as u8, ty);
        return Some(Instr::FpMovImm { ty, imm, rd });
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
