//! Scalar floating-point operations with ARM semantics.
//!
//! The host FPU computes the correctly rounded IEEE 754 results. What
//! differs between ARM and x86 is NaN handling, so every operation here
//! resolves NaN cases first, following the ARM pseudocode:
//!
//! - `FPProcessNaNs`: a signalling NaN operand wins (and is quieted), then
//!   the first quiet NaN operand. Payloads are kept.
//! - An invalid operation (inf - inf, 0 * inf, sqrt(-1), ...) produces the
//!   *default NaN*, which is positive on ARM (x86 produces a negative one).
//! - With FPCR.DN set, every NaN result is the default NaN.
//!
//! Values are passed around as raw bits (`u64`) so that NaN payloads are
//! never touched by the host.
//!
//! Not implemented yet: rounding modes other than round-to-nearest for
//! arithmetic (FPCR.RMode), flush-to-zero (FPCR.FZ), and the cumulative
//! exception flags in FPSR. Instructions with an explicit rounding mode
//! (FRINTx, FCVTxx) are exact.

use crate::Nzcv;

/// FPCR.DN: default NaN mode.
pub const FPCR_DN: u64 = 1 << 25;

/// Scalar floating-point precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpType {
    Single,
    Double,
}

impl FpType {
    pub fn bytes(self) -> usize {
        match self {
            FpType::Single => 4,
            FpType::Double => 8,
        }
    }

    pub fn mask(self) -> u64 {
        match self {
            FpType::Single => 0xFFFF_FFFF,
            FpType::Double => u64::MAX,
        }
    }

    fn sign(self) -> u64 {
        match self {
            FpType::Single => 1 << 31,
            FpType::Double => 1 << 63,
        }
    }

    fn exp_mask(self) -> u64 {
        match self {
            FpType::Single => 0x7F80_0000,
            FpType::Double => 0x7FF0_0000_0000_0000,
        }
    }

    fn mant_mask(self) -> u64 {
        match self {
            FpType::Single => 0x007F_FFFF,
            FpType::Double => 0x000F_FFFF_FFFF_FFFF,
        }
    }

    fn quiet_bit(self) -> u64 {
        match self {
            FpType::Single => 1 << 22,
            FpType::Double => 1 << 51,
        }
    }

    pub fn default_nan(self) -> u64 {
        match self {
            FpType::Single => 0x7FC0_0000,
            FpType::Double => 0x7FF8_0000_0000_0000,
        }
    }

    fn infinity(self, negative: bool) -> u64 {
        self.exp_mask() | if negative { self.sign() } else { 0 }
    }
}

/// Rounding used by FRINTx and FCVTxx.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// To nearest, ties to even.
    TieEven,
    /// Towards plus infinity.
    PosInf,
    /// Towards minus infinity.
    NegInf,
    /// Towards zero.
    Zero,
    /// To nearest, ties away from zero.
    TieAway,
}

fn round_f64(x: f64, mode: Rounding) -> f64 {
    match mode {
        Rounding::TieEven => x.round_ties_even(),
        Rounding::PosInf => x.ceil(),
        Rounding::NegInf => x.floor(),
        Rounding::Zero => x.trunc(),
        Rounding::TieAway => x.round(),
    }
}

fn round_f32(x: f32, mode: Rounding) -> f32 {
    match mode {
        Rounding::TieEven => x.round_ties_even(),
        Rounding::PosInf => x.ceil(),
        Rounding::NegInf => x.floor(),
        Rounding::Zero => x.trunc(),
        Rounding::TieAway => x.round(),
    }
}

/// Floating-point environment for one operation.
#[derive(Debug, Clone, Copy)]
pub struct Fp {
    pub ty: FpType,
    /// FPCR.DN.
    pub default_nan: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    MaxNum,
    MinNum,
    /// FNMUL: -(a * b).
    NMul,
}

impl Fp {
    pub fn new(ty: FpType, fpcr: u64) -> Self {
        Fp {
            ty,
            default_nan: fpcr & FPCR_DN != 0,
        }
    }

    pub fn is_nan(self, x: u64) -> bool {
        x & self.ty.exp_mask() == self.ty.exp_mask() && x & self.ty.mant_mask() != 0
    }

    fn is_snan(self, x: u64) -> bool {
        self.is_nan(x) && x & self.ty.quiet_bit() == 0
    }

    fn is_qnan(self, x: u64) -> bool {
        self.is_nan(x) && x & self.ty.quiet_bit() != 0
    }

    fn is_inf(self, x: u64) -> bool {
        x & !self.ty.sign() & self.ty.mask() == self.ty.exp_mask()
    }

    fn is_zero(self, x: u64) -> bool {
        x & !self.ty.sign() & self.ty.mask() == 0
    }

    fn is_negative(self, x: u64) -> bool {
        x & self.ty.sign() != 0
    }

    /// `FPProcessNaN`: quiets a NaN, or gives the default NaN in DN mode.
    fn process_nan(self, x: u64) -> u64 {
        if self.default_nan {
            self.ty.default_nan()
        } else {
            x | self.ty.quiet_bit()
        }
    }

    /// `FPProcessNaNs` for any number of operands, in priority order.
    fn process_nans(self, ops: &[u64]) -> Option<u64> {
        if let Some(&x) = ops.iter().find(|&&x| self.is_snan(x)) {
            return Some(self.process_nan(x));
        }
        ops.iter()
            .find(|&&x| self.is_qnan(x))
            .map(|&x| self.process_nan(x))
    }

    /// Replaces a NaN produced by the host (an invalid operation) with the
    /// ARM default NaN.
    fn finish(self, result: u64) -> u64 {
        if self.is_nan(result) {
            self.ty.default_nan()
        } else {
            result
        }
    }

    pub fn neg(self, x: u64) -> u64 {
        x ^ self.ty.sign()
    }

    pub fn abs(self, x: u64) -> u64 {
        x & !self.ty.sign()
    }

    fn host2(self, a: u64, b: u64, f32op: fn(f32, f32) -> f32, f64op: fn(f64, f64) -> f64) -> u64 {
        match self.ty {
            FpType::Single => {
                u64::from(f32op(f32::from_bits(a as u32), f32::from_bits(b as u32)).to_bits())
            }
            FpType::Double => f64op(f64::from_bits(a), f64::from_bits(b)).to_bits(),
        }
    }

    pub fn binop(self, op: BinOp, a: u64, b: u64) -> u64 {
        let (a, b) = (a & self.ty.mask(), b & self.ty.mask());
        match op {
            BinOp::MaxNum | BinOp::MinNum => {
                // `FPMaxNum`/`FPMinNum`: a lone quiet NaN loses to a number.
                let replace = if op == BinOp::MaxNum {
                    self.ty.infinity(true)
                } else {
                    self.ty.infinity(false)
                };
                let (mut a2, mut b2) = (a, b);
                if self.is_qnan(a) && !self.is_qnan(b) {
                    a2 = replace;
                }
                if self.is_qnan(b) && !self.is_qnan(a) {
                    b2 = replace;
                }
                let inner = if op == BinOp::MaxNum {
                    BinOp::Max
                } else {
                    BinOp::Min
                };
                return self.binop(inner, a2, b2);
            }
            BinOp::NMul => return self.neg(self.binop(BinOp::Mul, a, b)),
            _ => {}
        }
        if let Some(nan) = self.process_nans(&[a, b]) {
            return nan;
        }
        let result = match op {
            BinOp::Add => self.host2(a, b, |x, y| x + y, |x, y| x + y),
            BinOp::Sub => self.host2(a, b, |x, y| x - y, |x, y| x - y),
            BinOp::Mul => self.host2(a, b, |x, y| x * y, |x, y| x * y),
            BinOp::Div => self.host2(a, b, |x, y| x / y, |x, y| x / y),
            BinOp::Max | BinOp::Min => {
                let max = op == BinOp::Max;
                if self.is_zero(a) && self.is_zero(b) {
                    // max(-0, +0) = +0 and min(-0, +0) = -0.
                    let neg = if max {
                        self.is_negative(a) && self.is_negative(b)
                    } else {
                        self.is_negative(a) || self.is_negative(b)
                    };
                    return if neg { self.ty.sign() } else { 0 };
                }
                let a_greater = self.compare(a, b) == Some(std::cmp::Ordering::Greater);
                return if a_greater == max { a } else { b };
            }
            BinOp::MaxNum | BinOp::MinNum | BinOp::NMul => unreachable!(),
        };
        self.finish(result)
    }

    /// `FPMulAdd`: `addend + op1 * op2` with a single rounding.
    pub fn mul_add(self, addend: u64, op1: u64, op2: u64) -> u64 {
        let (a, x, y) = (
            addend & self.ty.mask(),
            op1 & self.ty.mask(),
            op2 & self.ty.mask(),
        );
        let inf_times_zero =
            (self.is_inf(x) && self.is_zero(y)) || (self.is_zero(x) && self.is_inf(y));
        let nan = self.process_nans(&[a, x, y]);
        // A quiet NaN addend does not hide an invalid inf * 0.
        if self.is_qnan(a) && inf_times_zero {
            return self.ty.default_nan();
        }
        if let Some(nan) = nan {
            return nan;
        }
        let result = match self.ty {
            FpType::Single => u64::from(
                f32::from_bits(x as u32)
                    .mul_add(f32::from_bits(y as u32), f32::from_bits(a as u32))
                    .to_bits(),
            ),
            FpType::Double => f64::from_bits(x)
                .mul_add(f64::from_bits(y), f64::from_bits(a))
                .to_bits(),
        };
        self.finish(result)
    }

    pub fn sqrt(self, x: u64) -> u64 {
        let x = x & self.ty.mask();
        if let Some(nan) = self.process_nans(&[x]) {
            return nan;
        }
        let result = match self.ty {
            FpType::Single => u64::from(f32::from_bits(x as u32).sqrt().to_bits()),
            FpType::Double => f64::from_bits(x).sqrt().to_bits(),
        };
        self.finish(result)
    }

    /// FRINTx: round to an integral value in floating-point format.
    pub fn round_int(self, x: u64, mode: Rounding) -> u64 {
        let x = x & self.ty.mask();
        if let Some(nan) = self.process_nans(&[x]) {
            return nan;
        }
        match self.ty {
            FpType::Single => u64::from(round_f32(f32::from_bits(x as u32), mode).to_bits()),
            FpType::Double => round_f64(f64::from_bits(x), mode).to_bits(),
        }
    }

    fn to_f64(self, x: u64) -> f64 {
        match self.ty {
            FpType::Single => f64::from(f32::from_bits(x as u32)),
            FpType::Double => f64::from_bits(x),
        }
    }

    /// IEEE comparison. `None` means unordered (a NaN is involved).
    pub fn compare(self, a: u64, b: u64) -> Option<std::cmp::Ordering> {
        self.to_f64(a & self.ty.mask())
            .partial_cmp(&self.to_f64(b & self.ty.mask()))
    }

    /// Flags set by FCMP/FCMPE.
    pub fn compare_flags(self, a: u64, b: u64) -> Nzcv {
        use std::cmp::Ordering::*;
        let bits = match self.compare(a, b) {
            None => 0b0011,
            Some(Equal) => 0b0110,
            Some(Less) => 0b1000,
            Some(Greater) => 0b0010,
        };
        Nzcv::from_nibble(bits)
    }

    /// FCVT between single and double precision (`self.ty` is the source).
    pub fn convert(self, x: u64, to: FpType) -> u64 {
        let x = x & self.ty.mask();
        if self.ty == to {
            return x;
        }
        if self.is_nan(x) {
            if self.default_nan {
                return to.default_nan();
            }
            let sign = self.is_negative(x);
            let mant = x & self.ty.mant_mask();
            // Keep the top payload bits and force the result quiet.
            let mant = match to {
                FpType::Double => mant << 29,
                FpType::Single => mant >> 29,
            };
            return to.exp_mask() | to.quiet_bit() | mant | if sign { to.sign() } else { 0 };
        }
        match to {
            FpType::Double => f64::from(f32::from_bits(x as u32)).to_bits(),
            FpType::Single => u64::from((f64::from_bits(x) as f32).to_bits()),
        }
    }

    /// `FPToFixed`: converts to a (possibly fixed-point) integer of `bits`
    /// width, saturating; NaN gives zero.
    pub fn to_int(self, x: u64, signed: bool, bits: u32, fbits: u32, mode: Rounding) -> u64 {
        let x = x & self.ty.mask();
        if self.is_nan(x) {
            return 0;
        }
        // Scaling by a power of two is exact in f64 for single inputs, and
        // overflow to infinity still saturates correctly for doubles.
        let scaled = self.to_f64(x) * 2f64.powi(fbits as i32);
        let r = round_f64(scaled, mode);
        // Rust's float-to-int casts saturate, exactly like FPToFixed.
        match (signed, bits) {
            (true, 32) => u64::from(r as i32 as u32),
            (true, _) => r as i64 as u64,
            (false, 32) => u64::from(r as u32),
            (false, _) => r as u64,
        }
    }

    /// `FixedToFP`: converts an integer of `bits` width, divided by
    /// `2^fbits`, rounding to nearest.
    pub fn from_int(self, value: u64, signed: bool, bits: u32, fbits: u32) -> u64 {
        let scale = 2f64.powi(fbits as i32);
        match self.ty {
            FpType::Single => {
                // One rounding in the int-to-float cast; the power-of-two
                // division afterwards is exact.
                let f = match (signed, bits) {
                    (true, 32) => value as u32 as i32 as f32,
                    (true, _) => value as i64 as f32,
                    (false, 32) => value as u32 as f32,
                    (false, _) => value as f32,
                };
                u64::from((f / scale as f32).to_bits())
            }
            FpType::Double => {
                let f = match (signed, bits) {
                    (true, 32) => f64::from(value as u32 as i32),
                    (true, _) => value as i64 as f64,
                    (false, 32) => f64::from(value as u32),
                    (false, _) => value as f64,
                };
                (f / scale).to_bits()
            }
        }
    }
}

/// `VFPExpandImm`: the 8-bit immediate of FMOV (scalar, immediate).
pub fn expand_imm(imm8: u8, ty: FpType) -> u64 {
    let sign = u64::from(imm8 >> 7);
    let b6 = u64::from((imm8 >> 6) & 1);
    let b54 = u64::from((imm8 >> 4) & 3);
    let frac = u64::from(imm8 & 0xF);
    match ty {
        FpType::Single => {
            // exp = NOT(b6) : Replicate(b6, 5) : b54
            let exp = ((b6 ^ 1) << 7) | (if b6 == 1 { 0b11111 } else { 0 } << 2) | b54;
            (sign << 31) | (exp << 23) | (frac << 19)
        }
        FpType::Double => {
            let exp = ((b6 ^ 1) << 10) | (if b6 == 1 { 0xFF } else { 0 } << 2) | b54;
            (sign << 63) | (exp << 52) | (frac << 48)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Fp = Fp {
        ty: FpType::Single,
        default_nan: false,
    };
    const D: Fp = Fp {
        ty: FpType::Double,
        default_nan: false,
    };

    fn s(x: f32) -> u64 {
        u64::from(x.to_bits())
    }

    #[test]
    fn invalid_operations_give_positive_default_nan() {
        assert_eq!(
            S.binop(BinOp::Sub, s(f32::INFINITY), s(f32::INFINITY)),
            0x7FC0_0000
        );
        assert_eq!(
            D.binop(BinOp::Mul, 0, f64::INFINITY.to_bits()),
            0x7FF8_0000_0000_0000
        );
        assert_eq!(S.sqrt(s(-1.0)), 0x7FC0_0000);
    }

    #[test]
    fn nan_priority_and_payloads() {
        let qnan = 0x7FC0_1234;
        let snan = 0xFF80_0042;
        // A signalling NaN wins over an earlier quiet NaN, and is quieted.
        assert_eq!(S.binop(BinOp::Add, qnan, snan), 0xFFC0_0042);
        // Otherwise the first quiet NaN is returned unchanged.
        assert_eq!(S.binop(BinOp::Add, s(1.0), qnan), qnan);
        let dn = Fp {
            ty: FpType::Single,
            default_nan: true,
        };
        assert_eq!(dn.binop(BinOp::Add, qnan, s(1.0)), 0x7FC0_0000);
    }

    #[test]
    fn max_min_rules() {
        assert_eq!(S.binop(BinOp::Max, s(-0.0), s(0.0)), s(0.0));
        assert_eq!(S.binop(BinOp::Min, s(0.0), s(-0.0)), s(-0.0));
        // FMAXNM ignores a single quiet NaN; FMAX does not.
        assert_eq!(S.binop(BinOp::MaxNum, 0x7FC0_0000, s(2.0)), s(2.0));
        assert_eq!(S.binop(BinOp::Max, 0x7FC0_0000, s(2.0)), 0x7FC0_0000);
    }

    #[test]
    fn fused_multiply_add_rounds_once() {
        // (1 + 2^-23)^2 - 1 is exactly 2^-22 + 2^-46 with a single rounding.
        let x = s(1.0 + f32::EPSILON);
        let r = S.mul_add(s(-1.0), x, x);
        assert_eq!(
            f32::from_bits(r as u32),
            2.0 * f32::EPSILON + f32::EPSILON * f32::EPSILON
        );
    }

    #[test]
    fn conversions_saturate() {
        assert_eq!(S.to_int(s(1e20), true, 32, 0, Rounding::Zero), 0x7FFF_FFFF);
        assert_eq!(S.to_int(s(-1.0), false, 64, 0, Rounding::Zero), 0);
        assert_eq!(S.to_int(0x7FC0_0000, true, 64, 0, Rounding::Zero), 0);
        assert_eq!(
            D.to_int(2.5f64.to_bits(), true, 64, 0, Rounding::TieEven),
            2
        );
        assert_eq!(
            D.to_int(2.5f64.to_bits(), true, 64, 0, Rounding::TieAway),
            3
        );
        assert_eq!(
            D.to_int((-2.5f64).to_bits(), true, 32, 0, Rounding::NegInf),
            (-3i32) as u32 as u64
        );
        assert_eq!(
            S.from_int(u64::MAX, false, 64, 0),
            s(18446744073709551616.0)
        );
        assert_eq!(D.from_int(3, true, 32, 1), 1.5f64.to_bits());
    }

    #[test]
    fn fmov_immediates() {
        // fmov s0, #1.0 uses imm8 = 0x70; #-2.0 uses 0x80; #0.5 uses 0x60.
        assert_eq!(expand_imm(0x70, FpType::Single), s(1.0));
        assert_eq!(expand_imm(0x80, FpType::Single), s(-2.0));
        assert_eq!(expand_imm(0x60, FpType::Double), 0.5f64.to_bits());
    }
}
