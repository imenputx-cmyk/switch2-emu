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
//! Half precision is computed in f64 and rounded once to 16 bits, which is
//! exact for +, -, *, / and sqrt (53 >= 2 * 11 + 2, so double rounding is
//! innocuous). Fused multiply-add and integer conversions to half precision
//! are computed exactly with 128-bit integers instead.
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
    Half,
    Single,
    Double,
}

impl FpType {
    pub fn bytes(self) -> usize {
        match self {
            FpType::Half => 2,
            FpType::Single => 4,
            FpType::Double => 8,
        }
    }

    pub fn mask(self) -> u64 {
        match self {
            FpType::Half => 0xFFFF,
            FpType::Single => 0xFFFF_FFFF,
            FpType::Double => u64::MAX,
        }
    }

    fn sign(self) -> u64 {
        match self {
            FpType::Half => 1 << 15,
            FpType::Single => 1 << 31,
            FpType::Double => 1 << 63,
        }
    }

    fn exp_mask(self) -> u64 {
        match self {
            FpType::Half => 0x7C00,
            FpType::Single => 0x7F80_0000,
            FpType::Double => 0x7FF0_0000_0000_0000,
        }
    }

    fn mant_mask(self) -> u64 {
        match self {
            FpType::Half => 0x03FF,
            FpType::Single => 0x007F_FFFF,
            FpType::Double => 0x000F_FFFF_FFFF_FFFF,
        }
    }

    fn quiet_bit(self) -> u64 {
        match self {
            FpType::Half => 1 << 9,
            FpType::Single => 1 << 22,
            FpType::Double => 1 << 51,
        }
    }

    pub fn default_nan(self) -> u64 {
        match self {
            FpType::Half => 0x7E00,
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
            FpType::Half => u64::from(f64_to_half(f64op(
                half_to_f64(a as u16),
                half_to_f64(b as u16),
            ))),
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
            FpType::Half => return self.half_mul_add(a, x, y),
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
            FpType::Half => u64::from(f64_to_half(half_to_f64(x as u16).sqrt())),
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
            FpType::Half => u64::from(f64_to_half(round_f64(half_to_f64(x as u16), mode))),
            FpType::Single => u64::from(round_f32(f32::from_bits(x as u32), mode).to_bits()),
            FpType::Double => round_f64(f64::from_bits(x), mode).to_bits(),
        }
    }

    fn to_f64(self, x: u64) -> f64 {
        match self.ty {
            FpType::Half => half_to_f64(x as u16),
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
        if self.ty == FpType::Half {
            return half_to(x as u16, to, self.default_nan);
        }
        if to == FpType::Half {
            return u64::from(to_half(x, self.ty, self.default_nan));
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
                _ => mant >> 29,
            };
            return to.exp_mask() | to.quiet_bit() | mant | if sign { to.sign() } else { 0 };
        }
        match to {
            FpType::Double => f64::from(f32::from_bits(x as u32)).to_bits(),
            _ => u64::from((f64::from_bits(x) as f32).to_bits()),
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
            (true, 16) => u64::from(r as i16 as u16),
            (true, 32) => u64::from(r as i32 as u32),
            (true, _) => r as i64 as u64,
            (false, 16) => u64::from(r as u16),
            (false, 32) => u64::from(r as u32),
            (false, _) => r as u64,
        }
    }

    /// `FixedToFP`: converts an integer of `bits` width, divided by
    /// `2^fbits`, rounding to nearest.
    pub fn from_int(self, value: u64, signed: bool, bits: u32, fbits: u32) -> u64 {
        let scale = 2f64.powi(fbits as i32);
        match self.ty {
            FpType::Half => {
                // Exact: the integer, scaled by 2^-fbits, rounded once.
                let v = match (signed, bits) {
                    (true, 16) => i128::from(value as u16 as i16),
                    (true, 32) => i128::from(value as u32 as i32),
                    (true, _) => i128::from(value as i64),
                    (false, 16) => i128::from(value as u16),
                    (false, 32) => i128::from(value as u32),
                    (false, _) => i128::from(value),
                };
                u64::from(round_scaled_to_half(v, -(fbits as i32), false))
            }
            FpType::Single => {
                // One rounding in the int-to-float cast; the power-of-two
                // division afterwards is exact.
                let f = match (signed, bits) {
                    (true, 16) => f32::from(value as u16 as i16),
                    (false, 16) => f32::from(value as u16),
                    (true, 32) => value as u32 as i32 as f32,
                    (true, _) => value as i64 as f32,
                    (false, 32) => value as u32 as f32,
                    (false, _) => value as f32,
                };
                u64::from((f / scale as f32).to_bits())
            }
            FpType::Double => {
                let f = match (signed, bits) {
                    (true, 16) => f64::from(value as u16 as i16),
                    (false, 16) => f64::from(value as u16),
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

impl Fp {
    fn two(self) -> u64 {
        match self.ty {
            FpType::Half => 0x4000,
            FpType::Single => u64::from(2f32.to_bits()),
            FpType::Double => 2f64.to_bits(),
        }
    }

    fn encode_f64(self, v: f64) -> u64 {
        match self.ty {
            FpType::Half => u64::from(f64_to_half(v)),
            FpType::Single => u64::from((v as f32).to_bits()),
            FpType::Double => v.to_bits(),
        }
    }

    /// FMULX: like FMUL, but infinity times zero gives +/-2.0.
    pub fn mulx(self, a: u64, b: u64) -> u64 {
        let (a, b) = (a & self.ty.mask(), b & self.ty.mask());
        if let Some(nan) = self.process_nans(&[a, b]) {
            return nan;
        }
        if (self.is_inf(a) && self.is_zero(b)) || (self.is_zero(a) && self.is_inf(b)) {
            let sign = (a ^ b) & self.ty.sign();
            return self.two() | sign;
        }
        self.binop(BinOp::Mul, a, b)
    }

    /// FRECPS (`FPRecipStepFused`): `2 - a * b`, fused.
    pub fn recip_step(self, a: u64, b: u64) -> u64 {
        let (a, b) = (self.neg(a & self.ty.mask()), b & self.ty.mask());
        if let Some(nan) = self.process_nans(&[a, b]) {
            return nan;
        }
        if (self.is_inf(a) && self.is_zero(b)) || (self.is_zero(a) && self.is_inf(b)) {
            return self.two();
        }
        self.mul_add(self.two(), a, b)
    }

    /// FRSQRTS (`FPRSqrtStepFused`): `(3 - a * b) / 2`, fused.
    pub fn rsqrt_step(self, a: u64, b: u64) -> u64 {
        let (a, b) = (self.neg(a & self.ty.mask()), b & self.ty.mask());
        if let Some(nan) = self.process_nans(&[a, b]) {
            return nan;
        }
        if (self.is_inf(a) && self.is_zero(b)) || (self.is_zero(a) && self.is_inf(b)) {
            return self.encode_f64(1.5);
        }
        // (3 + a*b) / 2 with one rounding: 1.5 + a * (b/2), halving
        // whichever operand halves exactly so that no precision is lost.
        let half = |x: u64| -> Option<u64> {
            let h = self.binop(BinOp::Div, x, self.two());
            (self.binop(BinOp::Mul, h, self.two()) == x).then_some(h)
        };
        let one_and_half = self.encode_f64(1.5);
        match (half(b), half(a)) {
            (Some(hb), _) => self.mul_add(one_and_half, a, hb),
            (None, Some(ha)) => self.mul_add(one_and_half, ha, b),
            // Both are tiny denormals: the product cannot affect 1.5.
            (None, None) => one_and_half,
        }
    }

    /// Splits into (sign, biased exponent, fraction widened to 52 bits).
    fn unpack52(self, x: u64) -> (u64, i64, u64) {
        match self.ty {
            FpType::Half => ((x >> 15) & 1, ((x >> 10) & 0x1F) as i64, (x & 0x3FF) << 42),
            FpType::Single => (
                (x >> 31) & 1,
                ((x >> 23) & 0xFF) as i64,
                (x & 0x7F_FFFF) << 29,
            ),
            FpType::Double => (
                (x >> 63) & 1,
                ((x >> 52) & 0x7FF) as i64,
                x & ((1 << 52) - 1),
            ),
        }
    }

    /// FRECPE (`FPRecipEstimate`): 8-bit reciprocal estimate.
    pub fn recip_estimate(self, x: u64) -> u64 {
        let x = x & self.ty.mask();
        if let Some(nan) = self.process_nans(&[x]) {
            return nan;
        }
        let negative = self.is_negative(x);
        let sign_bit = if negative { self.ty.sign() } else { 0 };
        if self.is_inf(x) {
            return sign_bit;
        }
        if self.is_zero(x) {
            return self.ty.infinity(negative);
        }
        let tiny = match self.ty {
            FpType::Half => half_to_f64(x as u16).abs() < 2f64.powi(-16),
            FpType::Single => f64::from(f32::from_bits(x as u32)).abs() < 2f64.powi(-128),
            FpType::Double => f64::from_bits(x).abs() < 2f64.powi(-1024),
        };
        if tiny {
            return self.ty.infinity(negative);
        }
        let (sign, mut exp, mut fraction) = self.unpack52(x);
        if exp == 0 {
            if fraction >> 51 & 1 == 0 {
                exp = -1;
                fraction = (fraction & ((1 << 50) - 1)) << 2;
            } else {
                fraction = (fraction & ((1 << 51) - 1)) << 1;
            }
        }
        let scaled = 0x100 | ((fraction >> 44) & 0xFF);
        let mut result_exp = match self.ty {
            FpType::Half => 29 - exp,
            FpType::Single => 253 - exp,
            FpType::Double => 2045 - exp,
        };
        let estimate = recip_estimate_int(scaled);
        let mut fraction = (estimate & 0xFF) << 44;
        if result_exp == 0 {
            fraction = (1 << 51) | (fraction >> 1);
        } else if result_exp == -1 {
            fraction = (1 << 50) | (fraction >> 2);
            result_exp = 0;
        }
        match self.ty {
            FpType::Half => (sign << 15) | (((result_exp as u64) & 0x1F) << 10) | (fraction >> 42),
            FpType::Single => {
                (sign << 31) | (((result_exp as u64) & 0xFF) << 23) | (fraction >> 29)
            }
            FpType::Double => (sign << 63) | (((result_exp as u64) & 0x7FF) << 52) | fraction,
        }
    }

    /// FRSQRTE (`FPRSqrtEstimate`): 8-bit reciprocal square root estimate.
    pub fn rsqrt_estimate(self, x: u64) -> u64 {
        let x = x & self.ty.mask();
        if let Some(nan) = self.process_nans(&[x]) {
            return nan;
        }
        if self.is_zero(x) {
            return self.ty.infinity(self.is_negative(x));
        }
        if self.is_negative(x) {
            return self.ty.default_nan();
        }
        if self.is_inf(x) {
            return 0;
        }
        let (_, mut exp, mut fraction) = self.unpack52(x);
        let mask52 = (1u64 << 52) - 1;
        if exp == 0 {
            while fraction >> 51 & 1 == 0 {
                fraction = (fraction << 1) & mask52;
                exp -= 1;
            }
            fraction = (fraction << 1) & mask52;
        }
        let scaled = if exp & 1 == 0 {
            0x100 | ((fraction >> 44) & 0xFF)
        } else {
            0x80 | ((fraction >> 45) & 0x7F)
        };
        let result_exp = match self.ty {
            FpType::Half => (44 - exp).div_euclid(2),
            FpType::Single => (380 - exp).div_euclid(2),
            FpType::Double => (3068 - exp).div_euclid(2),
        } as u64;
        let estimate = rsqrt_estimate_int(scaled);
        match self.ty {
            FpType::Half => ((result_exp & 0x1F) << 10) | ((estimate & 0xFF) << 2),
            FpType::Single => ((result_exp & 0xFF) << 23) | ((estimate & 0xFF) << 15),
            FpType::Double => ((result_exp & 0x7FF) << 52) | ((estimate & 0xFF) << 44),
        }
    }

    /// FRECPX: reciprocal exponent.
    pub fn recpx(self, x: u64) -> u64 {
        let x = x & self.ty.mask();
        if let Some(nan) = self.process_nans(&[x]) {
            return nan;
        }
        let sign = x & self.ty.sign();
        let exp_mask = self.ty.exp_mask();
        let lsb = 1u64 << exp_mask.trailing_zeros();
        if x & exp_mask == 0 {
            // Zeros and denormals: the largest finite exponent.
            sign | (exp_mask - lsb)
        } else {
            sign | (!x & exp_mask)
        }
    }

    /// FCVTXN: double to single with round-to-odd (`self.ty` is Double).
    pub fn convert_round_odd(self, x: u64) -> u64 {
        let x = x & self.ty.mask();
        if self.is_nan(x) {
            return self.convert(x, FpType::Single);
        }
        let d = f64::from_bits(x);
        let mut r = d as f32;
        if f64::from(r) != d && r.is_finite() || (r.is_infinite() && d.is_finite()) {
            // Truncate towards zero, then set the lowest bit (sticky).
            if f64::from(r).abs() > d.abs() {
                r = f32::from_bits(r.to_bits() - 1);
            }
            r = f32::from_bits(r.to_bits() | 1);
        }
        u64::from(r.to_bits())
    }
}

/// `RecipEstimate()` on a 9-bit fixed-point input in [256, 511].
pub(crate) fn recip_estimate_int(a: u64) -> u64 {
    let a = a * 2 + 1;
    let b = (1u64 << 19) / a;
    b.div_ceil(2)
}

/// `RecipSqrtEstimate()` on a 9-bit fixed-point input in [128, 511].
pub(crate) fn rsqrt_estimate_int(a: u64) -> u64 {
    let a = if a < 256 {
        a * 2 + 1
    } else {
        ((a >> 1) << 1) * 2 + 2
    };
    let mut b = 512u64;
    while a * (b + 1) * (b + 1) < (1 << 28) {
        b += 1;
    }
    b.div_ceil(2)
}

impl Fp {
    /// Half-precision `FPMulAdd`, computed exactly in fixed point: every
    /// half is an integer multiple of 2^-24, so products are multiples of
    /// 2^-48 and fit easily in 128 bits.
    fn half_mul_add(self, a: u64, x: u64, y: u64) -> u64 {
        let as_int = |h: u64| -> i128 {
            let mag = i128::from(h & 0x3FF) + if h & 0x7C00 != 0 { 1024 } else { 0 };
            let exp = ((h >> 10) & 0x1F).max(1) as u32;
            let v = mag << (exp - 1); // in units of 2^-24
            if h & 0x8000 != 0 {
                -v
            } else {
                v
            }
        };
        let (fa, fx, fy) = (
            half_to_f64(a as u16),
            half_to_f64(x as u16),
            half_to_f64(y as u16),
        );
        if !(fa.is_finite() && fx.is_finite() && fy.is_finite()) {
            // Infinities: the f64 result is exact (inf - inf gives NaN).
            return self.finish(u64::from(f64_to_half(fx.mul_add(fy, fa))));
        }
        let product = as_int(x) * as_int(y); // units of 2^-48
        let sum = (as_int(a) << 24) + product;
        if sum == 0 {
            // Exact zero: the sign follows IEEE (both negative gives -0).
            let product_negative = ((x ^ y) >> 15) & 1 == 1;
            let addend_negative = (a >> 15) & 1 == 1;
            let product_zero = product == 0;
            let neg = if product_zero && as_int(a) == 0 {
                product_negative && addend_negative
            } else {
                false
            };
            return if neg { 0x8000 } else { 0 };
        }
        u64::from(round_scaled_to_half(sum, -48, false))
    }
}

/// Rounds the exact value `v * 2^k` to half precision (nearest, ties to
/// even). `v` must not be zero unless `negative_zero` decides the sign.
fn round_scaled_to_half(v: i128, k: i32, negative_zero: bool) -> u16 {
    if v == 0 {
        return if negative_zero { 0x8000 } else { 0 };
    }
    let sign: u16 = if v < 0 { 0x8000 } else { 0 };
    let mag = v.unsigned_abs();
    let msb = 127 - mag.leading_zeros() as i32;
    let mut exp = msb + k; // value in [2^exp, 2^(exp+1))
    if exp > 15 {
        return sign | 0x7C00;
    }
    let lsb_exp = if exp >= -14 { exp - 10 } else { -24 };
    let shift = lsb_exp - k;
    let mut m = if shift <= 0 {
        mag << (-shift) as u32
    } else {
        let shift = shift as u32;
        let kept = if shift >= 128 { 0 } else { mag >> shift };
        let rem = if shift >= 128 {
            mag
        } else {
            mag & ((1u128 << shift) - 1)
        };
        let half = if shift > 128 { 0 } else { 1u128 << (shift - 1) };
        let round_up = shift <= 128 && (rem > half || (rem == half && kept & 1 == 1));
        kept + u128::from(round_up)
    };
    if lsb_exp == -24 && exp < -14 {
        // Subnormal (a carry into 0x400 is the smallest normal, encoded
        // the same way).
        return sign | m as u16;
    }
    if m == 2048 {
        m = 1024;
        exp += 1;
    }
    if exp > 15 {
        return sign | 0x7C00;
    }
    sign | (((exp + 15) as u16) << 10) | (m as u16 - 1024)
}

/// Converts a non-NaN double to half precision, rounding to nearest even.
pub fn f64_to_half(x: f64) -> u16 {
    if x.is_nan() {
        // Only reached for invalid operations: the caller turns any NaN
        // into the ARM default NaN.
        return 0x7E00;
    }
    let sign = ((x.to_bits() >> 63) as u16) << 15;
    let a = x.abs();
    if a.is_infinite() || a >= 65520.0 {
        return sign | 0x7C00;
    }
    if a < 2f64.powi(-14) {
        // Subnormal range: units of 2^-24, ties to even.
        let m = (a * 2f64.powi(24)).round_ties_even() as u16;
        return sign | m;
    }
    let mut exp = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let mant = a / 2f64.powi(exp);
    let mut m = ((mant - 1.0) * 1024.0).round_ties_even() as u16;
    if m == 1024 {
        m = 0;
        exp += 1;
    }
    if exp > 15 {
        return sign | 0x7C00;
    }
    sign | (((exp + 15) as u16) << 10) | m
}

/// Converts a half-precision value to double (exact; NaN payloads are not
/// kept, callers that need them handle NaN first).
pub fn half_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1F;
    let mant = f64::from(h & 0x3FF);
    sign * match exp {
        0 => mant * 2f64.powi(-24),
        31 if mant == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        e => (1.0 + mant / 1024.0) * 2f64.powi(i32::from(e) - 15),
    }
}

/// FCVT/FCVTL from half precision.
pub fn half_to(h: u16, to: FpType, default_nan: bool) -> u64 {
    let is_nan = h & 0x7C00 == 0x7C00 && h & 0x3FF != 0;
    if is_nan {
        if default_nan {
            return to.default_nan();
        }
        let sign = u64::from(h >> 15);
        let payload = u64::from(h & 0x1FF);
        return match to {
            FpType::Half => u64::from(h | 0x200),
            FpType::Single => (sign << 31) | 0x7FC0_0000 | (payload << 13),
            FpType::Double => (sign << 63) | 0x7FF8_0000_0000_0000 | (payload << 42),
        };
    }
    let v = half_to_f64(h);
    match to {
        FpType::Half => u64::from(h),
        FpType::Single => u64::from((v as f32).to_bits()),
        FpType::Double => v.to_bits(),
    }
}

/// FCVT/FCVTN to half precision (`from` is the source precision).
pub fn to_half(x: u64, from: FpType, default_nan: bool) -> u16 {
    let fp = Fp {
        ty: from,
        default_nan,
    };
    let x = x & from.mask();
    if fp.is_nan(x) {
        if default_nan {
            return 0x7E00;
        }
        let (sign, top) = match from {
            FpType::Half => return x as u16 | 0x200,
            FpType::Single => ((x >> 31) as u16, ((x >> 13) & 0x1FF) as u16),
            FpType::Double => ((x >> 63) as u16, ((x >> 42) & 0x1FF) as u16),
        };
        return (sign << 15) | 0x7E00 | top;
    }
    f64_to_half(fp.to_f64(x))
}

/// `VFPExpandImm`: the 8-bit immediate of FMOV (scalar, immediate).
pub fn expand_imm(imm8: u8, ty: FpType) -> u64 {
    let sign = u64::from(imm8 >> 7);
    let b6 = u64::from((imm8 >> 6) & 1);
    let b54 = u64::from((imm8 >> 4) & 3);
    let frac = u64::from(imm8 & 0xF);
    match ty {
        FpType::Half => {
            // exp = NOT(b6) : Replicate(b6, 2) : b54
            let exp = ((b6 ^ 1) << 4) | (if b6 == 1 { 0b11 } else { 0 } << 2) | b54;
            (sign << 15) | (exp << 10) | (frac << 6)
        }
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
    fn half_precision_basics() {
        let h = Fp {
            ty: FpType::Half,
            default_nan: false,
        };
        let one = 0x3C00;
        assert_eq!(h.binop(BinOp::Add, one, one), 0x4000); // 1 + 1 = 2
        assert_eq!(h.binop(BinOp::Mul, 0x7BFF, 0x4000), 0x7C00); // 65504 * 2 = inf
        assert_eq!(h.binop(BinOp::Sub, 0x7C00, 0x7C00), 0x7E00); // default NaN
        assert_eq!(h.from_int(65519, false, 32, 0), 0x7BFF); // rounds down to max
        assert_eq!(h.from_int(65520, false, 32, 0), 0x7C00); // ties to even: inf
        assert_eq!(h.from_int(1, false, 32, 24), 0x0001); // smallest subnormal
                                                          // (1 + 2^-10) * (1 + 2^-10) - 1 needs the fused single rounding.
        let x = 0x3C01;
        assert_eq!(
            h.mul_add(0xBC00, x, x),
            u64::from(f64_to_half(2f64.powi(-9) + 2f64.powi(-20)))
        );
        assert_eq!(expand_imm(0x70, FpType::Half), 0x3C00);
    }

    #[test]
    fn fmov_immediates() {
        // fmov s0, #1.0 uses imm8 = 0x70; #-2.0 uses 0x80; #0.5 uses 0x60.
        assert_eq!(expand_imm(0x70, FpType::Single), s(1.0));
        assert_eq!(expand_imm(0x80, FpType::Single), s(-2.0));
        assert_eq!(expand_imm(0x60, FpType::Double), 0.5f64.to_bits());
    }
}
