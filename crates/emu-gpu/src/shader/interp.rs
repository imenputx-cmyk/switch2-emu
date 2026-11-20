//! Runs decoded shaders on the CPU: the reference that translated
//! (SPIR-V) shaders are compared against, and the software renderer's
//! shader core.
//!
//! One invocation runs at a time, so divergence never happens: SSY
//! pushes a reconvergence point and SYNC pops it and jumps there.

use super::decode::{
    sr, Bop, FCmp, ICmp, Inst, IntFunc, IpaMode, LopOp, Mods, MufuOp, Op, Operand, Pred, Round,
    XmadMode, RZ,
};

/// What a shader reads and writes besides its registers.
pub trait ShaderIo {
    /// Input attribute word (vertex attributes for a VS).
    fn load_attr(&self, addr: u32) -> u32;
    /// Output attribute word (VS position and varyings).
    fn store_attr(&mut self, addr: u32, value: u32);
    /// Interpolated input (PS).
    fn interpolate(&self, addr: u32, mode: IpaMode) -> f32;
    fn cbuf(&self, index: u8, offset: u32) -> u32;
    /// Samples the 2D texture whose handle is at word `handle` of the
    /// bindless constant buffer.
    fn texture(&self, _handle: u32, _u: f32, _v: f32) -> [f32; 4] {
        [0.0; 4]
    }
    /// Fetches texel (x, y) of level 0; zero outside the texture.
    fn fetch(&self, _handle: u32, _x: i32, _y: i32) -> [f32; 4] {
        [0.0; 4]
    }
    /// Gathers component `comp` of the 2x2 texels sampled at (u, v), in
    /// the order (i0,j1), (i1,j1), (i1,j0), (i0,j0).
    fn gather(&self, _handle: u32, _u: f32, _v: f32, _comp: u8) -> [f32; 4] {
        [0.0; 4]
    }
    /// Width, height, depth and levels of level `lod`.
    fn texture_size(&self, _handle: u32, _lod: u32) -> [u32; 4] {
        [0; 4]
    }
    /// A special register that depends on the invocation (thread and
    /// workgroup ids); the others are [`sr::constant`].
    fn special(&self, _sr: u8) -> u32 {
        0
    }
    /// Global memory (GPU virtual addresses); unmapped reads give zero.
    fn load_global(&mut self, _va: u64, _buf: &mut [u8]) {}
    fn store_global(&mut self, _va: u64, _data: &[u8]) {}
}

pub struct Regs(pub [u32; 256]);

impl Regs {
    fn get(&self, r: u8) -> u32 {
        if r == RZ {
            0
        } else {
            self.0[r as usize]
        }
    }

    fn set(&mut self, r: u8, v: u32) {
        if r != RZ {
            self.0[r as usize] = v;
        }
    }

    pub fn f(&self, r: u8) -> f32 {
        f32::from_bits(self.get(r))
    }
}

/// Instructions run before a program is considered stuck.
pub const STEP_LIMIT: usize = 1 << 20;

/// Maximum SSY nesting (deeper pushes overwrite the top entry).
pub const SSY_DEPTH: usize = 16;

pub fn apply(v: f32, m: Mods) -> f32 {
    let v = if m.abs { v.abs() } else { v };
    if m.neg {
        -v
    } else {
        v
    }
}

fn sat(v: f32, on: bool) -> f32 {
    // NaN saturates to 0, like the hardware's clamp.
    if on {
        if v.is_nan() {
            0.0
        } else {
            v.clamp(0.0, 1.0)
        }
    } else {
        v
    }
}

pub fn fcmp(cmp: FCmp, a: f32, b: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        return cmp & 8 != 0;
    }
    (cmp & 1 != 0 && a < b) || (cmp & 2 != 0 && a == b) || (cmp & 4 != 0 && a > b)
}

pub fn icmp(cmp: ICmp, a: u32, b: u32, signed: bool) -> bool {
    let ord = if signed {
        (a as i32).cmp(&(b as i32))
    } else {
        a.cmp(&b)
    };
    let bit = match ord {
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 2,
        std::cmp::Ordering::Greater => 4,
    };
    cmp & bit != 0
}

pub fn bop(op: Bop, a: bool, b: bool) -> bool {
    match op {
        Bop::And => a && b,
        Bop::Or => a || b,
        Bop::Xor => a != b,
    }
}

fn round(v: f32, r: Round) -> f32 {
    match r {
        Round::Nearest => v.round_ties_even(),
        Round::Floor => v.floor(),
        Round::Ceil => v.ceil(),
        Round::Trunc => v.trunc(),
        Round::Pass => v,
    }
}

fn half(v: u32, hi: bool, signed: bool) -> u32 {
    let h = if hi { v >> 16 } else { v & 0xFFFF };
    if signed {
        h as u16 as i16 as i32 as u32
    } else {
        h
    }
}

/// The result of an XMAD (shared with tests of the translation).
#[allow(clippy::too_many_arguments)]
pub fn xmad(
    a: u32,
    a_hi: bool,
    a_signed: bool,
    b: u32,
    b_hi: bool,
    b_signed: bool,
    c: u32,
    mode: XmadMode,
    psl: bool,
    mrg: bool,
) -> u32 {
    let mut p = half(a, a_hi, a_signed).wrapping_mul(half(b, b_hi, b_signed));
    if psl {
        p <<= 16;
    }
    let c = match mode {
        XmadMode::C => c,
        XmadMode::Clo => c & 0xFFFF,
        XmadMode::Chi => c >> 16,
        XmadMode::Cbcc => c.wrapping_add(b << 16),
    };
    let r = p.wrapping_add(c);
    if mrg {
        (r & 0xFFFF) | b << 16
    } else {
        r
    }
}

struct Machine<'a> {
    regs: Regs,
    preds: [bool; 16],
    io: &'a mut dyn ShaderIo,
}

impl Machine<'_> {
    fn operand(&self, o: Operand) -> u32 {
        match o {
            Operand::Reg(r) => self.regs.get(r),
            Operand::Imm(v) => v,
            Operand::Cbuf { index, offset } => self.io.cbuf(index, offset),
        }
    }

    fn float(&self, o: Operand, m: Mods) -> f32 {
        apply(f32::from_bits(self.operand(o)), m)
    }

    fn pred(&self, p: Pred) -> bool {
        (p.index == 7 || self.preds[p.index as usize]) != p.neg
    }

    fn set_pred(&mut self, index: u8, v: bool) {
        if index != 7 {
            self.preds[index as usize] = v;
        }
    }

    /// A global address: `addr` (pair or zero-extended) + `uaddr` pair +
    /// offset.
    fn address(&self, addr: u8, wide: bool, uaddr: u8, offset: i32) -> u64 {
        let pair =
            |r: u8| u64::from(self.regs.get(r)) | u64::from(self.regs.get(r.wrapping_add(1))) << 32;
        let base = if addr == RZ {
            0
        } else if wide {
            pair(addr)
        } else {
            u64::from(self.regs.get(addr))
        };
        let u = if uaddr == RZ { 0 } else { pair(uaddr) };
        base.wrapping_add(u).wrapping_add(offset as i64 as u64)
    }

    fn set_f(&mut self, dst: u8, v: f32, s: bool) {
        self.regs.set(dst, sat(v, s).to_bits());
    }

    /// Runs a non-control-flow instruction.
    fn exec(&mut self, op: Op) {
        let reg_f = |m: &Self, r: u8, mods: Mods| apply(m.regs.f(r), mods);
        let neg = |v: bool| Mods { neg: v, abs: false };
        match op {
            Op::Mov { dst, src } => {
                let v = self.operand(src);
                self.regs.set(dst, v);
            }
            Op::Fadd {
                dst,
                a,
                a_mods,
                b,
                b_mods,
                sat: s,
            } => {
                let v = reg_f(self, a, a_mods) + self.float(b, b_mods);
                self.set_f(dst, v, s);
            }
            Op::Fmul {
                dst,
                a,
                b,
                neg_b,
                sat: s,
            } => {
                let v = self.regs.f(a) * self.float(b, neg(neg_b));
                self.set_f(dst, v, s);
            }
            Op::Ffma {
                dst,
                a,
                b,
                neg_b,
                c,
                neg_c,
                sat: s,
            } => {
                let v = self
                    .regs
                    .f(a)
                    .mul_add(self.float(b, neg(neg_b)), self.float(c, neg(neg_c)));
                self.set_f(dst, v, s);
            }
            Op::Fmnmx {
                dst,
                a,
                a_mods,
                b,
                b_mods,
                min,
            } => {
                let (x, y) = (reg_f(self, a, a_mods), self.float(b, b_mods));
                // Like GLSL's NMin/NMax: a NaN input yields the other one.
                let v = if self.pred(min) { x.min(y) } else { x.max(y) };
                self.set_f(dst, v, false);
            }
            Op::Mufu {
                dst,
                func,
                a,
                mods,
                sat: s,
            } => {
                let x = reg_f(self, a, mods);
                let v = match func {
                    MufuOp::Cos => x.cos(),
                    MufuOp::Sin => x.sin(),
                    MufuOp::Ex2 => x.exp2(),
                    MufuOp::Lg2 => x.log2(),
                    MufuOp::Rcp => 1.0 / x,
                    MufuOp::Rsq => 1.0 / x.sqrt(),
                    MufuOp::Sqrt => x.sqrt(),
                };
                self.set_f(dst, v, s);
            }
            Op::Rro { dst, src, mods } => {
                let v = self.float(src, mods);
                self.set_f(dst, v, false);
            }
            Op::Fsetp {
                dst,
                dst2,
                cmp,
                a,
                a_mods,
                b,
                b_mods,
                bop: o,
                c,
            } => {
                let r = fcmp(cmp, reg_f(self, a, a_mods), self.float(b, b_mods));
                let c = self.pred(c);
                self.set_pred(dst, bop(o, r, c));
                self.set_pred(dst2, bop(o, !r, c));
            }
            Op::Fset {
                dst,
                cmp,
                a,
                a_mods,
                b,
                b_mods,
                bop: o,
                c,
                bf,
            } => {
                let r = bop(
                    o,
                    fcmp(cmp, reg_f(self, a, a_mods), self.float(b, b_mods)),
                    self.pred(c),
                );
                self.regs.set(dst, set_value(r, bf));
            }
            Op::Isetp {
                dst,
                dst2,
                cmp,
                signed,
                a,
                b,
                bop: o,
                c,
            } => {
                let r = icmp(cmp, self.regs.get(a), self.operand(b), signed);
                let c = self.pred(c);
                self.set_pred(dst, bop(o, r, c));
                self.set_pred(dst2, bop(o, !r, c));
            }
            Op::Iset {
                dst,
                cmp,
                signed,
                a,
                b,
                bop: o,
                c,
                bf,
            } => {
                let r = bop(
                    o,
                    icmp(cmp, self.regs.get(a), self.operand(b), signed),
                    self.pred(c),
                );
                self.regs.set(dst, set_value(r, bf));
            }
            Op::Psetp {
                dst,
                dst2,
                op1,
                op2,
                a,
                b,
                c,
            } => {
                let r = bop(op1, self.pred(a), self.pred(b));
                let c = self.pred(c);
                self.set_pred(dst, bop(op2, r, c));
                self.set_pred(dst2, bop(op2, !r, c));
            }
            Op::Iadd {
                dst,
                a,
                neg_a,
                b,
                neg_b,
            } => {
                let x = negate(self.regs.get(a), neg_a);
                let y = negate(self.operand(b), neg_b);
                self.regs.set(dst, x.wrapping_add(y));
            }
            Op::Iscadd {
                dst,
                a,
                neg_a,
                b,
                neg_b,
                shift,
            } => {
                let x = negate(self.regs.get(a), neg_a) << shift;
                let y = negate(self.operand(b), neg_b);
                self.regs.set(dst, x.wrapping_add(y));
            }
            Op::Shl { dst, a, b } => {
                let s = self.operand(b);
                let v = if s >= 32 { 0 } else { self.regs.get(a) << s };
                self.regs.set(dst, v);
            }
            Op::Shr { dst, a, b, signed } => {
                let (x, s) = (self.regs.get(a), self.operand(b));
                let v = if signed {
                    ((x as i32) >> s.min(31)) as u32
                } else if s >= 32 {
                    0
                } else {
                    x >> s
                };
                self.regs.set(dst, v);
            }
            Op::Lop {
                dst,
                op,
                a,
                inv_a,
                b,
                inv_b,
            } => {
                let x = if inv_a {
                    !self.regs.get(a)
                } else {
                    self.regs.get(a)
                };
                let y = if inv_b {
                    !self.operand(b)
                } else {
                    self.operand(b)
                };
                let v = match op {
                    LopOp::And => x & y,
                    LopOp::Or => x | y,
                    LopOp::Xor => x ^ y,
                    LopOp::PassB => y,
                };
                self.regs.set(dst, v);
            }
            Op::Imnmx {
                dst,
                a,
                b,
                signed,
                min,
            } => {
                let (x, y) = (self.regs.get(a), self.operand(b));
                let v = match (signed, self.pred(min)) {
                    (true, true) => (x as i32).min(y as i32) as u32,
                    (true, false) => (x as i32).max(y as i32) as u32,
                    (false, true) => x.min(y),
                    (false, false) => x.max(y),
                };
                self.regs.set(dst, v);
            }
            Op::Iadd3 {
                dst,
                a,
                neg_a,
                b,
                neg_b,
                c,
                neg_c,
            } => {
                let n = |v: u32, on: bool| if on { v.wrapping_neg() } else { v };
                let v = n(self.regs.get(a), neg_a)
                    .wrapping_add(n(self.operand(b), neg_b))
                    .wrapping_add(n(self.operand(c), neg_c));
                self.regs.set(dst, v);
            }
            Op::Lop3 {
                dst,
                a,
                b,
                c,
                lut,
                pred,
            } => {
                let v = lop3(self.regs.get(a), self.operand(b), self.operand(c), lut);
                self.regs.set(dst, v);
                self.set_pred(pred, v != 0);
            }
            Op::Imad {
                dst,
                a,
                b,
                c,
                neg_c,
            } => {
                let c = self.operand(c);
                let c = if neg_c { c.wrapping_neg() } else { c };
                let v = self
                    .regs
                    .get(a)
                    .wrapping_mul(self.operand(b))
                    .wrapping_add(c);
                self.regs.set(dst, v);
            }
            Op::Shf {
                dst,
                a,
                b,
                c,
                right,
                hi,
                wrap,
                wide,
                signed,
            } => {
                let v = shf(
                    self.regs.get(a),
                    self.operand(b),
                    self.operand(c),
                    right,
                    hi,
                    wrap,
                    wide,
                    signed,
                );
                self.regs.set(dst, v);
            }
            Op::Sel { dst, a, b, cond } => {
                let v = if self.pred(cond) {
                    self.regs.get(a)
                } else {
                    self.operand(b)
                };
                self.regs.set(dst, v);
            }
            Op::Xmad {
                dst,
                a,
                a_hi,
                a_signed,
                b,
                b_hi,
                b_signed,
                c,
                mode,
                psl,
                mrg,
            } => {
                let v = xmad(
                    self.regs.get(a),
                    a_hi,
                    a_signed,
                    self.operand(b),
                    b_hi,
                    b_signed,
                    self.operand(c),
                    mode,
                    psl,
                    mrg,
                );
                self.regs.set(dst, v);
            }
            Op::F2i {
                dst,
                src,
                mods,
                signed,
                round: r,
            } => {
                // Rust's float-to-int casts saturate and map NaN to 0.
                let x = round(self.float(src, mods), r);
                let v = if signed { x as i32 as u32 } else { x as u32 };
                self.regs.set(dst, v);
            }
            Op::I2f {
                dst,
                src,
                signed,
                neg,
                abs,
            } => {
                let x = self.operand(src);
                let v = if signed {
                    let i = x as i32 as i64;
                    let i = if abs { i.abs() } else { i };
                    (if neg { -i } else { i }) as f32
                } else {
                    let f = x as f32;
                    if neg {
                        -f
                    } else {
                        f
                    }
                };
                self.set_f(dst, v, false);
            }
            Op::F2f {
                dst,
                src,
                mods,
                round: r,
                sat: s,
            } => {
                let v = round(self.float(src, mods), r);
                self.set_f(dst, v, s);
            }
            Op::Ald {
                dst,
                base,
                addr,
                count,
            } => {
                let addr = addr + self.regs.get(base);
                for i in 0..count {
                    let v = self.io.load_attr(addr + 4 * i);
                    self.regs.set(dst.wrapping_add(i as u8), v);
                }
            }
            Op::Ast {
                src,
                base,
                addr,
                count,
            } => {
                let addr = addr + self.regs.get(base);
                for i in 0..count {
                    let v = self.regs.get(src.wrapping_add(i as u8));
                    self.io.store_attr(addr + 4 * i, v);
                }
            }
            Op::Ipa {
                dst,
                addr,
                mode,
                sat: s,
            } => {
                let v = self.io.interpolate(addr, mode);
                self.set_f(dst, v, s);
            }
            Op::Texs {
                dst0,
                dst1,
                a,
                b,
                handle,
                comps,
                ..
            } => {
                let t = self.io.texture(handle, self.regs.f(a), self.regs.f(b));
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    self.regs.set(texs_dst(dst0, dst1, k), t[c].to_bits());
                }
            }
            Op::Tld {
                dst0,
                dst1,
                a,
                b,
                handle,
                comps,
            } => {
                let t = self
                    .io
                    .fetch(handle, self.regs.get(a) as i32, self.regs.get(b) as i32);
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    self.regs.set(texs_dst(dst0, dst1, k), t[c].to_bits());
                }
            }
            Op::Tex {
                dst,
                a,
                handle,
                comps,
                ..
            } => {
                let t = self
                    .io
                    .texture(handle, self.regs.f(a), self.regs.f(a.wrapping_add(1)));
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    if dst != RZ {
                        self.regs.set(dst.wrapping_add(k as u8), t[c].to_bits());
                    }
                }
            }
            Op::S2r { dst, sr: r } => {
                let v = sr::constant(r).unwrap_or_else(|| self.io.special(r));
                self.regs.set(dst, v);
            }
            Op::Ldc {
                dst,
                index,
                base,
                offset,
                wide,
            } => {
                let at = self.regs.get(base).wrapping_add(offset);
                self.regs.set(dst, self.io.cbuf(index, at));
                if wide && dst != RZ {
                    self.regs
                        .set(dst + 1, self.io.cbuf(index, at.wrapping_add(4)));
                }
            }
            Op::Ldg {
                dst,
                addr,
                wide,
                uaddr,
                offset,
                size,
            } => {
                let va = self.address(addr, wide, uaddr, offset);
                let bytes = access_bytes(size);
                let mut buf = [0u8; 16];
                self.io.load_global(va, &mut buf[..bytes]);
                let word = |k: usize| u32::from_le_bytes(buf[4 * k..4 * k + 4].try_into().unwrap());
                match size {
                    0 => self.regs.set(dst, u32::from(buf[0])),
                    1 => self.regs.set(dst, buf[0] as i8 as i32 as u32),
                    2 => self
                        .regs
                        .set(dst, u32::from(u16::from_le_bytes([buf[0], buf[1]]))),
                    3 => self
                        .regs
                        .set(dst, i16::from_le_bytes([buf[0], buf[1]]) as i32 as u32),
                    _ => {
                        for k in 0..bytes / 4 {
                            if dst != RZ {
                                self.regs.set(dst.wrapping_add(k as u8), word(k));
                            }
                        }
                    }
                }
            }
            Op::Stg {
                src,
                addr,
                wide,
                uaddr,
                offset,
                size,
            } => {
                let va = self.address(addr, wide, uaddr, offset);
                let bytes = access_bytes(size);
                let mut buf = Vec::with_capacity(16);
                for k in 0..bytes.div_ceil(4) {
                    let r = if src == RZ {
                        RZ
                    } else {
                        src.wrapping_add(k as u8)
                    };
                    buf.extend_from_slice(&self.regs.get(r).to_le_bytes());
                }
                self.io.store_global(va, &buf[..bytes]);
            }
            Op::Plop3 {
                dst,
                dst2,
                a,
                b,
                c,
                lut,
                lut2,
            } => {
                let i = (self.pred(a) as u8) << 2 | (self.pred(b) as u8) << 1 | self.pred(c) as u8;
                self.set_pred(dst, lut >> i & 1 != 0);
                self.set_pred(dst2, lut2 >> i & 1 != 0);
            }
            Op::IntUnary { dst, func, src } => {
                let x = self.operand(src);
                self.regs.set(dst, int_unary(func, x));
            }
            Op::Prmt { dst, a, sel, c } => {
                let v = prmt(self.regs.get(a), self.operand(sel), self.operand(c));
                self.regs.set(dst, v);
            }
            Op::ImadWide {
                dst,
                a,
                b,
                c,
                signed,
            } => {
                let (x, y) = (self.regs.get(a), self.operand(b));
                let p = if signed {
                    (i64::from(x as i32) * i64::from(y as i32)) as u64
                } else {
                    u64::from(x) * u64::from(y)
                };
                let z = if c == RZ {
                    0
                } else {
                    u64::from(self.regs.get(c)) | u64::from(self.regs.get(c + 1)) << 32
                };
                let r = p.wrapping_add(z);
                self.regs.set(dst, r as u32);
                if dst != RZ {
                    self.regs.set(dst + 1, (r >> 32) as u32);
                }
            }
            Op::ImadHi {
                dst,
                a,
                b,
                c,
                signed,
            } => {
                let (x, y) = (self.regs.get(a), self.operand(b));
                let hi = if signed {
                    ((i64::from(x as i32) * i64::from(y as i32)) >> 32) as u32
                } else {
                    ((u64::from(x) * u64::from(y)) >> 32) as u32
                };
                self.regs.set(dst, hi.wrapping_add(self.operand(c)));
            }
            Op::Vote { dst, pdst, src, eq } => {
                let p = self.pred(src);
                self.regs.set(dst, u32::from(p));
                self.set_pred(pdst, eq || p);
            }
            Op::Txq {
                dst,
                dst1,
                lod,
                handle,
                comps,
            } => {
                let size = self.io.texture_size(handle, self.regs.get(lod));
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    self.regs.set(texs_dst(dst, dst1, k), size[c]);
                }
            }
            Op::Tld4 {
                dst0,
                dst1,
                a,
                b,
                handle,
                comp,
                comps,
            } => {
                let t = self.io.gather(handle, self.regs.f(a), self.regs.f(b), comp);
                for (k, c) in (0..4).filter(|c| comps & (1 << c) != 0).enumerate() {
                    self.regs.set(texs_dst(dst0, dst1, k), t[c].to_bits());
                }
            }
            Op::Nop | Op::Bra { .. } | Op::Ssy { .. } | Op::Sync | Op::Kil | Op::Exit => {}
        }
    }
}

/// Bytes of a global access of Maxwell size `size`.
pub fn access_bytes(size: u8) -> usize {
    match size {
        0 | 1 => 1,
        2 | 3 => 2,
        4 => 4,
        5 => 8,
        _ => 16,
    }
}

pub fn int_unary(func: IntFunc, x: u32) -> u32 {
    match func {
        IntFunc::Popc => x.count_ones(),
        IntFunc::Brev => x.reverse_bits(),
        IntFunc::Iabs => (x as i32).unsigned_abs(),
        IntFunc::Flo { signed, shift } => {
            let m = if signed && (x as i32) < 0 { !x } else { x };
            if m == 0 {
                u32::MAX
            } else if shift {
                m.leading_zeros()
            } else {
                31 - m.leading_zeros()
            }
        }
    }
}

/// PRMT in its default mode.
pub fn prmt(a: u32, sel: u32, c: u32) -> u32 {
    let bytes = (u64::from(c) << 32 | u64::from(a)).to_le_bytes();
    (0..4).fold(0, |r, i| {
        let n = sel >> (4 * i) & 0xF;
        let b = bytes[(n & 7) as usize];
        let b = if n & 8 != 0 {
            if b & 0x80 != 0 {
                0xFF
            } else {
                0
            }
        } else {
            b
        };
        r | u32::from(b) << (8 * i)
    })
}

/// Register of the `k`-th written component of a TEXS (RZ: discarded).
pub fn texs_dst(dst0: u8, dst1: u8, k: usize) -> u8 {
    let (base, i) = if k < 2 { (dst0, k) } else { (dst1, k - 2) };
    if base == RZ {
        RZ
    } else {
        base.wrapping_add(i as u8)
    }
}

fn negate(v: u32, on: bool) -> u32 {
    if on {
        v.wrapping_neg()
    } else {
        v
    }
}

fn set_value(r: bool, bf: bool) -> u32 {
    match (r, bf) {
        (false, _) => 0,
        (true, true) => 1.0f32.to_bits(),
        (true, false) => u32::MAX,
    }
}

/// Runs a program to its EXIT (or past its last instruction); returns
/// the registers, or `None` if the invocation was discarded (KIL).
pub fn run(program: &[Inst], io: &mut dyn ShaderIo) -> Option<Regs> {
    let mut m = Machine {
        regs: Regs([0; 256]),
        preds: [false; 16],
        io,
    };
    let mut ssy = Vec::with_capacity(SSY_DEPTH);
    let mut pc = 0;
    for _ in 0..STEP_LIMIT {
        let Some(&inst) = program.get(pc) else { break };
        pc += 1;
        if !m.pred(inst.pred) {
            continue;
        }
        match inst.op {
            Op::Exit => break,
            Op::Kil => return None,
            Op::Bra { target } => pc = target,
            Op::Ssy { target } => {
                if ssy.len() == SSY_DEPTH {
                    ssy.pop();
                }
                ssy.push(target);
            }
            Op::Sync => match ssy.pop() {
                Some(t) => pc = t,
                None => break,
            },
            op => m.exec(op),
        }
    }
    Some(m.regs)
}

/// LOP3.LUT on three words.
pub fn lop3(a: u32, b: u32, c: u32, lut: u8) -> u32 {
    let mut r = 0;
    for i in 0..8u32 {
        if lut >> i & 1 != 0 {
            let pick = |v: u32, bit: u32| if i >> bit & 1 != 0 { v } else { !v };
            r |= pick(a, 2) & pick(b, 1) & pick(c, 0);
        }
    }
    r
}

/// SHF: shifts the 64-bit value `high:low` and returns its low or high word.
#[allow(clippy::too_many_arguments)]
pub fn shf(
    low: u32,
    amount: u32,
    high: u32,
    right: bool,
    hi: bool,
    wrap: bool,
    wide: bool,
    signed: bool,
) -> u32 {
    let width = if wide { 64 } else { 32 };
    let s = if wrap {
        amount & (width - 1)
    } else {
        amount.min(width)
    };
    let v = (u64::from(high) << 32) | u64::from(low);
    let r = if right {
        if signed {
            (v as i64)
                .checked_shr(s)
                .unwrap_or(if (v as i64) < 0 { -1 } else { 0 }) as u64
        } else {
            v.checked_shr(s).unwrap_or(0)
        }
    } else {
        v.checked_shl(s).unwrap_or(0)
    };
    if hi {
        (r >> 32) as u32
    } else {
        r as u32
    }
}

#[cfg(test)]
mod tests {
    use super::super::{asm, decode::decode_program, HEADER_SIZE};
    use super::*;

    struct Inputs([f32; 4]);

    impl ShaderIo for Inputs {
        fn load_attr(&self, _: u32) -> u32 {
            0
        }
        fn store_attr(&mut self, _: u32, _: u32) {}
        fn interpolate(&self, addr: u32, _: IpaMode) -> f32 {
            self.0[((addr - 0x80) / 4) as usize]
        }
        fn cbuf(&self, _: u8, _: u32) -> u32 {
            0
        }
    }

    fn colors(rgb: [f32; 3]) -> Option<[f32; 4]> {
        let code = asm::control_flow_sample();
        let program = decode_program(&code[HEADER_SIZE..]).unwrap();
        let regs = run(&program, &mut Inputs([rgb[0], rgb[1], rgb[2], 1.0]))?;
        Some([regs.f(0), regs.f(1), regs.f(2), regs.f(3)])
    }

    #[test]
    fn branches_loops_and_discards() {
        assert_eq!(
            colors([0.8, 0.6, 0.2]),
            Some([0.8, 0.6 * 0.25, 48.0 * (1.0 / 255.0), 0.75])
        );
        assert_eq!(
            colors([0.3, 0.5, 0.1]),
            Some([0.3, 0.5, 144.0 * (1.0 / 255.0), 0.25])
        );
        assert_eq!(colors([0.3, 0.1, 0.2]), None);
    }

    #[test]
    fn integer_helpers() {
        assert!(fcmp(7, 1.0, 2.0) && !fcmp(7, f32::NAN, 2.0));
        assert!(fcmp(9, f32::NAN, 0.0) && fcmp(9, -1.0, 0.0) && !fcmp(1, f32::NAN, 0.0));
        assert!(icmp(1, u32::MAX, 0, true) && !icmp(1, u32::MAX, 0, false));
        // (0x1234 * 0x0002) + (0x5678 << 16) with CBCC, then merge.
        assert_eq!(
            xmad(
                0x1234,
                false,
                false,
                2,
                false,
                false,
                0,
                XmadMode::C,
                false,
                false
            ),
            0x2468
        );
        assert_eq!(
            xmad(
                0xABCD_0003,
                true,
                false,
                0x0001_0002,
                true,
                false,
                5,
                XmadMode::C,
                true,
                false
            ),
            (0xABCD << 16) + 5
        );
        assert_eq!(
            xmad(
                0xFFFF,
                false,
                true,
                3,
                false,
                false,
                0,
                XmadMode::C,
                false,
                false
            ),
            (-3i32) as u32
        );
        assert_eq!(
            xmad(
                2,
                false,
                false,
                0x0007_0003,
                false,
                false,
                1,
                XmadMode::Cbcc,
                false,
                true
            ),
            (7) | 0x0003_0000
        );
    }
}
