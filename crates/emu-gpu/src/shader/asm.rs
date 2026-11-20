//! A tiny Maxwell assembler for the decoded subset, used by tests and to
//! build test programs' shaders (`cargo run -p emu-gpu --example shaders`).
//! Instructions are unpredicated (PT) unless wrapped in [`guard`].

use super::decode::byte_of;
use super::HEADER_SIZE;

const PT: u64 = 7 << 16;

fn op(top: u64) -> u64 {
    top << 48 | PT
}

fn r(reg: u8, lo: u32) -> u64 {
    u64::from(reg) << lo
}

/// A 20-bit float immediate: the value's top bits (low 12 bits dropped).
fn f20(value: f32) -> u64 {
    let bits = value.to_bits();
    assert_eq!(
        bits & 0xFFF,
        0,
        "{value} does not fit a 20-bit float immediate"
    );
    u64::from((bits >> 12) & 0x7FFFF) << 20 | u64::from(bits >> 31) << 56
}

pub fn mov(dst: u8, src: u8) -> u64 {
    op(0x5c98) | r(dst, 0) | r(src, 20) | 0xF << 39
}

pub fn mov32i(dst: u8, imm: u32) -> u64 {
    op(0x0100) | r(dst, 0) | u64::from(imm) << 20 | 0xF << 12
}

pub fn mov_cbuf(dst: u8, index: u8, offset: u32) -> u64 {
    op(0x4c98) | r(dst, 0) | u64::from(index) << 34 | u64::from(offset / 4) << 20 | 0xF << 39
}

pub fn fadd(dst: u8, a: u8, b: u8) -> u64 {
    op(0x5c58) | r(dst, 0) | r(a, 8) | r(b, 20)
}

pub fn fadd_imm(dst: u8, a: u8, b: f32) -> u64 {
    op(0x3858) | r(dst, 0) | r(a, 8) | f20(b)
}

pub fn fadd32i(dst: u8, a: u8, b: f32) -> u64 {
    op(0x0800) | r(dst, 0) | r(a, 8) | u64::from(b.to_bits()) << 20
}

pub fn fmul(dst: u8, a: u8, b: u8) -> u64 {
    op(0x5c68) | r(dst, 0) | r(a, 8) | r(b, 20)
}

/// dst = a * c[index][offset].
pub fn fmul_cbuf(dst: u8, a: u8, index: u8, offset: u32) -> u64 {
    op(0x4c68) | r(dst, 0) | r(a, 8) | u64::from(index) << 34 | u64::from(offset / 4) << 20
}

/// dst = a + c[index][offset].
pub fn fadd_cbuf(dst: u8, a: u8, index: u8, offset: u32) -> u64 {
    op(0x4c58) | r(dst, 0) | r(a, 8) | u64::from(index) << 34 | u64::from(offset / 4) << 20
}

pub fn fmul32i(dst: u8, a: u8, b: f32) -> u64 {
    op(0x1e00) | r(dst, 0) | r(a, 8) | u64::from(b.to_bits()) << 20
}

/// dst = a * b + c (registers).
pub fn ffma(dst: u8, a: u8, b: u8, c: u8) -> u64 {
    op(0x5980) | r(dst, 0) | r(a, 8) | r(b, 20) | r(c, 39)
}

/// dst = a * c[index][offset] + c.
pub fn ffma_cbuf(dst: u8, a: u8, index: u8, offset: u32, c: u8) -> u64 {
    op(0x4980)
        | r(dst, 0)
        | r(a, 8)
        | u64::from(index) << 34
        | u64::from(offset / 4) << 20
        | r(c, 39)
}

/// Loads `count` (1-4) attribute words at `addr` into dst...
pub fn ald(dst: u8, addr: u32, count: u32) -> u64 {
    op(0xefd8)
        | r(dst, 0)
        | r(255, 8)
        | u64::from(addr) << 20
        | u64::from(count - 1) << 47
        | r(255, 39)
}

/// Stores `count` (1-4) registers from src up to attributes at `addr`.
pub fn ast(src: u8, addr: u32, count: u32) -> u64 {
    op(0xeff0)
        | r(src, 0)
        | r(255, 8)
        | u64::from(addr) << 20
        | u64::from(count - 1) << 47
        | r(255, 39)
}

/// Interpolates the attribute at `addr` (perspective-correct, "pass").
pub fn ipa(dst: u8, addr: u32) -> u64 {
    op(0xe000) | r(dst, 0) | r(255, 8) | u64::from(addr) << 28 | r(255, 20) | r(255, 39) | 7 << 47
}

pub fn exit() -> u64 {
    op(0xe300) | 0xF
}

pub fn nop() -> u64 {
    op(0x50b0)
}

/// Predicates `inst` on P`p` (negated if `neg`); P7 is PT.
pub fn guard(inst: u64, p: u8, neg: bool) -> u64 {
    inst & !(0xF << 16) | u64::from(p) << 16 | u64::from(neg) << 19
}

/// Sets the immediate-form second operand: a 20-bit signed integer.
fn s20(value: i32) -> u64 {
    assert!(
        (-(1 << 19)..1 << 19).contains(&value),
        "{value} does not fit 20 bits"
    );
    (value as u64 & 0x7FFFF) << 20 | u64::from(value < 0) << 56
}

/// Predicate source at bit `lo` (index, then negation).
fn p(index: u8, neg: bool, lo: u32) -> u64 {
    (u64::from(index) | u64::from(neg) << 3) << lo
}

/// Branch offset of `target` from the instruction after index `at`.
fn rel(at: usize, target: usize) -> u64 {
    let off = byte_of(target) - (byte_of(at) + 8);
    (off as u64 & 0xFF_FFFF) << 20
}

/// Branch (instruction `at`) to instruction `target`.
pub fn bra(at: usize, target: usize) -> u64 {
    op(0xe240) | 0xF | rel(at, target)
}

/// Sets the reconvergence point for SYNC (instruction `at`).
pub fn ssy(at: usize, target: usize) -> u64 {
    op(0xe290) | rel(at, target)
}

pub fn sync() -> u64 {
    op(0xf0f8) | 0xF
}

pub fn kil() -> u64 {
    op(0xe330) | 0xF
}

/// Float compare codes (FSETP/FSET).
pub mod fcmp {
    pub const LT: u8 = 1;
    pub const EQ: u8 = 2;
    pub const LE: u8 = 3;
    pub const GT: u8 = 4;
    pub const NE: u8 = 5;
    pub const GE: u8 = 6;
    pub const NUM: u8 = 7;
    pub const NAN: u8 = 8;
    pub const LTU: u8 = 9;
}

/// Integer compare codes (ISETP/ISET).
pub mod icmp {
    pub const LT: u8 = 1;
    pub const EQ: u8 = 2;
    pub const LE: u8 = 3;
    pub const GT: u8 = 4;
    pub const NE: u8 = 5;
    pub const GE: u8 = 6;
}

/// P`dst` = cmp(a, b) (and !cmp to PT), combined with PT by AND.
pub fn fsetp(dst: u8, cmp: u8, a: u8, b: u8) -> u64 {
    op(0x5bb0)
        | u64::from(cmp) << 48
        | r(a, 8)
        | r(b, 20)
        | u64::from(dst) << 3
        | 7
        | p(7, false, 39)
}

pub fn fsetp_imm(dst: u8, cmp: u8, a: u8, b: f32) -> u64 {
    op(0x36b0) | u64::from(cmp) << 48 | r(a, 8) | f20(b) | u64::from(dst) << 3 | 7 | p(7, false, 39)
}

/// P`dst` = cmp(a, b) bop P`c` and P`dst2` = !cmp(a, b) bop P`c`
/// (bop: 0 and, 1 or, 2 xor).
#[allow(clippy::too_many_arguments)]
pub fn isetp(dst: u8, dst2: u8, cmp: u8, signed: bool, a: u8, b: u8, bop: u8, c: u8) -> u64 {
    op(0x5b60)
        | u64::from(cmp) << 49
        | u64::from(signed) << 48
        | r(a, 8)
        | r(b, 20)
        | u64::from(dst) << 3
        | u64::from(dst2)
        | u64::from(bop) << 45
        | p(c, false, 39)
}

pub fn isetp_imm(dst: u8, cmp: u8, signed: bool, a: u8, b: i32) -> u64 {
    op(0x3660)
        | u64::from(cmp) << 49
        | u64::from(signed) << 48
        | r(a, 8)
        | s20(b)
        | u64::from(dst) << 3
        | 7
        | p(7, false, 39)
}

/// P`dst` = (P`a` op P`b`) (op: 0 and, 1 or, 2 xor); `neg_b` negates b.
pub fn psetp(dst: u8, op1: u8, a: u8, b: u8, neg_b: bool) -> u64 {
    op(0x5090)
        | u64::from(op1) << 24
        | p(a, false, 12)
        | p(b, neg_b, 29)
        | p(7, false, 39)
        | u64::from(dst) << 3
        | 7
}

/// dst = 1.0 (`bf`) or ~0 if cmp(a, b), else 0.
pub fn fset(dst: u8, cmp: u8, a: u8, b: u8, bf: bool) -> u64 {
    op(0x5800)
        | u64::from(cmp) << 48
        | u64::from(bf) << 52
        | r(dst, 0)
        | r(a, 8)
        | r(b, 20)
        | p(7, false, 39)
}

pub fn iset(dst: u8, cmp: u8, signed: bool, a: u8, b: u8, bf: bool) -> u64 {
    op(0x5b50)
        | u64::from(cmp) << 49
        | u64::from(signed) << 48
        | u64::from(bf) << 44
        | r(dst, 0)
        | r(a, 8)
        | r(b, 20)
        | p(7, false, 39)
}

pub fn iadd(dst: u8, a: u8, b: u8) -> u64 {
    op(0x5c10) | r(dst, 0) | r(a, 8) | r(b, 20)
}

/// dst = a - b.
pub fn isub(dst: u8, a: u8, b: u8) -> u64 {
    iadd(dst, a, b) | 1 << 48
}

pub fn iadd_imm(dst: u8, a: u8, b: i32) -> u64 {
    op(0x3810) | r(dst, 0) | r(a, 8) | s20(b)
}

pub fn iadd32i(dst: u8, a: u8, b: u32) -> u64 {
    op(0x1c00) | r(dst, 0) | r(a, 8) | u64::from(b) << 20
}

/// dst = (a << shift) + b.
pub fn iscadd(dst: u8, a: u8, b: u8, shift: u32) -> u64 {
    op(0x5c18) | r(dst, 0) | r(a, 8) | r(b, 20) | u64::from(shift) << 39
}

pub fn shl(dst: u8, a: u8, b: u8) -> u64 {
    op(0x5c48) | r(dst, 0) | r(a, 8) | r(b, 20)
}

pub fn shl_imm(dst: u8, a: u8, b: i32) -> u64 {
    op(0x3848) | r(dst, 0) | r(a, 8) | s20(b)
}

pub fn shr(dst: u8, a: u8, b: u8, signed: bool) -> u64 {
    op(0x5c28) | r(dst, 0) | r(a, 8) | r(b, 20) | u64::from(signed) << 48
}

pub fn shr_imm(dst: u8, a: u8, b: i32, signed: bool) -> u64 {
    op(0x3828) | r(dst, 0) | r(a, 8) | s20(b) | u64::from(signed) << 48
}

/// Logic op (0 and, 1 or, 2 xor, 3 pass b) with optional inversions.
pub fn lop(dst: u8, lop: u8, a: u8, inv_a: bool, b: u8, inv_b: bool) -> u64 {
    op(0x5c40)
        | r(dst, 0)
        | r(a, 8)
        | r(b, 20)
        | u64::from(lop) << 41
        | u64::from(inv_a) << 39
        | u64::from(inv_b) << 40
}

pub fn lop32i(dst: u8, lop: u8, a: u8, b: u32) -> u64 {
    op(0x0400) | r(dst, 0) | r(a, 8) | u64::from(b) << 20 | u64::from(lop) << 53
}

pub fn imnmx(dst: u8, a: u8, b: u8, signed: bool, min: bool) -> u64 {
    op(0x5c20) | r(dst, 0) | r(a, 8) | r(b, 20) | u64::from(signed) << 48 | p(7, !min, 39)
}

pub fn fmnmx(dst: u8, a: u8, b: u8, min: bool) -> u64 {
    op(0x5c60) | r(dst, 0) | r(a, 8) | r(b, 20) | p(7, !min, 39)
}

/// dst = P`c` ? a : b.
pub fn sel(dst: u8, a: u8, b: u8, c: u8, neg: bool) -> u64 {
    op(0x5ca0) | r(dst, 0) | r(a, 8) | r(b, 20) | p(c, neg, 39)
}

/// XMAD with registers. `mode`: 0 C, 1 CLO, 2 CHI, 4 CBCC.
#[allow(clippy::too_many_arguments)]
pub fn xmad(
    dst: u8,
    a: u8,
    a_hi: bool,
    b: u8,
    b_hi: bool,
    c: u8,
    mode: u8,
    psl: bool,
    mrg: bool,
) -> u64 {
    op(0x5b00)
        | r(dst, 0)
        | r(a, 8)
        | r(b, 20)
        | r(c, 39)
        | u64::from(a_hi) << 53
        | u64::from(b_hi) << 35
        | u64::from(mode) << 50
        | u64::from(psl) << 36
        | u64::from(mrg) << 37
}

/// XMAD with a 16-bit immediate B.
pub fn xmad_imm(dst: u8, a: u8, b: u16, c: u8) -> u64 {
    op(0x3600) | r(dst, 0) | r(a, 8) | u64::from(b) << 20 | r(c, 39)
}

/// MUFU function codes.
pub mod mufu {
    pub const COS: u8 = 0;
    pub const SIN: u8 = 1;
    pub const EX2: u8 = 2;
    pub const LG2: u8 = 3;
    pub const RCP: u8 = 4;
    pub const RSQ: u8 = 5;
    pub const SQRT: u8 = 8;
}

pub fn mufu(dst: u8, func: u8, a: u8) -> u64 {
    op(0x5080) | r(dst, 0) | r(a, 8) | u64::from(func) << 20
}

pub fn rro(dst: u8, a: u8) -> u64 {
    op(0x5c90) | r(dst, 0) | r(a, 20)
}

/// f32 to 32-bit integer; `round`: 0 nearest, 1 floor, 2 ceil, 3 trunc.
pub fn f2i(dst: u8, a: u8, signed: bool, round: u8) -> u64 {
    op(0x5cb0)
        | r(dst, 0)
        | r(a, 20)
        | 2 << 10
        | 2 << 8
        | u64::from(signed) << 12
        | u64::from(round) << 39
}

pub fn i2f(dst: u8, a: u8, signed: bool) -> u64 {
    op(0x5cb8) | r(dst, 0) | r(a, 20) | 2 << 10 | 2 << 8 | u64::from(signed) << 13
}

/// Rounds to an integral value (same codes as `f2i`).
pub fn frnd(dst: u8, a: u8, round: u8) -> u64 {
    op(0x5ca8) | r(dst, 0) | r(a, 20) | 2 << 10 | 2 << 8 | 1 << 42 | u64::from(round) << 39
}

/// TEXS: samples texture `handle` (bindless constant buffer word) at
/// (a, b). `sel` picks the components: with dst1 = RZ 0..=7 for R, G,
/// B, A, RG, RA, GA, BA; otherwise 0..=4 for RGB, RGA, RBA, GBA, RGBA.
pub fn texs(dst0: u8, dst1: u8, a: u8, b: u8, handle: u32, sel: u8, lz: bool) -> u64 {
    op(0xd000)
        | r(dst0, 0)
        | r(a, 8)
        | r(b, 20)
        | r(dst1, 28)
        | u64::from(handle) << 36
        | u64::from(sel) << 50
        | (if lz { 2 } else { 1 }) << 53
}

/// TEX (2D): samples texture `handle` at (a, a + 1); the `comps` (RGBA
/// bits) go to consecutive registers from dst.
pub fn tex(dst: u8, a: u8, handle: u32, comps: u8, lz: bool) -> u64 {
    op(0xc038)
        | r(dst, 0)
        | r(a, 8)
        | r(255, 20)
        | 1 << 29
        | u64::from(comps) << 31
        | u64::from(handle) << 36
        | u64::from(lz) << 55
}

/// A neutral scheduling control word (no stalls, no barriers).
pub const SCHED: u64 = 0x001f_8000_fc00_07e0;

/// Builds a program: header, then the instructions with a scheduling
/// word before every three.
pub fn program(header: &[u8; HEADER_SIZE], insts: &[u64]) -> Vec<u8> {
    let mut out = header.to_vec();
    for group in insts.chunks(3) {
        out.extend_from_slice(&SCHED.to_le_bytes());
        for i in 0..3 {
            out.extend_from_slice(&group.get(i).copied().unwrap_or_else(nop).to_le_bytes());
        }
    }
    out
}

/// A vertex shader header (SPH type 1, VTG).
pub fn vertex_header() -> [u8; HEADER_SIZE] {
    let mut h = [0u8; HEADER_SIZE];
    let w0: u32 = 1 | 3 << 5 | 1 << 10; // VTG, version 3, VERTEX
    h[0..4].copy_from_slice(&w0.to_le_bytes());
    h
}

/// A pixel shader header (SPH type 2) writing RGBA of color target 0.
pub fn pixel_header() -> [u8; HEADER_SIZE] {
    let mut h = [0u8; HEADER_SIZE];
    let w0: u32 = 2 | 3 << 5 | 5 << 10; // PS, version 3, PIXEL
    h[0..4].copy_from_slice(&w0.to_le_bytes());
    h[0x48..0x4C].copy_from_slice(&0xFu32.to_le_bytes());
    h
}

/// A pixel shader exercising control flow and the integer, conversion
/// and predicate instructions (used by the translation tests): an
/// if/else through SSY/SYNC on the red input, a loop counting to
/// trunc(8 * red) for alpha, integer bit twiddling of blue, and a
/// discard where green is small.
pub fn control_flow_sample() -> Vec<u8> {
    program(
        &pixel_header(),
        &[
            ipa(0, 0x80),
            ipa(1, 0x84),
            ipa(2, 0x88),
            mov32i(3, 0x3F80_0000),
            fsetp_imm(0, fcmp::GT, 0, 0.5),
            ssy(5, 11),
            guard(bra(6, 9), 0, true),
            fmul32i(1, 1, 0.25),
            sync(),
            fadd_imm(2, 2, 0.5),
            sync(),
            fmul32i(7, 0, 8.0),
            f2i(6, 7, true, 3),
            mov32i(5, 0),
            isetp(1, 7, icmp::LT, true, 5, 6, 0, 7),
            guard(bra(15, 18), 1, true),
            iadd32i(5, 5, 1),
            bra(17, 14),
            i2f(8, 5, true),
            fmul32i(3, 8, 0.125),
            fmul32i(7, 2, 255.0),
            f2i(9, 7, false, 0),
            lop32i(9, 0, 9, 0xF0),
            xmad_imm(10, 9, 3, 9),
            shr_imm(10, 10, 2, false),
            i2f(11, 10, false),
            fmul32i(2, 11, 1.0 / 255.0),
            fsetp_imm(2, fcmp::LT, 1, 0.125),
            guard(kil(), 2, false),
            exit(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::super::decode::{decode, IpaMode, Mods, Op, Operand};
    use super::*;

    #[test]
    fn round_trips_through_the_decoder() {
        let cases: [(u64, Op); 12] = [
            (
                mov(1, 2),
                Op::Mov {
                    dst: 1,
                    src: Operand::Reg(2),
                },
            ),
            (
                mov32i(3, 0x3F80_0000),
                Op::Mov {
                    dst: 3,
                    src: Operand::Imm(0x3F80_0000),
                },
            ),
            (
                mov_cbuf(4, 2, 0x10),
                Op::Mov {
                    dst: 4,
                    src: Operand::Cbuf {
                        index: 2,
                        offset: 0x10,
                    },
                },
            ),
            (
                fadd(0, 1, 2),
                Op::Fadd {
                    dst: 0,
                    a: 1,
                    a_mods: Mods::default(),
                    b: Operand::Reg(2),
                    b_mods: Mods::default(),
                    sat: false,
                },
            ),
            (
                fadd_imm(0, 1, -0.5),
                Op::Fadd {
                    dst: 0,
                    a: 1,
                    a_mods: Mods::default(),
                    b: Operand::Imm((-0.5f32).to_bits()),
                    b_mods: Mods::default(),
                    sat: false,
                },
            ),
            (
                fadd32i(5, 6, 0.1),
                Op::Fadd {
                    dst: 5,
                    a: 6,
                    a_mods: Mods::default(),
                    b: Operand::Imm(0.1f32.to_bits()),
                    b_mods: Mods::default(),
                    sat: false,
                },
            ),
            (
                fmul(7, 8, 9),
                Op::Fmul {
                    dst: 7,
                    a: 8,
                    b: Operand::Reg(9),
                    neg_b: false,
                    sat: false,
                },
            ),
            (
                fmul32i(7, 8, 2.0),
                Op::Fmul {
                    dst: 7,
                    a: 8,
                    b: Operand::Imm(2.0f32.to_bits()),
                    neg_b: false,
                    sat: false,
                },
            ),
            (
                ffma(1, 2, 3, 4),
                Op::Ffma {
                    dst: 1,
                    a: 2,
                    b: Operand::Reg(3),
                    neg_b: false,
                    c: Operand::Reg(4),
                    neg_c: false,
                    sat: false,
                },
            ),
            (
                ald(0, 0x80, 4),
                Op::Ald {
                    dst: 0,
                    base: 255,
                    addr: 0x80,
                    count: 4,
                },
            ),
            (
                ast(4, 0x70, 4),
                Op::Ast {
                    src: 4,
                    base: 255,
                    addr: 0x70,
                    count: 4,
                },
            ),
            (
                ipa(2, 0x84),
                Op::Ipa {
                    dst: 2,
                    addr: 0x84,
                    mode: IpaMode::Pass,
                    sat: false,
                },
            ),
        ];
        for (bits, inst) in cases {
            assert_eq!(decode(bits, 0).map(|i| i.op), Ok(inst), "{bits:#018x}");
        }
        assert_eq!(
            decode(fmul_cbuf(1, 2, 3, 0x20), 0).map(|i| i.op),
            Ok(Op::Fmul {
                dst: 1,
                a: 2,
                b: Operand::Cbuf {
                    index: 3,
                    offset: 0x20
                },
                neg_b: false,
                sat: false
            })
        );
        assert_eq!(
            decode(fadd_cbuf(1, 2, 0, 4), 0).map(|i| i.op),
            Ok(Op::Fadd {
                dst: 1,
                a: 2,
                a_mods: Mods::default(),
                b: Operand::Cbuf {
                    index: 0,
                    offset: 4
                },
                b_mods: Mods::default(),
                sat: false
            })
        );
        assert_eq!(decode(exit(), 0).map(|i| i.op), Ok(Op::Exit));
        assert_eq!(decode(nop(), 0).map(|i| i.op), Ok(Op::Nop));
        assert_eq!(
            decode(ffma_cbuf(1, 2, 3, 8, 4), 0).map(|i| i.op),
            Ok(Op::Ffma {
                dst: 1,
                a: 2,
                b: Operand::Cbuf {
                    index: 3,
                    offset: 8
                },
                neg_b: false,
                c: Operand::Reg(4),
                neg_c: false,
                sat: false
            })
        );
    }

    #[test]
    fn new_instructions_round_trip() {
        use super::super::decode::{Bop, LopOp, MufuOp, Pred, Round, XmadMode, PT};
        let p = |index: u8, neg: bool| Pred { index, neg };
        let d = |bits: u64| decode(bits, 0).map(|i| i.op).unwrap();
        assert_eq!(d(sync()), Op::Sync);
        assert_eq!(d(kil()), Op::Kil);
        assert_eq!(decode(bra(4, 1), 4).unwrap().op, Op::Bra { target: 1 });
        assert_eq!(decode(bra(1, 7), 1).unwrap().op, Op::Bra { target: 7 });
        assert_eq!(decode(ssy(2, 3), 2).unwrap().op, Op::Ssy { target: 3 });
        let g = decode(guard(exit(), 3, true), 0).unwrap();
        assert_eq!((g.pred, g.op), (p(3, true), Op::Exit));
        assert_eq!(
            d(fsetp(2, fcmp::LE, 4, 5)),
            Op::Fsetp {
                dst: 2,
                dst2: 7,
                cmp: fcmp::LE,
                a: 4,
                a_mods: Mods::default(),
                b: Operand::Reg(5),
                b_mods: Mods::default(),
                bop: Bop::And,
                c: PT
            }
        );
        assert_eq!(
            d(fsetp_imm(1, fcmp::GT, 0, -0.5)),
            Op::Fsetp {
                dst: 1,
                dst2: 7,
                cmp: fcmp::GT,
                a: 0,
                a_mods: Mods::default(),
                b: Operand::Imm((-0.5f32).to_bits()),
                b_mods: Mods::default(),
                bop: Bop::And,
                c: PT
            }
        );
        assert_eq!(
            d(isetp(1, 2, icmp::GE, true, 3, 4, 1, 5)),
            Op::Isetp {
                dst: 1,
                dst2: 2,
                cmp: icmp::GE,
                signed: true,
                a: 3,
                b: Operand::Reg(4),
                bop: Bop::Or,
                c: p(5, false)
            }
        );
        assert_eq!(
            d(isetp_imm(0, icmp::NE, false, 3, -7)),
            Op::Isetp {
                dst: 0,
                dst2: 7,
                cmp: icmp::NE,
                signed: false,
                a: 3,
                b: Operand::Imm(-7i32 as u32),
                bop: Bop::And,
                c: PT
            }
        );
        assert_eq!(
            d(psetp(3, 2, 1, 2, true)),
            Op::Psetp {
                dst: 3,
                dst2: 7,
                op1: Bop::Xor,
                op2: Bop::And,
                a: p(1, false),
                b: p(2, true),
                c: PT
            }
        );
        assert_eq!(
            d(fset(1, fcmp::EQ, 2, 3, true)),
            Op::Fset {
                dst: 1,
                cmp: fcmp::EQ,
                a: 2,
                a_mods: Mods::default(),
                b: Operand::Reg(3),
                b_mods: Mods::default(),
                bop: Bop::And,
                c: PT,
                bf: true
            }
        );
        assert_eq!(
            d(iset(1, icmp::LT, false, 2, 3, false)),
            Op::Iset {
                dst: 1,
                cmp: icmp::LT,
                signed: false,
                a: 2,
                b: Operand::Reg(3),
                bop: Bop::And,
                c: PT,
                bf: false
            }
        );
        assert_eq!(
            d(iadd(1, 2, 3)),
            Op::Iadd {
                dst: 1,
                a: 2,
                neg_a: false,
                b: Operand::Reg(3),
                neg_b: false
            }
        );
        assert_eq!(
            d(isub(1, 2, 3)),
            Op::Iadd {
                dst: 1,
                a: 2,
                neg_a: false,
                b: Operand::Reg(3),
                neg_b: true
            }
        );
        assert_eq!(
            d(iadd_imm(1, 2, -3)),
            Op::Iadd {
                dst: 1,
                a: 2,
                neg_a: false,
                b: Operand::Imm(-3i32 as u32),
                neg_b: false
            }
        );
        assert_eq!(
            d(iadd32i(1, 2, 0xDEAD_BEEF)),
            Op::Iadd {
                dst: 1,
                a: 2,
                neg_a: false,
                b: Operand::Imm(0xDEAD_BEEF),
                neg_b: false
            }
        );
        assert_eq!(
            d(iscadd(1, 2, 3, 4)),
            Op::Iscadd {
                dst: 1,
                a: 2,
                neg_a: false,
                b: Operand::Reg(3),
                neg_b: false,
                shift: 4
            }
        );
        assert_eq!(
            d(shl(1, 2, 3)),
            Op::Shl {
                dst: 1,
                a: 2,
                b: Operand::Reg(3)
            }
        );
        assert_eq!(
            d(shl_imm(1, 2, 5)),
            Op::Shl {
                dst: 1,
                a: 2,
                b: Operand::Imm(5)
            }
        );
        assert_eq!(
            d(shr(1, 2, 3, true)),
            Op::Shr {
                dst: 1,
                a: 2,
                b: Operand::Reg(3),
                signed: true
            }
        );
        assert_eq!(
            d(shr_imm(1, 2, 9, false)),
            Op::Shr {
                dst: 1,
                a: 2,
                b: Operand::Imm(9),
                signed: false
            }
        );
        assert_eq!(
            d(lop(1, 2, 3, true, 4, false)),
            Op::Lop {
                dst: 1,
                op: LopOp::Xor,
                a: 3,
                inv_a: true,
                b: Operand::Reg(4),
                inv_b: false
            }
        );
        assert_eq!(
            d(lop32i(1, 0, 2, 0xFF00)),
            Op::Lop {
                dst: 1,
                op: LopOp::And,
                a: 2,
                inv_a: false,
                b: Operand::Imm(0xFF00),
                inv_b: false
            }
        );
        assert_eq!(
            d(imnmx(1, 2, 3, true, false)),
            Op::Imnmx {
                dst: 1,
                a: 2,
                b: Operand::Reg(3),
                signed: true,
                min: p(7, true)
            }
        );
        assert_eq!(
            d(fmnmx(1, 2, 3, true)),
            Op::Fmnmx {
                dst: 1,
                a: 2,
                a_mods: Mods::default(),
                b: Operand::Reg(3),
                b_mods: Mods::default(),
                min: PT
            }
        );
        assert_eq!(
            d(sel(1, 2, 3, 4, true)),
            Op::Sel {
                dst: 1,
                a: 2,
                b: Operand::Reg(3),
                cond: p(4, true)
            }
        );
        assert_eq!(
            d(xmad(1, 2, true, 3, true, 4, 4, true, true)),
            Op::Xmad {
                dst: 1,
                a: 2,
                a_hi: true,
                a_signed: false,
                b: Operand::Reg(3),
                b_hi: true,
                b_signed: false,
                c: Operand::Reg(4),
                mode: XmadMode::Cbcc,
                psl: true,
                mrg: true
            }
        );
        assert_eq!(
            d(xmad_imm(1, 2, 0x1234, 4)),
            Op::Xmad {
                dst: 1,
                a: 2,
                a_hi: false,
                a_signed: false,
                b: Operand::Imm(0x1234),
                b_hi: false,
                b_signed: false,
                c: Operand::Reg(4),
                mode: XmadMode::C,
                psl: false,
                mrg: false
            }
        );
        assert_eq!(
            d(mufu(1, mufu::RSQ, 2)),
            Op::Mufu {
                dst: 1,
                func: MufuOp::Rsq,
                a: 2,
                mods: Mods::default(),
                sat: false
            }
        );
        assert_eq!(
            d(rro(1, 2)),
            Op::Rro {
                dst: 1,
                src: Operand::Reg(2),
                mods: Mods::default()
            }
        );
        assert_eq!(
            d(f2i(1, 2, true, 1)),
            Op::F2i {
                dst: 1,
                src: Operand::Reg(2),
                mods: Mods::default(),
                signed: true,
                round: Round::Floor
            }
        );
        assert_eq!(
            d(i2f(1, 2, false)),
            Op::I2f {
                dst: 1,
                src: Operand::Reg(2),
                signed: false,
                neg: false,
                abs: false
            }
        );
        assert_eq!(
            d(frnd(1, 2, 2)),
            Op::F2f {
                dst: 1,
                src: Operand::Reg(2),
                mods: Mods::default(),
                round: Round::Ceil,
                sat: false
            }
        );
    }
}
