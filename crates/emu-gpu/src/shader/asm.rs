//! A tiny Maxwell assembler for the decoded subset, used by tests and to
//! build test programs' shaders (`cargo run -p emu-gpu --example shaders`).
//! Every instruction is unpredicated (PT).

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

#[cfg(test)]
mod tests {
    use super::super::decode::{decode, Inst, IpaMode, Mods, Operand};
    use super::*;

    #[test]
    fn round_trips_through_the_decoder() {
        let cases: [(u64, Inst); 12] = [
            (
                mov(1, 2),
                Inst::Mov {
                    dst: 1,
                    src: Operand::Reg(2),
                },
            ),
            (
                mov32i(3, 0x3F80_0000),
                Inst::Mov {
                    dst: 3,
                    src: Operand::Imm(0x3F80_0000),
                },
            ),
            (
                mov_cbuf(4, 2, 0x10),
                Inst::Mov {
                    dst: 4,
                    src: Operand::Cbuf {
                        index: 2,
                        offset: 0x10,
                    },
                },
            ),
            (
                fadd(0, 1, 2),
                Inst::Fadd {
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
                Inst::Fadd {
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
                Inst::Fadd {
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
                Inst::Fmul {
                    dst: 7,
                    a: 8,
                    b: Operand::Reg(9),
                    neg_b: false,
                    sat: false,
                },
            ),
            (
                fmul32i(7, 8, 2.0),
                Inst::Fmul {
                    dst: 7,
                    a: 8,
                    b: Operand::Imm(2.0f32.to_bits()),
                    neg_b: false,
                    sat: false,
                },
            ),
            (
                ffma(1, 2, 3, 4),
                Inst::Ffma {
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
                Inst::Ald {
                    dst: 0,
                    base: 255,
                    addr: 0x80,
                    count: 4,
                },
            ),
            (
                ast(4, 0x70, 4),
                Inst::Ast {
                    src: 4,
                    base: 255,
                    addr: 0x70,
                    count: 4,
                },
            ),
            (
                ipa(2, 0x84),
                Inst::Ipa {
                    dst: 2,
                    addr: 0x84,
                    mode: IpaMode::Pass,
                    sat: false,
                },
            ),
        ];
        for (bits, inst) in cases {
            assert_eq!(decode(bits), Ok(inst), "{bits:#018x}");
        }
        assert_eq!(
            decode(fmul_cbuf(1, 2, 3, 0x20)),
            Ok(Inst::Fmul {
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
            decode(fadd_cbuf(1, 2, 0, 4)),
            Ok(Inst::Fadd {
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
        assert_eq!(decode(exit()), Ok(Inst::Exit));
        assert_eq!(decode(nop()), Ok(Inst::Nop));
        assert_eq!(
            decode(ffma_cbuf(1, 2, 3, 8, 4)),
            Ok(Inst::Ffma {
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
}
