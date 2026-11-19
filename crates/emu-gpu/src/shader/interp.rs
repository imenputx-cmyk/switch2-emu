//! Runs decoded shaders on the CPU: the reference that translated
//! (SPIR-V) shaders are compared against, and the software renderer's
//! shader core.

use super::decode::{Inst, IpaMode, Mods, Operand, RZ};

/// What a shader reads and writes besides its registers.
pub trait ShaderIo {
    /// Input attribute word (vertex attributes for a VS).
    fn load_attr(&self, addr: u32) -> u32;
    /// Output attribute word (VS position and varyings).
    fn store_attr(&mut self, addr: u32, value: u32);
    /// Interpolated input (PS).
    fn interpolate(&self, addr: u32, mode: IpaMode) -> f32;
    fn cbuf(&self, index: u8, offset: u32) -> u32;
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

fn operand(regs: &Regs, io: &dyn ShaderIo, o: Operand) -> u32 {
    match o {
        Operand::Reg(r) => regs.get(r),
        Operand::Imm(v) => v,
        Operand::Cbuf { index, offset } => io.cbuf(index, offset),
    }
}

fn apply(v: f32, m: Mods) -> f32 {
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

/// Runs a straight-line program to its EXIT; returns the registers.
pub fn run(program: &[Inst], io: &mut dyn ShaderIo) -> Regs {
    let mut regs = Regs([0; 256]);
    for &inst in program {
        match inst {
            Inst::Mov { dst, src } => {
                let v = operand(&regs, io, src);
                regs.set(dst, v);
            }
            Inst::Fadd {
                dst,
                a,
                a_mods,
                b,
                b_mods,
                sat: s,
            } => {
                let x = apply(regs.f(a), a_mods);
                let y = apply(f32::from_bits(operand(&regs, io, b)), b_mods);
                regs.set(dst, sat(x + y, s).to_bits());
            }
            Inst::Fmul {
                dst,
                a,
                b,
                neg_b,
                sat: s,
            } => {
                let y = f32::from_bits(operand(&regs, io, b));
                let y = if neg_b { -y } else { y };
                regs.set(dst, sat(regs.f(a) * y, s).to_bits());
            }
            Inst::Ffma {
                dst,
                a,
                b,
                neg_b,
                c,
                neg_c,
                sat: s,
            } => {
                let y = f32::from_bits(operand(&regs, io, b));
                let z = f32::from_bits(operand(&regs, io, c));
                let (y, z) = (if neg_b { -y } else { y }, if neg_c { -z } else { z });
                regs.set(dst, sat(regs.f(a).mul_add(y, z), s).to_bits());
            }
            Inst::Ald {
                dst,
                base,
                addr,
                count,
            } => {
                let addr = addr + regs.get(base);
                for i in 0..count {
                    let v = io.load_attr(addr + 4 * i);
                    regs.set(dst.wrapping_add(i as u8), v);
                }
            }
            Inst::Ast {
                src,
                base,
                addr,
                count,
            } => {
                let addr = addr + regs.get(base);
                for i in 0..count {
                    let v = regs.get(src.wrapping_add(i as u8));
                    io.store_attr(addr + 4 * i, v);
                }
            }
            Inst::Ipa {
                dst,
                addr,
                mode,
                sat: s,
            } => {
                regs.set(dst, sat(io.interpolate(addr, mode), s).to_bits());
            }
            Inst::Nop => {}
            Inst::Exit => break,
        }
    }
    regs
}
