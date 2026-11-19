//! Guest CPU for the AArch64 (ARMv8-A) core.
//!
//! - [`decoder`] turns 32-bit instruction words into [`decoder::Instr`].
//! - [`interpreter`] executes them on a [`Cpu`] (see [`Cpu::step`]).

pub mod decoder;
pub mod fp;
pub mod interpreter;
pub mod simd;

pub use emu_memory::{Fault, Memory, Perm};
pub use interpreter::Exit;

/// Condition flags (PSTATE.NZCV).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Nzcv {
    pub n: bool,
    pub z: bool,
    pub c: bool,
    pub v: bool,
}

impl Nzcv {
    /// Packs the flags into bits [31:28], the layout of the NZCV register.
    pub fn to_bits(self) -> u64 {
        (u64::from(self.n) << 31)
            | (u64::from(self.z) << 30)
            | (u64::from(self.c) << 29)
            | (u64::from(self.v) << 28)
    }

    pub fn from_bits(value: u64) -> Self {
        Nzcv {
            n: value >> 31 & 1 == 1,
            z: value >> 30 & 1 == 1,
            c: value >> 29 & 1 == 1,
            v: value >> 28 & 1 == 1,
        }
    }

    /// Unpacks the 4-bit `nzcv` field used by CCMP/CCMN.
    pub fn from_nibble(nibble: u8) -> Self {
        Self::from_bits(u64::from(nibble) << 28)
    }
}

/// Index used for register 31, which means `XZR` (zero register) in most
/// instructions and `SP` in others. The caller decides which one applies.
pub const REG_31: u8 = 31;

/// Architectural state of one AArch64 core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cpu {
    /// General-purpose registers X0..X30.
    x: [u64; 31],
    /// Stack pointer.
    sp: u64,
    /// Program counter.
    pc: u64,
    /// SIMD&FP registers V0..V31.
    v: [u128; 32],
    /// Condition flags.
    pub nzcv: Nzcv,
    pub fpcr: u64,
    pub fpsr: u64,
    /// Thread pointer, writable by user code.
    pub tpidr_el0: u64,
    /// Read-only thread pointer, set by the kernel. Horizon (the Switch OS)
    /// points this at the thread-local region.
    pub tpidrro_el0: u64,
    /// Virtual counter (CNTVCT_EL0). Advanced by one per executed step.
    pub counter: u64,
    /// Counter frequency in Hz (CNTFRQ_EL0).
    pub counter_freq: u64,
    /// Address marked by the last load-exclusive, if any.
    pub exclusive: Option<u64>,
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu {
            x: [0; 31],
            sp: 0,
            pc: 0,
            v: [0; 32],
            nzcv: Nzcv::default(),
            fpcr: 0,
            fpsr: 0,
            tpidr_el0: 0,
            tpidrro_el0: 0,
            counter: 0,
            // The original Switch reports 19.2 MHz.
            counter_freq: 19_200_000,
            exclusive: None,
        }
    }
}

impl Cpu {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads a register used as a source operand in a data-processing
    /// instruction. Register 31 reads as zero (`XZR`).
    pub fn read_xzr(&self, reg: u8) -> u64 {
        match reg {
            0..=30 => self.x[reg as usize],
            REG_31 => 0,
            _ => panic!("invalid register index {reg}"),
        }
    }

    /// Writes a register used as a destination in a data-processing
    /// instruction. Writes to register 31 are discarded (`XZR`).
    pub fn write_xzr(&mut self, reg: u8, value: u64) {
        match reg {
            0..=30 => self.x[reg as usize] = value,
            REG_31 => {}
            _ => panic!("invalid register index {reg}"),
        }
    }

    /// Reads a register used as a base address, where register 31 is `SP`.
    pub fn read_sp(&self, reg: u8) -> u64 {
        match reg {
            0..=30 => self.x[reg as usize],
            REG_31 => self.sp,
            _ => panic!("invalid register index {reg}"),
        }
    }

    /// Writes a register used as a base address, where register 31 is `SP`.
    pub fn write_sp(&mut self, reg: u8, value: u64) {
        match reg {
            0..=30 => self.x[reg as usize] = value,
            REG_31 => self.sp = value,
            _ => panic!("invalid register index {reg}"),
        }
    }

    pub fn pc(&self) -> u64 {
        self.pc
    }

    pub fn set_pc(&mut self, value: u64) {
        self.pc = value;
    }

    pub fn sp(&self) -> u64 {
        self.sp
    }

    /// Reads a whole 128-bit SIMD&FP register.
    pub fn read_v(&self, reg: u8) -> u128 {
        self.v[reg as usize]
    }

    pub fn write_v(&mut self, reg: u8, value: u128) {
        self.v[reg as usize] = value;
    }

    /// Writes a scalar into the low `bytes` of a SIMD&FP register and
    /// clears the rest, as every scalar write does on AArch64.
    pub fn write_scalar(&mut self, reg: u8, value: u64, bytes: usize) {
        let mask = if bytes >= 8 {
            u64::MAX
        } else {
            (1u64 << (bytes * 8)) - 1
        };
        self.v[reg as usize] = u128::from(value & mask);
    }

    /// Reads the low 64 bits of a SIMD&FP register.
    pub fn read_d(&self, reg: u8) -> u64 {
        self.v[reg as usize] as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xzr_reads_zero_and_ignores_writes() {
        let mut cpu = Cpu::new();
        cpu.write_xzr(REG_31, 0xdead_beef);
        assert_eq!(cpu.read_xzr(REG_31), 0);
        assert_eq!(cpu.sp(), 0, "XZR write must not touch SP");
    }

    #[test]
    fn general_registers_roundtrip() {
        let mut cpu = Cpu::new();
        for reg in 0..=30u8 {
            cpu.write_xzr(reg, u64::from(reg) + 1000);
        }
        for reg in 0..=30u8 {
            assert_eq!(cpu.read_xzr(reg), u64::from(reg) + 1000);
        }
    }

    #[test]
    fn sp_is_register_31_in_address_context() {
        let mut cpu = Cpu::new();
        cpu.write_sp(REG_31, 0x8000_0000);
        assert_eq!(cpu.read_sp(REG_31), 0x8000_0000);
        assert_eq!(cpu.read_xzr(REG_31), 0, "data context still sees XZR");
    }

    #[test]
    #[should_panic(expected = "invalid register index")]
    fn out_of_range_register_panics() {
        Cpu::new().read_xzr(32);
    }

    #[test]
    fn nzcv_bits_roundtrip() {
        let f = Nzcv {
            n: true,
            z: false,
            c: true,
            v: false,
        };
        assert_eq!(f.to_bits(), 0xA000_0000);
        assert_eq!(Nzcv::from_bits(f.to_bits()), f);
        assert_eq!(
            Nzcv::from_nibble(0b0101),
            Nzcv {
                n: false,
                z: true,
                c: false,
                v: true
            }
        );
    }
}
