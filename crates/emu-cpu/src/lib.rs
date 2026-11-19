//! Guest CPU state for the AArch64 (ARMv8-A) core.
//!
//! - [`decoder`] turns 32-bit instruction words into [`decoder::Instr`].
//! - [`interpreter`] executes them on a [`Cpu`] (see [`Cpu::execute`]).

pub mod decoder;
pub mod interpreter;

/// Condition flags (PSTATE.NZCV).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Nzcv {
    pub n: bool,
    pub z: bool,
    pub c: bool,
    pub v: bool,
}

/// Index used for register 31, which means `XZR` (zero register) in most
/// instructions and `SP` in others. The caller decides which one applies.
pub const REG_31: u8 = 31;

/// Architectural register state of one AArch64 core.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Cpu {
    /// General-purpose registers X0..X30.
    x: [u64; 31],
    /// Stack pointer.
    sp: u64,
    /// Program counter.
    pc: u64,
    /// Condition flags.
    pub nzcv: Nzcv,
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
}
