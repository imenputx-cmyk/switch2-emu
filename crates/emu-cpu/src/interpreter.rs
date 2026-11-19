//! Reference interpreter: executes one decoded instruction at a time.
//!
//! Semantics follow the pseudocode in the ARM Architecture Reference
//! Manual. Speed is not a goal here; this is the ground truth that a later
//! JIT will be tested against.

use crate::decoder::{decode, BranchRegOp, Instr, MoveWideOp};
use crate::{Cpu, Nzcv};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecError {
    /// The instruction is unallocated or not implemented yet.
    Unimplemented(u32),
}

/// `AddWithCarry()` from the ARM pseudocode, for 32- or 64-bit operands.
fn add_with_carry(x: u64, y: u64, carry_in: bool, sf: bool) -> (u64, Nzcv) {
    let c = u64::from(carry_in);
    if sf {
        let unsigned_sum = u128::from(x) + u128::from(y) + u128::from(c);
        let signed_sum = i128::from(x as i64) + i128::from(y as i64) + i128::from(c);
        let result = unsigned_sum as u64;
        let flags = Nzcv {
            n: (result as i64) < 0,
            z: result == 0,
            c: u128::from(result) != unsigned_sum,
            v: i128::from(result as i64) != signed_sum,
        };
        (result, flags)
    } else {
        let (x, y) = (x as u32, y as u32);
        let unsigned_sum = u64::from(x) + u64::from(y) + c;
        let signed_sum = i64::from(x as i32) + i64::from(y as i32) + c as i64;
        let result = unsigned_sum as u32;
        let flags = Nzcv {
            n: (result as i32) < 0,
            z: result == 0,
            c: u64::from(result) != unsigned_sum,
            v: i64::from(result as i32) != signed_sum,
        };
        (u64::from(result), flags)
    }
}

fn truncate(value: u64, sf: bool) -> u64 {
    if sf {
        value
    } else {
        value & 0xFFFF_FFFF
    }
}

impl Cpu {
    /// Decodes and executes one instruction located at the current PC.
    /// On success the PC points to the next instruction to run.
    pub fn execute(&mut self, insn: u32) -> Result<(), ExecError> {
        let pc = self.pc();
        let mut next_pc = pc.wrapping_add(4);

        match decode(insn) {
            Instr::Nop => {}

            Instr::AddSubImm {
                sf,
                sub,
                set_flags,
                imm,
                rn,
                rd,
            } => {
                let operand1 = self.read_sp(rn);
                let (operand2, carry_in) = if sub { (!imm, true) } else { (imm, false) };
                let (result, flags) = add_with_carry(operand1, operand2, carry_in, sf);
                if set_flags {
                    self.nzcv = flags;
                    self.write_xzr(rd, result);
                } else {
                    self.write_sp(rd, result);
                }
            }

            Instr::MoveWide {
                sf,
                op,
                shift,
                imm16,
                rd,
            } => {
                let imm = u64::from(imm16) << shift;
                let result = match op {
                    MoveWideOp::Movz => imm,
                    MoveWideOp::Movn => !imm,
                    MoveWideOp::Movk => (self.read_xzr(rd) & !(0xFFFF_u64 << shift)) | imm,
                };
                self.write_xzr(rd, truncate(result, sf));
            }

            Instr::Branch { link, offset } => {
                if link {
                    self.write_xzr(30, pc.wrapping_add(4));
                }
                next_pc = pc.wrapping_add_signed(offset);
            }

            Instr::BranchReg { op, rn } => {
                // Read the target before writing X30, so `blr x30` works.
                let target = self.read_xzr(rn);
                if op == BranchRegOp::Blr {
                    self.write_xzr(30, pc.wrapping_add(4));
                }
                next_pc = target;
            }

            Instr::CompareBranch {
                sf,
                nonzero,
                offset,
                rt,
            } => {
                let is_zero = truncate(self.read_xzr(rt), sf) == 0;
                if is_zero != nonzero {
                    next_pc = pc.wrapping_add_signed(offset);
                }
            }

            Instr::Unknown(raw) => return Err(ExecError::Unimplemented(raw)),
        }

        self.set_pc(next_pc);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodings from `llvm-mc -triple=aarch64 -show-encoding`.
    const ADD_X0_X1_1: u32 = 0x9100_0420;
    const ADDS_W3_W4_4095: u32 = 0x313F_FC83;
    const SUB_SP_SP_32: u32 = 0xD100_83FF;
    const CMP_X5_7: u32 = 0xF100_1CBF;
    const MOVZ_X1_BEEF_LSL48: u32 = 0xD2F7_DDE1;
    const MOVN_W2_0: u32 = 0x1280_0002;
    const MOVK_X3_FFFF_LSL16: u32 = 0xF2BF_FFE3;
    const B_MINUS_4: u32 = 0x17FF_FFFF;
    const BL_256: u32 = 0x9400_0040;
    const RET: u32 = 0xD65F_03C0;
    const BLR_X9: u32 = 0xD63F_0120;
    const CBZ_X0_12: u32 = 0xB400_0060;
    const CBNZ_W1_MINUS_8: u32 = 0x35FF_FFC1;

    fn cpu_at(pc: u64) -> Cpu {
        let mut cpu = Cpu::new();
        cpu.set_pc(pc);
        cpu
    }

    #[test]
    fn add_immediate_advances_pc() {
        let mut cpu = cpu_at(0x1000);
        cpu.write_xzr(1, 41);
        cpu.execute(ADD_X0_X1_1).unwrap();
        assert_eq!(cpu.read_xzr(0), 42);
        assert_eq!(cpu.pc(), 0x1004);
        assert_eq!(cpu.nzcv, Nzcv::default(), "ADD must not touch flags");
    }

    #[test]
    fn adds_32bit_carry_and_zero() {
        // 0xFFFF_F001 + 4095 = 0x1_0000_0000 -> W3 = 0, C = 1, Z = 1.
        let mut cpu = Cpu::new();
        cpu.write_xzr(4, 0xFFFF_F001);
        cpu.execute(ADDS_W3_W4_4095).unwrap();
        assert_eq!(cpu.read_xzr(3), 0);
        assert_eq!(
            cpu.nzcv,
            Nzcv {
                n: false,
                z: true,
                c: true,
                v: false
            }
        );
    }

    #[test]
    fn adds_32bit_signed_overflow() {
        // 0x7FFF_FFFF + 4095 overflows the signed 32-bit range.
        let mut cpu = Cpu::new();
        cpu.write_xzr(4, 0x7FFF_FFFF);
        cpu.execute(ADDS_W3_W4_4095).unwrap();
        assert_eq!(cpu.read_xzr(3), 0x8000_0FFE);
        assert_eq!(
            cpu.nzcv,
            Nzcv {
                n: true,
                z: false,
                c: false,
                v: true
            }
        );
    }

    #[test]
    fn sub_uses_sp() {
        let mut cpu = Cpu::new();
        cpu.write_sp(31, 0x8000);
        cpu.execute(SUB_SP_SP_32).unwrap();
        assert_eq!(cpu.sp(), 0x8000 - 32);
    }

    #[test]
    fn cmp_sets_flags_and_discards_result() {
        let mut cpu = Cpu::new();
        cpu.write_xzr(5, 7);
        cpu.execute(CMP_X5_7).unwrap();
        // Equal: Z = 1, and C = 1 because no borrow occurred.
        assert_eq!(
            cpu.nzcv,
            Nzcv {
                n: false,
                z: true,
                c: true,
                v: false
            }
        );
        assert_eq!(cpu.sp(), 0, "CMP writes XZR, not SP");

        cpu.write_xzr(5, 3);
        cpu.execute(CMP_X5_7).unwrap();
        // 3 - 7: negative, borrow -> C = 0.
        assert_eq!(
            cpu.nzcv,
            Nzcv {
                n: true,
                z: false,
                c: false,
                v: false
            }
        );
    }

    #[test]
    fn move_wide_variants() {
        let mut cpu = Cpu::new();
        cpu.execute(MOVZ_X1_BEEF_LSL48).unwrap();
        assert_eq!(cpu.read_xzr(1), 0xBEEF_0000_0000_0000);

        cpu.write_xzr(2, u64::MAX);
        cpu.execute(MOVN_W2_0).unwrap();
        assert_eq!(cpu.read_xzr(2), 0xFFFF_FFFF, "W writes zero the top half");

        cpu.write_xzr(3, 0x1111_2222_3333_4444);
        cpu.execute(MOVK_X3_FFFF_LSL16).unwrap();
        assert_eq!(cpu.read_xzr(3), 0x1111_2222_FFFF_4444);
    }

    #[test]
    fn branch_and_link_then_return() {
        let mut cpu = cpu_at(0x2000);
        cpu.execute(BL_256).unwrap();
        assert_eq!(cpu.pc(), 0x2100);
        assert_eq!(cpu.read_xzr(30), 0x2004);
        cpu.execute(RET).unwrap();
        assert_eq!(cpu.pc(), 0x2004);

        cpu.execute(B_MINUS_4).unwrap();
        assert_eq!(cpu.pc(), 0x2000);
    }

    #[test]
    fn blr_links_and_jumps() {
        let mut cpu = cpu_at(0x3000);
        cpu.write_xzr(9, 0x4000);
        cpu.execute(BLR_X9).unwrap();
        assert_eq!(cpu.pc(), 0x4000);
        assert_eq!(cpu.read_xzr(30), 0x3004);
    }

    #[test]
    fn compare_and_branch() {
        let mut cpu = cpu_at(0x100);
        cpu.execute(CBZ_X0_12).unwrap();
        assert_eq!(cpu.pc(), 0x10C, "X0 is zero, branch taken");

        cpu.set_pc(0x100);
        cpu.write_xzr(0, 1);
        cpu.execute(CBZ_X0_12).unwrap();
        assert_eq!(cpu.pc(), 0x104, "X0 is non-zero, fall through");

        // CBNZ on W1 only looks at the low 32 bits.
        cpu.set_pc(0x100);
        cpu.write_xzr(1, 0x1_0000_0000);
        cpu.execute(CBNZ_W1_MINUS_8).unwrap();
        assert_eq!(cpu.pc(), 0x104, "W1 is zero, fall through");

        cpu.set_pc(0x100);
        cpu.write_xzr(1, 5);
        cpu.execute(CBNZ_W1_MINUS_8).unwrap();
        assert_eq!(cpu.pc(), 0xF8);
    }

    #[test]
    fn unknown_instruction_is_an_error() {
        let mut cpu = cpu_at(0x100);
        assert_eq!(cpu.execute(0), Err(ExecError::Unimplemented(0)));
        assert_eq!(cpu.pc(), 0x100, "PC must not move on error");
    }

    /// Runs a tiny program: count X0 down from 3 to 0 in a loop.
    ///   0x00: movz x0, #3
    ///   0x04: sub  x0, x0, #1
    ///   0x08: cbnz x0, #-4
    ///   0x0c: ret
    #[test]
    fn countdown_loop() {
        let program: [u32; 4] = [0xD280_0060, 0xD100_0400, 0xB5FF_FFE0, RET];
        let mut cpu = Cpu::new();
        cpu.write_xzr(30, 0xDEAD_0000);
        let mut steps = 0;
        while cpu.pc() != 0xDEAD_0000 {
            let insn = program[(cpu.pc() / 4) as usize];
            cpu.execute(insn).unwrap();
            steps += 1;
            assert!(steps < 100, "loop did not terminate");
        }
        assert_eq!(cpu.read_xzr(0), 0);
        assert_eq!(steps, 1 + 3 * 2 + 1);
    }
}
