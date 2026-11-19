//! The macro method expander (MME) of Maxwell-class GPUs: a tiny
//! processor in the 3D engine that runs uploaded macros. A pushbuffer
//! calls a macro with parameters; the macro computes, reads 3D registers
//! and sends method writes back into the engine. Drivers (NVN, deko3d)
//! use macros for draws, clears and state.
//!
//! Instruction format (32 bits):
//!
//! | bits  | field                                          |
//! |-------|------------------------------------------------|
//! | 2:0   | operation (ALU, add-imm, bitfield x3, read, branch) |
//! | 6:4   | result operation (where the result goes)       |
//! | 7     | exit (after the delay slot)                    |
//! | 10:8  | dst register                                   |
//! | 13:11 | src A                                          |
//! | 16:14 | src B                                          |
//! | 21:17 | ALU operation / bitfield source bit            |
//! | 26:22 | bitfield size                                  |
//! | 31:27 | bitfield destination bit                       |
//! | 31:14 | signed immediate                               |
//!
//! Branches (bit 4 = branch if non-zero, bit 5 = annul the delay slot)
//! and exits execute the following instruction (the delay slot) first.
//! r0 reads as zero; r1 starts with the first parameter. The method
//! address register holds a method (11:0) and an increment (17:12)
//! applied after each send.
//!
//! Sources: envytools' macro simulator and disassembler (demmt/macro.c,
//! envydis/macro.c, MIT) for the encodings, and deko3d's macro sources
//! (zlib) for how drivers use them.

/// Where macro sends and register reads go.
pub trait MacroTarget {
    fn send(&mut self, method: u32, value: u32);
    fn read(&self, method: u32) -> u32;
}

/// Instructions a macro may run before it is stopped (a hung macro
/// would otherwise hang the emulator).
const STEP_LIMIT: usize = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroError {
    /// The code ran past the end of macro memory.
    OutOfBounds(u32),
    /// More parameters were consumed than given.
    MissingParameter(u32),
    StepLimit,
    /// Operation 6 is not a valid instruction.
    InvalidOperation(u32),
}

fn field(v: u32, lo: u32, len: u32) -> u32 {
    (v >> lo) & ((1u64 << len) - 1) as u32
}

/// Runs the macro starting at `start` in `code` with `params`.
pub fn execute(
    code: &[u32],
    start: u32,
    params: &[u32],
    target: &mut impl MacroTarget,
) -> Result<(), MacroError> {
    let mut regs = [0u32; 8];
    let mut carry = false;
    let mut method_addr = 0u32; // method (11:0) | increment (17:12)
    let mut next_param = 0usize;
    regs[1] = params.first().copied().unwrap_or(0);
    next_param += 1;

    let mut pc = start;
    let mut delayed: Option<u32> = None; // branch target after the delay slot
    let mut exiting = false;
    for _ in 0..STEP_LIMIT {
        let inst = *code.get(pc as usize).ok_or(MacroError::OutOfBounds(pc))?;
        let in_delay_slot = delayed.is_some() || exiting;
        let operation = inst & 7;
        let dst = field(inst, 8, 3) as usize;
        let a = regs[field(inst, 11, 3) as usize];
        let b = regs[field(inst, 14, 3) as usize];
        let imm = (inst as i32 >> 14) as u32;
        let (src_bit, size, dst_bit) = (field(inst, 17, 5), field(inst, 22, 5), field(inst, 27, 5));
        let mask = ((1u64 << size) - 1) as u32;

        let mut fetch = || -> Result<u32, MacroError> {
            let p = params
                .get(next_param)
                .copied()
                .ok_or(MacroError::MissingParameter(pc))?;
            next_param += 1;
            Ok(p)
        };

        let mut next_pc = pc + 1;
        if operation == 7 {
            // Branch on src A being zero / non-zero.
            let taken = (a == 0) != (inst & 0x10 != 0);
            if taken && !in_delay_slot {
                let target_pc = pc.wrapping_add(imm);
                if inst & 0x20 != 0 {
                    next_pc = target_pc; // annulled: no delay slot
                } else {
                    delayed = Some(target_pc);
                    pc = next_pc;
                    continue;
                }
            }
        } else {
            let result = match operation {
                0 => {
                    // ALU
                    let (r, c) = match field(inst, 17, 5) {
                        0 => a.overflowing_add(b),
                        1 => {
                            let (r1, c1) = a.overflowing_add(b);
                            let (r2, c2) = r1.overflowing_add(carry as u32);
                            (r2, c1 || c2)
                        }
                        2 => {
                            let (r, borrow) = a.overflowing_sub(b);
                            (r, !borrow)
                        }
                        3 => {
                            let (r1, b1) = a.overflowing_sub(b);
                            let (r2, b2) = r1.overflowing_sub((!carry) as u32);
                            (r2, !(b1 || b2))
                        }
                        8 => (a ^ b, carry),
                        9 => (a | b, carry),
                        10 => (a & b, carry),
                        11 => (a & !b, carry),
                        12 => (!(a & b), carry),
                        _ => return Err(MacroError::InvalidOperation(inst)),
                    };
                    carry = c;
                    r
                }
                1 => a.wrapping_add(imm),
                // Extract-insert: replace `size` bits of A at dst_bit with
                // the bits of B at src_bit.
                2 => (a & !(mask << dst_bit)) | (((b >> src_bit) & mask) << dst_bit),
                // Extract `size` bits of B at bit A, shift left by dst_bit.
                3 => (b.checked_shr(a).unwrap_or(0) & mask) << dst_bit,
                // Extract `size` bits of B at src_bit, shift left by A.
                4 => ((b >> src_bit) & mask).checked_shl(a).unwrap_or(0),
                // Read a 3D register.
                5 => target.read(b.wrapping_add(imm) & 0xFFF),
                _ => return Err(MacroError::InvalidOperation(inst)),
            };
            let set = |r: usize, v: u32, regs: &mut [u32; 8]| {
                if r != 0 {
                    regs[r] = v;
                }
            };
            let send = |v: u32, addr: &mut u32, target: &mut dyn FnMut(u32, u32)| {
                target(*addr & 0xFFF, v);
                let incr = (*addr >> 12) & 0x3F;
                *addr = (*addr & !0xFFF) | ((*addr + incr) & 0xFFF);
            };
            let mut out = |m: u32, v: u32| target.send(m, v);
            match field(inst, 4, 3) {
                0 => set(dst, fetch()?, &mut regs),
                1 => set(dst, result, &mut regs),
                2 => {
                    set(dst, result, &mut regs);
                    method_addr = result;
                }
                3 => {
                    set(dst, fetch()?, &mut regs);
                    send(result, &mut method_addr, &mut out);
                }
                4 => {
                    set(dst, result, &mut regs);
                    send(result, &mut method_addr, &mut out);
                }
                5 => {
                    set(dst, fetch()?, &mut regs);
                    method_addr = result;
                }
                6 => {
                    set(dst, result, &mut regs);
                    method_addr = result;
                    let p = fetch()?;
                    send(p, &mut method_addr, &mut out);
                }
                _ => {
                    set(dst, result, &mut regs);
                    method_addr = result;
                    send((result >> 12) & 0x3F, &mut method_addr, &mut out);
                }
            }
        }

        if in_delay_slot {
            if exiting {
                return Ok(());
            }
            next_pc = delayed.take().unwrap();
        } else if inst & 0x80 != 0 {
            exiting = true;
        }
        pc = next_pc;
    }
    Err(MacroError::StepLimit)
}

#[cfg(test)]
pub mod asm {
    //! A small assembler for tests.

    pub const ADD: u32 = 0;
    pub const SUB: u32 = 2;
    pub const AND: u32 = 10;

    // Result operations.
    pub const FETCH: u32 = 0; // dst = next parameter
    pub const MOVE: u32 = 1;
    pub const MOVE_SET_METHOD: u32 = 2;
    pub const FETCH_SEND: u32 = 3;
    pub const MOVE_SEND: u32 = 4;
    pub const MOVE_SET_METHOD_FETCH_SEND: u32 = 6;

    fn base(op: u32, result: u32, dst: u32, a: u32, b: u32) -> u32 {
        op | result << 4 | dst << 8 | a << 11 | b << 14
    }

    pub fn alu(alu_op: u32, result: u32, dst: u32, a: u32, b: u32) -> u32 {
        base(0, result, dst, a, b) | alu_op << 17
    }

    pub fn addi(result: u32, dst: u32, a: u32, imm: i32) -> u32 {
        base(1, result, dst, a, 0) | (imm as u32) << 14
    }

    pub fn ext_ins(
        result: u32,
        dst: u32,
        a: u32,
        b: u32,
        src_bit: u32,
        size: u32,
        dst_bit: u32,
    ) -> u32 {
        base(2, result, dst, a, b) | src_bit << 17 | size << 22 | dst_bit << 27
    }

    pub fn read(result: u32, dst: u32, b: u32, imm: i32) -> u32 {
        base(5, result, dst, 0, b) | (imm as u32) << 14
    }

    /// Branch if `a` is zero (or non-zero), relative to this instruction.
    pub fn branch(a: u32, nonzero: bool, annul: bool, offset: i32) -> u32 {
        7 | (nonzero as u32) << 4 | (annul as u32) << 5 | a << 11 | (offset as u32) << 14
    }

    pub fn exit(inst: u32) -> u32 {
        inst | 0x80
    }

    pub const NOP: u32 = 0x11;

    /// A method address value: method (word index) and increment.
    pub fn maddr(method_byte_offset: u32, incr: u32) -> i32 {
        ((method_byte_offset / 4) | incr << 12) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::asm::*;
    use super::*;

    #[derive(Default)]
    struct Recorder {
        sent: Vec<(u32, u32)>,
        regs: Vec<u32>,
    }

    impl MacroTarget for Recorder {
        fn send(&mut self, method: u32, value: u32) {
            self.sent.push((method * 4, value));
        }
        fn read(&self, method: u32) -> u32 {
            self.regs.get(method as usize).copied().unwrap_or(0)
        }
    }

    fn run(code: &[u32], params: &[u32]) -> Recorder {
        let mut r = Recorder {
            regs: (0..0x1000).map(|i| i * 3).collect(),
            ..Default::default()
        };
        execute(code, 0, params, &mut r).unwrap();
        r
    }

    #[test]
    fn sends_parameters_with_increment() {
        // addr = 0xD80 incrementing by 1; send r1 and two more params.
        let code = [
            addi(MOVE_SET_METHOD, 0, 0, maddr(0xD80, 1)),
            alu(ADD, MOVE_SEND, 0, 1, 0),  // send r1
            alu(ADD, FETCH_SEND, 2, 0, 0), // r2 = param; send 0
            exit(alu(ADD, MOVE_SEND, 0, 2, 0)),
            NOP,
        ];
        let r = run(&code, &[7, 8]);
        assert_eq!(r.sent, [(0xD80, 7), (0xD84, 0), (0xD88, 8)]);
    }

    #[test]
    fn loop_with_delay_slot_and_read() {
        // r2 = count (param 1); loop: send r3 to 0x100 (non-incrementing),
        // r3 += 5, r2 -= 1 in the delay slot. Then send reg[4].
        let code = [
            addi(MOVE_SET_METHOD, 0, 0, maddr(0x100, 0)),
            alu(ADD, MOVE, 2, 1, 0), // r2 = r1 (count)
            // loop:
            alu(ADD, MOVE_SEND, 0, 3, 0), // send r3
            addi(MOVE, 3, 3, 5),          // r3 += 5
            branch(2, true, false, -2),   // if r2 != 0 goto loop (after delay slot)
            addi(MOVE, 2, 2, -1),         // delay slot: r2 -= 1
            read(MOVE, 4, 0, 4),          // r4 = regs[4] = 12
            exit(alu(ADD, MOVE_SEND, 0, 4, 0)),
            NOP,
        ];
        let r = run(&code, &[3]);
        // r2 = 3: sends at r2 = 3, 2, 1, 0 -> 4 iterations.
        assert_eq!(
            r.sent,
            [
                (0x100, 0),
                (0x100, 5),
                (0x100, 10),
                (0x100, 15),
                (0x100, 12)
            ]
        );
    }

    #[test]
    fn annulled_branch_skips_delay_slot() {
        let code = [
            addi(MOVE_SET_METHOD, 0, 0, maddr(0x200, 0)),
            branch(0, false, true, 3), // r0 == 0: jump to 4, annulled
            addi(MOVE_SEND, 0, 0, 1),  // skipped
            addi(MOVE_SEND, 0, 0, 2),  // skipped
            exit(addi(MOVE_SEND, 0, 0, 3)),
            NOP,
        ];
        assert_eq!(run(&code, &[0]).sent, [(0x200, 3)]);
    }

    #[test]
    fn bitfields_and_carry() {
        let code = [
            addi(MOVE_SET_METHOD, 0, 0, maddr(0x300, 1)),
            // Insert bits 8..16 of r1 at bit 4 of r0 (zero).
            alu(ADD, MOVE, 0, 0, 0),
            ext_ins(MOVE_SEND, 2, 0, 1, 8, 8, 4),
            // 0xFFFFFFFF + 1 carries; adc of 0 + 0 + carry = 1.
            addi(MOVE, 3, 0, -1),
            addi(MOVE, 4, 0, 1),
            alu(ADD, MOVE, 5, 3, 4),
            exit(alu(1, MOVE_SEND, 0, 0, 0)),
            NOP,
        ];
        assert_eq!(run(&code, &[0xAB_CD00]).sent, [(0x300, 0xCD0), (0x304, 1)]);
    }

    #[test]
    fn missing_parameter_is_an_error() {
        let code = [exit(alu(ADD, FETCH, 2, 0, 0)), NOP];
        let mut r = Recorder::default();
        assert_eq!(
            execute(&code, 0, &[1], &mut r),
            Err(MacroError::MissingParameter(0))
        );
        let _ = SUB;
        let _ = AND;
        let _ = MOVE_SET_METHOD_FETCH_SEND;
    }
}
