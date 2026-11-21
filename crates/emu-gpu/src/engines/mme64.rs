//! The Turing/Ampere macro method expander ("MME64"), which Switch 2's NVN2
//! driver uses for nearly all 3D state and draws. Each instruction is 96
//! bits and drives two ALUs and two method outputs.
//!
//! Ported from Lichii/Eden's `video_core/macro_turing.h` (GPL-3.0-or-later),
//! whose semantics come from Mesa's `mme_tu104_sim.c` / `mme_tu104.c`
//! (Copyright © 2022 Collabora Ltd., MIT).
//!
//! Instruction fields (bit offsets into the 96-bit word, whose low 32 bits
//! are the *last* of the three uploaded words):
//!
//! | bits   | field                      |
//! |--------|----------------------------|
//! | 0      | end (one more instruction runs after it) |
//! | 1..5   | predicate mode             |
//! | 5..10  | predicate register         |
//! | 10..15 / 46..51 | ALU op (lane 0 / 1) |
//! | 15..20 / 51..56 | destination        |
//! | 20..25 / 56..61 | source A           |
//! | 25..30 / 61..66 | source B           |
//! | 30..46 / 66..82 | immediate          |
//! | 82..85 / 89..92 | method output select |
//! | 85..89 / 92..96 | emit output select |

/// Where a macro's method writes and register reads go.
pub trait Mme64Target {
    /// Reads a 3D register (method index).
    fn read(&mut self, method: u32) -> u32;
    /// Writes a method. `fifo` is the load FIFO: a DMA read into it
    /// (`MME_DMA_READ_FIFOED`) appends words the macro then loads.
    /// Returns false for a method the macro may not write.
    fn emit(&mut self, method: u32, value: u32, fifo: &mut Vec<u32>) -> bool;
    /// The macro data RAM.
    fn data_ram(&mut self) -> &mut [u32];
}

/// The macro code and start addresses, shared by a process's channels
/// and by its 3D and compute classes (NVN2 uploads macros through one and
/// calls them through the other; Lichii NX2_NOTES row 53).
pub struct Shared {
    pub code: Vec<u32>,
    pub write: usize,
    pub starts: Vec<u32>,
    pub start_pointer: usize,
}

impl Default for Shared {
    fn default() -> Self {
        Shared {
            code: vec![0; 0x8000],
            write: 0,
            starts: vec![0; 0x100],
            start_pointer: 0,
        }
    }
}

impl Shared {
    /// MME_INSTRUCTION_RAM_POINTER (instructions).
    pub fn set_pointer(&mut self, value: u32) {
        self.write = (value as usize * 3).min(self.code.len());
    }

    /// MME_INSTRUCTION_RAM: the next code word.
    pub fn upload(&mut self, value: u32) {
        if let Some(slot) = self.code.get_mut(self.write) {
            *slot = value;
            self.write += 1;
        }
    }

    /// MME_START_ADDRESS_RAM_POINTER.
    pub fn set_start_pointer(&mut self, value: u32) {
        self.start_pointer = value as usize;
    }

    /// MME_START_ADDRESS_RAM: the next macro's start.
    pub fn bind(&mut self, value: u32) {
        if let Some(slot) = self.starts.get_mut(self.start_pointer) {
            *slot = value;
        }
        self.start_pointer += 1;
    }
}

/// Instructions a call may run before it is stopped.
const STEP_LIMIT: u32 = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mme64Error {
    pub pc: u32,
    pub message: &'static str,
    /// For a runaway macro: its last steps and the registers.
    pub detail: String,
}

#[derive(Clone, Copy)]
struct Decoded {
    imm: [u32; 2],
    op: [u32; 2],
    dst: [u32; 2],
    src_a: [u32; 2],
    src_b: [u32; 2],
    out_method: [u32; 2],
    out_emit: [u32; 2],
    pred_reg: u32,
    pred_mode: u32,
    mask_true: u32,
    mask_false: u32,
    end: bool,
    load: [bool; 2],
}

/// Predicate modes: per lane (ALU 0, ALU 1, output 0, output 1), run
/// when the predicate is True, False or Unconditionally.
const PREDICATES: [&[u8; 4]; 16] = [
    b"UUUU", b"TTTT", b"FFFF", b"TTUU", b"FFUU", b"TFUU", b"TUUU", b"FUUU", b"UUTT", b"UUTF",
    b"UUTU", b"UUFT", b"UUFF", b"UUFU", b"UUUT", b"UUUF",
];

fn decode(bits: [u32; 3]) -> Decoded {
    let field = |low: u32, count: u32| -> u32 {
        let word = (low / 32) as usize;
        let mut value = u64::from(bits[word]);
        if word < 2 {
            value |= u64::from(bits[word + 1]) << 32;
        }
        ((value >> (low % 32)) & ((1u64 << count) - 1)) as u32
    };
    let pred_mode = field(1, 4);
    let (mut mask_true, mut mask_false) = (15, 15);
    if pred_mode != 0 {
        mask_true = 0;
        mask_false = 0;
        for (i, &c) in PREDICATES[pred_mode as usize].iter().enumerate() {
            if c != b'T' {
                mask_true |= 1 << i;
            }
            if c != b'F' {
                mask_false |= 1 << i;
            }
        }
    }
    let mut d = Decoded {
        imm: [field(30, 16), field(66, 16)],
        op: [field(10, 5), field(46, 5)],
        dst: [field(15, 5), field(51, 5)],
        src_a: [field(20, 5), field(56, 5)],
        src_b: [field(25, 5), field(61, 5)],
        out_method: [field(82, 3), field(89, 3)],
        out_emit: [field(85, 4), field(92, 4)],
        pred_reg: field(5, 5),
        pred_mode,
        mask_true,
        mask_false,
        end: field(0, 1) != 0,
        load: [false; 2],
    };
    for i in 0..2 {
        let (reg, out) = (28 + i as u32, 3 + i as u32);
        d.load[i] = d.pred_reg == reg
            || d.src_a.contains(&reg)
            || d.src_b.contains(&reg)
            || d.out_method.contains(&out)
            || d.out_emit.contains(&out);
    }
    d
}

/// Steps kept for failure details.
const RING: usize = 32;

#[derive(Debug, Clone, Copy, Default)]
struct Step {
    valid: bool,
    pc: u32,
    ops: [u32; 2],
    xs: [u32; 2],
    ys: [u32; 2],
    alu: [u32; 2],
    next: u32,
    loop_count: u32,
    read: Option<(u32, u32)>,
    consumed: u32,
}

/// A failure's details: the parameters and the last steps.
fn describe(ring: &[Step; RING], step: usize, params: &[u32], consumed: usize) -> String {
    let mut s = format!(
        "  {} parameters, {consumed} consumed: {:x?}{}\n  last steps:\n",
        params.len(),
        &params[..params.len().min(24)],
        if params.len() > 24 { " ..." } else { "" }
    );
    for k in 0..RING {
        let st = &ring[(step + k) % RING];
        if !st.valid {
            continue;
        }
        s += &format!(
            "    pc {:#x}: ops {:?} x {:x?} y {:x?} -> {:x?}, next {:#x}, loop {}, used {}",
            st.pc, st.ops, st.xs, st.ys, st.alu, st.next, st.loop_count, st.consumed
        );
        if let Some((r, v)) = st.read {
            s += &format!(", read reg {r:#x} = {v:#x}");
        }
        s.push('\n');
    }
    s
}

/// Runs the macro at instruction `start` of `code` (three words per
/// instruction) with `params` as its load FIFO.
pub fn execute(
    code: &[u32],
    start: u32,
    params: &mut Vec<u32>,
    target: &mut impl Mme64Target,
) -> Result<(), Mme64Error> {
    let mut regs = [0u32; 24];
    let mut carry = 0u32;
    let (mut method, mut increment) = (0u32, 0u32);
    let mut pc = start;
    let (mut loop_count, mut loop_start, mut loop_end) = (0u32, 0u32, 0u32);
    let mut consumed = 0usize;
    let mut end_next = false;
    // The last steps, for the details of a failure.
    let mut ring = [Step::default(); RING];
    let mut read_note: Option<(u32, u32)> = None;

    for step in 0..STEP_LIMIT {
        let err = |message| Mme64Error {
            pc,
            message,
            detail: String::new(),
        };
        let base = pc as usize * 3;
        if base + 2 >= code.len() {
            return Err(err("instruction outside uploaded RAM"));
        }
        let d = decode([code[base + 2], code[base + 1], code[base]]);

        let mut load = [0u32; 2];
        for (slot, &wanted) in load.iter_mut().zip(&d.load) {
            if wanted {
                let Some(&v) = params.get(consumed) else {
                    return Err(Mme64Error {
                        pc,
                        message: "parameter FIFO underflow",
                        detail: describe(&ring, step as usize, params, consumed),
                    });
                };
                *slot = v;
                consumed += 1;
            }
        }
        let get = |reg: u32, lane: usize| -> Result<u32, &'static str> {
            Ok(match reg {
                0..=23 => regs[reg as usize],
                24 => 0,
                25 => d.imm[lane] as u16 as i16 as i32 as u32,
                26 => d.imm[1 - lane] as u16 as i16 as i32 as u32,
                27 => (d.imm[0] << 16) | d.imm[1],
                28 => load[0],
                29 => load[1],
                _ => return Err("invalid source register"),
            })
        };
        let mask = if d.pred_mode != 0 {
            match get(d.pred_reg, 0) {
                Ok(0) => d.mask_false,
                Ok(_) => d.mask_true,
                Err(e) => return Err(err(e)),
            }
        } else {
            15
        };
        // Both ALUs read their operands before either result is written.
        let mut xs = [0u32; 2];
        let mut ys = [0u32; 2];
        for lane in 0..2 {
            if mask & (1 << lane) != 0 {
                xs[lane] = get(d.src_a[lane], lane).map_err(err)?;
                ys[lane] = get(d.src_b[lane], lane).map_err(err)?;
            }
        }

        let mut next_pc = pc + 1;
        let mut branch_cancels_end = false;
        let mut stopped = false;
        let mut alu = [0u32; 2];
        for lane in 0..2 {
            if mask & (1 << lane) == 0 {
                continue;
            }
            let (x, y) = (xs[lane], ys[lane]);
            let imm = d.imm[lane];
            let condition = |op: u32| match op {
                17 | 25 => (x as i32) < (y as i32),
                18 | 26 => x < y,
                19 | 27 => (x as i32) <= (y as i32),
                20 | 28 => x <= y,
                _ => x == y,
            };
            let mut value = 0u32;
            match d.op[lane] {
                0 => {
                    value = x.wrapping_add(y);
                    carry = u32::from(value < x);
                }
                1 => value = x.wrapping_add(y).wrapping_add(carry),
                2 => {
                    value = x.wrapping_sub(y);
                    carry = u32::from(value > x);
                }
                3 => value = x.wrapping_sub(y).wrapping_sub(carry),
                4 => {
                    let p = i64::from(x as i32).wrapping_mul(i64::from(y as i32)) as u64;
                    value = p as u32;
                    carry = (p >> 32) as u32;
                }
                6 => {
                    let p = u64::from(x) * u64::from(y);
                    value = p as u32;
                    carry = (p >> 32) as u32;
                }
                5 => value = carry,
                // EXTENDED: only the load barrier (x = 0x1000), a no-op here.
                7 => {
                    if x != 0x1000 {
                        return Err(err("unsupported extended operation"));
                    }
                }
                8 => value = x.leading_zeros(),
                9 => value = x << (y & 31),
                10 => value = x >> (y & 31),
                11 => value = ((x as i32) >> (y & 31)) as u32,
                12 => value = x & y,
                13 => value = !(x & y),
                14 => value = x | y,
                15 => value = x ^ y,
                16 => {
                    // MERGE: insert `count` bits of y (from `source`) into x at `dest`.
                    let dest = (imm >> 10) & 63;
                    let count = (imm >> 5) & 31;
                    let source = imm & 31;
                    if dest >= 32 {
                        return Err(err("invalid merge destination"));
                    }
                    let bits = (1u32 << count).wrapping_sub(1);
                    value = (x & !(bits << dest)) | (((y >> source) & bits) << dest);
                }
                17..=21 => value = if condition(d.op[lane]) { u32::MAX } else { 0 },
                22 => {
                    value = target.read(x.wrapping_add(y) & 0xFFFF);
                    read_note = Some((x.wrapping_add(y) & 0xFFFF, value));
                }
                23 => {
                    if loop_count != 0 || imm < 2 {
                        return Err(err("invalid loop"));
                    }
                    // NVN emits zero-count loops around FIFO reads; they
                    // skip the body.
                    if x == 0 {
                        next_pc = pc + imm;
                    } else {
                        loop_count = x - 1;
                        loop_start = pc;
                        loop_end = pc + imm - 1;
                    }
                }
                24 => next_pc = pc + (imm & 0x7FFF),
                25..=29 => {
                    if condition(d.op[lane]) == (imm & 0x8000 != 0) {
                        let offset = ((imm << 19) as i32) >> 19;
                        if offset == -4096 {
                            stopped = true;
                        } else {
                            next_pc = (pc as i32 + offset) as u32;
                            // Bit 14 keeps a taken back edge from ending the call.
                            branch_cancels_end = imm & 0x4000 != 0;
                        }
                    }
                }
                30 => match target.data_ram().get(x as usize) {
                    Some(&v) => value = v,
                    None => return Err(err("data RAM read out of bounds")),
                },
                31 => match target.data_ram().get_mut(x as usize) {
                    Some(slot) => *slot = y,
                    None => return Err(err("data RAM write out of bounds")),
                },
                _ => unreachable!("5-bit opcode"),
            }
            alu[lane] = value;
            match d.dst[lane] {
                r @ 0..=23 => regs[r as usize] = value,
                24 => {}
                _ => return Err(err("invalid destination register")),
            }
        }

        let output = |selector: u32| -> Option<u32> {
            Some(match selector {
                1 => alu[0],
                2 => alu[1],
                3 => load[0],
                4 => load[1],
                5 => d.imm[0],
                6 => d.imm[1],
                8 => d.imm[0] >> 12,
                9 => d.imm[1] >> 12,
                10 => (d.imm[0] << 16) | d.imm[1],
                _ => return None,
            })
        };
        for lane in 0..2 {
            if mask & (4 << lane) == 0 {
                continue;
            }
            if d.out_method[lane] != 0 {
                let Some(address) = output(d.out_method[lane]) else {
                    return Err(err("invalid output selector"));
                };
                method = address & 0xFFF;
                increment = (address >> 12) & 15;
            }
            if d.out_emit[lane] != 0 {
                let Some(value) = output(d.out_emit[lane]) else {
                    return Err(err("invalid output selector"));
                };
                if !target.emit(method, value, params) {
                    return Err(err("unsupported emitted method"));
                }
                method = method.wrapping_add(increment);
            }
        }

        ring[step as usize % RING] = Step {
            valid: true,
            pc,
            ops: d.op,
            xs,
            ys,
            alu,
            next: next_pc,
            loop_count,
            read: read_note.take(),
            consumed: consumed as u32,
        };
        if end_next || stopped {
            return Ok(());
        }
        end_next = d.end && !branch_cancels_end;
        if loop_count != 0 && pc == loop_end {
            loop_count -= 1;
            next_pc = loop_start + 1;
        }
        pc = next_pc;
    }
    Err(Mme64Error {
        pc,
        message: "instruction limit reached (runaway macro)",
        detail: describe(&ring, STEP_LIMIT as usize, params, consumed),
    })
}

/// Builds MME64 instructions (for tests and examples).
pub mod asm {
    /// One ALU lane and its output.
    #[derive(Clone, Copy, Default)]
    pub struct Lane {
        pub op: u32,
        pub dst: u32,
        pub src_a: u32,
        pub src_b: u32,
        pub imm: u32,
        pub out_method: u32,
        pub out_emit: u32,
    }

    pub const ZERO: u32 = 24;
    pub const IMM: u32 = 25;
    pub const LOAD0: u32 = 28;
    /// A lane that does nothing.
    pub const NOP: Lane = Lane {
        op: 0,
        dst: ZERO,
        src_a: ZERO,
        src_b: ZERO,
        imm: 0,
        out_method: 0,
        out_emit: 0,
    };

    /// Encodes one instruction, returned in upload order.
    pub fn inst(end: bool, pred_mode: u32, pred_reg: u32, lanes: [Lane; 2]) -> [u32; 3] {
        let mut v: u128 =
            u128::from(end) | (u128::from(pred_mode) << 1) | (u128::from(pred_reg) << 5);
        let mut put = |low: u32, count: u32, value: u32| {
            v |= (u128::from(value) & ((1 << count) - 1)) << low;
        };
        for (i, l) in lanes.iter().enumerate() {
            let s = 36 * i as u32;
            put(10 + s, 5, l.op);
            put(15 + s, 5, l.dst);
            put(20 + s, 5, l.src_a);
            put(25 + s, 5, l.src_b);
            put(30 + s, 16, l.imm);
            put(82 + 7 * i as u32, 3, l.out_method);
            put(85 + 7 * i as u32, 4, l.out_emit);
        }
        [(v >> 64) as u32, (v >> 32) as u32, v as u32]
    }
}

#[cfg(test)]
mod tests {
    use super::asm::*;
    use super::*;

    #[derive(Default)]
    struct Recorder {
        regs: std::collections::HashMap<u32, u32>,
        writes: Vec<(u32, u32)>,
        dram: Vec<u32>,
    }

    impl Mme64Target for Recorder {
        fn read(&mut self, method: u32) -> u32 {
            self.regs.get(&method).copied().unwrap_or(0)
        }
        fn emit(&mut self, method: u32, value: u32, fifo: &mut Vec<u32>) -> bool {
            self.writes.push((method, value));
            // A fake FIFOED DMA read: append two words.
            if method == 0x158 {
                fifo.extend([7, 9]);
            }
            method < 0xE00
        }
        fn data_ram(&mut self) -> &mut [u32] {
            if self.dram.is_empty() {
                self.dram = vec![0; 0xC00];
            }
            &mut self.dram
        }
    }

    fn program(insts: &[[u32; 3]]) -> Vec<u32> {
        insts.iter().flatten().copied().collect()
    }

    fn run(code: &[u32], params: &[u32]) -> (Result<(), Mme64Error>, Recorder) {
        let mut r = Recorder::default();
        let mut fifo = params.to_vec();
        (execute(code, 0, &mut fifo, &mut r), r)
    }

    /// method = imm (lane 1's immediate), emit = lane 0's ALU result.
    fn emit_alu(method: u32, alu: Lane) -> [u32; 3] {
        inst(
            false,
            0,
            0,
            [
                Lane {
                    out_method: 6,
                    out_emit: 1,
                    ..alu
                },
                Lane { imm: method, ..NOP },
            ],
        )
    }

    #[test]
    fn add_and_emit_with_end_delay_slot() {
        let mut first = emit_alu(
            0x123,
            Lane {
                op: 0,
                dst: 0,
                src_a: IMM,
                src_b: LOAD0,
                imm: 5,
                ..NOP
            },
        );
        first = {
            let mut e = first;
            e[2] |= 1; // end
            e
        };
        // The instruction after END still runs.
        let second = emit_alu(
            0x124,
            Lane {
                op: 14,
                dst: 1,
                src_a: 0,
                src_b: ZERO,
                ..NOP
            },
        );
        let never = emit_alu(0x125, NOP);
        let (result, r) = run(&program(&[first, second, never]), &[10]);
        assert_eq!(result, Ok(()));
        assert_eq!(r.writes, vec![(0x123, 15), (0x124, 15)]);
    }

    #[test]
    fn loop_emits_parameters_with_increment() {
        // r0 = load (count); loop r0 times over one instruction that emits
        // the next parameter to method 0x200 + i (increment 1).
        let set_method = inst(
            false,
            0,
            0,
            [
                Lane {
                    op: 0,
                    dst: 0,
                    src_a: LOAD0,
                    src_b: ZERO,
                    ..NOP
                },
                Lane {
                    imm: 0x1200,
                    out_method: 6,
                    ..NOP
                },
            ],
        );
        let lp = inst(
            false,
            0,
            0,
            [
                Lane {
                    op: 23,
                    src_a: 0,
                    imm: 2,
                    ..NOP
                },
                NOP,
            ],
        );
        let body = inst(
            false,
            0,
            0,
            [
                Lane {
                    out_emit: 3,
                    src_a: LOAD0,
                    ..NOP
                },
                NOP,
            ],
        );
        let mut end = inst(false, 0, 0, [NOP, NOP]);
        end[2] |= 1;
        let tail = inst(false, 0, 0, [NOP, NOP]);
        let code = program(&[set_method, lp, body, end, tail]);
        let (result, r) = run(&code, &[3, 0xA, 0xB, 0xC]);
        assert_eq!(result, Ok(()));
        assert_eq!(r.writes, vec![(0x200, 0xA), (0x201, 0xB), (0x202, 0xC)]);
    }

    #[test]
    fn data_ram_register_reads_and_fifo_dma() {
        let mut r = Recorder::default();
        r.regs.insert(0x40, 0x99);
        // r0 = state(0x40); dwrite(r1=5 -> addr, r0); r2 = dread(5);
        // emit r2 to 0x300; emit to 0x158 (fake DMA into the FIFO);
        // emit load0 (+ load1 next) to 0x301.
        let i0 = inst(
            false,
            0,
            0,
            [
                Lane {
                    op: 22,
                    dst: 0,
                    src_a: IMM,
                    src_b: ZERO,
                    imm: 0x40,
                    ..NOP
                },
                Lane {
                    op: 0,
                    dst: 1,
                    src_a: IMM,
                    src_b: ZERO,
                    imm: 5,
                    ..NOP
                },
            ],
        );
        let i1 = inst(
            false,
            0,
            0,
            [
                Lane {
                    op: 31,
                    dst: ZERO,
                    src_a: 1,
                    src_b: 0,
                    ..NOP
                },
                NOP,
            ],
        );
        let i2 = emit_alu(
            0x300,
            Lane {
                op: 30,
                dst: 2,
                src_a: 1,
                src_b: ZERO,
                ..NOP
            },
        );
        let i3 = inst(
            false,
            0,
            0,
            [
                Lane {
                    imm: 0x158,
                    out_method: 5,
                    out_emit: 5,
                    ..NOP
                },
                NOP,
            ],
        );
        let mut i4 = inst(
            false,
            0,
            0,
            [
                Lane {
                    imm: 0x301,
                    out_method: 5,
                    out_emit: 3,
                    ..NOP
                },
                NOP,
            ],
        );
        i4[2] |= 1;
        let i5 = inst(false, 0, 0, [NOP, NOP]);
        let code = program(&[i0, i1, i2, i3, i4, i5]);
        let mut fifo = Vec::new();
        assert_eq!(execute(&code, 0, &mut fifo, &mut r), Ok(()));
        assert_eq!(r.dram[5], 0x99);
        assert_eq!(r.writes, vec![(0x300, 0x99), (0x158, 0x158), (0x301, 7)]);
    }

    #[test]
    fn predicated_branch_and_errors() {
        // r0 = load; if r0 == 0 skip the emit (BEQ +2), else emit 1.
        let i0 = inst(
            false,
            0,
            0,
            [
                Lane {
                    op: 0,
                    dst: 0,
                    src_a: LOAD0,
                    src_b: ZERO,
                    ..NOP
                },
                NOP,
            ],
        );
        // BEQ: taken when (x == y) == bit15; offset 2.
        let i1 = inst(
            false,
            0,
            0,
            [
                Lane {
                    op: 29,
                    src_a: 0,
                    src_b: ZERO,
                    imm: 0x8000 | 2,
                    ..NOP
                },
                NOP,
            ],
        );
        let i2 = inst(
            false,
            0,
            0,
            [
                Lane {
                    imm: 0x7,
                    out_method: 5,
                    out_emit: 5,
                    ..NOP
                },
                NOP,
            ],
        );
        let mut i3 = inst(false, 0, 0, [NOP, NOP]);
        i3[2] |= 1;
        let i4 = inst(false, 0, 0, [NOP, NOP]);
        let code = program(&[i0, i1, i2, i3, i4]);
        assert_eq!(run(&code, &[0]).1.writes, vec![]);
        assert_eq!(run(&code, &[4]).1.writes, vec![(7, 7)]);
        // Missing parameter.
        assert_eq!(
            run(&code, &[]).0.unwrap_err().message,
            "parameter FIFO underflow"
        );
    }
}
