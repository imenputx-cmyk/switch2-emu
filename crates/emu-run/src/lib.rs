//! Runs freestanding AArch64 test programs (ELF) on the emulated CPU.
//!
//! Programs talk to the host through a tiny test ABI, `svc #0` with the call
//! number in x8 (see `tests/programs/rt.h`). This is not the Switch OS
//! interface; that is a later milestone.

use emu_cpu::{Cpu, Exit};
use emu_loader::load_elf;
use emu_memory::{Memory, Perm};
use std::fmt;

pub const STACK_TOP: u64 = 0x7FFF_0000_0000;
pub const STACK_SIZE: u64 = 0x10_0000;

const SYS_WRITE: u64 = 1;
const SYS_EXIT: u64 = 2;

#[derive(Debug)]
pub struct RunResult {
    pub exit_code: u64,
    pub stdout: Vec<u8>,
    pub instructions: u64,
}

#[derive(Debug)]
pub enum RunError {
    Load(emu_loader::LoadError),
    /// The CPU stopped for a reason the test ABI does not handle.
    Stopped {
        exit: Exit,
        cpu: Box<Cpu>,
    },
    BadSyscall {
        number: u64,
        pc: u64,
    },
    StepLimit,
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Load(e) => write!(f, "load failed: {e}"),
            RunError::Stopped { exit, cpu } => {
                write!(
                    f,
                    "guest stopped: {exit:?} (pc={:#x}, sp={:#x})",
                    cpu.pc(),
                    cpu.sp()
                )
            }
            RunError::BadSyscall { number, pc } => {
                write!(f, "unknown test syscall {number} at {pc:#x}")
            }
            RunError::StepLimit => write!(f, "step limit reached"),
        }
    }
}

impl std::error::Error for RunError {}

pub fn run_elf(elf: &[u8], max_steps: u64) -> Result<RunResult, RunError> {
    let mut mem = Memory::new();
    let image = load_elf(&mut mem, elf).map_err(RunError::Load)?;
    mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE, Perm::RW)
        .expect("stack mapping");

    let mut cpu = Cpu::new();
    cpu.set_pc(image.entry);
    cpu.write_sp(31, STACK_TOP);

    let mut stdout = Vec::new();
    let mut steps = 0u64;
    loop {
        let budget = max_steps.saturating_sub(steps);
        if budget == 0 {
            return Err(RunError::StepLimit);
        }
        let before = cpu.counter;
        let exit = cpu.run(&mut mem, budget);
        steps += cpu.counter - before;
        match exit {
            None => return Err(RunError::StepLimit),
            Some(Exit::Svc(0)) => match cpu.read_xzr(8) {
                SYS_WRITE => {
                    let mut buf = vec![0u8; cpu.read_xzr(1) as usize];
                    if let Err(fault) = mem.read(cpu.read_xzr(0), &mut buf) {
                        let pc = cpu.pc() - 4;
                        return Err(RunError::Stopped {
                            exit: Exit::Fault { pc, fault },
                            cpu: Box::new(cpu),
                        });
                    }
                    stdout.extend_from_slice(&buf);
                    cpu.write_xzr(0, buf.len() as u64);
                }
                SYS_EXIT => {
                    return Ok(RunResult {
                        exit_code: cpu.read_xzr(0),
                        stdout,
                        instructions: steps,
                    })
                }
                number => {
                    return Err(RunError::BadSyscall {
                        number,
                        pc: cpu.pc() - 4,
                    })
                }
            },
            Some(exit) => {
                return Err(RunError::Stopped {
                    exit,
                    cpu: Box::new(cpu),
                })
            }
        }
    }
}
