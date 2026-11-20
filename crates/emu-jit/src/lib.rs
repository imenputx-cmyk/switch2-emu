//! AArch64 JIT backend built on [dynarmic](https://github.com/azahar-emu/dynarmic)
//! (0BSD), the recompiler used by yuzu, Eden and Lichii.
//!
//! [`Jit::run`] has the same contract as the interpreter's `Cpu::run`: it
//! executes guest code until an SVC, a breakpoint, a fault or the step
//! budget, and leaves the guest state in the given [`Cpu`]. Instructions
//! dynarmic does not handle fall back to our own interpreter.

use std::ffi::c_void;

use emu_cpu::{Cpu, Exit, Fault, Memory, Nzcv};

mod ffi {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct State {
        pub x: [u64; 31],
        pub sp: u64,
        pub pc: u64,
        pub v: [u64; 64],
        pub pstate: u32,
        pub fpcr: u32,
        pub fpsr: u32,
        pub pad: u32,
        pub tpidr_el0: u64,
        pub tpidrro_el0: u64,
    }

    #[repr(C)]
    pub struct Callbacks {
        pub ctx: *mut c_void,
        pub read: unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> bool,
        pub write: unsafe extern "C" fn(*mut c_void, u64, *const u8, usize) -> bool,
        pub fetch: unsafe extern "C" fn(*mut c_void, u64, *mut u32) -> bool,
        pub svc: unsafe extern "C" fn(*mut c_void, u32),
        pub exception: unsafe extern "C" fn(*mut c_void, u64, u32),
        pub interpret: unsafe extern "C" fn(*mut c_void, *mut State, u64, usize) -> bool,
        pub add_ticks: unsafe extern "C" fn(*mut c_void, u64),
        pub ticks_remaining: unsafe extern "C" fn(*mut c_void) -> u64,
        pub cntpct: unsafe extern "C" fn(*mut c_void) -> u64,
    }

    #[repr(C)]
    pub struct EmuJit {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn emu_jit_new(cb: *const Callbacks) -> *mut EmuJit;
        pub fn emu_jit_free(jit: *mut EmuJit);
        pub fn emu_jit_run(jit: *mut EmuJit, state: *mut State) -> u32;
        pub fn emu_jit_invalidate(jit: *mut EmuJit, addr: u64, size: u64);
        pub fn emu_jit_clear_cache(jit: *mut EmuJit);
    }
}

/// Placeholder PC for a fault seen inside a memory callback; replaced with
/// the PC dynarmic reports once it has stopped. That PC can be past the
/// faulting instruction (dynarmic only stops at the end of the block).
const PC_UNKNOWN: u64 = u64::MAX;

/// What the callbacks can reach while a run is in progress.
struct Ctx {
    cpu: *mut Cpu,
    mem: *mut Memory,
    exit: Option<Exit>,
    remaining: u64,
}

/// A dynarmic A64 JIT instance for one guest core.
pub struct Jit {
    raw: *mut ffi::EmuJit,
    ctx: Box<Ctx>,
}

impl Default for Jit {
    fn default() -> Self {
        Self::new()
    }
}

impl Jit {
    pub fn new() -> Self {
        let mut ctx = Box::new(Ctx {
            cpu: std::ptr::null_mut(),
            mem: std::ptr::null_mut(),
            exit: None,
            remaining: 0,
        });
        let cb = ffi::Callbacks {
            ctx: (&mut *ctx as *mut Ctx).cast(),
            read: cb_read,
            write: cb_write,
            fetch: cb_fetch,
            svc: cb_svc,
            exception: cb_exception,
            interpret: cb_interpret,
            add_ticks: cb_add_ticks,
            ticks_remaining: cb_ticks_remaining,
            cntpct: cb_cntpct,
        };
        // SAFETY: the shim copies the callback table; `ctx` is boxed, so
        // its address stays valid for the JIT's lifetime.
        let raw = unsafe { ffi::emu_jit_new(&cb) };
        Jit { raw, ctx }
    }

    /// Runs at most `max_steps` guest instructions. Returns `None` when the
    /// budget ran out, like the interpreter's `Cpu::run`.
    pub fn run(&mut self, cpu: &mut Cpu, mem: &mut Memory, max_steps: u64) -> Option<Exit> {
        if max_steps == 0 {
            return None;
        }
        let mut state = to_state(cpu);
        cpu.exclusive = None;
        self.ctx.cpu = cpu;
        self.ctx.mem = mem;
        self.ctx.exit = None;
        self.ctx.remaining = max_steps;
        // SAFETY: `raw` is live; the callbacks only touch `cpu` and `mem`
        // through `ctx` while this call runs, and we hold both borrows.
        unsafe { ffi::emu_jit_run(self.raw, &mut state) };
        self.ctx.cpu = std::ptr::null_mut();
        self.ctx.mem = std::ptr::null_mut();
        from_state(cpu, &state);
        match self.ctx.exit.take() {
            Some(Exit::Fault {
                pc: PC_UNKNOWN,
                fault,
            }) => Some(Exit::Fault {
                pc: state.pc,
                fault,
            }),
            exit => exit,
        }
    }

    /// Drops translated code for `[addr, addr + size)`, e.g. after the
    /// guest's code mappings change.
    pub fn invalidate(&mut self, addr: u64, size: u64) {
        // SAFETY: `raw` is live.
        unsafe { ffi::emu_jit_invalidate(self.raw, addr, size) }
    }

    /// Drops all translated code.
    pub fn clear_cache(&mut self) {
        // SAFETY: `raw` is live.
        unsafe { ffi::emu_jit_clear_cache(self.raw) }
    }
}

impl Drop for Jit {
    fn drop(&mut self) {
        // SAFETY: `raw` came from `emu_jit_new` and is freed once.
        unsafe { ffi::emu_jit_free(self.raw) }
    }
}

fn to_state(cpu: &Cpu) -> ffi::State {
    let mut s = ffi::State {
        x: [0; 31],
        sp: cpu.sp(),
        pc: cpu.pc(),
        v: [0; 64],
        pstate: cpu.nzcv.to_bits() as u32,
        fpcr: cpu.fpcr as u32,
        fpsr: cpu.fpsr as u32,
        pad: 0,
        tpidr_el0: cpu.tpidr_el0,
        tpidrro_el0: cpu.tpidrro_el0,
    };
    for i in 0..31 {
        s.x[i] = cpu.read_xzr(i as u8);
    }
    for i in 0..32 {
        let v = cpu.read_v(i as u8);
        s.v[2 * i] = v as u64;
        s.v[2 * i + 1] = (v >> 64) as u64;
    }
    s
}

fn from_state(cpu: &mut Cpu, s: &ffi::State) {
    for i in 0..31 {
        cpu.write_xzr(i as u8, s.x[i]);
    }
    cpu.write_sp(31, s.sp);
    cpu.set_pc(s.pc);
    for i in 0..32 {
        let v = u128::from(s.v[2 * i]) | (u128::from(s.v[2 * i + 1]) << 64);
        cpu.write_v(i as u8, v);
    }
    cpu.nzcv = Nzcv::from_bits(u64::from(s.pstate));
    cpu.fpcr = u64::from(s.fpcr);
    cpu.fpsr = u64::from(s.fpsr);
    cpu.tpidr_el0 = s.tpidr_el0;
    cpu.tpidrro_el0 = s.tpidrro_el0;
}

/// # Safety
/// `ctx` must be the `Ctx` of a JIT that is inside [`Jit::run`].
unsafe fn ctx<'a>(ctx: *mut c_void) -> &'a mut Ctx {
    &mut *ctx.cast::<Ctx>()
}

fn fail(c: &mut Ctx, fault: Fault) -> bool {
    if c.exit.is_none() {
        c.exit = Some(Exit::Fault {
            pc: PC_UNKNOWN,
            fault,
        });
    }
    true
}

unsafe extern "C" fn cb_read(c: *mut c_void, addr: u64, out: *mut u8, len: usize) -> bool {
    let c = ctx(c);
    let buf = std::slice::from_raw_parts_mut(out, len);
    match (*c.mem).read(addr, buf) {
        Ok(()) => false,
        Err(fault) => fail(c, fault),
    }
}

unsafe extern "C" fn cb_write(c: *mut c_void, addr: u64, data: *const u8, len: usize) -> bool {
    let c = ctx(c);
    let buf = std::slice::from_raw_parts(data, len);
    match (*c.mem).write(addr, buf) {
        Ok(()) => false,
        Err(fault) => fail(c, fault),
    }
}

unsafe extern "C" fn cb_fetch(c: *mut c_void, addr: u64, out: *mut u32) -> bool {
    let c = ctx(c);
    match (*c.mem).fetch_u32(addr) {
        Ok(insn) => {
            *out = insn;
            false
        }
        // Not an error yet: the shim turns this into a no-execute
        // exception only if the instruction is actually reached.
        Err(_) => true,
    }
}

unsafe extern "C" fn cb_svc(c: *mut c_void, imm: u32) {
    let c = ctx(c);
    c.exit.get_or_insert(Exit::Svc(imm as u16));
}

unsafe extern "C" fn cb_exception(c: *mut c_void, pc: u64, kind: u32) {
    let c = ctx(c);
    if c.exit.is_some() {
        return;
    }
    let mem = &*c.mem;
    c.exit = Some(match kind {
        0 => {
            let insn = mem.read_uint(pc, 4).unwrap_or(0) as u32;
            Exit::Brk((insn >> 5) as u16)
        }
        _ => match mem.fetch_u32(pc) {
            Err(fault) => Exit::Fault { pc, fault },
            Ok(insn) => Exit::Unimplemented { pc, insn },
        },
    });
}

unsafe extern "C" fn cb_interpret(
    c: *mut c_void,
    state: *mut ffi::State,
    pc: u64,
    count: usize,
) -> bool {
    let c = ctx(c);
    if c.exit.is_some() {
        // An earlier access already failed; stop before running more.
        return true;
    }
    let cpu = &mut *c.cpu;
    from_state(cpu, &*state);
    cpu.set_pc(pc);
    let exit = cpu.run(&mut *c.mem, count as u64);
    *state = to_state(cpu);
    match exit {
        None => false,
        Some(exit) => {
            c.exit.get_or_insert(exit);
            true
        }
    }
}

unsafe extern "C" fn cb_add_ticks(c: *mut c_void, ticks: u64) {
    let c = ctx(c);
    c.remaining = c.remaining.saturating_sub(ticks);
    let cpu = &mut *c.cpu;
    cpu.counter = cpu.counter.wrapping_add(ticks);
}

unsafe extern "C" fn cb_ticks_remaining(c: *mut c_void) -> u64 {
    ctx(c).remaining
}

unsafe extern "C" fn cb_cntpct(c: *mut c_void) -> u64 {
    (*ctx(c).cpu).counter
}

#[cfg(test)]
mod tests {
    use super::*;
    use emu_cpu::Perm;

    const CODE: u64 = 0x1_0000;
    const DATA: u64 = 0x2_0000;

    fn setup(code: &[u32]) -> (Cpu, Memory) {
        let mut mem = Memory::new();
        mem.map(CODE, 0x1000, Perm::RX).unwrap();
        mem.map(DATA, 0x1000, Perm::RW).unwrap();
        let bytes: Vec<u8> = code.iter().flat_map(|i| i.to_le_bytes()).collect();
        mem.write_privileged(CODE, &bytes).unwrap();
        let mut cpu = Cpu::new();
        cpu.set_pc(CODE);
        (cpu, mem)
    }

    #[test]
    fn loop_then_svc() {
        // x0 = 0; x1 = 10; loop: x0 += x1; x1 -= 1; cbnz x1, loop; str x0,[x2]; svc #0x26
        let code = [
            0xD280_0000, // mov x0, #0
            0xD280_0141, // mov x1, #10
            0x8B01_0000, // add x0, x0, x1
            0xD100_0421, // sub x1, x1, #1
            0xB5FF_FFC1, // cbnz x1, -8
            0xF900_0040, // str x0, [x2]
            0xD400_04C1, // svc #0x26
        ];
        let (mut cpu, mut mem) = setup(&code);
        cpu.write_xzr(2, DATA);
        let mut jit = Jit::new();
        let exit = jit.run(&mut cpu, &mut mem, 1_000);
        assert_eq!(exit, Some(Exit::Svc(0x26)));
        assert_eq!(cpu.read_xzr(0), 55);
        assert_eq!(mem.read_u64(DATA).unwrap(), 55);
        assert_eq!(cpu.pc(), CODE + 7 * 4);
    }

    #[test]
    fn step_budget_and_resume() {
        let code = [0x9100_0400, 0x17FF_FFFF]; // add x0, x0, #1; b -4
        let (mut cpu, mut mem) = setup(&code);
        let mut jit = Jit::new();
        assert_eq!(jit.run(&mut cpu, &mut mem, 100), None);
        let first = cpu.read_xzr(0);
        assert!(first > 0 && first <= 100, "ran {first} iterations");
        assert_eq!(jit.run(&mut cpu, &mut mem, 100), None);
        assert!(cpu.read_xzr(0) > first);
    }

    #[test]
    fn fault_and_breakpoint() {
        let (mut cpu, mut mem) = setup(&[0xF940_0000]); // ldr x0, [x0]
        cpu.write_xzr(0, 0xDEAD_0000);
        let mut jit = Jit::new();
        match jit.run(&mut cpu, &mut mem, 10) {
            Some(Exit::Fault { fault, .. }) => assert_eq!(fault.addr, 0xDEAD_0000),
            other => panic!("expected a fault, got {other:?}"),
        }

        let (mut cpu, mut mem) = setup(&[0xD420_0000 | (0x4C44 << 5)]); // brk #0x4c44
        let mut jit = Jit::new();
        assert_eq!(jit.run(&mut cpu, &mut mem, 10), Some(Exit::Brk(0x4C44)));
        assert_eq!(cpu.pc(), CODE);
    }

    #[test]
    fn matches_interpreter_on_simd_and_flags() {
        // fmov d0, #1.0; fadd d1, d0, d0; cmp x1, x2; cset x3, eq; svc #1
        let code = [
            0x1E6E_1000,
            0x1E60_2801,
            0xEB02_003F,
            0x9A9F_17E3,
            0xD400_0021,
        ];
        let (mut a, mut mem_a) = setup(&code);
        a.write_xzr(1, 7);
        a.write_xzr(2, 7);
        let (mut b, mut mem_b) = (a.clone(), {
            let (_, m) = setup(&code);
            m
        });
        let mut jit = Jit::new();
        let ja = jit.run(&mut a, &mut mem_a, 100);
        let ib = b.run(&mut mem_b, 100);
        assert_eq!(ja, ib);
        assert_eq!(a.read_v(1), b.read_v(1));
        assert_eq!(a.read_xzr(3), 1);
        assert_eq!(a.nzcv, b.nzcv);
        assert_eq!(a.pc(), b.pc());
    }
}
