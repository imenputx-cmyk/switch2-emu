//! Guest threads, the scheduler, and the kernel's synchronization
//! primitives: mutex arbitration, condition variables (process-wide
//! keys), the address arbiter and WaitSynchronization.
//!
//! Threads run one at a time on the single interpreter, round-robin with
//! a fixed time slice; a thread that blocks gives up the CPU at once.
//! Time is the CPU counter (19.2 MHz ticks). When every thread is blocked
//! the clock jumps to the earliest timeout; with no timeout left the
//! process is deadlocked.
//!
//! The semantics follow the switchbrew SVC pages and the behaviour libnx
//! (ISC) relies on in `mutex.c`, `condvar.c` and `levent.c`; Atmosphère's
//! kernel documentation was used for SignalToAddress's value update.

use crate::handles::{KObject, CURRENT_THREAD};
use crate::memmap::{MemState, PERM_R, PERM_W};
use crate::result::*;
use crate::{Horizon, Stop, PAGE};
use emu_cpu::Cpu;

/// Instructions a thread runs before the next one gets the CPU.
pub const TIME_SLICE: u64 = 20_000;
/// Bit set in a mutex word when other threads wait for it.
pub const HANDLE_WAIT_MASK: u32 = 0x4000_0000;
/// Size of a thread's TLS area.
pub const TLS_SIZE: u64 = 0x200;
pub const MAX_THREADS: usize = 256;

const INVALID_PRIORITY: ResultCode = 1 | (112 << 9);
const INVALID_CORE_ID: ResultCode = 1 | (113 << 9);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaitOn {
    /// SleepThread with a positive duration.
    Sleep,
    /// WaitSynchronization on these handles.
    Sync(Vec<u32>),
    /// ArbitrateLock: waits to be handed the mutex at `addr`.
    Mutex { addr: u64, tag: u32 },
    /// WaitProcessWideKeyAtomic: waits on `key`, then retakes `mutex`.
    CondVar { key: u64, mutex: u64, tag: u32 },
    /// WaitForAddress.
    Address { addr: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreadState {
    /// Created, not started yet.
    Created,
    Runnable,
    Waiting {
        on: WaitOn,
        deadline: Option<u64>,
    },
    Exited,
}

pub struct Thread {
    /// Saved registers (stale for the running thread, whose registers are
    /// in `Horizon::cpu`).
    pub cpu: Cpu,
    pub tid: u64,
    pub priority: u32,
    pub state: ThreadState,
    pub tls: u64,
    /// Order in which the thread started waiting, for FIFO wake-ups.
    wait_seq: u64,
}

impl Horizon {
    /// Thread index of a thread handle (or the current-thread pseudo
    /// handle).
    pub(crate) fn thread_of(&self, handle: u32) -> Option<usize> {
        if handle == CURRENT_THREAD {
            return Some(self.current);
        }
        match self.handles.get(handle) {
            Some(&KObject::Thread { id }) => Some(id),
            _ => None,
        }
    }

    /// Registers of thread `i`, live or saved.
    fn thread_cpu(&mut self, i: usize) -> &mut Cpu {
        if i == self.current {
            &mut self.cpu
        } else {
            &mut self.threads[i].cpu
        }
    }

    fn tick(&self) -> u64 {
        self.cpu.counter
    }

    fn deadline(&self, timeout_ns: i64) -> Option<u64> {
        (timeout_ns >= 0).then(|| self.tick() + (timeout_ns as u64).saturating_mul(192) / 10_000)
    }

    /// Allocates a TLS area, mapping a new page when needed.
    pub(crate) fn alloc_tls(&mut self) -> u64 {
        let base = self.layout.tls_page;
        let mut slot = 0u64;
        loop {
            let addr = base + slot * TLS_SIZE;
            if !self
                .threads
                .iter()
                .any(|t| t.tls == addr && t.state != ThreadState::Exited)
            {
                if addr.is_multiple_of(PAGE) && self.map.is_free(addr, PAGE) {
                    self.map_block(addr, PAGE, MemState::ThreadLocal, PERM_R | PERM_W);
                }
                // A reused area starts zeroed, like a fresh one.
                self.mem
                    .write_privileged(addr, &[0u8; TLS_SIZE as usize])
                    .unwrap();
                return addr;
            }
            slot += 1;
        }
    }

    /// Adds a thread; returns its index.
    pub(crate) fn add_thread(
        &mut self,
        cpu: Cpu,
        priority: u32,
        tls: u64,
        state: ThreadState,
    ) -> usize {
        let tid = self.next_tid;
        self.next_tid += 1;
        self.threads.push(Thread {
            cpu,
            tid,
            priority,
            state,
            tls,
            wait_seq: 0,
        });
        self.threads.len() - 1
    }

    /// Blocks the current thread.
    fn block(&mut self, on: WaitOn, deadline: Option<u64>) {
        self.wait_counter += 1;
        let t = &mut self.threads[self.current];
        t.wait_seq = self.wait_counter;
        t.state = ThreadState::Waiting { on, deadline };
    }

    /// Makes thread `i` runnable with result `x0` (and `x1` if given).
    fn wake(&mut self, i: usize, x0: ResultCode, x1: Option<u64>) {
        self.threads[i].state = ThreadState::Runnable;
        let cpu = self.thread_cpu(i);
        cpu.write_xzr(0, u64::from(x0));
        if let Some(v) = x1 {
            cpu.write_xzr(1, v);
        }
    }

    /// Waiting threads matching `pred`, highest priority first, then in
    /// the order they started waiting.
    fn waiters(&self, pred: impl Fn(&WaitOn) -> bool) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.threads.len())
            .filter(
                |&i| matches!(&self.threads[i].state, ThreadState::Waiting { on, .. } if pred(on)),
            )
            .collect();
        v.sort_by_key(|&i| (self.threads[i].priority, self.threads[i].wait_seq));
        v
    }

    fn read32(&self, addr: u64) -> Option<u32> {
        self.mem.read_uint(addr, 4).ok().map(|v| v as u32)
    }

    fn write32(&mut self, addr: u64, v: u32) -> bool {
        self.mem.write_uint(addr, 4, u64::from(v)).is_ok()
    }

    // ---- SVCs ----

    /// CreateThread(entry X1, arg X2, stack top X3, priority W4, core W5).
    pub(crate) fn svc_create_thread(&mut self) {
        let (entry, arg, stack, prio, core) = (
            self.x(1),
            self.x(2),
            self.x(3),
            self.x(4) as u32,
            self.x(5) as i32,
        );
        if prio > 0x3F {
            return self.ret(INVALID_PRIORITY);
        }
        // Switch 2 programs use six cores (0-5); one host thread runs
        // them all here, so any of them is accepted.
        if !(-2..=5).contains(&core) {
            return self.ret(INVALID_CORE_ID);
        }
        if self
            .threads
            .iter()
            .filter(|t| t.state != ThreadState::Exited)
            .count()
            >= MAX_THREADS
        {
            return self.ret(OUT_OF_RESOURCE);
        }
        let tls = self.alloc_tls();
        let mut cpu = Cpu::new();
        cpu.set_pc(entry);
        cpu.write_xzr(0, arg);
        cpu.write_sp(31, stack);
        cpu.tpidrro_el0 = tls;
        let i = self.add_thread(cpu, prio, tls, ThreadState::Created);
        match self.handles.insert(KObject::Thread { id: i }) {
            Some(h) => {
                self.ret(SUCCESS);
                self.set_x(1, u64::from(h));
            }
            None => {
                self.threads[i].state = ThreadState::Exited;
                self.ret(OUT_OF_HANDLES);
            }
        }
    }

    pub(crate) fn svc_start_thread(&mut self) {
        let Some(i) = self.thread_of(self.x(0) as u32) else {
            return self.ret(INVALID_HANDLE);
        };
        if self.threads[i].state != ThreadState::Created {
            return self.ret(INVALID_STATE);
        }
        self.threads[i].state = ThreadState::Runnable;
        self.ret(SUCCESS);
    }

    pub(crate) fn svc_exit_thread(&mut self) {
        self.threads[self.current].state = ThreadState::Exited;
        self.recheck_sync_waiters();
    }

    /// SleepThread: 0 and negative values yield, a positive value sleeps.
    pub(crate) fn svc_sleep_thread(&mut self) {
        let ns = self.x(0) as i64;
        if ns > 0 {
            let deadline = self.deadline(ns);
            self.block(WaitOn::Sleep, deadline);
        } else {
            self.yield_requested = true;
        }
    }

    /// WaitSynchronization(handles X1, count X2, timeout X3) -> index W1.
    pub(crate) fn svc_wait_synchronization(&mut self) {
        let (list, count, timeout) = (self.x(1), self.x(2), self.x(3) as i64);
        if count > 0x40 {
            return self.ret(OUT_OF_RANGE);
        }
        let mut handles = Vec::new();
        for i in 0..count {
            let Some(h) = self.read32(list + 4 * i) else {
                return self.ret(INVALID_ADDRESS);
            };
            match self.handles.get(h) {
                Some(KObject::Event { .. } | KObject::Thread { .. }) => handles.push(h),
                _ if h == CURRENT_THREAD => handles.push(h),
                _ => return self.ret(INVALID_HANDLE),
            }
        }
        self.last_wait = Some((
            handles
                .iter()
                .map(|&h| match self.handles.get(h) {
                    Some(&KObject::Event { id }) => self
                        .system
                        .event_names
                        .get(&id)
                        .map_or(format!("event {id}"), |n| format!("event {id} ({n})")),
                    Some(&KObject::Thread { id }) => format!("thread {id}"),
                    _ => format!("handle {h:#x}"),
                })
                .collect(),
            timeout,
        ));
        if let Some(i) = handles.iter().position(|&h| self.is_signaled(h)) {
            self.ret(SUCCESS);
            self.set_x(1, i as u64);
            return;
        }
        if timeout == 0 {
            return self.ret(TIMED_OUT);
        }
        let deadline = self.deadline(timeout);
        self.block(WaitOn::Sync(handles), deadline);
    }

    fn is_signaled(&self, handle: u32) -> bool {
        match self.handles.get(handle) {
            Some(&KObject::Event { id }) => self.kobjects.events[id],
            Some(&KObject::Thread { id }) => self.threads[id].state == ThreadState::Exited,
            _ => false,
        }
    }

    /// Wakes WaitSynchronization waiters whose objects became signaled.
    pub(crate) fn recheck_sync_waiters(&mut self) {
        for i in 0..self.threads.len() {
            let ThreadState::Waiting {
                on: WaitOn::Sync(handles),
                ..
            } = &self.threads[i].state
            else {
                continue;
            };
            if let Some(idx) = handles.iter().position(|&h| self.is_signaled(h)) {
                self.wake(i, SUCCESS, Some(idx as u64));
            }
        }
    }

    /// ArbitrateLock(owner handle W0, mutex X1, tag W2).
    pub(crate) fn svc_arbitrate_lock(&mut self) {
        let (owner, addr, tag) = (self.x(0) as u32, self.x(1), self.x(2) as u32);
        if !addr.is_multiple_of(4) {
            return self.ret(INVALID_ADDRESS);
        }
        let Some(value) = self.read32(addr) else {
            return self.ret(INVALID_ADDRESS);
        };
        // The owner may already have released it.
        if value != owner | HANDLE_WAIT_MASK {
            return self.ret(SUCCESS);
        }
        if self.thread_of(owner).is_none() {
            return self.ret(INVALID_HANDLE);
        }
        self.block(WaitOn::Mutex { addr, tag }, None);
        self.set_x(0, u64::from(SUCCESS));
    }

    /// ArbitrateUnlock(mutex X0).
    pub(crate) fn svc_arbitrate_unlock(&mut self) {
        let addr = self.x(0);
        if !addr.is_multiple_of(4) {
            return self.ret(INVALID_ADDRESS);
        }
        let r = if self.release_mutex(addr) {
            SUCCESS
        } else {
            INVALID_ADDRESS
        };
        self.ret(r);
    }

    /// Hands the mutex at `addr` to its best waiter, or frees it.
    fn release_mutex(&mut self, addr: u64) -> bool {
        let waiters = self.waiters(|on| matches!(on, WaitOn::Mutex { addr: a, .. } if *a == addr));
        let Some(&next) = waiters.first() else {
            return self.write32(addr, 0);
        };
        let ThreadState::Waiting {
            on: WaitOn::Mutex { tag, .. },
            ..
        } = self.threads[next].state
        else {
            unreachable!()
        };
        let more = if waiters.len() > 1 {
            HANDLE_WAIT_MASK
        } else {
            0
        };
        if !self.write32(addr, tag | more) {
            return false;
        }
        self.wake(next, SUCCESS, None);
        true
    }

    /// WaitProcessWideKeyAtomic(mutex X0, key X1, tag W2, timeout X3).
    pub(crate) fn svc_wait_process_wide_key(&mut self) {
        let (mutex, key, tag, timeout) = (self.x(0), self.x(1), self.x(2) as u32, self.x(3) as i64);
        if !mutex.is_multiple_of(4) {
            return self.ret(INVALID_ADDRESS);
        }
        if !self.release_mutex(mutex) || !self.write32(key, 1) {
            return self.ret(INVALID_ADDRESS);
        }
        if timeout == 0 {
            return self.ret(TIMED_OUT);
        }
        let deadline = self.deadline(timeout);
        self.block(WaitOn::CondVar { key, mutex, tag }, deadline);
    }

    /// SignalProcessWideKey(key X0, count W1; <= 0 means all).
    pub(crate) fn svc_signal_process_wide_key(&mut self) {
        let (key, count) = (self.x(0), self.x(1) as i32);
        let waiters = self.waiters(|on| matches!(on, WaitOn::CondVar { key: k, .. } if *k == key));
        let n = if count <= 0 {
            waiters.len()
        } else {
            (count as usize).min(waiters.len())
        };
        for &i in &waiters[..n] {
            let ThreadState::Waiting {
                on: WaitOn::CondVar { mutex, tag, .. },
                ..
            } = self.threads[i].state
            else {
                unreachable!()
            };
            // Retake the mutex for the woken thread, or queue it there.
            match self.read32(mutex) {
                Some(0) => {
                    self.write32(mutex, tag);
                    self.wake(i, SUCCESS, None);
                }
                Some(v) => {
                    self.write32(mutex, v | HANDLE_WAIT_MASK);
                    self.threads[i].state = ThreadState::Waiting {
                        on: WaitOn::Mutex { addr: mutex, tag },
                        deadline: None,
                    };
                }
                None => self.wake(i, INVALID_ADDRESS, None),
            }
        }
        if n == waiters.len() {
            self.write32(key, 0);
        }
        self.ret(SUCCESS);
    }

    /// WaitForAddress(addr X0, type W1, value W2, timeout X3).
    pub(crate) fn svc_wait_for_address(&mut self) {
        let (addr, kind, value, timeout) = (
            self.x(0),
            self.x(1) as u32,
            self.x(2) as i32,
            self.x(3) as i64,
        );
        if !addr.is_multiple_of(4) {
            return self.ret(INVALID_ADDRESS);
        }
        let Some(cur) = self.read32(addr).map(|v| v as i32) else {
            return self.ret(INVALID_ADDRESS);
        };
        let wait = match kind {
            0 => cur < value,
            1 => {
                if cur < value {
                    self.write32(addr, cur.wrapping_sub(1) as u32);
                }
                cur < value
            }
            2 => cur == value,
            _ => return self.ret(INVALID_ENUM_VALUE),
        };
        if !wait {
            return self.ret(INVALID_STATE);
        }
        if timeout == 0 {
            return self.ret(TIMED_OUT);
        }
        let deadline = self.deadline(timeout);
        self.block(WaitOn::Address { addr }, deadline);
    }

    /// SignalToAddress(addr X0, type W1, value W2, count W3; <= 0: all).
    pub(crate) fn svc_signal_to_address(&mut self) {
        let (addr, kind, value, count) = (
            self.x(0),
            self.x(1) as u32,
            self.x(2) as i32,
            self.x(3) as i32,
        );
        if !addr.is_multiple_of(4) {
            return self.ret(INVALID_ADDRESS);
        }
        let waiters = self.waiters(|on| *on == WaitOn::Address { addr });
        let n = if count <= 0 {
            waiters.len()
        } else {
            (count as usize).min(waiters.len())
        };
        if kind != 0 {
            let Some(cur) = self.read32(addr).map(|v| v as i32) else {
                return self.ret(INVALID_ADDRESS);
            };
            if cur != value {
                return self.ret(INVALID_STATE);
            }
            let new = match kind {
                1 => value.wrapping_add(1),
                2 => {
                    // Tells waiters how many of them remain blocked.
                    if waiters.is_empty() {
                        value.wrapping_add(1)
                    } else if count <= 0 {
                        value.wrapping_sub(2)
                    } else if waiters.len() <= count as usize {
                        value.wrapping_sub(1)
                    } else {
                        value
                    }
                }
                _ => return self.ret(INVALID_ENUM_VALUE),
            };
            self.write32(addr, new as u32);
        }
        for &i in &waiters[..n] {
            self.wake(i, SUCCESS, None);
        }
        self.ret(SUCCESS);
    }

    // ---- scheduling ----

    /// Wakes threads whose timeout passed.
    fn expire_timeouts(&mut self) {
        let now = self.tick();
        for i in 0..self.threads.len() {
            let ThreadState::Waiting {
                on,
                deadline: Some(d),
            } = &self.threads[i].state
            else {
                continue;
            };
            if *d > now {
                continue;
            }
            let result = match on {
                WaitOn::Sleep => SUCCESS,
                _ => TIMED_OUT,
            };
            self.wake(i, result, None);
        }
    }

    /// Picks the next thread to run (round-robin after the current one)
    /// and makes it current. Advances the clock if all threads sleep.
    pub(crate) fn schedule(&mut self, pc: u64) -> Result<(), Stop> {
        self.sample_input();
        loop {
            self.expire_timeouts();
            self.vblank();
            self.audio_frame();
            self.check_syncpoint_waits();
            let n = self.threads.len();
            let next = (1..=n)
                .map(|k| (self.current + k) % n)
                .find(|&i| self.threads[i].state == ThreadState::Runnable);
            if let Some(next) = next {
                self.switch_to(next);
                return Ok(());
            }
            if self
                .threads
                .iter()
                .all(|t| matches!(t.state, ThreadState::Exited | ThreadState::Created))
            {
                return Err(Stop::ExitProcess);
            }
            let earliest = self
                .threads
                .iter()
                .filter_map(|t| match t.state {
                    ThreadState::Waiting { deadline, .. } => deadline,
                    _ => None,
                })
                .min();
            // With vblank events, the next vblank can wake someone.
            let next_vblank = (!self.system.vblank_events.is_empty())
                .then(|| (self.cpu.counter / VBLANK_PERIOD + 1) * VBLANK_PERIOD);
            let next_audio = (!self.system.audio_events.is_empty())
                .then(|| (self.cpu.counter / AUDIO_PERIOD + 1) * AUDIO_PERIOD);
            match earliest
                .into_iter()
                .chain(next_vblank)
                .chain(next_audio)
                .min()
            {
                Some(d) => self.cpu.counter = self.cpu.counter.max(d),
                None => return Err(Stop::Deadlock { pc }),
            }
        }
    }

    /// Signals the vblank events once per 1/60 s of guest time and wakes
    /// the threads waiting on them.
    fn vblank(&mut self) {
        let due = self.cpu.counter / VBLANK_PERIOD;
        if due <= self.last_vblank || self.system.vblank_events.is_empty() {
            return;
        }
        self.last_vblank = due;
        for &e in &self.system.vblank_events {
            self.kobjects.events[e] = true;
        }
        self.recheck_sync_waiters();
    }

    /// Signals the audio renderers' events once per audio frame.
    fn audio_frame(&mut self) {
        let due = self.cpu.counter / AUDIO_PERIOD;
        if due <= self.last_audio_frame || self.system.audio_events.is_empty() {
            return;
        }
        self.last_audio_frame = due;
        for &e in &self.system.audio_events {
            self.kobjects.events[e] = true;
        }
        self.recheck_sync_waiters();
    }

    /// Signals nx2:syncpt slot events whose GPU syncpoint was reached.
    pub(crate) fn check_syncpoint_waits(&mut self) {
        if self.system.syncpt_waits.is_empty() {
            return;
        }
        let points = &self.system.gpu.syncpoints;
        let mut fired = false;
        let events = &mut self.kobjects.events;
        self.system.syncpt_waits.retain(|&(sp, value, e)| {
            let reached = points
                .get(sp as usize)
                .is_some_and(|&now| now.wrapping_sub(value) as i32 >= 0);
            if reached {
                events[e] = true;
                fired = true;
            }
            !reached
        });
        if fired {
            self.recheck_sync_waiters();
        }
    }

    /// Writes a new controller sample to hid shared memory every 4 ms of
    /// guest time, like the hid sysmodule.
    fn sample_input(&mut self) {
        const PERIOD: u64 = 19_200_000 / 250;
        let Some(id) = self.system.hid_shared_memory else {
            return;
        };
        let Some(base) = self.kobjects.shared_memory[id].mapped_at else {
            return;
        };
        let due = self.cpu.counter / PERIOD;
        if due <= self.last_input_sample {
            return;
        }
        self.last_input_sample = due;
        let sampling = self.system.hid_sampling;
        self.system.hid_sampling += 1;
        for (off, bytes) in crate::services::hid::sample_writes(sampling, &self.system.input) {
            let _ = self.mem.write_privileged(base + off as u64, &bytes);
        }
    }

    fn switch_to(&mut self, next: usize) {
        if next == self.current {
            return;
        }
        let now = self.cpu.counter;
        std::mem::swap(&mut self.cpu, &mut self.threads[self.current].cpu);
        std::mem::swap(&mut self.cpu, &mut self.threads[next].cpu);
        self.cpu.counter = now;
        // A context switch clears the exclusive monitor.
        self.cpu.exclusive = None;
        self.current = next;
    }

    /// The current thread's id (GetThreadId on the pseudo handle).
    pub(crate) fn thread_id(&self, i: usize) -> u64 {
        self.threads[i].tid
    }
}

/// Guest ticks (19.2 MHz) per vblank at 60 Hz.
pub const VBLANK_PERIOD: u64 = 19_200_000 / 60;
/// One audio frame: 5 ms of guest time.
pub const AUDIO_PERIOD: u64 = 19_200_000 / 200;
