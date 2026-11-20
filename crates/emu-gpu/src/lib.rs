//! The GPU of the Switch family: an NVIDIA Tegra GPU that shares memory
//! with the CPU. Programs talk to it through channels: they write
//! commands ("methods") into pushbuffers in GPU memory and submit GPFIFO
//! entries pointing at them. Each method goes to the engine bound to a
//! subchannel (3D, compute, 2D, DMA copy, inline-to-memory) or to the
//! channel itself (host methods: semaphores, syncpoints, binding).
//!
//! Commands are processed synchronously when submitted, with a software
//! backend for now: clears and copies write guest memory directly. The
//! classes are the Maxwell ones (B197 3D, B0B5 copy, ...) that the
//! Switch's driver reports and that homebrew (deko3d) and Switch games
//! use; their method layouts come from NVIDIA's open-gpu-doc class
//! headers (MIT).

pub mod backend;
pub mod blocklinear;
pub mod draw;
pub mod engines;
pub mod mmu;
pub mod pipeline;
pub mod shader;
pub mod surface;
pub mod texture;

use emu_memory::Memory;
use engines::{copy::CopyEngine, inline::InlineToMemory, maxwell3d::Maxwell3D, EngineCtx};
use mmu::AddressSpace;
use std::collections::BTreeSet;

pub const NUM_SYNCPOINTS: usize = 192;

/// Class numbers.
pub mod class {
    pub const TWOD: u32 = 0x902D;
    pub const THREED: u32 = 0xB197;
    pub const COMPUTE: u32 = 0xB1C0;
    pub const GPFIFO: u32 = 0xB06F;
    pub const INLINE_TO_MEMORY: u32 = 0xA140;
    pub const DMA_COPY: u32 = 0xB0B5;
    /// Switch 2 (Ampere, GA10B-class) equivalents. Their method layouts
    /// extend the Maxwell ones, so they run on the same engines (as in
    /// Lichii); the 3D class uses the Turing+ macro ISA (MME64).
    pub const AMPERE_THREED: u32 = 0xC797;
    pub const AMPERE_COMPUTE: u32 = 0xC7C0;
    pub const AMPERE_DMA_COPY: u32 = 0xC7B5;
    pub const AMPERE_GPFIFO: u32 = 0xC76F;
}

/// An engine bound to a subchannel.
pub enum Engine {
    ThreeD(Box<Maxwell3D>),
    /// A copy engine and the class it was bound as.
    Copy(Box<CopyEngine>, u32),
    Inline(InlineToMemory),
    /// A class that is bound but not emulated yet; methods are recorded
    /// as unimplemented.
    Other(u32),
}

impl Engine {
    fn new(class: u32) -> Engine {
        match class {
            class::THREED => Engine::ThreeD(Box::default()),
            class::AMPERE_THREED => Engine::ThreeD(Box::new(Maxwell3D::ampere())),
            class::DMA_COPY | class::AMPERE_DMA_COPY => Engine::Copy(Box::default(), class),
            class::INLINE_TO_MEMORY => Engine::Inline(InlineToMemory::default()),
            c => Engine::Other(c),
        }
    }

    pub fn class(&self) -> u32 {
        match self {
            Engine::ThreeD(e) => e.class(),
            Engine::Copy(_, c) => *c,
            Engine::Inline(_) => class::INLINE_TO_MEMORY,
            Engine::Other(c) => *c,
        }
    }
}

/// A method write being expanded from a pushbuffer header.
#[derive(Debug, Clone, Copy)]
struct Pending {
    subchannel: usize,
    method: u32,
    remaining: u32,
    mode: Mode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Increasing,
    NonIncreasing,
    IncreaseOnce,
}

#[derive(Default)]
struct HostState {
    semaphore_va: u64,
    semaphore_payload: u32,
    syncpoint_payload: u32,
}

/// Queued work on a channel.
#[derive(Debug, Clone, Copy)]
enum Work {
    /// Command words at a GPU address.
    Words { va: u64, words: usize },
    /// Raise the channel's syncpoint (a fence) once the work before is done.
    Increment(u32),
}

/// A semaphore acquire the channel waits on.
#[derive(Debug, Clone, Copy)]
struct Acquire {
    va: u64,
    value: u32,
    /// SEMAPHORED operation: 1 equal, 4 greater or equal, 8 mask.
    op: u32,
}

pub struct Channel {
    pub address_space: Option<usize>,
    /// The syncpoint this channel's fences use.
    pub syncpoint: u32,
    subchannels: [Option<Engine>; 8],
    pending: Option<Pending>,
    host: HostState,
    /// Work not run yet (behind an unmet semaphore acquire).
    queue: std::collections::VecDeque<Work>,
    stalled: Option<Acquire>,
    /// A Switch 2 (Ampere) class was bound: semaphore acquires wait.
    ampere: bool,
    /// Host (GPFIFO class) method values, by method index.
    host_regs: [u32; 0x40],
}

pub struct Gpu {
    pub address_spaces: Vec<AddressSpace>,
    pub channels: Vec<Channel>,
    pub syncpoints: Vec<u32>,
    /// (class, method byte offset) pairs that were not emulated.
    pub unimplemented: BTreeSet<(u32, u32)>,
    /// Faults while processing commands (unmapped addresses etc.).
    pub faults: Vec<String>,
    /// Log methods that are not emulated, once each.
    pub trace: bool,
    /// Carries out rendering work (software by default).
    pub backend: Box<dyn backend::Backend>,
}

impl Default for Gpu {
    fn default() -> Self {
        Gpu {
            address_spaces: Vec::new(),
            channels: Vec::new(),
            syncpoints: vec![0; NUM_SYNCPOINTS],
            unimplemented: BTreeSet::new(),
            faults: Vec::new(),
            trace: false,
            backend: Box::new(backend::SoftwareBackend),
        }
    }
}

/// Submission flags (nvgpu's NVGPU_SUBMIT_GPFIFO_FLAGS_*).
pub mod submit_flags {
    /// Wait for the given fence first.
    pub const FENCE_WAIT: u32 = 1 << 0;
    /// The kernel increments the channel's syncpoint after the work.
    pub const FENCE_INCREMENT: u32 = 1 << 1;
}

impl Gpu {
    pub fn new_address_space(&mut self) -> usize {
        self.address_spaces.push(AddressSpace::new());
        self.address_spaces.len() - 1
    }

    pub fn new_channel(&mut self) -> usize {
        let id = self.channels.len();
        self.channels.push(Channel {
            address_space: None,
            // Low syncpoint ids belong to the system; one per channel.
            syncpoint: 32 + id as u32,
            subchannels: Default::default(),
            pending: None,
            host: HostState::default(),
            queue: Default::default(),
            stalled: None,
            ampere: false,
            host_regs: [0; 0x40],
        });
        id
    }

    /// Whether a channel waits on a semaphore acquire.
    pub fn has_stalled(&self) -> bool {
        self.channels.iter().any(|c| c.stalled.is_some())
    }

    /// Runs channels whose semaphore acquire is met by now (the CPU or
    /// another channel released it). Returns whether any made progress.
    pub fn poll(&mut self, mem: &mut Memory) -> bool {
        let mut progress = false;
        for c in 0..self.channels.len() {
            if self.channels[c].stalled.is_some() && self.acquire_met(c, mem) {
                self.pump(c, mem);
                progress = true;
            }
        }
        progress
    }

    fn acquire_met(&self, channel: usize, mem: &Memory) -> bool {
        let ch = &self.channels[channel];
        let (Some(a), Some(asid)) = (ch.stalled, ch.address_space) else {
            return true;
        };
        let mut b = [0u8; 4];
        if self.address_spaces[asid].read(mem, a.va, &mut b).is_err() {
            return true; // nothing to wait on
        }
        let word = u32::from_le_bytes(b);
        match a.op {
            1 => word == a.value,
            4 => word >= a.value,
            8 => word & a.value != 0,
            _ => true,
        }
    }

    /// Runs queued work on a channel until it is empty or an acquire is
    /// not met.
    fn pump(&mut self, channel: usize, mem: &mut Memory) {
        loop {
            if self.channels[channel].stalled.is_some() {
                if !self.acquire_met(channel, mem) {
                    return;
                }
                self.channels[channel].stalled = None;
            }
            let Some(work) = self.channels[channel].queue.pop_front() else {
                return;
            };
            let (va, words) = match work {
                Work::Increment(n) => {
                    let sp = self.channels[channel].syncpoint as usize;
                    self.syncpoints[sp] = self.syncpoints[sp].wrapping_add(n);
                    continue;
                }
                Work::Words { va, words } => (va, words),
            };
            let Some(asid) = self.channels[channel].address_space else {
                self.faults
                    .push("submit on a channel without an address space".into());
                self.channels[channel].queue.clear();
                return;
            };
            // Read when run, not when submitted: NVN2 writes the commands
            // behind an acquire after submitting them.
            let mut raw = vec![0u8; words * 4];
            if let Err(f) = self.address_spaces[asid].read(mem, va, &mut raw) {
                self.faults
                    .push(format!("pushbuffer fetch at {va:#x}: {f:?}"));
                continue;
            }
            for (i, w) in raw.chunks_exact(4).enumerate() {
                self.process_word(channel, u32::from_le_bytes(w.try_into().unwrap()), mem);
                if self.channels[channel].stalled.is_some() && i + 1 < words {
                    // Resume after this word; the method in progress (if
                    // any) stays pending.
                    self.channels[channel].queue.push_front(Work::Words {
                        va: va + 4 * (i as u64 + 1),
                        words: words - i - 1,
                    });
                    break;
                }
            }
        }
    }

    /// Queues GPFIFO entries on a channel and runs them (all of them,
    /// unless a semaphore acquire is not met yet). `increment` raises the
    /// syncpoint after the work (FENCE_INCREMENT adds one more). Returns
    /// the channel's syncpoint id and the value it reaches once the work
    /// is done.
    pub fn submit_with_increment(
        &mut self,
        channel: usize,
        entries: &[u64],
        flags: u32,
        increment: u32,
        mem: &mut Memory,
    ) -> (u32, u32) {
        let ch = &mut self.channels[channel];
        for &entry in entries {
            // Entry: address bits 31:2 in the low word, 39:32 in bits
            // 7:0 of the high word; length in words in bits 30:10.
            let va = (entry & 0xFFFF_FFFC) | ((entry >> 32) & 0xFF) << 32;
            let words = ((entry >> 42) & 0x1F_FFFF) as usize;
            if words != 0 {
                ch.queue.push_back(Work::Words { va, words });
            } // else a control entry (NOP etc.)
        }
        let increment =
            increment.wrapping_add(u32::from(flags & submit_flags::FENCE_INCREMENT != 0));
        if increment != 0 {
            ch.queue.push_back(Work::Increment(increment));
        }
        self.pump(channel, mem);
        // Results must be in memory before the fence signals.
        if let Some(asid) = self.channels[channel].address_space {
            let mut g = backend::GuestMemory {
                mem,
                aspace: &self.address_spaces[asid],
                asid,
            };
            if let Err(e) = self.backend.flush(&mut g) {
                self.faults
                    .push(format!("{} backend: {e}", self.backend.name()));
            }
        }
        let ch = &self.channels[channel];
        let sp = ch.syncpoint as usize;
        let queued: u32 = ch
            .queue
            .iter()
            .map(|w| match w {
                Work::Increment(n) => *n,
                Work::Words { .. } => 0,
            })
            .fold(0, u32::wrapping_add);
        (sp as u32, self.syncpoints[sp].wrapping_add(queued))
    }

    /// [`Gpu::submit_with_increment`] without an extra increment.
    pub fn submit(
        &mut self,
        channel: usize,
        entries: &[u64],
        flags: u32,
        mem: &mut Memory,
    ) -> (u32, u32) {
        self.submit_with_increment(channel, entries, flags, 0, mem)
    }

    fn process_word(&mut self, channel: usize, word: u32, mem: &mut Memory) {
        if let Some(mut p) = self.channels[channel].pending {
            self.method(channel, p.subchannel, p.method, word, p.remaining == 1, mem);
            match p.mode {
                Mode::Increasing => p.method += 1,
                Mode::IncreaseOnce => {
                    p.method += 1;
                    p.mode = Mode::NonIncreasing;
                }
                Mode::NonIncreasing => {}
            }
            p.remaining -= 1;
            self.channels[channel].pending = (p.remaining > 0).then_some(p);
            return;
        }
        let method = word & 0xFFF;
        let subchannel = ((word >> 13) & 7) as usize;
        let count = (word >> 16) & 0x1FFF;
        let mode = match word >> 29 {
            1 => Mode::Increasing,
            3 => Mode::NonIncreasing,
            5 => Mode::IncreaseOnce,
            4 => {
                // Immediate: the count field is the data.
                self.method(channel, subchannel, method, count, true, mem);
                return;
            }
            _ => return, // NOPs and legacy/unsupported headers
        };
        if count > 0 {
            self.channels[channel].pending = Some(Pending {
                subchannel,
                method,
                remaining: count,
                mode,
            });
        }
    }

    /// One method write. `method` is a word index (byte offset / 4).
    fn method(
        &mut self,
        channel: usize,
        subchannel: usize,
        method: u32,
        value: u32,
        last: bool,
        mem: &mut Memory,
    ) {
        // Methods below 0x40 go to the channel itself.
        if method < 0x40 {
            self.host_method(channel, subchannel, method, value, mem);
            return;
        }
        let Some(asid) = self.channels[channel].address_space else {
            return;
        };
        let Gpu {
            address_spaces,
            channels,
            syncpoints,
            unimplemented,
            faults,
            trace,
            backend,
        } = self;
        let mut ctx = EngineCtx {
            guest: backend::GuestMemory {
                mem,
                aspace: &address_spaces[asid],
                asid,
            },
            syncpoints,
            faults,
            backend: backend.as_mut(),
        };
        let mut from_macros = Vec::new();
        // NVN2 writes inline-to-memory methods on subchannel 2 without
        // binding a class first (Lichii NX2_NOTES row 49).
        let slot = &mut channels[channel].subchannels[subchannel];
        if slot.is_none() && subchannel == 2 {
            *slot = Some(Engine::Inline(InlineToMemory::default()));
        }
        let handled = match slot.as_mut() {
            Some(Engine::ThreeD(e)) => e.write(method, value, last, &mut ctx, &mut from_macros),
            Some(Engine::Copy(e, _)) => e.write(method, value, &mut ctx),
            Some(Engine::Inline(e)) => e.write(method, value, &mut ctx),
            Some(Engine::Other(_)) | None => false,
        };
        if !handled {
            let class = channels[channel].subchannels[subchannel]
                .as_ref()
                .map_or(0, Engine::class);
            if unimplemented.insert((class, method * 4)) && *trace {
                eprintln!(
                    "[gpu] class {class:#06x}: method {:#06x} = {value:#x} not emulated",
                    method * 4
                );
            }
        }
    }

    fn host_method(
        &mut self,
        channel: usize,
        subchannel: usize,
        method: u32,
        value: u32,
        mem: &mut Memory,
    ) {
        let ch = &mut self.channels[channel];
        if let Some(r) = ch.host_regs.get_mut(method as usize) {
            *r = value;
        }
        match method * 4 {
            // SET_OBJECT: bind a class to the subchannel.
            0x00 => {
                let class = value & 0xFFFF;
                if class >> 8 >= 0xC5 {
                    ch.ampere = true; // Ampere-era classes (C5xx and later)
                }
                if ch.subchannels[subchannel].as_ref().map(Engine::class) != Some(class) {
                    ch.subchannels[subchannel] = Some(Engine::new(class));
                }
            }
            0x10 => {
                ch.host.semaphore_va =
                    (ch.host.semaphore_va & 0xFFFF_FFFF) | u64::from(value & 0xFF) << 32
            }
            0x14 => {
                ch.host.semaphore_va = (ch.host.semaphore_va & !0xFFFF_FFFF) | u64::from(value & !3)
            }
            0x18 => ch.host.semaphore_payload = value,
            0x1C => {
                // SEMAPHORED: operation in bits 4:0; release writes the
                // payload (4 bytes, or 16 with a timestamp).
                if value & 0x1F == 2 {
                    let (va, payload) = (ch.host.semaphore_va, ch.host.semaphore_payload);
                    let mut data = payload.to_le_bytes().to_vec();
                    if value & (1 << 24) == 0 {
                        data.extend_from_slice(&[0; 4]);
                        data.extend_from_slice(&0u64.to_le_bytes());
                    }
                    if let Some(asid) = ch.address_space {
                        if let Err(f) = self.address_spaces[asid].write(mem, va, &data) {
                            self.faults
                                .push(format!("semaphore release at {va:#x}: {f:?}"));
                        }
                    }
                }
                // Acquires: on Switch 2 the CPU releases them once it has
                // written the commands that follow, so the channel waits
                // (Lichii NX2_NOTES row 50). Elsewhere work is synchronous
                // and they count as met.
                let op = value & 0xF;
                if ch.ampere && matches!(op, 1 | 4 | 8) {
                    ch.stalled = Some(Acquire {
                        va: ch.host.semaphore_va,
                        value: ch.host.semaphore_payload,
                        op,
                    });
                }
            }
            // Ampere SEM_ADDR_LO/HI, SEM_PAYLOAD_LO/HI, SEM_EXECUTE.
            0x5C | 0x60 | 0x64 | 0x68 => {}
            0x6C => {
                // Release (operation 1) writes the payload, 8 bytes with
                // PAYLOAD_SIZE (bit 24). Acquires do not wait here, as in
                // Lichii's ProcessSemaphoreExecute.
                if value & 7 == 1 {
                    let regs = &ch.host_regs;
                    let va = (u64::from(regs[0x60 / 4] & 0x1FF_FFFF) << 32)
                        | u64::from(regs[0x5C / 4] & !3);
                    let mut data = regs[0x64 / 4].to_le_bytes().to_vec();
                    if value & (1 << 24) != 0 {
                        data.extend_from_slice(&regs[0x68 / 4].to_le_bytes());
                    }
                    if let Some(asid) = ch.address_space {
                        if let Err(f) = self.address_spaces[asid].write(mem, va, &data) {
                            self.faults
                                .push(format!("semaphore release at {va:#x}: {f:?}"));
                        }
                    }
                }
            }
            0x70 => ch.host.syncpoint_payload = value,
            // SYNCPOINTB: operation 1:0 (1 = increment), index 15:8.
            0x74 if value & 3 == 1 => {
                let id = ((value >> 8) & 0xFF) as usize;
                if let Some(s) = self.syncpoints.get_mut(id) {
                    *s = s.wrapping_add(1);
                }
            }
            // NOP, non-stall interrupt, WFI, memory ops, etc.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use emu_memory::Perm;

    /// Pushbuffer header: increasing methods.
    pub fn incr(subch: u32, method: u32, count: u32) -> u32 {
        (1 << 29) | (count << 16) | (subch << 13) | (method / 4)
    }

    pub fn immd(subch: u32, method: u32, data: u32) -> u32 {
        (4 << 29) | (data << 16) | (subch << 13) | (method / 4)
    }

    /// A GPU with one channel whose address space maps GPU 0x1000_0000
    /// to guest 0x100_0000 (1 MiB).
    pub fn setup() -> (Gpu, Memory) {
        let mut mem = Memory::new();
        mem.map(0x100_0000, 0x10_0000, Perm::RW).unwrap();
        let mut gpu = Gpu::default();
        let a = gpu.new_address_space();
        assert!(gpu.address_spaces[a].map(0x1000_0000, 0x100_0000, 0x10_0000));
        let c = gpu.new_channel();
        gpu.channels[c].address_space = Some(a);
        (gpu, mem)
    }

    /// Writes words at guest 0x100_0000 and submits them.
    pub fn run(gpu: &mut Gpu, mem: &mut Memory, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.write(0x100_0000, &bytes).unwrap();
        let entry = 0x1000_0000u64 | ((words.len() as u64) << 42);
        gpu.submit(0, &[entry], 0, mem);
        assert!(gpu.faults.is_empty(), "{:?}", gpu.faults);
    }

    #[test]
    fn host_semaphore_and_syncpoint() {
        let (mut gpu, mut mem) = setup();
        run(
            &mut gpu,
            &mut mem,
            &[
                incr(0, 0x10, 4),
                0,
                0x1000_8000,
                0xCAFE_F00D,
                2 | (1 << 24), // release, 4 bytes
                incr(0, 0x70, 2),
                0,
                1 | (40 << 8), // increment syncpoint 40
            ],
        );
        let mut b = [0u8; 4];
        mem.read(0x100_8000, &mut b).unwrap();
        assert_eq!(u32::from_le_bytes(b), 0xCAFE_F00D);
        assert_eq!(gpu.syncpoints[40], 1);
    }

    #[test]
    fn method_modes() {
        let (mut gpu, mut mem) = setup();
        // Bind 3D, then non-increasing and increase-once writes to the
        // 3D register file.
        run(
            &mut gpu,
            &mut mem,
            &[
                incr(0, 0, 1),
                class::THREED,
                (3 << 29) | (2 << 16) | (0x0d80 / 4),
                1,
                2,
                (5 << 29) | (3 << 16) | (0x0d84 / 4),
                3,
                4,
                5,
            ],
        );
        let Some(Engine::ThreeD(e)) = &gpu.channels[0].subchannels[0] else {
            panic!()
        };
        assert_eq!(e.reg(0x0d80), 2);
        assert_eq!(e.reg(0x0d84), 3);
        assert_eq!(e.reg(0x0d88), 5);
    }

    #[test]
    fn ampere_3d_runs_mme64_macros() {
        use engines::mme64::asm::{inst, Lane, LOAD0, NOP, ZERO};
        let (mut gpu, mut mem) = setup();
        // Macro: r0 = load + 1; write it to 3D method 0xd80/4 (lane 1 imm),
        // then also DMA-read a word into the data RAM via MME_DMA_READ.
        let mut first = inst(
            false,
            0,
            0,
            [
                Lane {
                    op: 0,
                    dst: 0,
                    src_a: LOAD0,
                    src_b: 25,
                    imm: 1,
                    out_method: 6,
                    out_emit: 1,
                },
                Lane {
                    imm: 0x0d80 / 4,
                    ..NOP
                },
            ],
        );
        first[2] |= 1; // end
        let second = inst(false, 0, 0, [NOP, Lane { dst: ZERO, ..NOP }]);
        let mut words = vec![incr(0, 0, 1), class::AMPERE_THREED];
        // Upload: instruction pointer 0, six code words; start 0 at index 0.
        words.extend([incr(0, 0x114, 1), 0, (3 << 29) | (6 << 16) | (0x118 / 4)]);
        words.extend(first);
        words.extend(second);
        words.extend([incr(0, 0x11C, 2), 0, 0]);
        // Data for the DMA read, at GPU 0x1000_9000.
        mem.write(0x100_9000, &0xABCD_u32.to_le_bytes()).unwrap();
        words.extend([incr(0, 0x550, 4), 0, 0x1000_9000, 3, 1]);
        // Call macro 0 with parameter 41, then write subchannel 2 unbound.
        words.extend([incr(0, 0x3800, 1), 41, incr(2, 0x180, 1), 4]);
        run(&mut gpu, &mut mem, &words);
        let Some(Engine::ThreeD(e)) = &gpu.channels[0].subchannels[0] else {
            panic!()
        };
        assert_eq!(e.class(), class::AMPERE_THREED);
        assert_eq!(e.reg(0x0d80), 42);
        assert_eq!(e.data_ram[3], 0xABCD);
        assert_eq!(e.macro_calls, 1);
    }

    #[test]
    fn switch2_acquire_waits_for_the_cpu() {
        let (mut gpu, mut mem) = setup();
        // Bind Ampere 3D; acquire (GEQ 1) on 0x1000_8000, which holds 0;
        // then write a 3D register and release 7 via SEM_EXECUTE.
        let words = [
            incr(0, 0, 1),
            class::AMPERE_THREED,
            incr(0, 0x10, 4),
            0,
            0x1000_8000,
            1,
            4, // acquire >=
            incr(0, 0x0d80, 1),
            5,
            incr(0, 0x5C, 3),
            0x1000_8010,
            0,
            7,
            incr(0, 0x6C, 1),
            1, // release, 4 bytes
        ];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        mem.write(0x100_0000, &bytes).unwrap();
        let entry = 0x1000_0000u64 | ((words.len() as u64) << 42);
        let sp = gpu.channels[0].syncpoint as usize;
        let (_, fence) = gpu.submit(0, &[entry], submit_flags::FENCE_INCREMENT, &mut mem);
        assert!(gpu.faults.is_empty(), "{:?}", gpu.faults);
        assert!(gpu.has_stalled());
        assert_eq!(fence, 1);
        assert_eq!(gpu.syncpoints[sp], 0);
        let reg = |gpu: &Gpu| match &gpu.channels[0].subchannels[0] {
            Some(Engine::ThreeD(e)) => e.reg(0x0d80),
            _ => panic!(),
        };
        assert_eq!(reg(&gpu), 0);
        assert!(!gpu.poll(&mut mem));
        // The CPU releases the semaphore.
        mem.write(0x100_8000, &1u32.to_le_bytes()).unwrap();
        assert!(gpu.poll(&mut mem));
        assert!(!gpu.has_stalled());
        assert_eq!(reg(&gpu), 5);
        assert_eq!(gpu.syncpoints[sp], 1);
        let mut b = [0u8; 4];
        mem.read(0x100_8010, &mut b).unwrap();
        assert_eq!(u32::from_le_bytes(b), 7);
    }
}
