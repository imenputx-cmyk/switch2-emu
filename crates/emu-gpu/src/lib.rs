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

pub mod blocklinear;
pub mod engines;
pub mod mmu;
pub mod surface;

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
}

/// An engine bound to a subchannel.
pub enum Engine {
    ThreeD(Box<Maxwell3D>),
    Copy(Box<CopyEngine>),
    Inline(InlineToMemory),
    /// A class that is bound but not emulated yet; methods are recorded
    /// as unimplemented.
    Other(u32),
}

impl Engine {
    fn new(class: u32) -> Engine {
        match class {
            class::THREED => Engine::ThreeD(Box::default()),
            class::DMA_COPY => Engine::Copy(Box::default()),
            class::INLINE_TO_MEMORY => Engine::Inline(InlineToMemory::default()),
            c => Engine::Other(c),
        }
    }

    pub fn class(&self) -> u32 {
        match self {
            Engine::ThreeD(_) => class::THREED,
            Engine::Copy(_) => class::DMA_COPY,
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

pub struct Channel {
    pub address_space: Option<usize>,
    /// The syncpoint this channel's fences use.
    pub syncpoint: u32,
    subchannels: [Option<Engine>; 8],
    pending: Option<Pending>,
    host: HostState,
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
        });
        id
    }

    /// Processes GPFIFO entries on a channel. Returns the channel's
    /// syncpoint id and its value afterwards.
    pub fn submit(
        &mut self,
        channel: usize,
        entries: &[u64],
        flags: u32,
        mem: &mut Memory,
    ) -> (u32, u32) {
        for &entry in entries {
            // Entry: address bits 31:2 in the low word, 39:32 in bits
            // 7:0 of the high word; length in words in bits 30:10.
            let va = (entry & 0xFFFF_FFFC) | ((entry >> 32) & 0xFF) << 32;
            let words = ((entry >> 42) & 0x1F_FFFF) as usize;
            if words == 0 {
                continue; // control entry (NOP etc.)
            }
            let Some(asid) = self.channels[channel].address_space else {
                self.faults
                    .push("submit on a channel without an address space".into());
                break;
            };
            let mut raw = vec![0u8; words * 4];
            if let Err(f) = self.address_spaces[asid].read(mem, va, &mut raw) {
                self.faults
                    .push(format!("pushbuffer fetch at {va:#x}: {f:?}"));
                continue;
            }
            for w in raw.chunks_exact(4) {
                self.process_word(channel, u32::from_le_bytes(w.try_into().unwrap()), mem);
            }
        }
        let sp = self.channels[channel].syncpoint as usize;
        if flags & submit_flags::FENCE_INCREMENT != 0 {
            self.syncpoints[sp] = self.syncpoints[sp].wrapping_add(1);
        }
        (sp as u32, self.syncpoints[sp])
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
        } = self;
        let mut ctx = EngineCtx {
            mem,
            aspace: &address_spaces[asid],
            syncpoints,
            faults,
        };
        let mut from_macros = Vec::new();
        let handled = match channels[channel].subchannels[subchannel].as_mut() {
            Some(Engine::ThreeD(e)) => e.write(method, value, last, &mut ctx, &mut from_macros),
            Some(Engine::Copy(e)) => e.write(method, value, &mut ctx),
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
        match method * 4 {
            // SET_OBJECT: bind a class to the subchannel.
            0x00 => {
                let class = value & 0xFFFF;
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
                // Acquires are always satisfied: work is synchronous.
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
}
