//! Engines (GPU classes).

pub mod copy;
pub mod inline;
pub mod maxwell3d;

use crate::mmu::AddressSpace;
use emu_memory::Memory;

/// What an engine can touch while executing a method.
pub struct EngineCtx<'a> {
    pub mem: &'a mut Memory,
    pub aspace: &'a AddressSpace,
    pub syncpoints: &'a mut Vec<u32>,
    pub faults: &'a mut Vec<String>,
}

impl EngineCtx<'_> {
    pub fn write(&mut self, va: u64, data: &[u8]) {
        if let Err(f) = self.aspace.write(self.mem, va, data) {
            self.faults.push(format!("write at {va:#x}: {f:?}"));
        }
    }

    pub fn read(&mut self, va: u64, buf: &mut [u8]) {
        if let Err(f) = self.aspace.read(self.mem, va, buf) {
            self.faults.push(format!("read at {va:#x}: {f:?}"));
        }
    }

    pub fn increment_syncpoint(&mut self, id: u32) {
        if let Some(s) = self.syncpoints.get_mut(id as usize) {
            *s = s.wrapping_add(1);
        }
    }
}

/// Bits `hi:lo` of `v`.
pub(crate) fn bits(v: u32, hi: u32, lo: u32) -> u32 {
    (v >> lo) & ((1u64 << (hi - lo + 1)) - 1) as u32
}
