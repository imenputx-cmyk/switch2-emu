//! Engines (GPU classes).

pub mod copy;
pub mod inline;
pub mod maxwell3d;
pub mod mme;
pub mod mme64;

use crate::backend::{Backend, GuestMemory};

/// What an engine can touch while executing a method.
pub struct EngineCtx<'a> {
    pub guest: GuestMemory<'a>,
    pub syncpoints: &'a mut Vec<u32>,
    pub faults: &'a mut Vec<String>,
    pub backend: &'a mut dyn Backend,
}

impl EngineCtx<'_> {
    pub fn write(&mut self, va: u64, data: &[u8]) {
        if let Err(f) = self.guest.write(va, data) {
            self.faults.push(format!("write at {va:#x}: {f:?}"));
        }
    }

    pub fn read(&mut self, va: u64, buf: &mut [u8]) {
        if let Err(f) = self.guest.read(va, buf) {
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
