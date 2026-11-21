//! Engines (GPU classes).

pub mod compute;
pub mod copy;
pub mod inline;
pub mod maxwell3d;
pub mod mme;
pub mod mme64;
pub mod qmd;

use crate::backend::{Backend, GuestMemory};

/// What an engine can touch while executing a method.
pub struct EngineCtx<'a> {
    pub guest: GuestMemory<'a>,
    pub syncpoints: &'a mut Vec<u32>,
    pub faults: &'a mut Vec<String>,
    pub backend: &'a mut dyn Backend,
    /// MME64 macros (Ampere classes), shared by every channel.
    pub mme: &'a mut super::engines::mme64::Shared,
    /// The channel's MME data RAM, shared by its 3D and compute engines
    /// (Lichii channel_state.cpp: compute uses the 3D engine's).
    pub data_ram: &'a mut Vec<u32>,
}

impl EngineCtx<'_> {
    /// Writes guest memory; a fault is recorded with the calling engine
    /// source line (the fault kind is the message up to the colon).
    #[track_caller]
    pub fn write(&mut self, va: u64, data: &[u8]) {
        if let Err(f) = self.guest.write(va, data) {
            let at = std::panic::Location::caller();
            self.faults.push(format!(
                "write at {va:#x} from {}:{}: {f:?}",
                at.file().rsplit(['/', '\\']).next().unwrap_or(""),
                at.line()
            ));
        }
    }

    #[track_caller]
    pub fn read(&mut self, va: u64, buf: &mut [u8]) {
        if let Err(f) = self.guest.read(va, buf) {
            let at = std::panic::Location::caller();
            self.faults.push(format!(
                "read at {va:#x} from {}:{}: {f:?}",
                at.file().rsplit(['/', '\\']).next().unwrap_or(""),
                at.line()
            ));
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
