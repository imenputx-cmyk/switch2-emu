//! Scheduling of QMDs (compute launch descriptors, version 3.0) as NVN2
//! drives them through the compute class's PCAS actions: a QMD is loaded
//! with a dependency count, other QMDs (or the host) decrement it, and it
//! runs once it reaches zero. A finished QMD fires its release slots
//! (semaphore writes or reductions) and may signal a dependent QMD.
//!
//! Ported from Lichii's `video_core/native_qmd.h` (GPL-3.0-or-later).

use std::collections::HashMap;

/// A QMD: 64 words.
pub type Descriptor = [u32; 64];

/// Release slots (word offsets): address low, flags (address high in
/// bits 0..8, enable bit 23, reduction op 20..23, signed 24, reduce 26,
/// structure size 30..32), payload.
pub const RELEASE_SLOTS: [usize; 3] = [24, 28, 52];

/// Combines a semaphore word with a payload by reduction `op`.
pub fn reduce(op: u32, signed: bool, value: u32, payload: u32) -> u32 {
    match op {
        0 => value.wrapping_add(payload),
        1 if signed => (value as i32).min(payload as i32) as u32,
        1 => value.min(payload),
        2 if signed => (value as i32).max(payload as i32) as u32,
        2 => value.max(payload),
        3 => {
            if value >= payload {
                0
            } else {
                value + 1
            }
        }
        4 => {
            if value == 0 || value > payload {
                payload
            } else {
                value - 1
            }
        }
        5 => value & payload,
        6 => value | payload,
        _ => value ^ payload,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct State {
    dependencies: i32,
    deferred: i32,
    loaded: bool,
    scheduled: bool,
    completed: bool,
}

/// What the scheduler needs from the GPU.
pub trait QmdHost {
    /// Reads the QMD at a GPU address.
    fn read(&mut self, address: u64) -> Option<Descriptor>;
    /// Writes a release (`flags` as in the release slot).
    fn release(&mut self, address: u64, payload: u32, flags: u32) -> bool;
    /// Runs the QMD's shader grid; false if it could not.
    fn dispatch(&mut self, qmd: &Descriptor) -> bool;
}

#[derive(Default)]
pub struct Scheduler {
    states: HashMap<u64, State>,
}

/// Whether a QMD launches a grid (non-zero dimensions).
pub fn has_grid(q: &Descriptor) -> bool {
    q[12] != 0 && q[13] & 0xFFFF != 0 && q[14] & 0xFFFF != 0
}

fn decodable_release(flags: u32) -> bool {
    let size = flags >> 30;
    if flags & (1 << 29) != 0 {
        return false;
    }
    if flags & (1 << 26) != 0 {
        return size == 1 && (flags >> 24) & 3 < 2;
    }
    size <= 1
}

impl Scheduler {
    /// PCAS action `argument` (bits 0..4: 2 schedule, 3 load and
    /// schedule, 7 decrement, 12 increment by bits 14..24 + 1) on the QMD
    /// at `address`. Returns an error message, empty when all went well.
    pub fn signal(&mut self, address: u64, argument: u32, host: &mut dyn QmdHost) -> String {
        let mut error = String::new();
        let mut pending = vec![(address, argument & 15, ((argument >> 14) & 0x3FF) + 1)];
        let mut i = 0;
        while i < pending.len() {
            let (address, action, count) = pending[i];
            i += 1;
            if self.states.len() >= 4096 && !self.states.contains_key(&address) {
                self.states.retain(|_, s| !(s.completed && s.deferred == 0));
                if self.states.len() >= 4096 {
                    return "QMD cache limit".into();
                }
            }
            let state = self.states.entry(address).or_default();
            match action {
                2 | 3 => {
                    if action == 3 || !state.loaded {
                        let Some(q) = host.read(address) else {
                            return "unmapped QMD".into();
                        };
                        if q[18] & 0xFF != 0x30 || q[4] & 0x100 != 0 {
                            return format!(
                                "unsupported QMD version or queue (QMD {address:#x}: version word {:#x}, word 4 {:#x})",
                                q[18], q[4]
                            );
                        }
                        state.dependencies = ((q[4] >> 14) & 0xFFFF) as i32 + state.deferred;
                        state.deferred = 0;
                        state.loaded = true;
                        state.completed = false;
                    }
                    state.scheduled = true;
                }
                7 | 12 => {
                    let delta = if action == 7 {
                        -(count as i32)
                    } else {
                        count as i32
                    };
                    if !state.loaded || state.completed {
                        state.deferred += delta;
                        continue;
                    }
                    state.dependencies += delta;
                }
                0 | 1 => continue,
                _ => return format!("unsupported PCAS action {action}"),
            }
            if !state.loaded || !state.scheduled || state.completed || state.dependencies > 0 {
                continue;
            }
            if state.dependencies < 0 {
                return "QMD dependency underflow".into();
            }
            state.completed = true;
            let Some(q) = host.read(address) else {
                return "unmapped QMD".into();
            };
            if has_grid(&q) && !host.dispatch(&q) && error.is_empty() {
                error = "compute dispatch failed".into();
            }
            for &base in &RELEASE_SLOTS {
                let flags = q[base + 1];
                if flags & (1 << 23) == 0 {
                    continue;
                }
                if !decodable_release(flags) {
                    if error.is_empty() {
                        error = "unsupported QMD release format".into();
                    }
                    continue;
                }
                let destination = u64::from(flags & 0xFF) << 32 | u64::from(q[base]);
                if !host.release(destination, q[base + 2], flags) {
                    return "unmapped QMD release".into();
                }
            }
            if q[16] & 1 != 0 {
                let action = (q[16] >> 1) & 7;
                let next = match action {
                    4 => 7,
                    3 => 3,
                    1 => 2,
                    _ => return "unsupported dependent QMD action".into(),
                };
                if pending.len() == 256 {
                    return "QMD chain limit".into();
                }
                pending.push((u64::from(q[15]) << 8, next, 1));
            }
            if q[16] & (1 << 5) != 0 {
                return "second dependent QMD unsupported".into();
            }
        }
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Host {
        qmds: HashMap<u64, Descriptor>,
        releases: Vec<(u64, u32)>,
        dispatched: Vec<u64>,
    }

    impl QmdHost for Host {
        fn read(&mut self, address: u64) -> Option<Descriptor> {
            self.qmds.get(&address).copied()
        }
        fn release(&mut self, address: u64, payload: u32, _: u32) -> bool {
            self.releases.push((address, payload));
            true
        }
        fn dispatch(&mut self, q: &Descriptor) -> bool {
            self.dispatched.push(u64::from(q[48]));
            true
        }
    }

    fn qmd(deps: u32, program: u32) -> Descriptor {
        let mut q = [0; 64];
        q[18] = 0x30;
        q[4] = deps << 14;
        q[12] = 1;
        q[13] = 1;
        q[14] = 1;
        q[48] = program;
        q
    }

    #[test]
    fn dependencies_and_releases() {
        let mut a = qmd(1, 0xA);
        a[24] = 0x1000;
        a[25] = 1 << 23 | 1 << 30; // one-word release
        a[26] = 7;
        let mut host = Host {
            qmds: HashMap::from([(0x100, a)]),
            releases: Vec::new(),
            dispatched: Vec::new(),
        };
        let mut s = Scheduler::default();
        // Load and schedule: one dependency left, nothing runs.
        assert_eq!(s.signal(0x100, 3, &mut host), "");
        assert!(host.dispatched.is_empty());
        // Decrement: it runs and releases.
        assert_eq!(s.signal(0x100, 7, &mut host), "");
        assert_eq!(host.dispatched, [0xA]);
        assert_eq!(host.releases, [(0x1000, 7)]);
    }

    #[test]
    fn reductions() {
        assert_eq!(reduce(0, false, 5, 3), 8);
        assert_eq!(reduce(2, false, 5, 3), 5);
        assert_eq!(reduce(1, true, 5, (-3i32) as u32), (-3i32) as u32);
        assert_eq!(reduce(3, false, 3, 3), 0);
    }
}
