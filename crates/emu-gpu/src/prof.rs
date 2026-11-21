//! Host-time profile of the GPU side, shown in the status line, and the
//! reasons draws leave the Vulkan path.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Profiled parts.
#[derive(Clone, Copy)]
pub enum Part {
    /// Command processing (methods, macros, draws as a whole).
    Submit = 0,
    /// Vulkan draws (state, upload, recording).
    VkDraw,
    /// Vulkan flushes (wait for the GPU, write images back).
    VkFlush,
    /// Texture lookups, hashing and decoding.
    Textures,
    /// Draws rendered on the CPU.
    Software,
    /// Compute grids run on the CPU.
    Compute,
    /// Presenting (reading and converting the image).
    Present,
}

const NAMES: [&str; 7] = [
    "submit",
    "vk draw",
    "vk flush",
    "textures",
    "software draw",
    "cpu compute",
    "present",
];
static NANOS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static COUNTS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static FALLBACKS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

/// Runs `f`, adding its time to `part`.
pub fn time<R>(part: Part, f: impl FnOnce() -> R) -> R {
    let t = Instant::now();
    let r = f();
    NANOS[part as usize].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    COUNTS[part as usize].fetch_add(1, Ordering::Relaxed);
    r
}

/// Counts a draw that left the Vulkan path; prints each new reason once.
pub fn fallback(why: &str) {
    let Ok(mut m) = FALLBACKS.lock() else { return };
    let n = m.entry(why.to_string()).or_insert(0);
    if *n == 0 {
        eprintln!("[gpu] draw on the CPU (slow): {why}");
    }
    *n += 1;
}

/// Times since the last call, as "part ms (count)", and the fallbacks
/// so far; resets the times.
pub fn report() -> String {
    let mut parts = Vec::new();
    for (i, name) in NAMES.iter().enumerate() {
        let ns = NANOS[i].swap(0, Ordering::Relaxed);
        let n = COUNTS[i].swap(0, Ordering::Relaxed);
        if n != 0 {
            parts.push(format!("{name} {}ms/{n}", ns / 1_000_000));
        }
    }
    let mut s = parts.join(", ");
    let Ok(m) = FALLBACKS.lock() else { return s };
    if !m.is_empty() {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        s += "; CPU draws: ";
        s += &v
            .iter()
            .take(4)
            .map(|(k, n)| format!("{n}x {k}"))
            .collect::<Vec<_>>()
            .join(", ");
    }
    s
}

/// Adds the time until it is dropped to a part.
pub struct Guard(Part, Instant);

impl Guard {
    pub fn new(part: Part) -> Guard {
        Guard(part, Instant::now())
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        NANOS[self.0 as usize].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTS[self.0 as usize].fetch_add(1, Ordering::Relaxed);
    }
}
