//! GPU dumps for debugging: the MME64 macro RAM (disassembled), the start
//! table and its upload history, data RAM, 3D registers, recent macro
//! calls and recent command buffers, written to a text file when a macro
//! first fails (`--gpu-dump=PATH`).

use std::collections::VecDeque;
use std::fmt::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::engines::mme64;
use crate::{Engine, Gpu};

/// Where to write a dump; set by the host before the GPU is created.
pub static DUMP_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Recent command buffers kept for a dump.
const ENTRIES: usize = 48;
/// Words kept of each.
const ENTRY_WORDS: usize = 1024;

pub struct Recorder {
    path: PathBuf,
    /// (channel, GPU address, words).
    entries: VecDeque<(usize, u64, Vec<u32>, usize)>,
    written: bool,
}

impl Recorder {
    pub fn new(path: PathBuf) -> Recorder {
        Recorder {
            path,
            entries: VecDeque::new(),
            written: false,
        }
    }

    /// A recorder when the host asked for dumps.
    pub fn from_host() -> Option<Recorder> {
        Some(Recorder::new(DUMP_PATH.lock().ok()?.clone()?))
    }

    pub fn entry(&mut self, channel: usize, va: u64, raw: &[u8]) {
        if self.written {
            return;
        }
        if self.entries.len() == ENTRIES {
            self.entries.pop_front();
        }
        let words = raw.len() / 4;
        let kept = raw
            .chunks_exact(4)
            .take(ENTRY_WORDS)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        self.entries.push_back((channel, va, kept, words));
    }
}

impl Gpu {
    /// Writes the dump once, after the first macro failure.
    pub(crate) fn maybe_dump(&mut self) {
        let Some(rec) = self.recorder.as_ref() else {
            return;
        };
        if rec.written || !self.mme.failed {
            return;
        }
        let text = self.dump_text();
        let rec = self.recorder.as_mut().unwrap();
        rec.written = true;
        match std::fs::write(&rec.path, text) {
            Ok(()) => self
                .faults
                .push(format!("GPU dump written to {}", rec.path.display())),
            Err(e) => self
                .faults
                .push(format!("cannot write GPU dump {}: {e}", rec.path.display())),
        }
    }

    pub(crate) fn dump_text(&mut self) -> String {
        self.mme.end_run();
        let mut s = String::from("# imenput-emu GPU dump\n\n## Faults so far\n");
        for f in self.faults.iter().rev().take(40).rev() {
            let _ = writeln!(s, "{}", f.lines().next().unwrap_or(""));
        }
        s += "\n## Macro starts (index: instruction)\n";
        for (i, &st) in self.mme.starts.iter().enumerate() {
            if st != 0 || i == 0 {
                let _ = writeln!(s, "{i}: {st:#x}");
            }
        }
        s += "\n## Macro upload history\n";
        for line in self.mme.log.iter().flatten() {
            let _ = writeln!(s, "{line}");
        }
        s += "\n## Macro RAM (nonzero instructions)\n";
        let n = self.mme.code.len() / 3;
        let starts: std::collections::BTreeMap<u32, Vec<usize>> = self
            .mme
            .starts
            .iter()
            .enumerate()
            .fold(Default::default(), |mut m, (i, &st)| {
                if st != 0 || i == 0 {
                    m.entry(st).or_default().push(i);
                }
                m
            });
        for pc in 0..n {
            let w = &self.mme.code[pc * 3..pc * 3 + 3];
            if w.iter().all(|&x| x == 0) {
                continue;
            }
            if let Some(ix) = starts.get(&(pc as u32)) {
                let _ = writeln!(s, "-- macro {ix:?}");
            }
            let _ = writeln!(
                s,
                "{pc:#06x}: {}",
                mme64::disassemble(&self.mme.code, pc as u32)
            );
        }
        if let Some(c) = &self.mme.compute {
            s += "\n## Compute macro RAM (own RAM after compute uploaded code)\n";
            for (i, &st) in c.starts.iter().enumerate() {
                if st != 0 || i == 0 {
                    let _ = writeln!(s, "start[{i}] = {st:#x}");
                }
            }
            for pc in 0..c.code.len() / 3 {
                if c.code[pc * 3..pc * 3 + 3].iter().any(|&x| x != 0) {
                    let _ = writeln!(s, "{pc:#06x}: {}", mme64::disassemble(&c.code, pc as u32));
                }
            }
        }
        s += "\n## Recent macro calls (index, start, parameters; index | 0x1000 = compute)\n";
        for (i, st, p) in &self.mme.calls {
            let _ = writeln!(s, "macro {i:#x} at {st:#x}: {p:x?}");
        }
        for (c, ch) in self.channels.iter().enumerate() {
            let _ = writeln!(s, "\n## Channel {c}: data RAM (nonzero)");
            for (i, &v) in ch.data_ram.iter().enumerate() {
                if v != 0 {
                    let _ = writeln!(s, "{i:#05x}: {v:#x}");
                }
            }
            for (sc, e) in ch.subchannels.iter().enumerate() {
                if let Some(e) = e {
                    let _ = writeln!(s, "subchannel {sc}: class {:#06x}", e.class());
                }
                if let Some(Engine::ThreeD(e)) = e {
                    let _ = writeln!(
                        s,
                        "## Channel {c} subchannel {sc}: 3D registers (nonzero, byte offset)"
                    );
                    for (i, &v) in e.regs().iter().enumerate() {
                        if v != 0 {
                            let _ = writeln!(
                                s,
                                "{:#06x} {} = {v:#x}",
                                i * 4,
                                crate::regnames::name(false, i as u32 * 4)
                            );
                        }
                    }
                }
            }
        }
        if self.mme.trace.on {
            s += "\n## Trace tail (last 400 events; macro steps included when traced)\n";
            let ring = &self.mme.trace.ring;
            for (seq, ev) in ring.iter().skip(ring.len().saturating_sub(400)) {
                let _ = writeln!(s, "{}", crate::debug::format(*seq, ev));
            }
            s += "\n## Most written registers\n";
            let mut w: Vec<_> = self.mme.trace.writes.iter().collect();
            w.sort_by_key(|&(_, &c)| std::cmp::Reverse(c));
            for (&(compute, o), &c) in w.into_iter().take(40) {
                let _ = writeln!(
                    s,
                    "{c:>10}  {o:#06x} {}{}",
                    crate::regnames::name(compute, o),
                    if compute { " (compute)" } else { "" }
                );
            }
        }
        s += "\n## Recent command buffers (channel, GPU address, words)\n";
        if let Some(rec) = &self.recorder {
            for (c, va, w, total) in &rec.entries {
                let _ = writeln!(
                    s,
                    "channel {c} at {va:#x}, {total} words{}:",
                    if *total > w.len() { " (cut)" } else { "" }
                );
                for chunk in w.chunks(8) {
                    s += " ";
                    for x in chunk {
                        let _ = write!(s, " {x:08x}");
                    }
                    s.push('\n');
                }
            }
        }
        s
    }
}
