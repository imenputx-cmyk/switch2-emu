//! GPU debugging support: a trace of every command the GPU runs (methods
//! written from command buffers and from macros, macro calls, optionally
//! every macro step), kept in a ring for the debugger and dumps and
//! streamed to a file for `--gpu-trace`.

use std::collections::{HashSet, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::regnames;

/// What the host asked for, read when the GPU is created.
#[derive(Default, Clone)]
pub struct Config {
    /// `--gpu-trace=PATH`: stream the trace to a file.
    pub trace_path: Option<PathBuf>,
    /// Most lines streamed (the file stops there).
    pub trace_limit: usize,
    /// `--gpu-trace-macro=1,7`: macros to trace step by step.
    pub macros: Vec<usize>,
    /// Calls traced per macro.
    pub macro_calls: usize,
    /// Keep the ring (the interactive `gpu trace` and dumps) even with no file.
    pub ring: bool,
}

pub static CONFIG: Mutex<Option<Config>> = Mutex::new(None);

/// Where a method came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Pushbuffer,
    /// Written by this macro (3D class index, or `index | 0x1000` for compute).
    Macro(usize),
}

#[derive(Debug, Clone)]
pub enum Event {
    Method {
        channel: usize,
        sub: usize,
        class: u32,
        /// Byte offset.
        offset: u32,
        value: u32,
        source: Source,
    },
    Call {
        channel: usize,
        index: usize,
        compute: bool,
        start: u32,
        params: Vec<u32>,
    },
    Step(String),
    Note(String),
}

const RING: usize = 16384;

#[derive(Default)]
pub struct Trace {
    pub on: bool,
    pub ring: VecDeque<(u64, Event)>,
    pub seq: u64,
    file: Option<std::io::BufWriter<std::fs::File>>,
    lines_left: usize,
    /// Macros to trace step by step, and calls left to trace per macro.
    pub step_macros: HashSet<usize>,
    pub step_calls: std::collections::HashMap<usize, usize>,
    pub stepping: bool,
    /// Channel and subchannel of the method being run (for macro writes).
    pub context: (usize, usize),
    /// Calls and methods counted per macro index, for the interactive view.
    pub macro_counts: std::collections::BTreeMap<usize, u64>,
    /// How often each register was written, by (class is compute, offset).
    pub writes: std::collections::HashMap<(bool, u32), u64>,
}

impl Trace {
    pub fn from_host() -> Trace {
        let cfg = CONFIG.lock().ok().and_then(|c| c.clone());
        let Some(cfg) = cfg else {
            return Trace::default();
        };
        let mut t = Trace {
            on: cfg.ring || cfg.trace_path.is_some() || !cfg.macros.is_empty(),
            lines_left: cfg.trace_limit,
            step_macros: cfg.macros.iter().copied().collect(),
            ..Default::default()
        };
        for &m in &cfg.macros {
            t.step_calls.insert(m, cfg.macro_calls.max(1));
        }
        if let Some(p) = &cfg.trace_path {
            match std::fs::File::create(p) {
                Ok(f) => t.file = Some(std::io::BufWriter::new(f)),
                Err(e) => eprintln!("[gpu] cannot create trace {}: {e}", p.display()),
            }
        }
        t
    }

    pub fn push(&mut self, ev: Event) {
        if !self.on {
            return;
        }
        self.seq += 1;
        if let Event::Method { class, offset, .. } = &ev {
            *self
                .writes
                .entry((is_compute(*class), *offset))
                .or_default() += 1;
        }
        if let Event::Call { index, compute, .. } = &ev {
            *self
                .macro_counts
                .entry(*index | if *compute { 0x1000 } else { 0 })
                .or_default() += 1;
        }
        if self.seq.is_multiple_of(2048) {
            self.flush();
        }
        if self.lines_left > 0 {
            if let Some(f) = self.file.as_mut() {
                let _ = writeln!(f, "{}", format(self.seq, &ev));
                self.lines_left -= 1;
                if self.lines_left == 0 {
                    let _ = writeln!(f, "-- trace limit reached");
                    let _ = f.flush();
                }
            }
        }
        if self.ring.len() == RING {
            self.ring.pop_front();
        }
        self.ring.push_back((self.seq, ev));
    }

    pub fn note(&mut self, text: impl Into<String>) {
        self.push(Event::Note(text.into()));
    }

    /// A macro call starts: decides whether its steps are traced.
    pub fn begin_call(&mut self, index: usize) {
        self.stepping = false;
        if let Some(left) = self.step_calls.get_mut(&index) {
            if *left > 0 {
                *left -= 1;
                self.stepping = true;
            }
        }
    }

    pub fn flush(&mut self) {
        if let Some(f) = self.file.as_mut() {
            let _ = f.flush();
        }
    }
}

fn is_compute(class: u32) -> bool {
    class == crate::class::AMPERE_COMPUTE
}

/// One event as a line of text.
pub fn format(seq: u64, ev: &Event) -> String {
    match ev {
        Event::Method {
            channel,
            sub,
            class,
            offset,
            value,
            source,
        } => {
            let name = regnames::name(is_compute(*class), *offset);
            let from = match source {
                Source::Pushbuffer => String::new(),
                Source::Macro(m) if *m & 0x1000 != 0 => format!("  <- compute macro {}", m & 0xFFF),
                Source::Macro(m) => format!("  <- macro {m}"),
            };
            format!(
                "#{seq} ch{channel} sc{sub} {class:#06x} {offset:#06x} {name} = {value:#x}{from}"
            )
        }
        Event::Call {
            channel,
            index,
            compute,
            start,
            params,
        } => {
            let shown: Vec<String> = params.iter().take(32).map(|p| format!("{p:#x}")).collect();
            format!(
                "#{seq} ch{channel} CALL {}macro {index} at {start:#x} ({} params): [{}{}]",
                if *compute { "compute " } else { "" },
                params.len(),
                shown.join(", "),
                if params.len() > 32 { ", ..." } else { "" }
            )
        }
        Event::Step(s) => format!("#{seq}     {s}"),
        Event::Note(s) => format!("#{seq} -- {s}"),
    }
}

pub const COMMANDS: &str = "\
GPU commands (type them after `gpu`):
  status                 channels, classes, macro and trace state, recent faults
  on | off               turn the trace ring on or off
  trace [N] [TEXT]       last N traced events (default 40), only those containing TEXT
  step mN [CALLS]        trace macro N step by step for the next CALLS calls (3D; cN for compute)
  calls [N]              last N macro calls with parameters
  macros                 macro start table: first instruction and calls seen
  dis mN|PC [COUNT] [c]  disassemble macro N or from instruction PC (c: compute RAM)
  regs [TEXT]            nonzero 3D registers (by name filter)
  reg NAME|OFFSET        one 3D register: value and times written
  writes [N]             most written registers
  dram [START] [N]       MME data RAM words
  faults [N]             last N faults
  mem CH ADDR [LEN]      hex dump of GPU virtual memory on a channel
  dump FILE              write the full GPU dump now
";

fn num(s: &str) -> Option<u64> {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

impl crate::Gpu {
    /// The first 3D engine's registers, if a channel has one.
    fn first_3d(&self) -> Option<&crate::engines::maxwell3d::Maxwell3D> {
        self.channels
            .iter()
            .flat_map(|c| c.subchannels.iter().flatten())
            .find_map(|e| match e {
                crate::Engine::ThreeD(e) => Some(&**e),
                _ => None,
            })
    }

    /// Runs one debugger command (without the leading `gpu`).
    pub fn debug_command(&mut self, mem: &emu_memory::Memory, args: &[&str]) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let arg = |i: usize| args.get(i).copied();
        match arg(0).unwrap_or("status") {
            "help" | "?" => out += COMMANDS,
            "status" => {
                let t = &self.mme.trace;
                let _ = writeln!(
                    out,
                    "trace {} ({} events seen, ring {}), step-traced macros {:?}",
                    if t.on { "on" } else { "off" },
                    t.seq,
                    t.ring.len(),
                    t.step_macros
                );
                for (c, ch) in self.channels.iter().enumerate() {
                    let classes: Vec<String> = ch
                        .subchannels
                        .iter()
                        .enumerate()
                        .filter_map(|(i, e)| {
                            e.as_ref().map(|e| format!("sc{i}={:#06x}", e.class()))
                        })
                        .collect();
                    let _ = writeln!(
                        out,
                        "channel {c}: {} {}",
                        if ch.ampere { "Switch 2" } else { "classic" },
                        classes.join(" ")
                    );
                    if let Some(crate::Engine::ThreeD(e)) = ch.subchannels[0].as_ref() {
                        let _ = writeln!(out, "  3D macro calls {}", e.macro_calls);
                    }
                }
                let _ = writeln!(
                    out,
                    "macro RAM: {} instructions, {} starts bound; compute RAM {}",
                    self.mme
                        .code
                        .chunks(3)
                        .filter(|w| w.iter().any(|&x| x != 0))
                        .count(),
                    self.mme.starts.iter().filter(|&&s| s != 0).count(),
                    if self.mme.compute.is_some() {
                        "separate"
                    } else {
                        "shared with 3D"
                    }
                );
                let _ = writeln!(out, "{} faults; last:", self.faults.len());
                for f in self.faults.iter().rev().take(5).rev() {
                    let _ = writeln!(out, "  {}", f.lines().next().unwrap_or(""));
                }
            }
            "on" => {
                self.mme.trace.on = true;
                out += "trace on";
            }
            "off" => {
                self.mme.trace.on = false;
                out += "trace off";
            }
            "trace" => {
                let n = arg(1).and_then(num).unwrap_or(40) as usize;
                let filter = args.get(2..).map(|a| a.join(" ")).unwrap_or_default();
                if !self.mme.trace.on {
                    out += "(trace is off: `gpu on`, or start with --gpu-trace)\n";
                }
                let lines: Vec<String> = self
                    .mme
                    .trace
                    .ring
                    .iter()
                    .map(|(seq, ev)| format(*seq, ev))
                    .filter(|l| filter.is_empty() || l.contains(&filter))
                    .collect();
                for l in &lines[lines.len().saturating_sub(n)..] {
                    out += l;
                    out.push('\n');
                }
            }
            "step" => {
                let which = arg(1).unwrap_or("");
                let (compute, idx) = match which.strip_prefix('c') {
                    Some(r) => (true, r.parse::<usize>().ok()),
                    None => (
                        false,
                        which
                            .strip_prefix('m')
                            .and_then(|r| r.parse::<usize>().ok()),
                    ),
                };
                match idx {
                    Some(i) => {
                        let key = i | if compute { 0x1000 } else { 0 };
                        let calls = arg(2).and_then(num).unwrap_or(3) as usize;
                        self.mme.trace.on = true;
                        self.mme.trace.step_macros.insert(key);
                        self.mme.trace.step_calls.insert(key, calls);
                        let _ = write!(
                            out,
                            "tracing the next {calls} calls of {which} (see `gpu trace`)"
                        );
                    }
                    None => out += "usage: gpu step mN [CALLS]   (cN for compute)",
                }
            }
            "calls" => {
                let n = arg(1).and_then(num).unwrap_or(20) as usize;
                let calls: Vec<_> = self.mme.calls.iter().collect();
                for (i, st, p) in &calls[calls.len().saturating_sub(n)..] {
                    let shown: Vec<String> = p.iter().take(16).map(|x| format!("{x:#x}")).collect();
                    let _ = writeln!(
                        out,
                        "{}macro {} at {st:#x}: {} params [{}{}]",
                        if i & 0x1000 != 0 { "compute " } else { "" },
                        i & 0xFFF,
                        p.len(),
                        shown.join(", "),
                        if p.len() > 16 { ", .." } else { "" }
                    );
                }
            }
            "macros" => {
                for (i, &st) in self.mme.starts.iter().enumerate() {
                    if st == 0 && i != 0 {
                        continue;
                    }
                    let calls = self.mme.trace.macro_counts.get(&i).copied().unwrap_or(0);
                    let _ = writeln!(
                        out,
                        "m{i:<3} start {st:#06x}  calls {calls:<7} {}",
                        crate::engines::mme64::disassemble(&self.mme.code, st)
                    );
                }
                if let Some(c) = &self.mme.compute {
                    for (i, &st) in c.starts.iter().enumerate() {
                        if st == 0 && i != 0 {
                            continue;
                        }
                        let calls = self
                            .mme
                            .trace
                            .macro_counts
                            .get(&(i | 0x1000))
                            .copied()
                            .unwrap_or(0);
                        let _ = writeln!(
                            out,
                            "c{i:<3} start {st:#06x}  calls {calls:<7} {}",
                            crate::engines::mme64::disassemble(&c.code, st)
                        );
                    }
                }
            }
            "dis" => {
                let which = arg(1).unwrap_or("");
                let compute = args.contains(&"c") || which.starts_with('c');
                let ram = match (compute, &self.mme.compute) {
                    (true, Some(c)) => &**c,
                    _ => &self.mme,
                };
                let (start, label) = if let Some(i) = which
                    .strip_prefix(['m', 'c'])
                    .and_then(|r| r.parse::<usize>().ok())
                {
                    (
                        ram.starts.get(i).copied().unwrap_or(0),
                        format!("macro {i}"),
                    )
                } else {
                    (num(which).unwrap_or(0) as u32, "instruction".to_string())
                };
                let count = args.get(2).and_then(|a| num(a)).unwrap_or(24).min(512) as u32;
                let _ = writeln!(out, "{label} at {start:#x}:");
                for pc in start..start + count {
                    let _ = writeln!(
                        out,
                        "{pc:#06x}: {}",
                        crate::engines::mme64::disassemble(&ram.code, pc)
                    );
                }
            }
            "regs" => {
                let filter = arg(1).unwrap_or("");
                match self.first_3d() {
                    Some(e) => {
                        for (i, &v) in e.regs().iter().enumerate() {
                            if v == 0 {
                                continue;
                            }
                            let name = crate::regnames::name(false, i as u32 * 4);
                            if filter.is_empty() || name.contains(filter) {
                                let _ = writeln!(out, "{:#06x} {name} = {v:#x}", i * 4);
                            }
                        }
                    }
                    None => out += "no 3D engine yet",
                }
            }
            "reg" => {
                let Some(what) = arg(1) else {
                    return "usage: gpu reg NAME|OFFSET".into();
                };
                let offset = num(what)
                    .map(|o| o as u32)
                    .or_else(|| crate::regnames::find(false, what));
                match (offset, self.first_3d()) {
                    (Some(o), Some(e)) => {
                        let n = self.mme.trace.writes.get(&(false, o)).copied().unwrap_or(0);
                        let _ = write!(
                            out,
                            "{o:#06x} {} = {:#x}  (written {n} times{})",
                            crate::regnames::name(false, o),
                            e.reg(o),
                            if self.mme.trace.on { "" } else { "; trace off" }
                        );
                    }
                    (None, _) => out += "unknown register",
                    _ => out += "no 3D engine yet",
                }
            }
            "writes" => {
                let n = arg(1).and_then(num).unwrap_or(25) as usize;
                let mut w: Vec<_> = self.mme.trace.writes.iter().collect();
                w.sort_by_key(|&(_, &c)| std::cmp::Reverse(c));
                for (&(compute, o), &c) in w.into_iter().take(n) {
                    let _ = writeln!(
                        out,
                        "{c:>10}  {o:#06x} {}{}",
                        crate::regnames::name(compute, o),
                        if compute { " (compute)" } else { "" }
                    );
                }
            }
            "dram" => {
                let start = arg(1).and_then(num).unwrap_or(0) as usize;
                let n = arg(2).and_then(num).unwrap_or(32) as usize;
                for (c, ch) in self.channels.iter().enumerate() {
                    let _ = writeln!(out, "channel {c}:");
                    for i in start..(start + n).min(ch.data_ram.len()) {
                        let _ = writeln!(out, "  [{i:#05x}] = {:#010x}", ch.data_ram[i]);
                    }
                }
            }
            "faults" => {
                let n = arg(1).and_then(num).unwrap_or(10) as usize;
                for f in self.faults.iter().rev().take(n).rev() {
                    let _ = writeln!(out, "{}", f.lines().next().unwrap_or(""));
                }
            }
            "mem" => {
                let (Some(ch), Some(addr)) = (arg(1).and_then(num), arg(2).and_then(num)) else {
                    return "usage: gpu mem CH ADDR [LEN]".into();
                };
                let len = arg(3).and_then(num).unwrap_or(64).min(4096) as usize;
                let Some(asid) = self.channels.get(ch as usize).and_then(|c| c.address_space)
                else {
                    return "no such channel".into();
                };
                let mut buf = vec![0u8; len];
                match self.address_spaces[asid].read(mem, addr, &mut buf) {
                    Ok(()) => {
                        for (i, row) in buf.chunks(16).enumerate() {
                            let hex: Vec<String> = row.iter().map(|b| format!("{b:02x}")).collect();
                            let _ =
                                writeln!(out, "{:#014x}: {}", addr + i as u64 * 16, hex.join(" "));
                        }
                    }
                    Err(e) => {
                        let _ = write!(out, "unreadable: {e:?}");
                    }
                }
            }
            "dump" => match arg(1) {
                Some(path) => {
                    self.mme.failed = true;
                    let text = self.dump_text();
                    match std::fs::write(path, text) {
                        Ok(()) => {
                            let _ = write!(out, "wrote {path}");
                        }
                        Err(e) => {
                            let _ = write!(out, "cannot write {path}: {e}");
                        }
                    }
                }
                None => out += "usage: gpu dump FILE",
            },
            other => {
                let _ = writeln!(out, "unknown GPU command `{other}`\n{COMMANDS}");
            }
        }
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out
    }
}
