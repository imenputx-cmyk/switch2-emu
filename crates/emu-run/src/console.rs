//! An interactive debugger in the terminal (`--debug`): when the program
//! stops (a breakpoint, a single step, Ctrl+C, or an error such as an
//! unimplemented service command), it opens a prompt to look around and
//! carry on. Type `help` at the prompt for the commands.
//!
//! Addresses can be written as hex (`0x8000123` or `8000123`), relative
//! to a module (`m2+0x17b128`: module 2 plus an offset), or as a register
//! of the selected thread (`pc`, `lr`, `sp`, `x0`..`x30`), optionally
//! with `+off` / `-off`.

use emu_horizon::{Horizon, Stop};
use std::io::{BufRead, Write};

const HELP: &str = "\
commands:
  c                  continue
  s [n]              step n instructions (default 1)
  b ADDR             add a breakpoint      bd ADDR  remove    bl  list
  r                  registers of the selected thread
  t [N]              list threads / select thread N
  bt                 backtrace (frame pointer chain)
  x ADDR [LEN]       hex dump (default 0x40 bytes)
  xs ADDR            C string at ADDR
  w4 ADDR VAL / w8 ADDR VAL   write 4 / 8 bytes
  set REG VAL        set a register (x0..x30, sp, pc) of the selected thread
  dis [ADDR] [N]     decode N instructions (default: 8 at pc)
  mem ADDR           memory map entry for ADDR
  mods               loaded modules
  ipc / svc          recent IPC commands / SVCs
  stubs              stubbed (auto-answered) IPC commands
  trace on|off       print every SVC
  dump               the full stop report
  q                  quit";

/// Runs the program under the console: stops are reported and handled at
/// the prompt. Returns how the program ended (or the stop it was quit at).
pub fn run(hos: &mut Horizon, max_steps: u64, break_at_start: bool) -> Stop {
    let start = hos.cpu.counter;
    let mut console = Console {
        selected: hos.current,
    };
    let mut stop = if break_at_start {
        eprintln!("[debug] stopped before the first instruction; type `help` for commands");
        Some(Stop::Interrupted)
    } else {
        None
    };
    loop {
        if let Some(s) = stop.take() {
            match &s {
                Stop::ExitProcess | Stop::ReturnedToLoader(_) | Stop::StepLimit => return s,
                _ => {}
            }
            console.report(hos, &s);
            match console.prompt(hos) {
                Action::Continue => {}
                Action::Step(n) => {
                    // Steps run on the interpreter one by one.
                    for _ in 0..n {
                        hos.single_step = true;
                        let pc = hos.cpu.pc();
                        hos.skip_breakpoint_once = hos.breakpoints.contains(&pc);
                        match hos.run(1_000_000) {
                            Stop::Stepped { .. } => {}
                            other => {
                                hos.single_step = false;
                                stop = Some(other);
                                break;
                            }
                        }
                    }
                    if stop.is_none() {
                        stop = Some(Stop::Stepped { pc: hos.cpu.pc() });
                    }
                    continue;
                }
                Action::Quit => return s,
            }
            // Clear a pending Ctrl+C before running on.
            if let Some(flag) = &hos.interrupt {
                flag.store(false, std::sync::atomic::Ordering::Relaxed);
            }
            let pc = hos.cpu.pc();
            hos.skip_breakpoint_once = hos.breakpoints.contains(&pc);
            console.selected = hos.current;
        }
        let left = max_steps.saturating_sub(hos.cpu.counter - start);
        if left == 0 {
            return Stop::StepLimit;
        }
        stop = Some(hos.run(left));
    }
}

enum Action {
    Continue,
    Step(u64),
    Quit,
}

struct Console {
    selected: usize,
}

fn parse_num(s: &str) -> Option<u64> {
    let s = s.trim();
    let s = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(s, 16).ok()
}

impl Console {
    fn cpu<'h>(&self, hos: &'h Horizon) -> &'h emu_cpu::Cpu {
        if self.selected == hos.current || self.selected >= hos.threads.len() {
            &hos.cpu
        } else {
            &hos.threads[self.selected].cpu
        }
    }

    fn cpu_mut<'h>(&self, hos: &'h mut Horizon) -> &'h mut emu_cpu::Cpu {
        if self.selected == hos.current || self.selected >= hos.threads.len() {
            &mut hos.cpu
        } else {
            &mut hos.threads[self.selected].cpu
        }
    }

    /// An address expression (see the module docs).
    fn addr(&self, hos: &Horizon, expr: &str) -> Option<u64> {
        let expr = expr.trim();
        let (head, off): (&str, i64) = match expr.rfind(['+', '-']) {
            Some(i) if i > 0 => {
                let n = parse_num(&expr[i + 1..])? as i64;
                (&expr[..i], if &expr[i..=i] == "-" { -n } else { n })
            }
            _ => (expr, 0),
        };
        let cpu = self.cpu(hos);
        let base = match head {
            "pc" => cpu.pc(),
            "lr" => cpu.read_xzr(30),
            "sp" => cpu.read_sp(31),
            h if h.starts_with('x') && h[1..].parse::<u8>().is_ok_and(|n| n <= 30) => {
                cpu.read_xzr(h[1..].parse().ok()?)
            }
            h if h.starts_with('m') && h[1..].parse::<usize>().is_ok() => {
                hos.modules.get(h[1..].parse::<usize>().ok()?)?.0
            }
            h => parse_num(h)?,
        };
        Some(base.wrapping_add(off as u64))
    }

    fn report(&mut self, hos: &Horizon, stop: &Stop) {
        self.selected = hos.current;
        let why = match stop {
            Stop::Breakpoint { .. } => "breakpoint".to_string(),
            Stop::Stepped { .. } => "step".to_string(),
            Stop::Interrupted => "interrupted".to_string(),
            other => format!("{other:?}"),
        };
        eprintln!(
            "[debug] {why}: thread {} at {}",
            hos.current,
            hos.locate(hos.cpu.pc())
        );
        if let Ok(w) = hos.mem.read_uint(hos.cpu.pc(), 4) {
            eprintln!(
                "        {:#x}: {:08x}  {:?}",
                hos.cpu.pc(),
                w,
                emu_cpu::decoder::decode(w as u32)
            );
        }
    }

    fn regs(&self, hos: &Horizon) {
        let c = self.cpu(hos);
        for row in 0..8 {
            let line: Vec<String> = (0..4)
                .map(|k| row * 4 + k)
                .filter(|&n| n < 31)
                .map(|n| format!("x{n:<2}={:#018x}", c.read_xzr(n as u8)))
                .collect();
            eprintln!("  {}", line.join("  "));
        }
        eprintln!(
            "  sp ={:#018x}  pc ={:#018x}  nzcv={:#x}",
            c.read_sp(31),
            c.pc(),
            c.nzcv.to_bits() >> 28
        );
        eprintln!(
            "  pc at {}, lr at {}",
            hos.locate(c.pc()),
            hos.locate(c.read_xzr(30))
        );
    }

    fn backtrace(&self, hos: &Horizon) {
        let c = self.cpu(hos);
        eprintln!("  #0  {}", hos.locate(c.pc()));
        eprintln!("  #1  {} (lr)", hos.locate(c.read_xzr(30)));
        let mut fp = c.read_xzr(29);
        for depth in 2..32 {
            if fp == 0 || !fp.is_multiple_of(8) {
                break;
            }
            let (Ok(next), Ok(ret)) = (hos.mem.read_u64(fp), hos.mem.read_u64(fp + 8)) else {
                break;
            };
            if ret == 0 {
                break;
            }
            eprintln!("  #{depth:<2} {}", hos.locate(ret));
            if next <= fp {
                break;
            }
            fp = next;
        }
    }

    fn hexdump(hos: &Horizon, addr: u64, len: u64) {
        let mut buf = vec![0u8; len.min(0x1000) as usize];
        if let Err(e) = hos.mem.read(addr, &mut buf) {
            eprintln!("  cannot read: {e}");
            return;
        }
        for (i, chunk) in buf.chunks(16).enumerate() {
            let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
            let text: String = chunk
                .iter()
                .map(|&b| {
                    if (0x20..0x7F).contains(&b) {
                        b as char
                    } else {
                        '.'
                    }
                })
                .collect();
            eprintln!(
                "  {:#014x}: {:<48} {text}",
                addr + 16 * i as u64,
                hex.join(" ")
            );
        }
    }

    fn prompt(&mut self, hos: &mut Horizon) -> Action {
        let stdin = std::io::stdin();
        loop {
            eprint!("(emu) ");
            let _ = std::io::stderr().flush();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                return Action::Quit;
            }
            let words: Vec<&str> = line.split_whitespace().collect();
            let Some(&cmd) = words.first() else { continue };
            let arg = |i: usize| words.get(i).copied();
            match cmd {
                "c" | "continue" => return Action::Continue,
                "s" | "step" => {
                    return Action::Step(arg(1).and_then(|n| n.parse().ok()).unwrap_or(1));
                }
                "q" | "quit" => return Action::Quit,
                "h" | "help" | "?" => eprintln!("{HELP}"),
                "b" | "bd" => match arg(1).and_then(|a| self.addr(hos, a)) {
                    Some(a) => {
                        if cmd == "b" {
                            hos.breakpoints.insert(a);
                            eprintln!("  breakpoint at {}", hos.locate(a));
                        } else {
                            hos.breakpoints.remove(&a);
                        }
                    }
                    None => eprintln!("  usage: {cmd} ADDR"),
                },
                "bl" => {
                    for &a in &hos.breakpoints {
                        eprintln!("  {}", hos.locate(a));
                    }
                }
                "r" | "regs" => self.regs(hos),
                "t" | "threads" => match arg(1).and_then(|n| n.parse::<usize>().ok()) {
                    Some(n) if n < hos.threads.len() => {
                        self.selected = n;
                        self.regs(hos);
                    }
                    Some(_) => eprintln!("  no such thread"),
                    None => {
                        for (i, t) in hos.threads.iter().enumerate() {
                            let cpu = if i == hos.current { &hos.cpu } else { &t.cpu };
                            let mark = if i == self.selected { '*' } else { ' ' };
                            eprintln!(
                                " {mark}{i:<3} tid {:<4} prio {:<3} {:?} at {}",
                                t.tid,
                                t.priority,
                                t.state,
                                hos.locate(cpu.pc())
                            );
                        }
                    }
                },
                "bt" => self.backtrace(hos),
                "x" => match arg(1).and_then(|a| self.addr(hos, a)) {
                    Some(a) => {
                        let len = arg(2).and_then(parse_num).unwrap_or(0x40);
                        Self::hexdump(hos, a, len);
                    }
                    None => eprintln!("  usage: x ADDR [LEN]"),
                },
                "xs" => match arg(1).and_then(|a| self.addr(hos, a)) {
                    Some(a) => {
                        let mut s = Vec::new();
                        for k in 0..0x400 {
                            match hos.mem.read_uint(a + k, 1) {
                                Ok(0) | Err(_) => break,
                                Ok(b) => s.push(b as u8),
                            }
                        }
                        eprintln!("  \"{}\"", String::from_utf8_lossy(&s));
                    }
                    None => eprintln!("  usage: xs ADDR"),
                },
                "w4" | "w8" => {
                    let a = arg(1).and_then(|a| self.addr(hos, a));
                    let v = arg(2).and_then(parse_num);
                    match (a, v) {
                        (Some(a), Some(v)) => {
                            let size = if cmd == "w4" { 4 } else { 8 };
                            let bytes = v.to_le_bytes();
                            match hos.mem.write_privileged(a, &bytes[..size]) {
                                Ok(()) => hos.invalidate_all_code(),
                                Err(e) => eprintln!("  cannot write: {e}"),
                            }
                        }
                        _ => eprintln!("  usage: {cmd} ADDR VALUE"),
                    }
                }
                "set" => {
                    let v = arg(2).and_then(parse_num);
                    match (arg(1), v) {
                        (Some(r), Some(v)) => {
                            let c = self.cpu_mut(hos);
                            match r {
                                "pc" => c.set_pc(v),
                                "sp" => c.write_sp(31, v),
                                r if r.starts_with('x')
                                    && r[1..].parse::<u8>().is_ok_and(|n| n <= 30) =>
                                {
                                    c.write_xzr(r[1..].parse().unwrap(), v)
                                }
                                _ => eprintln!("  unknown register {r}"),
                            }
                        }
                        _ => eprintln!("  usage: set REG VALUE"),
                    }
                }
                "dis" => {
                    let a = arg(1)
                        .and_then(|a| self.addr(hos, a))
                        .unwrap_or_else(|| self.cpu(hos).pc());
                    let n = arg(2).and_then(|n| n.parse().ok()).unwrap_or(8u64);
                    for k in 0..n.min(256) {
                        let at = a + 4 * k;
                        match hos.mem.read_uint(at, 4) {
                            Ok(w) => eprintln!(
                                "  {}{:#x}: {:08x}  {:?}",
                                if hos.breakpoints.contains(&at) {
                                    "*"
                                } else {
                                    " "
                                },
                                at,
                                w,
                                emu_cpu::decoder::decode(w as u32)
                            ),
                            Err(e) => {
                                eprintln!("  {at:#x}: {e}");
                                break;
                            }
                        }
                    }
                }
                "mem" => match arg(1).and_then(|a| self.addr(hos, a)) {
                    Some(a) => {
                        let m = hos.map.query(a);
                        eprintln!(
                            "  {:#x}..{:#x} {:?} perm {:#x} attr {:#x}",
                            m.base,
                            m.base + m.size,
                            m.state,
                            m.perm,
                            m.attr
                        );
                    }
                    None => eprintln!("  usage: mem ADDR"),
                },
                "mods" => {
                    for (i, (base, size)) in hos.modules.iter().enumerate() {
                        eprintln!("  m{i}: {base:#x}..{:#x}", base + size);
                    }
                }
                "ipc" => {
                    for (service, command, result) in &hos.recent_ipc {
                        eprintln!("  {service} command {command} -> {result:#x}");
                    }
                }
                "svc" => {
                    for (id, pc) in &hos.recent_svcs {
                        eprintln!("  svc {id:#x} at {}", hos.locate(*pc));
                    }
                }
                "stubs" => {
                    for ((service, command), n) in &hos.stubbed {
                        eprintln!("  {service} {command}: {n} calls");
                    }
                }
                "trace" => match arg(1) {
                    Some("on") => hos.trace_svcs = true,
                    Some("off") => hos.trace_svcs = false,
                    _ => eprintln!("  usage: trace on|off"),
                },
                "dump" => eprintln!("{}\n{}", hos.status(), hos.dump()),
                other => eprintln!("  unknown command {other:?}; type `help`"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(parse_num("0x10"), Some(16));
        assert_eq!(parse_num("ff"), Some(255));
        assert_eq!(parse_num("zz"), None);
    }

    #[test]
    fn address_expressions() {
        let mut hos = Horizon::new();
        hos.modules = vec![(0x8000000, 0x1000), (0xa000000, 0x2000)];
        hos.cpu.set_pc(0x1234);
        hos.cpu.write_xzr(3, 0x50);
        let c = Console { selected: 0 };
        assert_eq!(c.addr(&hos, "m1+0x10"), Some(0xa000010));
        assert_eq!(c.addr(&hos, "pc"), Some(0x1234));
        assert_eq!(c.addr(&hos, "pc-4"), Some(0x1230));
        assert_eq!(c.addr(&hos, "x3+8"), Some(0x58));
        assert_eq!(c.addr(&hos, "0x99"), Some(0x99));
    }
}
