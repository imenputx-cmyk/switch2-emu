//! A GDB remote stub: lets GDB, LLDB, IDA or Ghidra debug the guest over
//! TCP (`target remote 127.0.0.1:PORT`). Supports registers (AArch64:
//! X0-X30, SP, PC, CPSR, V0-V31, FPSR, FPCR), memory reads and writes,
//! software breakpoints, single steps, continue, interrupting with Ctrl+C
//! and the guest's threads.
//!
//! While breakpoints are set or a step is pending the emulator runs on
//! the interpreter one instruction at a time; otherwise at full speed.

use emu_horizon::{Horizon, Stop};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};

/// Instructions run between checks for a Ctrl+C from the debugger.
const CHUNK: u64 = 20_000_000;

const TARGET_XML: &str = r#"<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target version="1.0">
  <architecture>aarch64</architecture>
  <feature name="org.gnu.gdb.aarch64.core">
    <reg name="x0" bitsize="64"/><reg name="x1" bitsize="64"/><reg name="x2" bitsize="64"/>
    <reg name="x3" bitsize="64"/><reg name="x4" bitsize="64"/><reg name="x5" bitsize="64"/>
    <reg name="x6" bitsize="64"/><reg name="x7" bitsize="64"/><reg name="x8" bitsize="64"/>
    <reg name="x9" bitsize="64"/><reg name="x10" bitsize="64"/><reg name="x11" bitsize="64"/>
    <reg name="x12" bitsize="64"/><reg name="x13" bitsize="64"/><reg name="x14" bitsize="64"/>
    <reg name="x15" bitsize="64"/><reg name="x16" bitsize="64"/><reg name="x17" bitsize="64"/>
    <reg name="x18" bitsize="64"/><reg name="x19" bitsize="64"/><reg name="x20" bitsize="64"/>
    <reg name="x21" bitsize="64"/><reg name="x22" bitsize="64"/><reg name="x23" bitsize="64"/>
    <reg name="x24" bitsize="64"/><reg name="x25" bitsize="64"/><reg name="x26" bitsize="64"/>
    <reg name="x27" bitsize="64"/><reg name="x28" bitsize="64"/><reg name="x29" bitsize="64"/>
    <reg name="x30" bitsize="64"/>
    <reg name="sp" bitsize="64" type="data_ptr"/>
    <reg name="pc" bitsize="64" type="code_ptr"/>
    <reg name="cpsr" bitsize="32"/>
  </feature>
  <feature name="org.gnu.gdb.aarch64.fpu">
    <reg name="v0" bitsize="128" type="uint128"/><reg name="v1" bitsize="128" type="uint128"/>
    <reg name="v2" bitsize="128" type="uint128"/><reg name="v3" bitsize="128" type="uint128"/>
    <reg name="v4" bitsize="128" type="uint128"/><reg name="v5" bitsize="128" type="uint128"/>
    <reg name="v6" bitsize="128" type="uint128"/><reg name="v7" bitsize="128" type="uint128"/>
    <reg name="v8" bitsize="128" type="uint128"/><reg name="v9" bitsize="128" type="uint128"/>
    <reg name="v10" bitsize="128" type="uint128"/><reg name="v11" bitsize="128" type="uint128"/>
    <reg name="v12" bitsize="128" type="uint128"/><reg name="v13" bitsize="128" type="uint128"/>
    <reg name="v14" bitsize="128" type="uint128"/><reg name="v15" bitsize="128" type="uint128"/>
    <reg name="v16" bitsize="128" type="uint128"/><reg name="v17" bitsize="128" type="uint128"/>
    <reg name="v18" bitsize="128" type="uint128"/><reg name="v19" bitsize="128" type="uint128"/>
    <reg name="v20" bitsize="128" type="uint128"/><reg name="v21" bitsize="128" type="uint128"/>
    <reg name="v22" bitsize="128" type="uint128"/><reg name="v23" bitsize="128" type="uint128"/>
    <reg name="v24" bitsize="128" type="uint128"/><reg name="v25" bitsize="128" type="uint128"/>
    <reg name="v26" bitsize="128" type="uint128"/><reg name="v27" bitsize="128" type="uint128"/>
    <reg name="v28" bitsize="128" type="uint128"/><reg name="v29" bitsize="128" type="uint128"/>
    <reg name="v30" bitsize="128" type="uint128"/><reg name="v31" bitsize="128" type="uint128"/>
    <reg name="fpsr" bitsize="32"/>
    <reg name="fpcr" bitsize="32"/>
  </feature>
</target>
"#;

/// Register numbers in the target description.
const REG_SP: usize = 31;
const REG_PC: usize = 32;
const REG_CPSR: usize = 33;
const REG_V0: usize = 34;
const REG_FPSR: usize = 66;
const REG_FPCR: usize = 67;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

fn checksum(data: &[u8]) -> u8 {
    data.iter().fold(0u8, |a, &b| a.wrapping_add(b))
}

/// The packet layer over one connection.
struct Conn {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl Conn {
    fn send(&mut self, data: &str) -> std::io::Result<()> {
        let packet = format!("${data}#{:02x}", checksum(data.as_bytes()));
        self.stream.write_all(packet.as_bytes())
    }

    /// The next packet (or `"\x03"` for an interrupt); `None` when the
    /// connection closed.
    fn recv(&mut self) -> Option<String> {
        loop {
            if let Some(p) = self.take_packet() {
                return Some(p);
            }
            let mut chunk = [0u8; 4096];
            match self.stream.read(&mut chunk) {
                Ok(0) => return None,
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(_) => return None,
            }
        }
    }

    fn take_packet(&mut self) -> Option<String> {
        // Skip acks; an interrupt byte stands alone.
        while let Some(&b) = self.buf.first() {
            match b {
                b'+' | b'-' => {
                    self.buf.remove(0);
                }
                0x03 => {
                    self.buf.remove(0);
                    return Some("\x03".into());
                }
                b'$' => break,
                _ => {
                    self.buf.remove(0);
                }
            }
        }
        let end = self.buf.iter().position(|&b| b == b'#')?;
        if self.buf.len() < end + 3 {
            return None;
        }
        let body = self.buf[1..end].to_vec();
        self.buf.drain(..end + 3);
        let _ = self.stream.write_all(b"+");
        Some(String::from_utf8_lossy(&body).into_owned())
    }

    /// Whether the debugger sent an interrupt (non-blocking).
    fn interrupted(&mut self) -> bool {
        if self.stream.set_nonblocking(true).is_err() {
            return false;
        }
        let mut chunk = [0u8; 256];
        let got = match self.stream.read(&mut chunk) {
            Ok(n) => {
                self.buf.extend_from_slice(&chunk[..n]);
                true
            }
            Err(_) => false,
        };
        let _ = self.stream.set_nonblocking(false);
        if got {
            if let Some(i) = self.buf.iter().position(|&b| b == 0x03) {
                self.buf.remove(i);
                return true;
            }
        }
        false
    }
}

struct Debugger<'h> {
    hos: &'h mut Horizon,
    /// Thread whose registers `g`/`p` show (index into `hos.threads`).
    selected: usize,
}

impl Debugger<'_> {
    fn cpu(&self) -> &emu_cpu::Cpu {
        if self.selected == self.hos.current || self.selected >= self.hos.threads.len() {
            &self.hos.cpu
        } else {
            &self.hos.threads[self.selected].cpu
        }
    }

    fn cpu_mut(&mut self) -> &mut emu_cpu::Cpu {
        if self.selected == self.hos.current || self.selected >= self.hos.threads.len() {
            &mut self.hos.cpu
        } else {
            &mut self.hos.threads[self.selected].cpu
        }
    }

    fn reg(&self, n: usize) -> Option<Vec<u8>> {
        let c = self.cpu();
        Some(match n {
            0..=30 => c.read_xzr(n as u8).to_le_bytes().to_vec(),
            REG_SP => c.read_sp(31).to_le_bytes().to_vec(),
            REG_PC => c.pc().to_le_bytes().to_vec(),
            REG_CPSR => (c.nzcv.to_bits() as u32).to_le_bytes().to_vec(),
            REG_V0..=65 => c.read_v((n - REG_V0) as u8).to_le_bytes().to_vec(),
            REG_FPSR => (c.fpsr as u32).to_le_bytes().to_vec(),
            REG_FPCR => (c.fpcr as u32).to_le_bytes().to_vec(),
            _ => return None,
        })
    }

    fn set_reg(&mut self, n: usize, v: &[u8]) -> bool {
        let mut b = [0u8; 16];
        let len = v.len().min(16);
        b[..len].copy_from_slice(&v[..len]);
        let u64v = u64::from_le_bytes(b[..8].try_into().unwrap());
        let c = self.cpu_mut();
        match n {
            0..=30 => c.write_xzr(n as u8, u64v),
            REG_SP => c.write_sp(31, u64v),
            REG_PC => c.set_pc(u64v),
            REG_CPSR => c.nzcv = emu_cpu::Nzcv::from_bits(u64v & 0xFFFF_FFFF),
            REG_V0..=65 => c.write_v((n - REG_V0) as u8, u128::from_le_bytes(b)),
            REG_FPSR => c.fpsr = u64v & 0xFFFF_FFFF,
            REG_FPCR => c.fpcr = u64v & 0xFFFF_FFFF,
            _ => return false,
        }
        true
    }

    fn all_regs(&self) -> String {
        (0..=REG_FPCR)
            .filter_map(|n| self.reg(n))
            .map(|r| hex(&r))
            .collect()
    }

    fn stop_reply(&self, signal: u8, breakpoint: bool) -> String {
        let tid = self.hos.current + 1;
        let pc = hex(&self.hos.cpu.pc().to_le_bytes());
        let kind = if breakpoint { "swbreak:;" } else { "" };
        format!("T{signal:02x}{REG_PC:02x}:{pc};thread:{tid:x};{kind}")
    }

    /// Handles a packet that does not resume execution; `None` for the
    /// resuming ones.
    fn handle(&mut self, p: &str) -> Option<String> {
        let reply = match p.as_bytes().first()? {
            b'?' => self.stop_reply(5, false),
            b'g' => self.all_regs(),
            b'G' => {
                let bytes = unhex(&p[1..]).unwrap_or_default();
                let mut at = 0;
                for n in 0..=REG_FPCR {
                    let size = match n {
                        REG_CPSR | REG_FPSR | REG_FPCR => 4,
                        REG_V0..=65 => 16,
                        _ => 8,
                    };
                    if let Some(v) = bytes.get(at..at + size) {
                        self.set_reg(n, v);
                    }
                    at += size;
                }
                "OK".into()
            }
            b'p' => match usize::from_str_radix(&p[1..], 16)
                .ok()
                .and_then(|n| self.reg(n))
            {
                Some(v) => hex(&v),
                None => "E01".into(),
            },
            b'P' => {
                let (n, v) = p[1..].split_once('=')?;
                let ok = usize::from_str_radix(n, 16)
                    .ok()
                    .zip(unhex(v))
                    .is_some_and(|(n, v)| self.set_reg(n, &v));
                if ok { "OK" } else { "E01" }.into()
            }
            b'm' => {
                let (a, l) = p[1..].split_once(',')?;
                let (a, l) = (
                    u64::from_str_radix(a, 16).ok()?,
                    usize::from_str_radix(l, 16).ok()?,
                );
                let mut buf = vec![0u8; l.min(0x4000)];
                match self.hos.mem.read(a, &mut buf) {
                    Ok(()) => hex(&buf),
                    Err(_) => "E14".into(),
                }
            }
            b'M' => {
                let (head, data) = p[1..].split_once(':')?;
                let (a, _) = head.split_once(',')?;
                let a = u64::from_str_radix(a, 16).ok()?;
                match unhex(data).map(|d| self.hos.mem.write_privileged(a, &d)) {
                    Some(Ok(())) => {
                        self.hos.invalidate_all_code();
                        "OK".into()
                    }
                    _ => "E14".into(),
                }
            }
            b'Z' | b'z' => {
                let mut parts = p[1..].split(',');
                let kind = parts.next()?;
                let addr = u64::from_str_radix(parts.next()?, 16).ok()?;
                if kind == "0" || kind == "1" {
                    if p.starts_with('Z') {
                        self.hos.breakpoints.insert(addr);
                    } else {
                        self.hos.breakpoints.remove(&addr);
                    }
                    "OK".into()
                } else {
                    String::new() // watchpoints: not supported
                }
            }
            b'H' => {
                // Hg<tid>: registers of that thread; Hc: ignored (all run).
                if let Some(t) = p.strip_prefix("Hg") {
                    if let Ok(t) = i64::from_str_radix(t, 16) {
                        if t > 0 {
                            self.selected = (t - 1) as usize;
                        } else {
                            self.selected = self.hos.current;
                        }
                    }
                }
                "OK".into()
            }
            b'T' => {
                let alive = usize::from_str_radix(&p[1..], 16)
                    .is_ok_and(|t| t >= 1 && t <= self.hos.threads.len());
                if alive { "OK" } else { "E01" }.into()
            }
            b'q' => self.query(p),
            b'v' => {
                if p == "vCont?" {
                    "vCont;c;C;s;S".into()
                } else if p.starts_with("vCont") {
                    return None;
                } else {
                    String::new()
                }
            }
            b'c' | b's' | b'C' | b'S' => return None,
            _ => String::new(),
        };
        Some(reply)
    }

    fn query(&self, p: &str) -> String {
        if p.starts_with("qSupported") {
            return "PacketSize=4000;qXfer:features:read+;swbreak+;vContSupported+".into();
        }
        if let Some(rest) = p.strip_prefix("qXfer:features:read:target.xml:") {
            let Some((off, len)) = rest.split_once(',') else {
                return "E01".into();
            };
            let off = usize::from_str_radix(off, 16).unwrap_or(0);
            let len = usize::from_str_radix(len, 16).unwrap_or(0);
            let xml = TARGET_XML.as_bytes();
            if off >= xml.len() {
                return "l".into();
            }
            let end = (off + len).min(xml.len());
            let more = if end < xml.len() { 'm' } else { 'l' };
            return format!("{more}{}", String::from_utf8_lossy(&xml[off..end]));
        }
        match p {
            "qAttached" => "1".into(),
            "qC" => format!("QC{:x}", self.hos.current + 1),
            "qfThreadInfo" => {
                let ids: Vec<String> = (1..=self.hos.threads.len().max(1))
                    .map(|t| format!("{t:x}"))
                    .collect();
                format!("m{}", ids.join(","))
            }
            "qsThreadInfo" => "l".into(),
            _ => {
                if let Some(t) = p.strip_prefix("qThreadExtraInfo,") {
                    let t = usize::from_str_radix(t, 16).unwrap_or(1).saturating_sub(1);
                    let text = match self.hos.threads.get(t) {
                        Some(th) => format!("tid {} prio {} {:?}", th.tid, th.priority, th.state),
                        None => "thread".into(),
                    };
                    return hex(text.as_bytes());
                }
                String::new()
            }
        }
    }
}

/// Waits for a debugger on `port`, then runs the guest under its control
/// until it detaches, kills or the program stops. Returns how it ended.
pub fn serve(hos: &mut Horizon, port: u16, max_steps: u64) -> Stop {
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[gdb] cannot listen on port {port}: {e}");
            return hos.run(max_steps);
        }
    };
    eprintln!("[gdb] waiting for a debugger: target remote 127.0.0.1:{port}");
    let stream = match listener.accept() {
        Ok((s, addr)) => {
            eprintln!("[gdb] debugger connected from {addr}");
            s
        }
        Err(e) => {
            eprintln!("[gdb] accept failed: {e}");
            return hos.run(max_steps);
        }
    };
    let _ = stream.set_nodelay(true);
    let mut conn = Conn {
        stream,
        buf: Vec::new(),
    };
    let start = hos.cpu.counter;
    let mut dbg = Debugger {
        selected: hos.current,
        hos,
    };
    loop {
        let Some(p) = conn.recv() else {
            eprintln!("[gdb] debugger disconnected; running on");
            dbg.hos.breakpoints.clear();
            let left = max_steps.saturating_sub(dbg.hos.cpu.counter - start);
            return dbg.hos.run(left);
        };
        if p == "\x03" {
            let r = dbg.stop_reply(2, false);
            let _ = conn.send(&r);
            continue;
        }
        if p == "k" {
            return Stop::Interrupted;
        }
        if p.starts_with('D') {
            let _ = conn.send("OK");
            dbg.hos.breakpoints.clear();
            let left = max_steps.saturating_sub(dbg.hos.cpu.counter - start);
            return dbg.hos.run(left);
        }
        if let Some(reply) = dbg.handle(&p) {
            if conn.send(&reply).is_err() {
                return Stop::Interrupted;
            }
            continue;
        }
        // Resume: step or continue.
        let step = p.starts_with('s') || p.starts_with('S') || p.contains(";s") || p.contains(";S");
        dbg.hos.single_step = step;
        let pc = dbg.hos.cpu.pc();
        dbg.hos.skip_breakpoint_once = dbg.hos.breakpoints.contains(&pc);
        dbg.selected = dbg.hos.current;
        let reply = loop {
            let left = max_steps.saturating_sub(dbg.hos.cpu.counter - start);
            if left == 0 {
                return Stop::StepLimit;
            }
            match dbg.hos.run(left.min(CHUNK)) {
                Stop::StepLimit => {
                    if conn.interrupted() {
                        dbg.hos.single_step = false;
                        break dbg.stop_reply(2, false);
                    }
                }
                Stop::Breakpoint { .. } => break dbg.stop_reply(5, true),
                Stop::Stepped { .. } => break dbg.stop_reply(5, false),
                Stop::Interrupted => break dbg.stop_reply(2, false),
                other => {
                    // The program stopped for good: report it, let the
                    // debugger look around, then end.
                    eprintln!("[gdb] program stopped: {other:?}");
                    let r = match other {
                        Stop::ExitProcess | Stop::ReturnedToLoader(_) => "W00".to_string(),
                        _ => dbg.stop_reply(11, false),
                    };
                    let _ = conn.send(&r);
                    if r.starts_with('W') {
                        return other;
                    }
                    // Keep serving inspection requests until detach/kill.
                    loop {
                        let Some(p) = conn.recv() else { return other };
                        if p == "k" || p.starts_with('D') {
                            let _ = conn.send("OK");
                            return other;
                        }
                        let reply = dbg.handle(&p).unwrap_or_else(|| dbg.stop_reply(11, false));
                        let _ = conn.send(&reply);
                    }
                }
            }
        };
        dbg.selected = dbg.hos.current;
        if conn.send(&reply).is_err() {
            return Stop::Interrupted;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_helpers() {
        assert_eq!(hex(&[0xAB, 0x01]), "ab01");
        assert_eq!(unhex("ab01"), Some(vec![0xAB, 0x01]));
        assert_eq!(unhex("abc"), None);
        assert_eq!(checksum(b"OK"), 0x9A);
        assert!(TARGET_XML.contains("org.gnu.gdb.aarch64.core"));
    }
}
