//! The Switch 2 compute class (Ampere, C7C0) as NVN2 uses it: MME64
//! macros, inline uploads, and QMD launches scheduled through PCAS
//! actions (see [`super::qmd`]). Grids run on the shader interpreter, one
//! workgroup at a time, with shared memory and barriers.
//!
//! Method layout from Lichii's `kepler_compute.cpp` native path
//! (GPL-3.0-or-later) and NVIDIA's open-gpu-doc class headers (MIT).

use super::inline::InlineToMemory;
use super::mme64::{self, Mme64Target};
use super::qmd::{self, Descriptor, QmdHost, Scheduler};
use super::EngineCtx;
use crate::backend::GuestMemory;
use crate::shader::decode::{is_complete, sr, Inst, IpaMode};
use crate::shader::interp::{resume, ShaderIo, Status, Thread};
use std::collections::HashMap;

pub mod reg {
    pub const MME_INSTRUCTION_RAM_POINTER: u32 = 0x114;
    pub const MME_INSTRUCTION_RAM: u32 = 0x118;
    pub const MME_START_ADDRESS_RAM_POINTER: u32 = 0x11C;
    pub const MME_START_ADDRESS_RAM: u32 = 0x120;
    /// QMD address >> 8.
    pub const QMD_ADDRESS: u32 = 0x2B4;
    pub const SEND_PCAS_B: u32 = 0x2B8;
    pub const SEND_SIGNALING_PCAS_B: u32 = 0x2BC;
    pub const SEND_SIGNALING_PCAS2_B: u32 = 0x2C0;
    pub const MACRO_CALLS: u32 = 0x3800;
}

const REGS: usize = 0x1000;

/// Most invocations a dispatch may run on the interpreter; larger grids
/// are skipped (their releases still fire).
pub const MAX_INVOCATIONS: u64 = 1 << 21;

pub struct Compute {
    regs: Vec<u32>,
    inline: InlineToMemory,
    data_ram: Vec<u32>,
    pending_call: Option<(usize, Vec<u32>)>,
    scheduler: Scheduler,
    programs: HashMap<u64, Vec<Inst>>,
    /// Failed programs: first bytes (to retry on change) and the error.
    failed_programs: HashMap<u64, ([u8; 256], String)>,
    /// Grids run, and grids skipped (too large or failed to decode).
    pub dispatches: u64,
    pub skipped: u64,
    /// Grids the backend ran on the GPU.
    pub gpu_dispatches: u64,
    /// Programs the backend could not run (interpreted from then on).
    gpu_failed: std::collections::HashSet<u64>,
}

impl Default for Compute {
    fn default() -> Self {
        Compute {
            regs: vec![0; REGS],
            inline: InlineToMemory::default(),
            data_ram: vec![0; 0xC00],
            pending_call: None,
            scheduler: Scheduler::default(),
            programs: HashMap::new(),
            failed_programs: HashMap::new(),
            dispatches: 0,
            skipped: 0,
            gpu_dispatches: 0,
            gpu_failed: Default::default(),
        }
    }
}

struct Target<'e, 'c, 'm> {
    engine: &'e mut Compute,
    ctx: &'c mut EngineCtx<'m>,
}

impl Mme64Target for Target<'_, '_, '_> {
    fn read(&mut self, method: u32) -> u32 {
        self.engine.regs.get(method as usize).copied().unwrap_or(0)
    }

    fn emit(&mut self, method: u32, value: u32, _fifo: &mut Vec<u32>) -> bool {
        if method * 4 >= reg::MACRO_CALLS {
            return false;
        }
        self.engine.write_method(method, value, self.ctx);
        true
    }

    fn data_ram(&mut self) -> &mut [u32] {
        &mut self.engine.data_ram
    }
}

impl Compute {
    pub fn reg(&self, offset: u32) -> u32 {
        self.regs.get(offset as usize / 4).copied().unwrap_or(0)
    }

    /// A method write from a pushbuffer (`last`: the end of a method run).
    pub fn write(&mut self, method: u32, value: u32, last: bool, ctx: &mut EngineCtx) -> bool {
        if method * 4 < reg::MACRO_CALLS {
            return self.write_method(method, value, ctx);
        }
        let index = ((method - reg::MACRO_CALLS / 4) / 2) as usize;
        if method.is_multiple_of(2) {
            self.pending_call = Some((index, vec![value]));
        } else if let Some((_, params)) = self.pending_call.as_mut() {
            params.push(value);
        }
        if last {
            if let Some((index, mut params)) = self.pending_call.take() {
                let Some(&start) = ctx.mme.starts.get(index) else {
                    ctx.faults
                        .push(format!("compute macro {index} out of range"));
                    return true;
                };
                let code = std::mem::take(&mut ctx.mme.code);
                let mut target = Target { engine: self, ctx };
                let result = mme64::execute(&code, start, &mut params, &mut target);
                let Target { ctx, .. } = target;
                if let Err(e) = result {
                    ctx.faults.push(format!(
                        "compute MME64 macro {index}: {} (pc {:#x})",
                        e.message, e.pc
                    ));
                }
                ctx.mme.code = code;
            }
        }
        true
    }

    fn write_method(&mut self, method: u32, value: u32, ctx: &mut EngineCtx) -> bool {
        let offset = method * 4;
        if (0x180..=0x1B4).contains(&offset) {
            return self.inline.write(method, value, ctx);
        }
        if let Some(r) = self.regs.get_mut(method as usize) {
            *r = value;
        }
        match offset {
            reg::MME_INSTRUCTION_RAM_POINTER => ctx.mme.set_pointer(value),
            reg::MME_INSTRUCTION_RAM => ctx.mme.upload(value),
            reg::MME_START_ADDRESS_RAM_POINTER => ctx.mme.set_start_pointer(value),
            reg::MME_START_ADDRESS_RAM => ctx.mme.bind(value),
            reg::SEND_PCAS_B | reg::SEND_SIGNALING_PCAS_B | reg::SEND_SIGNALING_PCAS2_B => {
                let action = value & 15;
                let skip = offset == reg::SEND_PCAS_B
                    || (offset == reg::SEND_SIGNALING_PCAS_B && value & 2 == 0)
                    || (offset == reg::SEND_SIGNALING_PCAS2_B && action <= 1);
                if !skip {
                    let address = u64::from(self.reg(reg::QMD_ADDRESS)) << 8;
                    let mut scheduler = std::mem::take(&mut self.scheduler);
                    let error = {
                        let mut host = Host {
                            engine: self,
                            ctx: &mut *ctx,
                        };
                        scheduler.signal(address, value, &mut host)
                    };
                    self.scheduler = scheduler;
                    if !error.is_empty() {
                        ctx.faults
                            .push(format!("QMD {address:#x} action {value:#x}: {error}"));
                    }
                }
            }
            _ => {}
        }
        true
    }

    /// The decoded compute program at `address` (SM86, no header).
    fn program(&mut self, address: u64, g: &GuestMemory) -> Result<Vec<Inst>, String> {
        if let Some(p) = self.programs.get(&address) {
            return Ok(p.clone());
        }
        let mut head = [0u8; 256];
        let _ = g.read_prefix(address, &mut head);
        if let Some((h, e)) = self.failed_programs.get(&address) {
            if *h == head {
                return Err(e.clone());
            }
        }
        let r = self.decode_program_at(address, g);
        if let Err(e) = &r {
            self.failed_programs.insert(address, (head, e.clone()));
        }
        r
    }

    fn decode_program_at(&mut self, address: u64, g: &GuestMemory) -> Result<Vec<Inst>, String> {
        let mut len = 0x400usize;
        loop {
            let mut code = vec![0u8; len];
            let got = g
                .read_prefix(address, &mut code)
                .map_err(|f| format!("compute program at {address:#x}: {f:?}"))?;
            let insts =
                match crate::shader::sm86::decode_program_ext(&code, got < len || len >= 1 << 20) {
                    Err(crate::shader::decode::DecodeError::Truncated) if got == len => Vec::new(),
                    r => r.map_err(|e| format!("compute program at {address:#x}: {e:x?}"))?,
                };
            if is_complete(&insts) {
                self.programs.insert(address, insts.clone());
                return Ok(insts);
            }
            if len >= 1 << 20 || got < len {
                return Err(format!("compute program at {address:#x}: no EXIT"));
            }
            len *= 4;
        }
    }

    /// Runs a QMD's grid.
    fn dispatch(&mut self, q: &Descriptor, ctx: &mut EngineCtx) -> bool {
        let grid = [q[12], q[13] & 0xFFFF, q[14] & 0xFFFF];
        let block = [q[18] >> 16, q[19] & 0xFFFF, q[19] >> 16];
        let invocations = grid
            .iter()
            .chain(&block)
            .map(|&v| u64::from(v))
            .product::<u64>();
        if invocations == 0 {
            return true;
        }
        let address = u64::from(q[48]) | u64::from(q[49] & 0x1_FFFF) << 32;
        let program = match self.program(address, &ctx.guest) {
            Ok(p) => p,
            Err(e) => {
                self.skipped += 1;
                ctx.faults.push(e);
                return false;
            }
        };
        let mut cbufs = vec![Vec::new(); 8];
        for (slot, cbuf) in cbufs.iter_mut().enumerate() {
            if q[20] >> slot & 1 == 0 {
                continue;
            }
            let high = q[33 + 2 * slot];
            let va = u64::from(q[32 + 2 * slot]) | u64::from(high & 0xFF) << 32;
            let size = (((high >> 19) & 0x1FFF) << 4) as usize;
            let mut data = vec![0u8; size];
            if ctx.guest.read(va, &mut data).is_ok() {
                *cbuf = data;
            }
        }
        // The backend first (Vulkan compute); the interpreter otherwise.
        if !self.gpu_failed.contains(&address) {
            let d = crate::backend::Dispatch {
                address,
                program: &program,
                grid,
                block,
                shared_bytes: q[17] & 0x3_FFFF,
                cbufs: &cbufs,
            };
            match ctx.backend.dispatch(&mut ctx.guest, &d) {
                Ok(true) => {
                    self.dispatches += 1;
                    self.gpu_dispatches += 1;
                    return true;
                }
                Ok(false) => {}
                Err(e) => {
                    self.gpu_failed.insert(address);
                    ctx.faults
                        .push(format!("compute on the GPU: {e}; using the interpreter"));
                }
            }
        }
        if invocations > MAX_INVOCATIONS {
            self.skipped += 1;
            if self.skipped <= 4 {
                ctx.faults.push(format!(
                    "compute grid {grid:?} x {block:?} too large for the interpreter; skipped"
                ));
            }
            return true;
        }
        let shared_size = (q[17] & 0x3_FFFF) as usize;
        let threads = (block[0] * block[1] * block[2]) as usize;
        for gz in 0..grid[2] {
            for gy in 0..grid[1] {
                for gx in 0..grid[0] {
                    let mut io = ComputeIo {
                        mem: &mut ctx.guest,
                        cbufs: &cbufs,
                        shared: vec![0; shared_size],
                        tid: [0; 3],
                        ctaid: [gx, gy, gz],
                    };
                    let mut state: Vec<Option<Thread>> =
                        (0..threads).map(|_| Some(Thread::new())).collect();
                    loop {
                        let mut waiting = false;
                        for (t, slot) in state.iter_mut().enumerate() {
                            let Some(thread) = slot else { continue };
                            let t = t as u32;
                            io.tid = [
                                t % block[0],
                                t / block[0] % block[1],
                                t / (block[0] * block[1]),
                            ];
                            match resume(&program, thread, &mut io) {
                                Status::Barrier => waiting = true,
                                Status::Exited | Status::Killed => *slot = None,
                            }
                        }
                        if !waiting {
                            break;
                        }
                    }
                }
            }
        }
        self.dispatches += 1;
        true
    }
}

struct Host<'e, 'c, 'm> {
    engine: &'e mut Compute,
    ctx: &'c mut EngineCtx<'m>,
}

impl QmdHost for Host<'_, '_, '_> {
    fn read(&mut self, address: u64) -> Option<Descriptor> {
        let mut bytes = [0u8; 256];
        self.ctx.guest.read(address, &mut bytes).ok()?;
        Some(std::array::from_fn(|i| {
            u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
        }))
    }

    fn release(&mut self, address: u64, payload: u32, flags: u32) -> bool {
        let g = &mut self.ctx.guest;
        let mut old = [0u8; 4];
        let result = if flags & (1 << 26) != 0 {
            if g.read(address, &mut old).is_err() {
                return false;
            }
            let v = qmd::reduce(
                (flags >> 20) & 7,
                flags >> 24 & 1 != 0,
                u32::from_le_bytes(old),
                payload,
            );
            g.write(address, &v.to_le_bytes())
        } else if flags >> 30 == 0 {
            // Four words: the payload, zero, then a timestamp.
            let mut data = [0u8; 16];
            data[..4].copy_from_slice(&payload.to_le_bytes());
            g.write(address, &data)
        } else if flags & (1 << 19) != 0 {
            // A max release (NVN2 tracks QMD completion this way).
            if g.read(address, &mut old).is_err() {
                return false;
            }
            let v = u32::from_le_bytes(old).max(payload);
            g.write(address, &v.to_le_bytes())
        } else {
            g.write(address, &payload.to_le_bytes())
        };
        result.is_ok()
    }

    fn dispatch(&mut self, q: &Descriptor) -> bool {
        self.engine.dispatch(q, self.ctx)
    }
}

struct ComputeIo<'a, 'g> {
    mem: &'a mut GuestMemory<'g>,
    cbufs: &'a [Vec<u8>],
    shared: Vec<u8>,
    tid: [u32; 3],
    ctaid: [u32; 3],
}

impl ShaderIo for ComputeIo<'_, '_> {
    fn load_attr(&self, _: u32) -> u32 {
        0
    }
    fn store_attr(&mut self, _: u32, _: u32) {}
    fn interpolate(&self, _: u32, _: IpaMode) -> f32 {
        0.0
    }
    fn cbuf(&self, index: u8, offset: u32) -> u32 {
        self.cbufs
            .get(index as usize)
            .and_then(|b| b.get(offset as usize..offset as usize + 4))
            .map_or(0, |w| u32::from_le_bytes(w.try_into().unwrap()))
    }
    fn special(&self, r: u8) -> u32 {
        match r {
            sr::TID_X => self.tid[0],
            sr::TID_Y => self.tid[1],
            sr::TID_Z => self.tid[2],
            sr::CTAID_X => self.ctaid[0],
            sr::CTAID_Y => self.ctaid[1],
            sr::CTAID_Z => self.ctaid[2],
            // SR_TID packed: x | y << 16 | z << 26.
            32 => self.tid[0] | self.tid[1] << 16 | self.tid[2] << 26,
            _ => 0,
        }
    }
    fn load_global(&mut self, va: u64, buf: &mut [u8]) {
        if self.mem.read(va, buf).is_err() {
            buf.fill(0);
        }
    }
    fn store_global(&mut self, va: u64, data: &[u8]) {
        let _ = self.mem.write(va, data);
    }
    fn load_shared(&mut self, offset: u32, buf: &mut [u8]) {
        let at = offset as usize;
        for (k, b) in buf.iter_mut().enumerate() {
            *b = self.shared.get(at + k).copied().unwrap_or(0);
        }
    }
    fn store_shared(&mut self, offset: u32, data: &[u8]) {
        let at = offset as usize;
        if let Some(dst) = self.shared.get_mut(at..at + data.len()) {
            dst.copy_from_slice(data);
        }
    }
}
