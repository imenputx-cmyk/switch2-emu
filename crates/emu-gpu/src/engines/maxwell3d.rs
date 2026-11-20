//! The 3D engine (class B197, MAXWELL_B). Most methods only set state in
//! a register file; a few trigger work. Emulated so far: render target
//! clears, inline data uploads, syncpoint increments and report
//! semaphores, and storing macro (MME) code. Draws and macro calls are
//! reported as not emulated.

use super::inline::InlineToMemory;
use super::mme::{self, MacroTarget};
use super::{bits, EngineCtx};
use crate::backend::{Layout, Rect, Surface};
use crate::draw::{
    ConstBuffer, Draw, IndexBuffer, VertexAttribute, VertexStream, Viewport, CBUF_SLOTS,
};
use crate::shader::decode::{decode_program, Inst};
use crate::shader::{Header, Stage, HEADER_SIZE};
use crate::surface::ColorFormat;

/// Register file size in words: methods 0x0000..0x3800 (bytes).
const REGS: usize = 0xE00;

/// Register byte offsets.
mod reg {
    pub const MME_INSTRUCTION_RAM_POINTER: u32 = 0x114;
    pub const MME_INSTRUCTION_RAM: u32 = 0x118;
    pub const MME_START_ADDRESS_RAM_POINTER: u32 = 0x11C;
    pub const MME_START_ADDRESS_RAM: u32 = 0x120;
    pub const INCREMENT_SYNC_POINT: u32 = 0x2C8;
    pub const COLOR_TARGET: u32 = 0x800; // + 64 * j: A, B, WIDTH, HEIGHT, FORMAT, MEMORY
    pub const CLEAR_RECT_HORIZONTAL: u32 = 0xD6C;
    pub const CLEAR_RECT_VERTICAL: u32 = 0xD70;
    pub const COLOR_CLEAR_VALUE: u32 = 0xD80;
    pub const SCISSOR_ENABLE: u32 = 0xE00;
    pub const SCISSOR_HORIZONTAL: u32 = 0xE04;
    pub const SCISSOR_VERTICAL: u32 = 0xE08;
    pub const SURFACE_CLIP_HORIZONTAL: u32 = 0xFF4;
    pub const SURFACE_CLIP_VERTICAL: u32 = 0xFF8;
    pub const CLEAR_SURFACE: u32 = 0x19D0;
    pub const REPORT_SEMAPHORE_A: u32 = 0x1B00;
    pub const REPORT_SEMAPHORE_C: u32 = 0x1B08;
    pub const REPORT_SEMAPHORE_D: u32 = 0x1B0C;
    pub const MACRO_CALLS: u32 = 0x3800;
    pub const VIEWPORT_SCALE_X: u32 = 0xA00; // scale x, y, z, offset x, y, z
    pub const VERTEX_ARRAY_START: u32 = 0xD74;
    pub const DRAW_VERTEX_ARRAY: u32 = 0xD78;
    pub const VERTEX_ATTRIBUTE: u32 = 0x1160; // + 4 * i
    pub const DRAW_ARRAY_FIRST: u32 = 0x1214;
    pub const DRAW_ARRAY_SUBSEQUENT: u32 = 0x1218;
    pub const CT_SELECT: u32 = 0x121C;
    pub const PROGRAM_REGION_A: u32 = 0x1608;
    pub const END: u32 = 0x1614;
    pub const BEGIN: u32 = 0x1618;
    pub const VERTEX_STREAM: u32 = 0x1C00; // + 16 * j: format, location a, b
    pub const PIPELINE: u32 = 0x2000; // + 64 * j: shader, program
    pub const GLOBAL_BASE_VERTEX: u32 = 0x1434;
    pub const INDEX_BUFFER_A: u32 = 0x17C8; // A/B address, C/D limit, E size, F first
    pub const INDEX_BUFFER_E: u32 = 0x17D8;
    pub const INDEX_BUFFER_F: u32 = 0x17DC;
    pub const DRAW_INDEX_BUFFER: u32 = 0x17E0;
    pub const CBUF_SELECTOR: u32 = 0x2380; // size, address upper, lower
    pub const LOAD_CBUF_OFFSET: u32 = 0x238C;
    pub const LOAD_CBUF: u32 = 0x2390; // ..0x23CC
    pub const BIND_GROUP_CBUF: u32 = 0x2410; // + 32 * stage
}

/// Draws that are not emulated yet (indexed, auto, inline indices).
const DRAW_TRIGGERS: [u32; 5] = [0x304, 0x123C, 0x1304, 0x15E8, 0x15F0];

/// Pipeline slots: 0 vertex A, 1 vertex B, 2-4 tessellation/geometry,
/// 5 pixel.
const PIPELINE_SLOTS: usize = 6;

pub struct Maxwell3D {
    regs: Vec<u32>,
    inline: InlineToMemory,
    /// Macro (MME) code and entry points, as uploaded.
    pub mme_code: Vec<u32>,
    pub mme_starts: Vec<u32>,
    /// Render target clears performed (for diagnostics and tests).
    pub clears: u64,
    /// The macro being called: index and parameters so far.
    pending_call: Option<(usize, Vec<u32>)>,
    /// Macros run (for diagnostics and tests).
    pub macro_calls: u64,
    /// Topology from BEGIN and vertex/index runs queued until END.
    topology: u32,
    ranges: Vec<(u32, u32)>,
    index_ranges: Vec<(u32, u32)>,
    /// Constant buffers bound per stage group (0 VS, 1-3 tess/geometry,
    /// 4 PS) and slot.
    cbufs: [[Option<ConstBuffer>; CBUF_SLOTS]; 5],
    /// Decoded programs by GPU address.
    programs: std::collections::HashMap<u64, (Header, Vec<Inst>)>,
    /// Draws executed (for diagnostics and tests).
    pub draws: u64,
}

/// Lets a running macro write methods into the engine.
struct Target<'e, 'c, 'm> {
    engine: &'e mut Maxwell3D,
    ctx: &'c mut EngineCtx<'m>,
    /// Methods that were not emulated, reported after the macro.
    unhandled: Vec<(u32, u32)>,
}

impl MacroTarget for Target<'_, '_, '_> {
    fn send(&mut self, method: u32, value: u32) {
        if !self.engine.write_method(method, value, self.ctx) {
            self.unhandled.push((method * 4, value));
        }
    }

    fn read(&self, method: u32) -> u32 {
        self.engine.regs.get(method as usize).copied().unwrap_or(0)
    }
}

impl Default for Maxwell3D {
    fn default() -> Self {
        Maxwell3D {
            regs: vec![0; REGS],
            inline: InlineToMemory::default(),
            mme_code: vec![0; 0x2000],
            mme_starts: vec![0; 0x80],
            clears: 0,
            pending_call: None,
            macro_calls: 0,
            topology: 0,
            ranges: Vec::new(),
            index_ranges: Vec::new(),
            cbufs: [[None; CBUF_SLOTS]; 5],
            programs: Default::default(),
            draws: 0,
        }
    }
}

impl Maxwell3D {
    /// A register by byte offset.
    pub fn reg(&self, offset: u32) -> u32 {
        self.regs.get(offset as usize / 4).copied().unwrap_or(0)
    }

    /// A method write from a pushbuffer. `last` marks the final word of
    /// a method run, which ends a macro call's parameters. Returns false
    /// for methods that are not emulated; `unhandled` collects those that
    /// macros wrote.
    pub fn write(
        &mut self,
        method: u32,
        value: u32,
        last: bool,
        ctx: &mut EngineCtx,
        unhandled: &mut Vec<(u32, u32)>,
    ) -> bool {
        if method * 4 < reg::MACRO_CALLS {
            return self.write_method(method, value, ctx);
        }
        // CALL_MME_MACRO(j) (even words) starts a call with its first
        // parameter; CALL_MME_DATA(j) (odd) adds parameters.
        let index = ((method - reg::MACRO_CALLS / 4) / 2) as usize;
        if method.is_multiple_of(2) {
            self.pending_call = Some((index, vec![value]));
        } else if let Some((_, params)) = self.pending_call.as_mut() {
            params.push(value);
        }
        if last {
            if let Some((index, params)) = self.pending_call.take() {
                self.call_macro(index, &params, ctx, unhandled);
            }
        }
        true
    }

    fn call_macro(
        &mut self,
        index: usize,
        params: &[u32],
        ctx: &mut EngineCtx,
        unhandled: &mut Vec<(u32, u32)>,
    ) {
        let Some(&start) = self.mme_starts.get(index) else {
            ctx.faults.push(format!("macro {index} out of range"));
            return;
        };
        let code = std::mem::take(&mut self.mme_code);
        let mut target = Target {
            engine: self,
            ctx,
            unhandled: Vec::new(),
        };
        let result = mme::execute(&code, start, params, &mut target);
        let Target {
            unhandled: u, ctx, ..
        } = target;
        unhandled.extend(u);
        if let Err(e) = result {
            ctx.faults
                .push(format!("macro {index} at {start:#x}: {e:?}"));
        }
        self.mme_code = code;
        self.macro_calls += 1;
    }

    /// A write to a 3D method (not a macro call).
    fn write_method(&mut self, method: u32, value: u32, ctx: &mut EngineCtx) -> bool {
        let offset = method * 4;
        if offset >= reg::MACRO_CALLS {
            return false;
        }
        if (0x180..=0x1B4).contains(&offset) {
            return self.inline.write(method, value, ctx);
        }
        self.regs[method as usize] = value;
        match offset {
            reg::MME_INSTRUCTION_RAM => {
                let p = self.reg(reg::MME_INSTRUCTION_RAM_POINTER);
                if let Some(slot) = self.mme_code.get_mut(p as usize) {
                    *slot = value;
                }
                self.regs[reg::MME_INSTRUCTION_RAM_POINTER as usize / 4] = p.wrapping_add(1);
            }
            reg::MME_START_ADDRESS_RAM => {
                let p = self.reg(reg::MME_START_ADDRESS_RAM_POINTER);
                if let Some(slot) = self.mme_starts.get_mut(p as usize) {
                    *slot = value;
                }
                self.regs[reg::MME_START_ADDRESS_RAM_POINTER as usize / 4] = p.wrapping_add(1);
            }
            reg::INCREMENT_SYNC_POINT => ctx.increment_syncpoint(bits(value, 11, 0)),
            reg::CLEAR_SURFACE => self.clear(value, ctx),
            reg::REPORT_SEMAPHORE_D => self.report_semaphore(value, ctx),
            reg::BEGIN => {
                self.topology = value & 0xFFFF;
                self.ranges.clear();
                self.index_ranges.clear();
            }
            reg::DRAW_VERTEX_ARRAY => self.ranges.push((self.reg(reg::VERTEX_ARRAY_START), value)),
            reg::DRAW_INDEX_BUFFER => self
                .index_ranges
                .push((self.reg(reg::INDEX_BUFFER_F), value)),
            reg::END => {
                let ranges = std::mem::take(&mut self.ranges);
                if !ranges.is_empty() {
                    self.draw(self.topology, ranges, None, ctx);
                }
                let ranges = std::mem::take(&mut self.index_ranges);
                if !ranges.is_empty() {
                    let size = 1 << (self.reg(reg::INDEX_BUFFER_E) & 3);
                    self.draw(self.topology, ranges, Some(size), ctx);
                }
            }
            // DRAW_INDEX_BUFFER{32,16,8}_BEGIN_END_INSTANCE_{FIRST,SUBSEQUENT}
            0x17E4..=0x17F8 => {
                let size = [4, 2, 1][((offset - 0x17E4) / 4 % 3) as usize];
                self.draw(
                    value >> 28,
                    vec![(value & 0xFFFF, (value >> 16) & 0xFFF)],
                    Some(size),
                    ctx,
                );
            }
            reg::LOAD_CBUF..=0x23CC => {
                // Constant buffer update: written to memory at the
                // selected buffer + offset, which advances.
                let off = self.reg(reg::LOAD_CBUF_OFFSET);
                let addr = self.selected_cbuf().addr + u64::from(off);
                ctx.write(addr, &value.to_le_bytes());
                self.regs[reg::LOAD_CBUF_OFFSET as usize / 4] = off.wrapping_add(4);
            }
            o if (reg::BIND_GROUP_CBUF..reg::BIND_GROUP_CBUF + 5 * 32).contains(&o)
                && (o - reg::BIND_GROUP_CBUF).is_multiple_of(32) =>
            {
                let stage = ((o - reg::BIND_GROUP_CBUF) / 32) as usize;
                let slot = bits(value, 8, 4) as usize;
                if slot < CBUF_SLOTS {
                    self.cbufs[stage][slot] = (value & 1 != 0).then(|| self.selected_cbuf());
                }
            }
            reg::DRAW_ARRAY_FIRST | reg::DRAW_ARRAY_SUBSEQUENT => {
                self.draw(
                    value >> 28,
                    vec![(value & 0xFFFF, (value >> 16) & 0xFFF)],
                    None,
                    ctx,
                );
            }
            o if DRAW_TRIGGERS.contains(&o) => return false,
            _ => {}
        }
        true
    }

    /// The decoded program at `addr` (header and instructions).
    fn program(&mut self, addr: u64, ctx: &mut EngineCtx) -> Result<(Header, Vec<Inst>), String> {
        if let Some(p) = self.programs.get(&addr) {
            return Ok(p.clone());
        }
        // Read in growing chunks until the program's EXIT is decoded.
        let mut len = 0x400usize;
        loop {
            let mut code = vec![0u8; len];
            ctx.guest
                .read(addr, &mut code)
                .map_err(|f| format!("shader at {addr:#x}: {f:?}"))?;
            let insts = decode_program(&code[HEADER_SIZE..])
                .map_err(|e| format!("shader at {addr:#x}: {e:x?}"))?;
            if insts.last() == Some(&Inst::Exit) {
                let p = (Header::parse(&code[..HEADER_SIZE]), insts);
                self.programs.insert(addr, p.clone());
                return Ok(p);
            }
            if len >= 1 << 20 {
                return Err(format!("shader at {addr:#x}: no EXIT"));
            }
            len *= 4;
        }
    }

    /// The program in a pipeline slot, if enabled.
    fn pipeline(
        &mut self,
        slot: usize,
        ctx: &mut EngineCtx,
    ) -> Result<Option<(Header, Vec<Inst>)>, String> {
        let base = reg::PIPELINE + 64 * slot as u32;
        if self.reg(base) & 1 == 0 {
            return Ok(None);
        }
        let region = u64::from(self.reg(reg::PROGRAM_REGION_A) & 0xFF) << 32
            | u64::from(self.reg(reg::PROGRAM_REGION_A + 4));
        self.program(region + u64::from(self.reg(base + 4)), ctx)
            .map(Some)
    }

    /// Color target `index` as a surface.
    fn color_target(&self, index: u32) -> Option<Surface> {
        let rt = reg::COLOR_TARGET + 64 * index;
        let format = ColorFormat::from_raw(self.reg(rt + 0x10))?;
        let memory = self.reg(rt + 0x14);
        let pitch_layout = memory & (1 << 12) != 0;
        let bpp = format.bytes_per_pixel();
        let w = self.reg(rt + 8);
        Some(Surface {
            addr: u64::from(self.reg(rt) & 0xFF) << 32 | u64::from(self.reg(rt + 4)),
            width: if pitch_layout { w / bpp } else { w },
            height: self.reg(rt + 0xC),
            format,
            layout: if pitch_layout {
                Layout::Pitch { pitch: w }
            } else {
                Layout::BlockLinear {
                    block_height_log2: bits(memory, 7, 4),
                }
            },
        })
    }

    /// The constant buffer named by the selector registers.
    fn selected_cbuf(&self) -> ConstBuffer {
        ConstBuffer {
            addr: u64::from(self.reg(reg::CBUF_SELECTOR + 4) & 0xFF) << 32
                | u64::from(self.reg(reg::CBUF_SELECTOR + 8)),
            size: self.reg(reg::CBUF_SELECTOR) & 0x1_FFFF,
        }
    }

    /// `index_size`: bytes per index for indexed draws.
    fn draw(
        &mut self,
        topology: u32,
        ranges: Vec<(u32, u32)>,
        index_size: Option<u32>,
        ctx: &mut EngineCtx,
    ) {
        if let Err(e) = self.try_draw(topology, ranges, index_size, ctx) {
            ctx.faults.push(format!("draw: {e}"));
        }
    }

    fn try_draw(
        &mut self,
        topology: u32,
        ranges: Vec<(u32, u32)>,
        index_size: Option<u32>,
        ctx: &mut EngineCtx,
    ) -> Result<(), String> {
        let mut vs = None;
        let mut ps = None;
        for slot in 0..PIPELINE_SLOTS {
            if let Some((h, insts)) = self.pipeline(slot, ctx)? {
                match h.stage {
                    Stage::Vertex => vs = Some(insts),
                    Stage::Pixel => ps = Some((h, insts)),
                    Stage::Other(t) => return Err(format!("shader stage {t} not supported")),
                }
            }
        }
        let vs = vs.ok_or("no vertex shader")?;
        let (ps_header, ps) = ps.ok_or("no pixel shader")?;
        let ct = self.reg(reg::CT_SELECT);
        let target = if ct & 0xF == 0 {
            None
        } else {
            self.color_target(bits(ct, 6, 4))
        };
        let attributes = (0..32)
            .map(|i| {
                let v = self.reg(reg::VERTEX_ATTRIBUTE + 4 * i);
                VertexAttribute {
                    stream: bits(v, 4, 0),
                    constant: v & (1 << 6) != 0,
                    offset: bits(v, 20, 7),
                    widths: bits(v, 26, 21),
                    kind: bits(v, 29, 27),
                    swap_rb: v & (1 << 31) != 0,
                }
            })
            .collect();
        let streams = (0..32)
            .map(|j| {
                let base = reg::VERTEX_STREAM + 16 * j;
                let format = self.reg(base);
                VertexStream {
                    enabled: format & (1 << 12) != 0,
                    stride: format & 0xFFF,
                    addr: u64::from(self.reg(base + 4) & 0xFF) << 32
                        | u64::from(self.reg(base + 8)),
                }
            })
            .collect();
        let f = |o: u32| f32::from_bits(self.reg(o));
        let v = reg::VIEWPORT_SCALE_X;
        let viewport = Viewport {
            scale: [f(v), f(v + 4), f(v + 8)],
            offset: [f(v + 12), f(v + 16), f(v + 20)],
        };
        let (mut x0, mut y0, mut x1, mut y1) = (0, 0, u32::MAX, u32::MAX);
        let ch = self.reg(reg::SURFACE_CLIP_HORIZONTAL);
        let cv = self.reg(reg::SURFACE_CLIP_VERTICAL);
        if ch >> 16 != 0 && cv >> 16 != 0 {
            (x0, x1) = (ch & 0xFFFF, (ch & 0xFFFF) + (ch >> 16));
            (y0, y1) = (cv & 0xFFFF, (cv & 0xFFFF) + (cv >> 16));
        }
        if self.reg(reg::SCISSOR_ENABLE) & 1 != 0 {
            let (h, vv) = (
                self.reg(reg::SCISSOR_HORIZONTAL),
                self.reg(reg::SCISSOR_VERTICAL),
            );
            (x0, x1) = (x0.max(h & 0xFFFF), x1.min(h >> 16));
            (y0, y1) = (y0.max(vv & 0xFFFF), y1.min(vv >> 16));
        }
        let d = Draw {
            target,
            vs,
            ps,
            ps_header,
            topology,
            ranges,
            index: index_size.map(|size| IndexBuffer {
                addr: u64::from(self.reg(reg::INDEX_BUFFER_A) & 0xFF) << 32
                    | u64::from(self.reg(reg::INDEX_BUFFER_A + 4)),
                size,
                base_vertex: self.reg(reg::GLOBAL_BASE_VERTEX) as i32,
            }),
            vs_cbufs: self.cbufs[0],
            ps_cbufs: self.cbufs[4],
            attributes,
            streams,
            viewport,
            clip: Rect { x0, y0, x1, y1 },
        };
        ctx.backend.draw(&mut ctx.guest, &d)?;
        self.draws += 1;
        Ok(())
    }

    fn report_semaphore(&mut self, d: u32, ctx: &mut EngineCtx) {
        let operation = d & 3;
        if operation == 1 || operation == 3 {
            return; // acquire (always satisfied) / trap
        }
        let va = u64::from(self.reg(reg::REPORT_SEMAPHORE_A) & 0xFF) << 32
            | u64::from(self.reg(reg::REPORT_SEMAPHORE_A + 4));
        // Release writes the payload; a report writes the counter (none
        // are tracked yet, so 0).
        let value = if operation == 0 {
            self.reg(reg::REPORT_SEMAPHORE_C)
        } else {
            0
        };
        if d & (1 << 28) != 0 {
            ctx.write(va, &value.to_le_bytes());
        } else {
            let mut data = u64::from(value).to_le_bytes().to_vec();
            data.extend_from_slice(&0u64.to_le_bytes()); // timestamp
            ctx.write(va, &data);
        }
    }

    /// CLEAR_SURFACE: fills the selected render target with the clear
    /// color, limited by the surface clip, scissor 0 and the clear rect.
    fn clear(&mut self, v: u32, ctx: &mut EngineCtx) {
        let mask = [v & 4 != 0, v & 8 != 0, v & 16 != 0, v & 32 != 0];
        if !mask.iter().any(|&m| m) {
            return; // depth/stencil only: no depth buffers yet
        }
        let rt = reg::COLOR_TARGET + 64 * bits(v, 9, 6);
        let Some(format) = ColorFormat::from_raw(self.reg(rt + 0x10)) else {
            ctx.faults.push(format!(
                "clear: render target format {:#x} not supported",
                self.reg(rt + 0x10)
            ));
            return;
        };
        let base = u64::from(self.reg(rt) & 0xFF) << 32 | u64::from(self.reg(rt + 4));
        let memory = self.reg(rt + 0x14);
        let pitch_layout = memory & (1 << 12) != 0;
        let bpp = format.bytes_per_pixel();
        // For pitch layout WIDTH holds the pitch in bytes.
        let (width, row_bytes) = if pitch_layout {
            (self.reg(rt + 8) / bpp, self.reg(rt + 8))
        } else {
            (self.reg(rt + 8), self.reg(rt + 8) * bpp)
        };
        let height = self.reg(rt + 0xC);
        let block_height = bits(memory, 7, 4);

        // Clear area.
        let (mut x0, mut y0, mut x1, mut y1) = (0, 0, width, height);
        let clip = |lo: u32, hi: u32, a: &mut u32, b: &mut u32| {
            *a = (*a).max(lo);
            *b = (*b).min(hi);
        };
        let ch = self.reg(reg::SURFACE_CLIP_HORIZONTAL);
        let cv = self.reg(reg::SURFACE_CLIP_VERTICAL);
        if ch >> 16 != 0 && cv >> 16 != 0 {
            clip(ch & 0xFFFF, (ch & 0xFFFF) + (ch >> 16), &mut x0, &mut x1);
            clip(cv & 0xFFFF, (cv & 0xFFFF) + (cv >> 16), &mut y0, &mut y1);
        }
        if self.reg(reg::SCISSOR_ENABLE) & 1 != 0 {
            let (h, vv) = (
                self.reg(reg::SCISSOR_HORIZONTAL),
                self.reg(reg::SCISSOR_VERTICAL),
            );
            clip(h & 0xFFFF, h >> 16, &mut x0, &mut x1);
            clip(vv & 0xFFFF, vv >> 16, &mut y0, &mut y1);
        }
        let (rh, rv) = (
            self.reg(reg::CLEAR_RECT_HORIZONTAL),
            self.reg(reg::CLEAR_RECT_VERTICAL),
        );
        if rh >> 16 != 0 && rv >> 16 != 0 {
            clip(rh & 0xFFFF, rh >> 16, &mut x0, &mut x1);
            clip(rv & 0xFFFF, rv >> 16, &mut y0, &mut y1);
        }
        if x0 >= x1 || y0 >= y1 {
            return;
        }

        let color = [0, 1, 2, 3].map(|i| f32::from_bits(self.reg(reg::COLOR_CLEAR_VALUE + 4 * i)));
        let surface = Surface {
            addr: base,
            width,
            height,
            format,
            layout: if pitch_layout {
                Layout::Pitch { pitch: row_bytes }
            } else {
                Layout::BlockLinear {
                    block_height_log2: block_height,
                }
            },
        };
        let rect = Rect { x0, y0, x1, y1 };
        if let Err(e) = ctx
            .backend
            .clear(&mut ctx.guest, &surface, rect, color, mask)
        {
            ctx.faults.push(format!("clear: {e}"));
        }
        self.clears += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{immd, incr, run, setup};
    use crate::{blocklinear, class};

    fn rt_setup(target: u64, width: u32, height: u32, format: u32, memory: u32) -> Vec<u32> {
        vec![
            incr(0, 0, 1),
            class::THREED,
            incr(0, 0x800, 6),
            (target >> 32) as u32,
            target as u32,
            width,
            height,
            format,
            memory,
        ]
    }

    #[test]
    fn clear_pitch_target_with_scissor() {
        let (mut gpu, mut mem) = setup();
        let mut cmds = rt_setup(0x1001_0000, 64 * 4, 16, 0xD5, 1 << 12);
        cmds.extend([
            incr(0, 0xD80, 4),
            1.0f32.to_bits(),
            0.5f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            immd(0, 0xE00, 1),
            incr(0, 0xE04, 2),
            2 | (10 << 16), // x 2..10
            3 | (5 << 16),  // y 3..5
            immd(0, 0x19D0, 0x3C),
        ]);
        run(&mut gpu, &mut mem, &cmds);
        let px = |x: u64, y: u64| {
            let mut b = [0u8; 4];
            mem.read(0x101_0000 + y * 256 + x * 4, &mut b).unwrap();
            b
        };
        assert_eq!(px(2, 3), [255, 128, 0, 255]);
        assert_eq!(px(9, 4), [255, 128, 0, 255]);
        assert_eq!(px(10, 4), [0; 4]);
        assert_eq!(px(2, 5), [0; 4]);
        assert_eq!(px(1, 3), [0; 4]);
    }

    #[test]
    fn clear_block_linear_target() {
        let (mut gpu, mut mem) = setup();
        // 32x16 BGRA, 2 GOBs high.
        let mut cmds = rt_setup(0x1002_0000, 32, 16, 0xCF, 1 << 4);
        cmds.extend([
            incr(0, 0xD80, 4),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            immd(0, 0x19D0, 0x3C),
        ]);
        run(&mut gpu, &mut mem, &cmds);
        let mut raw = vec![0u8; blocklinear::size(128, 16, 1)];
        mem.read(0x102_0000, &mut raw).unwrap();
        let linear = blocklinear::deswizzle(&raw, 128, 16, 1);
        assert!(linear.chunks(4).all(|p| p == [255, 0, 0, 255]));
    }

    #[test]
    fn inline_upload_and_syncpoint() {
        let (mut gpu, mut mem) = setup();
        run(
            &mut gpu,
            &mut mem,
            &[
                incr(0, 0, 1),
                class::THREED,
                incr(0, 0x180, 2),
                8,
                1,
                incr(0, 0x188, 2),
                0x0,
                0x1003_0000,
                immd(0, 0x1B0, 1),
                (3 << 29) | (2 << 16) | (0x1B4 / 4),
                0x1122_3344,
                0x5566_7788,
                immd(0, 0x2C8, 50),
            ],
        );
        let mut b = [0u8; 8];
        mem.read(0x103_0000, &mut b).unwrap();
        assert_eq!(b, [0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55]);
        assert_eq!(gpu.syncpoints[50], 1);
    }
}
