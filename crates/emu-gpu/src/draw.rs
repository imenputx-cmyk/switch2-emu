//! Draws: the state a draw call needs, collected from the 3D engine, and
//! the software renderer that executes it (vertex fetch, the vertex
//! shader on each vertex, triangle setup, rasterization with
//! perspective-correct interpolation, the pixel shader, and the color
//! write). The software renderer is the reference for the Vulkan one.
//!
//! Not yet: instancing, stencil, culling, clipping against the near
//! plane (depth is clamped to the viewport's range instead, like Vulkan
//! with depth clamping); textures are 2D, level 0 only; blending and the
//! depth test apply to color target 0.

use crate::backend::{GuestMemory, Rect, Surface};
use crate::pipeline::{Blend, Depth};
use crate::shader::decode::Op;
use crate::shader::decode::{Inst, IpaMode};
use crate::shader::interp::{self, ShaderIo};
use crate::shader::{attr, Header};
use crate::texture::{self, Pools};

/// Primitive topologies (BEGIN op).
pub mod topology {
    pub const POINTS: u32 = 0;
    pub const TRIANGLES: u32 = 4;
    pub const TRIANGLE_STRIP: u32 = 5;
    pub const TRIANGLE_FAN: u32 = 6;
    pub const QUADS: u32 = 7;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    pub stream: u32,
    /// Constant (not fetched) attributes read as (0, 0, 0, 1).
    pub constant: bool,
    pub offset: u32,
    /// COMPONENT_BIT_WIDTHS value.
    pub widths: u32,
    /// NUMERICAL_TYPE value.
    pub kind: u32,
    pub swap_rb: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VertexStream {
    pub enabled: bool,
    pub stride: u32,
    pub addr: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub scale: [f32; 3],
    pub offset: [f32; 3],
}

/// A bound constant buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstBuffer {
    pub addr: u64,
    pub size: u32,
}

/// Number of constant buffer slots per shader stage.
pub const CBUF_SLOTS: usize = 18;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexBuffer {
    pub addr: u64,
    /// Bytes per index: 1, 2 or 4.
    pub size: u32,
    /// Added to every index.
    pub base_vertex: i32,
}

#[derive(Debug, Clone)]
pub struct Draw {
    /// Color target 0 (pixel shader output 0), if color is written.
    pub target: Option<Surface>,
    /// Pixel shader outputs 1..8 (multiple render targets).
    pub more_targets: [Option<Surface>; 7],
    pub vs: Vec<Inst>,
    pub ps: Vec<Inst>,
    pub ps_header: Header,
    pub topology: u32,
    /// (first vertex, count) runs; with an index buffer, (first index,
    /// count) runs.
    pub ranges: Vec<(u32, u32)>,
    pub index: Option<IndexBuffer>,
    pub vs_cbufs: [Option<ConstBuffer>; CBUF_SLOTS],
    pub ps_cbufs: [Option<ConstBuffer>; CBUF_SLOTS],
    pub attributes: Vec<VertexAttribute>,
    pub streams: Vec<VertexStream>,
    pub viewport: Viewport,
    /// Pixels that may be written (scissor and surface clip).
    pub clip: Rect,
    pub textures: Pools,
    pub depth: Depth,
    pub blend: Blend,
    pub blend_constant: [f32; 4],
    pub cull: crate::pipeline::Cull,
    /// Switch 2: NVN2's root tables (8 x 256 bytes), read by shaders as
    /// constant bank [`crate::shader::sm86::ROOT_TABLE_BANK`].
    pub root_tables: Option<Vec<u8>>,
}

impl Draw {
    /// (pixel shader output, surface) of each bound color target.
    pub fn color_targets(&self) -> Vec<(usize, Surface)> {
        std::iter::once(self.target)
            .chain(self.more_targets)
            .enumerate()
            .filter_map(|(k, t)| t.map(|t| (k, t)))
            .collect()
    }
}

/// The bytes a shader sees in constant buffer `slot`: the bound buffer,
/// or the root tables in their bank.
pub fn cbuf_bytes(
    g: &GuestMemory,
    bound: Option<ConstBuffer>,
    slot: usize,
    root_tables: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    if slot == crate::shader::sm86::ROOT_TABLE_BANK as usize {
        if let Some(t) = root_tables {
            return Ok(t.to_vec());
        }
    }
    match bound {
        Some(b) => {
            let mut data = vec![0u8; b.size as usize];
            if g.read(b.addr, &mut data).is_err() {
                // The buffer may reach past its mapping: read what is
                // mapped, page by page, and leave the rest zero.
                let mut ok = false;
                let mut off = 0usize;
                while off < data.len() {
                    let page = (0x1000 - ((b.addr as usize + off) & 0xFFF)).min(data.len() - off);
                    ok |= g
                        .read(b.addr + off as u64, &mut data[off..off + page])
                        .is_ok();
                    off += page;
                }
                if !ok {
                    return Err(format!("constant buffer: Unmapped({:#x})", b.addr));
                }
            }
            Ok(data)
        }
        None => Ok(Vec::new()),
    }
}

/// Handles (bindless constant buffer word offsets) a program samples.
pub fn texture_handles(program: &[Inst]) -> Vec<u32> {
    let mut out = Vec::new();
    for i in program {
        if let Op::Texs { handle, .. }
        | Op::Tex { handle, .. }
        | Op::Tld { handle, .. }
        | Op::Txq { handle, .. }
        | Op::TexN { handle, .. }
        | Op::Tld4 { handle, .. } = i.op
        {
            if !out.contains(&handle) {
                out.push(handle);
            }
        }
    }
    out
}

/// The handle word a stage's texture `handle` refers to.
pub fn texture_handle_word(
    g: &GuestMemory,
    pools: &Pools,
    cbufs: &[Option<ConstBuffer>; CBUF_SLOTS],
    handle: u32,
) -> Result<u32, String> {
    // SM86 handles name their bank.
    let (slot, handle) = if handle & crate::shader::decode::HANDLE_IN_BANK != 0 {
        ((handle >> 16) & 0x1F, handle & 0xFFFF)
    } else {
        (pools.cbuf_slot, handle)
    };
    let cb = cbufs
        .get(slot as usize)
        .copied()
        .flatten()
        .ok_or("texture handle constant buffer not bound")?;
    if 4 * handle + 4 > cb.size {
        return Err(format!(
            "texture handle {handle} outside its constant buffer"
        ));
    }
    let mut w = [0u8; 4];
    g.read(cb.addr + 4 * u64::from(handle), &mut w)
        .map_err(|f| format!("texture handle: {f:?}"))?;
    Ok(u32::from_le_bytes(w))
}

/// Decodes the textures a program samples.
fn bind_textures(
    g: &GuestMemory,
    pools: &Pools,
    cbufs: &[Option<ConstBuffer>; CBUF_SLOTS],
    program: &[Inst],
) -> Result<texture::Bound, String> {
    let mut bound = texture::Bound::default();
    for handle in texture_handles(program) {
        let word = texture_handle_word(g, pools, cbufs, handle)?;
        let (tic, tsc) = texture::resolve(g, pools, word)?;
        bound
            .0
            .insert(handle, (texture::Texture::decode(g, &tic)?, tsc));
    }
    Ok(bound)
}

/// Reads one vertex attribute as 4 words (floats as their bits; integer
/// formats as integers).
fn fetch(
    g: &GuestMemory,
    a: &VertexAttribute,
    s: &VertexStream,
    vertex: u32,
) -> Result<[u32; 4], String> {
    let one = 1.0f32.to_bits();
    // Unconfigured attributes (no format) read like constant ones.
    if a.constant || !s.enabled || a.widths == 0 {
        return Ok([0, 0, 0, one]);
    }
    // (component count, bytes per component)
    let (n, size) = match a.widths {
        0x01 => (4, 4),
        0x02 => (3, 4),
        0x04 => (2, 4),
        0x12 => (1, 4),
        0x03 => (4, 2),
        0x05 => (3, 2),
        0x0F => (2, 2),
        0x1B => (1, 2),
        0x0A => (4, 1),
        0x13 => (3, 1),
        0x18 => (2, 1),
        0x1D => (1, 1),
        w => return Err(format!("vertex attribute format {w:#x} not supported")),
    };
    let addr = s.addr + u64::from(vertex) * u64::from(s.stride) + u64::from(a.offset);
    let mut raw = vec![0u8; n * size];
    g.read(addr, &mut raw)
        .map_err(|f| format!("vertex fetch: {f:?}"))?;
    let mut out = [0, 0, 0, one];
    for (i, c) in raw.chunks(size).enumerate() {
        let u = match size {
            4 => u32::from_le_bytes(c.try_into().unwrap()),
            2 => u32::from(u16::from_le_bytes(c.try_into().unwrap())),
            _ => u32::from(c[0]),
        };
        let max = ((1u64 << (8 * size)) - 1) as f32;
        out[i] = match a.kind {
            7 => match size {
                4 => u,
                2 => half_to_f32(u as u16).to_bits(),
                _ => return Err("8-bit float vertex attributes do not exist".into()),
            },
            2 => (u as f32 / max).to_bits(),
            1 => {
                let signed = ((u << (32 - 8 * size)) as i32 >> (32 - 8 * size)) as f32;
                (signed / (max / 2.0).floor()).max(-1.0).to_bits()
            }
            5 => (u as f32).to_bits(),
            6 => (((u << (32 - 8 * size)) as i32 >> (32 - 8 * size)) as f32).to_bits(),
            3 => ((u << (32 - 8 * size)) as i32 >> (32 - 8 * size)) as u32,
            4 => u,
            k => return Err(format!("vertex attribute type {k} not supported")),
        };
    }
    if a.swap_rb {
        out.swap(0, 2);
    }
    Ok(out)
}

pub fn half_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1F) as u32,
        (h & 0x3FF) as u32,
    );
    let bits = match e {
        0 if m == 0 => s << 31,
        0 => {
            let v = m as f32 / 1024.0 * 2f32.powi(-14);
            return if s == 1 { -v } else { v };
        }
        31 => s << 31 | 0xFF << 23 | m << 13,
        _ => s << 31 | (e + 112) << 23 | m << 13,
    };
    f32::from_bits(bits)
}

/// A shaded vertex: its outputs, by attribute word (0x70.. as words).
#[derive(Clone)]
struct Varyings {
    words: [u32; 128],
}

impl Varyings {
    fn get(&self, addr: u32) -> u32 {
        self.words.get((addr / 4) as usize).copied().unwrap_or(0)
    }
    fn f(&self, addr: u32) -> f32 {
        f32::from_bits(self.get(addr))
    }
}

/// Constant buffer contents, read once per draw.
struct Cbufs(Vec<Vec<u8>>);

impl Cbufs {
    fn load(
        g: &GuestMemory,
        bound: &[Option<ConstBuffer>; CBUF_SLOTS],
        root_tables: Option<&[u8]>,
    ) -> Result<Cbufs, String> {
        bound
            .iter()
            .enumerate()
            .map(|(slot, b)| cbuf_bytes(g, *b, slot, root_tables))
            .collect::<Result<_, _>>()
            .map(Cbufs)
    }

    /// Out-of-range reads return 0.
    fn get(&self, index: u8, offset: u32) -> u32 {
        self.0
            .get(index as usize)
            .and_then(|b| b.get(offset as usize..offset as usize + 4))
            .map_or(0, |w| u32::from_le_bytes(w.try_into().unwrap()))
    }
}

struct VertexIo<'a, 'g> {
    mem: &'a mut GuestMemory<'g>,
    inputs: &'a [[u32; 4]],
    out: Varyings,
    cbufs: &'a Cbufs,
    tex: &'a texture::Bound,
}

impl ShaderIo for VertexIo<'_, '_> {
    fn sample_any(&self, handle: u32, c: [f32; 3], layer: f32) -> [f32; 4] {
        self.tex.sample_any(handle, c, layer)
    }
    fn fetch_layer(&self, handle: u32, x: i32, y: i32, layer: i32) -> [f32; 4] {
        self.tex.fetch_layer(handle, x, y, layer)
    }
    fn gather(&self, handle: u32, u: f32, v: f32, comp: u8) -> [f32; 4] {
        self.tex.gather(handle, u, v, comp)
    }
    fn texture_size(&self, handle: u32, lod: u32) -> [u32; 4] {
        self.tex.size(handle, lod)
    }
    fn load_global(&mut self, va: u64, buf: &mut [u8]) {
        if self.mem.read(va, buf).is_err() {
            buf.fill(0);
        }
    }
    fn store_global(&mut self, va: u64, data: &[u8]) {
        let _ = self.mem.write(va, data);
    }
    fn load_attr(&self, addr: u32) -> u32 {
        if addr >= attr::generic(0) {
            let i = ((addr - attr::generic(0)) / 16) as usize;
            self.inputs
                .get(i)
                .map_or(0, |v| v[((addr / 4) % 4) as usize])
        } else {
            0 // system values (vertex id etc.) are not modelled yet
        }
    }
    fn store_attr(&mut self, addr: u32, value: u32) {
        if let Some(w) = self.out.words.get_mut((addr / 4) as usize) {
            *w = value;
        }
    }
    fn interpolate(&self, _: u32, _: IpaMode) -> f32 {
        0.0
    }
    fn cbuf(&self, index: u8, offset: u32) -> u32 {
        self.cbufs.get(index, offset)
    }
    fn texture(&self, handle: u32, u: f32, v: f32) -> [f32; 4] {
        self.tex.sample(handle, u, v)
    }

    fn fetch(&self, handle: u32, x: i32, y: i32) -> [f32; 4] {
        self.tex.fetch(handle, x, y)
    }
}

struct PixelIo<'a, 'g> {
    mem: &'a mut GuestMemory<'g>,
    v: [&'a Varyings; 3],
    cbufs: &'a Cbufs,
    tex: &'a texture::Bound,
    /// Perspective-correct barycentrics, and screen-space ones.
    persp: [f32; 3],
    frag: [f32; 4],
}

impl ShaderIo for PixelIo<'_, '_> {
    fn sample_any(&self, handle: u32, c: [f32; 3], layer: f32) -> [f32; 4] {
        self.tex.sample_any(handle, c, layer)
    }
    fn fetch_layer(&self, handle: u32, x: i32, y: i32, layer: i32) -> [f32; 4] {
        self.tex.fetch_layer(handle, x, y, layer)
    }
    fn gather(&self, handle: u32, u: f32, v: f32, comp: u8) -> [f32; 4] {
        self.tex.gather(handle, u, v, comp)
    }
    fn texture_size(&self, handle: u32, lod: u32) -> [u32; 4] {
        self.tex.size(handle, lod)
    }
    fn load_global(&mut self, va: u64, buf: &mut [u8]) {
        if self.mem.read(va, buf).is_err() {
            buf.fill(0);
        }
    }
    fn store_global(&mut self, va: u64, data: &[u8]) {
        let _ = self.mem.write(va, data);
    }
    fn load_attr(&self, addr: u32) -> u32 {
        self.interpolate(addr, IpaMode::Pass).to_bits()
    }
    fn store_attr(&mut self, _: u32, _: u32) {}
    fn interpolate(&self, addr: u32, mode: IpaMode) -> f32 {
        if (attr::POSITION..attr::POSITION + 16).contains(&addr) {
            return self.frag[((addr - attr::POSITION) / 4) as usize];
        }
        match mode {
            IpaMode::Constant => self.v[0].f(addr),
            _ => (0..3).map(|i| self.persp[i] * self.v[i].f(addr)).sum(),
        }
    }
    fn cbuf(&self, index: u8, offset: u32) -> u32 {
        self.cbufs.get(index, offset)
    }
    fn texture(&self, handle: u32, u: f32, v: f32) -> [f32; 4] {
        self.tex.sample(handle, u, v)
    }

    fn fetch(&self, handle: u32, x: i32, y: i32) -> [f32; 4] {
        self.tex.fetch(handle, x, y)
    }
}

/// Reads `count` indices from `first` on.
pub fn read_indices(
    g: &GuestMemory,
    ib: &IndexBuffer,
    first: u32,
    count: u32,
) -> Result<Vec<u32>, String> {
    let mut raw = vec![0u8; (count * ib.size) as usize];
    g.read(ib.addr + u64::from(first) * u64::from(ib.size), &mut raw)
        .map_err(|f| format!("index buffer: {f:?}"))?;
    Ok(raw
        .chunks(ib.size as usize)
        .map(|c| match c.len() {
            1 => u32::from(c[0]),
            2 => u32::from(u16::from_le_bytes([c[0], c[1]])),
            _ => u32::from_le_bytes(c.try_into().unwrap()),
        })
        .collect())
}

/// Triangles (vertex index triples) for a topology and vertex count.
pub fn triangles(topology: u32, count: u32) -> Result<Vec<[u32; 3]>, String> {
    Ok(match topology {
        topology::TRIANGLES => (0..count / 3)
            .map(|t| [3 * t, 3 * t + 1, 3 * t + 2])
            .collect(),
        topology::TRIANGLE_STRIP => (0..count.saturating_sub(2))
            .map(|i| {
                if i % 2 == 0 {
                    [i, i + 1, i + 2]
                } else {
                    [i + 1, i, i + 2]
                }
            })
            .collect(),
        topology::TRIANGLE_FAN => (1..count.saturating_sub(1))
            .map(|i| [0, i, i + 1])
            .collect(),
        topology::QUADS => (0..count / 4)
            .flat_map(|q| [[4 * q, 4 * q + 1, 4 * q + 2], [4 * q, 4 * q + 2, 4 * q + 3]])
            .collect(),
        t => return Err(format!("topology {t} not supported")),
    })
}

/// Executes a draw on guest memory.
pub fn draw_software(g: &mut GuestMemory, d: &Draw) -> Result<(), String> {
    let Some(target) = d.target else {
        return Ok(());
    };
    let vs_cbufs = Cbufs::load(g, &d.vs_cbufs, d.root_tables.as_deref())?;
    let ps_cbufs = Cbufs::load(g, &d.ps_cbufs, d.root_tables.as_deref())?;
    let vs_tex = bind_textures(g, &d.textures, &d.vs_cbufs, &d.vs)?;
    let ps_tex = bind_textures(g, &d.textures, &d.ps_cbufs, &d.ps)?;
    for &(first, count) in &d.ranges {
        // The vertices to shade, in order.
        let vertices: Vec<u32> = match &d.index {
            Some(ib) => read_indices(g, ib, first, count)?
                .into_iter()
                .map(|i| (i as i64 + i64::from(ib.base_vertex)) as u32)
                .collect(),
            None => (first..first + count).collect(),
        };
        // Vertex stage.
        let mut shaded = Vec::with_capacity(vertices.len());
        for &v in &vertices {
            let inputs = d
                .attributes
                .iter()
                .map(|a| {
                    let s = d
                        .streams
                        .get(a.stream as usize)
                        .copied()
                        .unwrap_or_default();
                    fetch(g, a, &s, v)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut io = VertexIo {
                mem: g,
                inputs: &inputs,
                out: Varyings { words: [0; 128] },
                cbufs: &vs_cbufs,
                tex: &vs_tex,
            };
            interp::run(&d.vs, &mut io);
            shaded.push(io.out);
        }
        for tri in triangles(d.topology, count)? {
            let v = tri.map(|i| &shaded[i as usize]);
            rasterize(g, d, &target, v, &ps_cbufs, &ps_tex)?;
        }
    }
    Ok(())
}

fn rasterize(
    g: &mut GuestMemory,
    d: &Draw,
    target: &Surface,
    v: [&Varyings; 3],
    cbufs: &Cbufs,
    tex: &texture::Bound,
) -> Result<(), String> {
    // Clip space -> window space.
    let mut win = [[0f32; 4]; 3];
    for (i, vert) in v.iter().enumerate() {
        let (x, y, z, w) = (vert.f(0x70), vert.f(0x74), vert.f(0x78), vert.f(0x7C));
        if w <= 0.0 || !w.is_finite() {
            return Ok(()); // needs clipping: not implemented, skip
        }
        win[i] = [
            x / w * d.viewport.scale[0] + d.viewport.offset[0],
            y / w * d.viewport.scale[1] + d.viewport.offset[1],
            z / w * d.viewport.scale[2] + d.viewport.offset[2],
            1.0 / w,
        ];
    }
    let edge = |a: [f32; 4], b: [f32; 4], px: f32, py: f32| {
        (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
    };
    let area = edge(win[0], win[1], win[2][0], win[2][1]);
    if area == 0.0 || d.cull.culls(area) {
        return Ok(());
    }
    let front = if d.cull.front_ccw {
        area < 0.0
    } else {
        area > 0.0
    };
    let stencil_on = d.depth.stencil.enabled;
    let face = if front {
        d.depth.stencil.front
    } else {
        d.depth.stencil.back
    };
    // Pixel coverage uses the top-left rule on the triangle oriented so
    // its area is positive.
    let sign = area.signum();
    // With the interior on the positive side in y-down window space, a
    // top edge runs in +x and a left edge runs in -y.
    let top_left = |a: [f32; 4], b: [f32; 4]| {
        let (dx, dy) = ((b[0] - a[0]) * sign, (b[1] - a[1]) * sign);
        (dy == 0.0 && dx > 0.0) || dy < 0.0
    };
    let edges = [(1, 2, 0), (2, 0, 1), (0, 1, 2)];
    let tl = edges.map(|(a, b, _)| top_left(win[a], win[b]));

    let (minx, maxx) = (
        win.iter().map(|p| p[0]).fold(f32::MAX, f32::min),
        win.iter().map(|p| p[0]).fold(f32::MIN, f32::max),
    );
    let (miny, maxy) = (
        win.iter().map(|p| p[1]).fold(f32::MAX, f32::min),
        win.iter().map(|p| p[1]).fold(f32::MIN, f32::max),
    );
    let x0 = (minx.floor().max(0.0) as u32).max(d.clip.x0);
    let y0 = (miny.floor().max(0.0) as u32).max(d.clip.y0);
    let x1 = ((maxx.ceil() as i64).clamp(0, i64::from(target.width)) as u32).min(d.clip.x1);
    let y1 = ((maxy.ceil() as i64).clamp(0, i64::from(target.height)) as u32).min(d.clip.y1);

    let near = d.viewport.offset[2];
    let far = near + d.viewport.scale[2];
    let (z_lo, z_hi) = (near.min(far), near.max(far));
    for y in y0..y1 {
        for x in x0..x1 {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let mut b = [0f32; 3];
            let mut inside = true;
            for (k, &(a, bb, opp)) in edges.iter().enumerate() {
                let e = edge(win[a], win[bb], px, py) * sign;
                if e < 0.0 || (e == 0.0 && !tl[k]) {
                    inside = false;
                    break;
                }
                b[opp] = e / (area * sign);
            }
            if !inside {
                continue;
            }
            // Perspective correction: weight by 1/w.
            let inv_w: f32 = (0..3).map(|i| b[i] * win[i][3]).sum();
            let persp = [0, 1, 2].map(|i| b[i] * win[i][3] / inv_w);
            let z: f32 = (0..3).map(|i| b[i] * win[i][2]).sum();
            let z = z.clamp(z_lo, z_hi);
            let mut io = PixelIo {
                mem: g,
                v,
                cbufs,
                tex,
                persp,
                frag: [px, py, z, inv_w],
            };
            let Some(regs) = interp::run(&d.ps, &mut io) else {
                continue; // discarded
            };
            // Stencil and depth tests against the stored values (depth
            // in the buffer's precision), then their writes.
            if let (Some(ds), true) = (&d.depth.target, d.depth.test || stencil_on) {
                if x >= ds.width || y >= ds.height {
                    continue;
                }
                let bpp = ds.format.bytes_per_pixel() as usize;
                let at = ds.byte_addr(x, y);
                let mut cur = [0u8; 8];
                g.read(at, &mut cur[..bpp])
                    .map_err(|f| format!("depth read: {f:?}"))?;
                let mut new = cur;
                let sb = ds.format.stencil_byte().filter(|_| stencil_on);
                if let Some(sb) = sb {
                    let stored = cur[sb];
                    if !face.test(stored) {
                        new[sb] = face.update(face.fail, stored);
                        if new != cur {
                            g.write(at, &new[..bpp])
                                .map_err(|f| format!("stencil write: {f:?}"))?;
                        }
                        continue;
                    }
                }
                let mut znew = new;
                ds.format.encode(z, &mut znew);
                let depth_pass = !d.depth.test
                    || d.depth
                        .func
                        .test(ds.format.decode(&znew), ds.format.decode(&cur));
                if let Some(sb) = sb {
                    let op = if depth_pass { face.zpass } else { face.zfail };
                    new[sb] = face.update(op, cur[sb]);
                    znew[sb] = new[sb];
                }
                if depth_pass && d.depth.write {
                    new = znew;
                }
                if new != cur {
                    g.write(at, &new[..bpp])
                        .map_err(|f| format!("depth write: {f:?}"))?;
                }
                if !depth_pass {
                    continue;
                }
            }
            // Outputs: the enabled components of each target, in
            // target order, in r0, r1, ...
            let mut next = 0u8;
            for k in 0..8 {
                let mask = d.ps_header.omap_targets[k];
                let mut color = [0f32; 4];
                for (c, slot) in color.iter_mut().enumerate() {
                    if mask & (1 << c) != 0 {
                        *slot = regs.f(next);
                        next += 1;
                    }
                }
                let surface = if k == 0 {
                    Some(*target)
                } else {
                    d.more_targets[k - 1]
                };
                let Some(t) = surface else {
                    continue;
                };
                if mask == 0 || x >= t.width || y >= t.height {
                    continue;
                }
                let bpp = t.format.bytes_per_pixel();
                let at = t.byte_addr(x * bpp, y);
                if d.blend.enabled {
                    let mut cur = vec![0u8; bpp as usize];
                    g.read(at, &mut cur)
                        .map_err(|f| format!("blend read: {f:?}"))?;
                    let unorm = !t.format.is_float();
                    color = d
                        .blend
                        .apply(color, t.format.decode(&cur), d.blend_constant, unorm);
                }
                let pixel = t.format.encode(color);
                g.write(at, &pixel)
                    .map_err(|f| format!("color write: {f:?}"))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topologies() {
        assert_eq!(
            triangles(topology::TRIANGLES, 7).unwrap(),
            [[0, 1, 2], [3, 4, 5]]
        );
        assert_eq!(
            triangles(topology::TRIANGLE_STRIP, 5).unwrap(),
            [[0, 1, 2], [2, 1, 3], [2, 3, 4]]
        );
        assert_eq!(
            triangles(topology::TRIANGLE_FAN, 4).unwrap(),
            [[0, 1, 2], [0, 2, 3]]
        );
        assert_eq!(
            triangles(topology::QUADS, 4).unwrap(),
            [[0, 1, 2], [0, 2, 3]]
        );
    }

    #[test]
    fn halves() {
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xC000), -2.0);
        assert_eq!(half_to_f32(0x3555), 0.333_251_95);
    }
}

#[cfg(test)]
mod raster_tests {
    use super::*;
    use crate::backend::Layout;
    use crate::mmu::AddressSpace;
    use crate::shader::{asm, decode::decode_program, Header};
    use crate::surface::ColorFormat;
    use emu_memory::{Memory, Perm};

    /// Two triangles sharing an edge must cover every pixel of their
    /// union exactly once (top-left rule), whatever their winding.
    #[test]
    fn shared_edges_are_drawn_once() {
        let mut mem = Memory::new();
        mem.map(0x10_0000, 0x1_0000, Perm::RW).unwrap();
        let mut aspace = AddressSpace::new();
        assert!(aspace.map(0x10_0000, 0x10_0000, 0x1_0000));
        let mut g = GuestMemory {
            mem: &mut mem,
            aspace: &aspace,
            asid: 0,
        };
        // VS: position = attribute 0. PS: color = (1/16, 0, 0, 0): with
        // additive... no blending, so count by drawing each triangle in
        // its own pass and summing.
        let vs = decode_program(
            &asm::program(
                &asm::vertex_header(),
                &[asm::ald(0, 0x80, 4), asm::ast(0, 0x70, 4), asm::exit()],
            )[0x50..],
        )
        .unwrap();
        let ps = decode_program(
            &asm::program(
                &asm::pixel_header(),
                &[
                    asm::mov32i(0, 0x3F80_0000),
                    asm::mov32i(1, 0),
                    asm::mov32i(2, 0),
                    asm::mov32i(3, 0),
                    asm::exit(),
                ],
            )[0x50..],
        )
        .unwrap();
        let target = Surface {
            addr: 0x10_0000,
            width: 16,
            height: 16,
            format: ColorFormat::Rgba8Unorm,
            layout: Layout::Pitch { pitch: 64 },
        };
        // Quad corners in clip space covering x 2..14, y 3..13 (viewport 16x16).
        let p = |x: f32, y: f32| {
            [
                ((x / 8.0 - 1.0).to_bits()),
                ((y / 8.0 - 1.0).to_bits()),
                0,
                1.0f32.to_bits(),
            ]
        };
        let quad = [p(2.0, 3.0), p(14.0, 3.0), p(14.0, 13.0), p(2.0, 13.0)];
        let mut count = [0u32; 256];
        for tri in [[0usize, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]] {
            g.write(0x10_0000, &[0u8; 1024]).unwrap();
            // Vertices stored in memory after the target.
            let verts: Vec<u8> = tri
                .iter()
                .flat_map(|&i| quad[i].iter().flat_map(|w| w.to_le_bytes()))
                .collect();
            g.write(0x10_1000, &verts).unwrap();
            let d = Draw {
                target: Some(target),
                more_targets: Default::default(),
                cull: Default::default(),
                vs: vs.clone(),
                ps: ps.clone(),
                ps_header: Header::parse(&asm::pixel_header()),
                topology: topology::TRIANGLES,
                ranges: vec![(0, 3)],
                index: None,
                vs_cbufs: Default::default(),
                ps_cbufs: Default::default(),
                attributes: vec![VertexAttribute {
                    stream: 0,
                    constant: false,
                    offset: 0,
                    widths: 1,
                    kind: 7,
                    swap_rb: false,
                }],
                streams: vec![VertexStream {
                    enabled: true,
                    stride: 16,
                    addr: 0x10_1000,
                }],
                viewport: Viewport {
                    scale: [8.0, 8.0, 1.0],
                    offset: [8.0, 8.0, 0.0],
                },
                clip: Rect {
                    x0: 0,
                    y0: 0,
                    x1: 16,
                    y1: 16,
                },
                textures: Default::default(),
                depth: Default::default(),
                blend: Default::default(),
                blend_constant: [0.0; 4],
                root_tables: None,
            };
            draw_software(&mut g, &d).unwrap();
            let mut px = vec![0u8; 1024];
            g.read(0x10_0000, &mut px).unwrap();
            for i in 0..256 {
                count[i] += u32::from(px[4 * i] == 255);
            }
        }
        // Each pixel of the quad is covered by exactly one of the two
        // triangles of each winding pair: 2 in total.
        for y in 0..16 {
            for x in 0..16 {
                let inside = (2..14).contains(&x) && (3..13).contains(&y);
                assert_eq!(
                    count[y * 16 + x],
                    if inside { 2 } else { 0 },
                    "pixel ({x}, {y})"
                );
            }
        }
    }
}
