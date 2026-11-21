//! Rendering backends. The engines decode methods into operations on
//! surfaces in GPU memory; a backend carries them out. The software
//! backend writes guest memory directly and is the reference; other
//! backends (Vulkan) must produce the same memory contents by the time
//! `flush` returns.

use crate::blocklinear;
use crate::mmu::{AddressSpace, GpuFault};
use crate::surface::ColorFormat;
use emu_memory::Memory;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Rows of `pitch` bytes.
    Pitch { pitch: u32 },
    /// Tegra block-linear with blocks `1 << block_height_log2` GOBs high.
    BlockLinear { block_height_log2: u32 },
}

/// A 2D color surface in GPU memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Surface {
    /// GPU virtual address.
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub layout: Layout,
}

impl Surface {
    /// Bytes per row of pixels (for block-linear, the row as stored
    /// before rounding to whole GOBs).
    pub fn row_bytes(&self) -> u32 {
        match self.layout {
            Layout::Pitch { pitch } => pitch,
            Layout::BlockLinear { .. } => self.width * self.format.bytes_per_pixel(),
        }
    }

    /// GPU address of byte `x` of row `y`.
    pub fn byte_addr(&self, x: u32, y: u32) -> u64 {
        match self.layout {
            Layout::Pitch { pitch } => self.addr + u64::from(y) * u64::from(pitch) + u64::from(x),
            Layout::BlockLinear { block_height_log2 } => {
                self.addr + blocklinear::offset(x, y, self.row_bytes(), block_height_log2) as u64
            }
        }
    }

    /// Bytes the surface covers in memory.
    pub fn size_bytes(&self) -> usize {
        match self.layout {
            Layout::Pitch { pitch } => pitch as usize * self.height as usize,
            Layout::BlockLinear { block_height_log2 } => {
                blocklinear::size(self.row_bytes(), self.height, block_height_log2)
            }
        }
    }

    /// Reads the surface as tightly packed rows (`width * bpp` bytes each).
    pub fn read_linear(&self, g: &GuestMemory) -> Result<Vec<u8>, GpuFault> {
        let row = (self.width * self.format.bytes_per_pixel()) as usize;
        let mut raw = vec![0u8; self.size_bytes()];
        g.read(self.addr, &mut raw)?;
        Ok(match self.layout {
            Layout::Pitch { pitch } => {
                let mut out = Vec::with_capacity(row * self.height as usize);
                for y in 0..self.height as usize {
                    out.extend_from_slice(&raw[y * pitch as usize..y * pitch as usize + row]);
                }
                out
            }
            Layout::BlockLinear { block_height_log2 } => {
                let padded = (row as u32).next_multiple_of(16);
                blocklinear::deswizzle_width(
                    &raw,
                    row as u32,
                    padded,
                    self.height,
                    block_height_log2,
                )
            }
        })
    }

    /// Writes tightly packed rows back to the surface.
    pub fn write_linear(&self, g: &mut GuestMemory, data: &[u8]) -> Result<(), GpuFault> {
        let row = (self.width * self.format.bytes_per_pixel()) as usize;
        match self.layout {
            Layout::Pitch { pitch } => {
                for y in 0..self.height as usize {
                    g.write(
                        self.addr + (y as u64) * u64::from(pitch),
                        &data[y * row..(y + 1) * row],
                    )?;
                }
            }
            Layout::BlockLinear { block_height_log2 } => {
                let mut raw = vec![0u8; self.size_bytes()];
                g.read(self.addr, &mut raw)?;
                blocklinear::swizzle_into(
                    &mut raw,
                    data,
                    row as u32,
                    self.height,
                    block_height_log2,
                );
                g.write(self.addr, &raw)?;
            }
        }
        Ok(())
    }
}

/// A clear rectangle in pixels, end exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

/// Guest memory as seen through one GPU address space.
pub struct GuestMemory<'a> {
    pub mem: &'a mut Memory,
    pub aspace: &'a AddressSpace,
    /// Index of the address space (backends key caches by it).
    pub asid: usize,
}

impl GuestMemory<'_> {
    pub fn read(&self, va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        self.aspace.read(self.mem, va, buf)
    }

    pub fn write(&mut self, va: u64, data: &[u8]) -> Result<(), GpuFault> {
        self.aspace.write(self.mem, va, data)
    }

    /// Reads as much of `buf` as is mapped from `va` on (page by page),
    /// returning the bytes read; the rest stays zero.
    pub fn read_prefix(&self, va: u64, buf: &mut [u8]) -> Result<usize, GpuFault> {
        let mut done = 0usize;
        while done < buf.len() {
            let at = va + done as u64;
            let n = ((0x1000 - (at & 0xFFF)) as usize).min(buf.len() - done);
            match self.read(at, &mut buf[done..done + n]) {
                Ok(()) => done += n,
                Err(f) if done == 0 => return Err(f),
                Err(_) => break,
            }
        }
        Ok(done)
    }
}

pub trait Backend {
    fn name(&self) -> &str;

    /// For tests and diagnostics: the concrete backend.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Fills `rect` of `surface` with `color`, writing only the
    /// components in `mask` (RGBA).
    fn clear(
        &mut self,
        g: &mut GuestMemory,
        surface: &Surface,
        rect: Rect,
        color: [f32; 4],
        mask: [bool; 4],
    ) -> Result<(), String>;

    /// Fills `rect` of a depth buffer with depth `z`.
    fn clear_depth(
        &mut self,
        g: &mut GuestMemory,
        target: &crate::pipeline::DepthSurface,
        rect: Rect,
        z: f32,
    ) -> Result<(), String> {
        clear_depth_software(g, target, rect, z)
    }

    /// Fills the stencil bits of `rect` of a depth/stencil buffer.
    fn clear_stencil(
        &mut self,
        g: &mut GuestMemory,
        target: &crate::pipeline::DepthSurface,
        rect: Rect,
        s: u8,
    ) -> Result<(), String> {
        clear_stencil_software(g, target, rect, s)
    }

    /// Runs a compute grid; `Ok(false)` leaves it to the interpreter.
    fn dispatch(&mut self, g: &mut GuestMemory, d: &Dispatch) -> Result<bool, String> {
        let _ = (g, d);
        Ok(false)
    }

    /// Executes a draw.
    fn draw(&mut self, g: &mut GuestMemory, d: &crate::draw::Draw) -> Result<(), String>;

    /// Makes all results visible in guest memory (end of a submission).
    fn flush(&mut self, g: &mut GuestMemory) -> Result<(), String>;
}

/// Clears a depth buffer in guest memory (keeping stencil bits).
pub fn clear_depth_software(
    g: &mut GuestMemory,
    t: &crate::pipeline::DepthSurface,
    r: Rect,
    z: f32,
) -> Result<(), String> {
    let mut data = t.read_linear(g)?;
    let bpp = t.format.bytes_per_pixel() as usize;
    for y in r.y0..r.y1.min(t.height) {
        for x in r.x0..r.x1.min(t.width) {
            let i = (y * t.width + x) as usize * bpp;
            t.format.encode(z, &mut data[i..i + bpp]);
        }
    }
    t.write_linear(g, &data)
}

pub fn clear_stencil_software(
    g: &mut GuestMemory,
    t: &crate::pipeline::DepthSurface,
    r: Rect,
    s: u8,
) -> Result<(), String> {
    let Some(at) = t.format.stencil_byte() else {
        return Ok(());
    };
    let mut data = t.read_linear(g)?;
    let bpp = t.format.bytes_per_pixel() as usize;
    for y in r.y0..r.y1.min(t.height) {
        for x in r.x0..r.x1.min(t.width) {
            data[(y * t.width + x) as usize * bpp + at] = s;
        }
    }
    t.write_linear(g, &data)
}

/// A compute grid (from a QMD) for a backend.
pub struct Dispatch<'a> {
    /// Program address (with the block size, identifies the program).
    pub address: u64,
    pub program: &'a [crate::shader::decode::Inst],
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
    /// Constant buffer contents by slot (empty: not bound).
    pub cbufs: &'a [Vec<u8>],
}

/// The reference backend: works directly on guest memory.
#[derive(Debug, Default)]
pub struct SoftwareBackend;

impl Backend for SoftwareBackend {
    fn name(&self) -> &str {
        "software"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clear(
        &mut self,
        g: &mut GuestMemory,
        s: &Surface,
        r: Rect,
        color: [f32; 4],
        mask: [bool; 4],
    ) -> Result<(), String> {
        let bpp = s.format.bytes_per_pixel();
        let pixel = s.format.encode(color);
        let row: Vec<u8> = pixel
            .iter()
            .copied()
            .cycle()
            .take(((r.x1 - r.x0) * bpp) as usize)
            .collect();
        let keep = if mask.iter().all(|&m| m) {
            None
        } else {
            Some(
                s.format
                    .component_bytes(mask)
                    .ok_or_else(|| format!("masked clear of {:?} not supported", s.format))?,
            )
        };
        let err = |f: GpuFault| format!("clear: {f:?}");
        for y in r.y0..r.y1 {
            let mut x = r.x0 * bpp;
            let end = r.x1 * bpp;
            while x < end {
                // Runs of up to 16 bytes are contiguous in both layouts.
                let n = (16 - x % 16).min(end - x);
                let src = &row[(x - r.x0 * bpp) as usize..(x - r.x0 * bpp + n) as usize];
                let va = s.byte_addr(x, y);
                match &keep {
                    None => g.write(va, src).map_err(err)?,
                    Some(keep) => {
                        let mut cur = vec![0u8; n as usize];
                        g.read(va, &mut cur).map_err(err)?;
                        for (i, b) in cur.iter_mut().enumerate() {
                            if keep[(x as usize + i) % bpp as usize] {
                                *b = src[i];
                            }
                        }
                        g.write(va, &cur).map_err(err)?;
                    }
                }
                x += n;
            }
        }
        Ok(())
    }

    fn draw(&mut self, g: &mut GuestMemory, d: &crate::draw::Draw) -> Result<(), String> {
        crate::draw::draw_software(g, d)
    }

    fn flush(&mut self, _g: &mut GuestMemory) -> Result<(), String> {
        Ok(())
    }
}
