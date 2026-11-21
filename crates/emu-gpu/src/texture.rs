//! Textures: the texture header (TIC) and sampler (TSC) pools, decoding
//! texels, and sampling.
//!
//! Shaders name a texture by a word offset into the "bindless" constant
//! buffer (the slot set with BINDLESS_TEXTURE_CONST_BUFFER_SLOT); that
//! word holds the TIC index in bits 19:0 and the TSC index in 31:20.
//! TIC and TSC entries are 32 bytes each. Field layouts follow deko3d's
//! `texture_image_control_block.h` and `texture_sampler_control_block.h`
//! (zlib; derived from envytools' rnndb).
//!
//! Supported for now: 2D (and 2D-no-mipmap) textures, level 0 only, the
//! common uncompressed formats, BC1-BC7 and ASTC (decoded with the
//! texture2ddecoder crate, MIT/Apache-2.0), nearest and linear
//! filtering, all address modes (mirror-clamp ones treated as mirror
//! then clamp).

use crate::backend::GuestMemory;
use crate::blocklinear;
use std::collections::HashMap;

/// Where the pools are, from the 3D engine registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Pools {
    pub tic: u64,
    pub tic_max: u32,
    pub tsc: u64,
    pub tsc_max: u32,
    /// Constant buffer slot holding the texture handles.
    pub cbuf_slot: u32,
}

/// Component encodings (TIC format word).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Snorm,
    Unorm,
    Sint,
    Uint,
    Float,
}

/// A decoded texture header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tic {
    /// TIC image format (ImageFormat).
    pub format: u32,
    pub kinds: [Kind; 4],
    /// Per output component: 0 zero, 2..=5 R/G/B/A, 6/7 one.
    pub swizzle: [u8; 4],
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    /// `None`: pitch layout with this pitch; otherwise block height.
    pub pitch: Option<u32>,
    pub block_height_log2: u32,
    pub srgb: bool,
    /// Texture type (see [`ty`]).
    pub ty: u32,
    /// Slices of a 3D texture, layers of an array, cubes of a cube array.
    pub depth: u32,
    pub block_depth_log2: u32,
    /// Mip levels the view covers (for the layer stride).
    pub levels: u32,
    /// Layer stride of the whole texture, for a single mip level's
    /// header (see [`Tic::level`]).
    pub stride_override: Option<u64>,
}

/// Texture types (TIC word 4).
pub mod ty {
    pub const D1: u32 = 0;
    pub const D2: u32 = 1;
    pub const D3: u32 = 2;
    pub const CUBE: u32 = 3;
    pub const D1_ARRAY: u32 = 4;
    pub const D2_ARRAY: u32 = 5;
    pub const D1_BUFFER: u32 = 6;
    pub const D2_NO_MIPMAP: u32 = 7;
    pub const CUBE_ARRAY: u32 = 8;
}

/// Bit widths of the R, G, B, A components from the least significant
/// bit up, for the supported formats.
fn layout(format: u32) -> Option<&'static [u32]> {
    Some(match format {
        0x01 => &[32, 32, 32, 32],
        0x02 => &[32, 32, 32],
        0x03 => &[16, 16, 16, 16],
        0x04 => &[32, 32],
        0x08 => &[8, 8, 8, 8],
        0x09 => &[10, 10, 10, 2],
        0x0C => &[16, 16],
        0x0F => &[32],
        0x12 => &[4, 4, 4, 4],
        0x13 => &[1, 5, 5, 5],
        0x14 => &[5, 5, 5, 1],
        0x15 => &[5, 6, 5],
        0x18 => &[8, 8],
        0x1B => &[16],
        0x1D => &[8],
        // B10G11R11 float and E5B9G9R9 shared-exponent.
        0x21 => &[11, 11, 10],
        0x20 => &[9, 9, 9, 5],
        _ => return None,
    })
}

/// Block-compressed formats: (block width, height, bytes per block).
fn compressed(format: u32) -> Option<(u32, u32, u32)> {
    Some(match format {
        0x24 | 0x27 => (4, 4, 8),         // DXT1 (BC1), DXN1 (BC4)
        0x25 | 0x26 | 0x28 => (4, 4, 16), // DXT23, DXT45, DXN2 (BC2, BC3, BC5)
        0x10 | 0x11 | 0x17 => (4, 4, 16), // BC6H signed/unsigned, BC7
        0x40 => (4, 4, 16),
        0x41 => (5, 5, 16),
        0x42 => (6, 6, 16),
        0x44 => (8, 8, 16),
        0x45 => (10, 10, 16),
        0x46 => (12, 12, 16),
        0x50 => (5, 4, 16),
        0x51 => (6, 5, 16),
        0x52 => (8, 6, 16),
        0x53 => (10, 8, 16),
        0x54 => (12, 10, 16),
        0x55 => (8, 5, 16),
        0x56 => (10, 5, 16),
        0x57 => (10, 6, 16),
        _ => return None,
    })
}

/// Decodes one compressed block to BGRA8 pixels (`bw * bh` of them).
fn decode_block(format: u32, bw: u32, bh: u32, data: &[u8], out: &mut [u32]) {
    use texture2ddecoder as t;
    match format {
        0x24 => t::decode_bc1a_block(data, out),
        0x25 => t::decode_bc2_block(data, out),
        0x26 => t::decode_bc3_block(data, out),
        0x27 => t::decode_bc4_block(data, out),
        0x28 => t::decode_bc5_block(data, out),
        0x10 => t::decode_bc6_block(data, out, true),
        0x11 => t::decode_bc6_block(data, out, false),
        0x17 => t::decode_bc7_block(data, out),
        _ => t::decode_astc_block(data, bw as usize, bh as usize, out),
    }
}

fn bits(v: u32, lo: u32, len: u32) -> u32 {
    (v >> lo) & ((1u64 << len) - 1) as u32
}

impl Tic {
    pub fn parse(e: &[u8; 32]) -> Result<Tic, String> {
        let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().unwrap());
        let format = bits(w(0), 0, 7);
        if layout(format).is_none() && compressed(format).is_none() {
            return Err(format!("texture format {format:#x}"));
        }
        let kind = |k| match k {
            1 => Ok(Kind::Snorm),
            2 => Ok(Kind::Unorm),
            3 => Ok(Kind::Sint),
            4 => Ok(Kind::Uint),
            5 => Ok(Kind::Snorm),
            6 => Ok(Kind::Unorm),
            7 => Ok(Kind::Float),
            k => Err(format!("texture component type {k}")),
        };
        let kinds = [
            kind(bits(w(0), 7, 3))?,
            kind(bits(w(0), 10, 3))?,
            kind(bits(w(0), 13, 3))?,
            kind(bits(w(0), 16, 3))?,
        ];
        let swizzle = [0, 1, 2, 3].map(|i| bits(w(0), 19 + 3 * i, 3) as u8);
        let version = bits(w(2), 21, 3);
        let ty = bits(w(4), 23, 4);
        if ty == ty::D1_BUFFER || ty > ty::CUBE_ARRAY {
            return Err(format!("texture type {ty} not supported"));
        }
        let pitch = match version {
            2 => Some(bits(w(3), 0, 16) << 5),
            3 => None,
            v => return Err(format!("texture header version {v} not supported")),
        };
        Ok(Tic {
            format,
            kinds,
            swizzle,
            addr: u64::from(w(1)) | u64::from(bits(w(2), 0, 16)) << 32,
            width: bits(w(4), 0, 16) + 1,
            height: bits(w(5), 0, 16) + 1,
            pitch,
            block_height_log2: bits(w(3), 3, 3),
            srgb: w(4) >> 22 & 1 != 0,
            ty,
            depth: bits(w(5), 16, 14) + 1,
            block_depth_log2: bits(w(3), 6, 3),
            levels: bits(w(7), 4, 4) + 1,
            stride_override: None,
        })
    }

    /// The storage element: (block width, block height, bytes); 1x1
    /// texels for uncompressed formats.
    pub fn element(&self) -> (u32, u32, u32) {
        compressed(self.format)
            .unwrap_or_else(|| (1, 1, layout(self.format).unwrap().iter().sum::<u32>() / 8))
    }

    /// Images stored: slices (3D), layers, or 6 faces per cube.
    pub fn layers(&self) -> u32 {
        match self.ty {
            ty::D3 | ty::D1_ARRAY | ty::D2_ARRAY => self.depth,
            ty::CUBE => 6,
            ty::CUBE_ARRAY => 6 * self.depth,
            _ => 1,
        }
    }

    /// Block height (log2 GOBs) and byte size of mip `level` of one
    /// layer (block-linear).
    fn level_layout(&self, level: u32) -> (u32, u64) {
        let (bw, bh, bytes) = self.element();
        let w = (self.width >> level).max(1);
        let h = (self.height >> level).max(1);
        let rows = h.div_ceil(bh);
        // Small levels use smaller blocks.
        let mut gobs = self.block_height_log2;
        while gobs > 0 && rows <= (4 << gobs) {
            gobs -= 1;
        }
        (
            gobs,
            blocklinear::size(w.div_ceil(bw) * bytes, rows, gobs) as u64,
        )
    }

    /// Bytes between array layers (cube faces): all mip levels of one
    /// layer, aligned to a level-0 block.
    pub fn layer_stride(&self) -> u64 {
        if let Some(s) = self.stride_override {
            return s;
        }
        let (_, bh, _) = self.element();
        if let Some(p) = self.pitch {
            return u64::from(p) * u64::from(self.height.div_ceil(bh));
        }
        let total: u64 = (0..self.levels).map(|l| self.level_layout(l).1).sum();
        let align = 512u64 << self.block_height_log2;
        total.next_multiple_of(align)
    }

    /// Mip levels that can be decoded: block-linear 2D, array and cube
    /// textures; down to 1x1 at most.
    pub fn mip_levels(&self) -> u32 {
        if self.pitch.is_some() || self.ty == ty::D3 || self.stride_override.is_some() {
            return 1;
        }
        let full = 32 - self.width.max(self.height).max(1).leading_zeros();
        self.levels.clamp(1, full)
    }

    /// The header of mip `level` alone (same layers and layer stride),
    /// with the block height the level really uses (small levels,
    /// level 0 included, use smaller blocks; yuzu AdjustMipBlockSize).
    pub fn level(&self, level: u32) -> Tic {
        if self.pitch.is_some() || self.ty == ty::D3 {
            return *self;
        }
        let offset: u64 = (0..level).map(|l| self.level_layout(l).1).sum();
        let (gobs, _) = self.level_layout(level);
        Tic {
            addr: self.addr + offset,
            width: (self.width >> level).max(1),
            height: (self.height >> level).max(1),
            block_height_log2: gobs,
            levels: 1,
            stride_override: Some(self.layer_stride()),
            ..*self
        }
    }

    /// Bytes of guest memory the whole texture (all layers and levels)
    /// spans, for detecting changes.
    pub fn byte_span(&self) -> u64 {
        let (bw, bh, bytes) = self.element();
        let rows = self.height.div_ceil(bh);
        let row = self.width.div_ceil(bw) * bytes;
        let layers = u64::from(self.layers().clamp(1, 2048));
        if self.ty == ty::D3 && self.pitch.is_none() {
            let block_rows = 8u32 << self.block_height_log2;
            return u64::from(row.div_ceil(64) * 64)
                * u64::from(rows.next_multiple_of(block_rows))
                * layers.next_multiple_of(1 << self.block_depth_log2);
        }
        if self.ty == ty::D3 {
            return u64::from(self.pitch.unwrap_or(row)) * u64::from(rows) * layers;
        }
        self.layer_stride() * layers
    }

    /// Bytes of level 0 (an upper bound for block-linear layouts).
    pub fn size_bytes(&self) -> u64 {
        let (bw, bh, bytes) = self.element();
        let row = u64::from(self.width.div_ceil(bw) * bytes);
        let rows = u64::from(self.height.div_ceil(bh));
        match self.pitch {
            Some(p) => u64::from(p) * rows,
            None => blocklinear::size(row as u32, rows as u32, self.block_height_log2) as u64,
        }
    }
}

/// Address modes (TSC).
pub mod wrap {
    pub const REPEAT: u8 = 0;
    pub const MIRRORED_REPEAT: u8 = 1;
    pub const CLAMP_TO_EDGE: u8 = 2;
    pub const CLAMP_TO_BORDER: u8 = 3;
    pub const CLAMP: u8 = 4;
}

/// A decoded sampler.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Tsc {
    pub wrap: [u8; 3],
    /// Linear (true) or nearest magnification; the software renderer
    /// samples level 0 with it for minification too.
    pub linear: bool,
    pub border: [f32; 4],
    /// Linear minification.
    pub min_linear: bool,
    /// Mip filter: 1 none, 2 nearest, 3 linear.
    pub mip_filter: u8,
    pub lod_bias: f32,
    pub min_lod: f32,
    pub max_lod: f32,
}

impl Tsc {
    pub fn parse(e: &[u8; 32]) -> Tsc {
        let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().unwrap());
        Tsc {
            wrap: [0, 1, 2].map(|i| bits(w(0), 3 * i, 3) as u8),
            linear: bits(w(1), 0, 2) == 2,
            border: [4, 5, 6, 7].map(|i| f32::from_bits(w(i))),
            min_linear: bits(w(1), 4, 2) == 2,
            mip_filter: bits(w(1), 6, 2) as u8,
            lod_bias: ((bits(w(1), 12, 13) << 19) as i32 >> 19) as f32 / 256.0,
            min_lod: bits(w(2), 0, 12) as f32 / 256.0,
            max_lod: bits(w(2), 12, 12) as f32 / 256.0,
        }
    }
}

/// Texels of level 0 as RGBA floats (after the header's swizzle).
#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    /// Images (slices, layers or cube faces) stored one after another.
    pub layers: u32,
    pub ty: u32,
    pub texels: Vec<[f32; 4]>,
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Decodes one texel's components (R, G, B, A, missing ones 0).
fn components(tic: &Tic, raw: &[u8]) -> [f32; 4] {
    if tic.format == 0x20 {
        let v = u32::from_le_bytes(raw[..4].try_into().unwrap());
        let scale = 2f32.powi((v >> 27) as i32 - 15 - 9);
        return [
            (v & 0x1FF) as f32 * scale,
            (v >> 9 & 0x1FF) as f32 * scale,
            (v >> 18 & 0x1FF) as f32 * scale,
            1.0,
        ];
    }
    let widths = layout(tic.format).unwrap();
    let mut out = [0f32; 4];
    let mut bit = 0u32;
    for (i, &w) in widths.iter().enumerate() {
        let v = if w == 32 {
            u32::from_le_bytes(
                raw[(bit / 8) as usize..(bit / 8 + 4) as usize]
                    .try_into()
                    .unwrap(),
            )
        } else {
            let mut word = 0u64;
            for (k, &b) in raw.iter().enumerate().take(8) {
                word |= u64::from(b) << (8 * k);
            }
            ((word >> bit) & ((1u64 << w) - 1)) as u32
        };
        let max = ((1u64 << w) - 1) as f32;
        out[i] = match tic.kinds[i] {
            Kind::Unorm => {
                let f = v as f32 / max;
                if tic.srgb && i < 3 {
                    srgb_to_linear(f)
                } else {
                    f
                }
            }
            Kind::Snorm => {
                let s = ((v << (32 - w)) as i32 >> (32 - w)) as f32;
                (s / (max / 2.0).floor()).max(-1.0)
            }
            Kind::Float => match w {
                32 => f32::from_bits(v),
                16 => crate::draw::half_to_f32(v as u16),
                // Unsigned small floats: 5 exponent bits, no sign.
                11 | 10 => {
                    let m = w - 5;
                    crate::draw::half_to_f32(
                        (((v >> m) << 10) | ((v & ((1 << m) - 1)) << (10 - m))) as u16,
                    )
                }
                _ => 0.0,
            },
            // Integer textures return the integer bits.
            Kind::Uint => f32::from_bits(v),
            Kind::Sint => f32::from_bits(((v << (32 - w)) as i32 >> (32 - w)) as u32),
        };
        bit += w;
    }
    out
}

/// Texels of one image from its tightly packed storage rows.
fn convert(tic: &Tic, linear: &[u8]) -> Vec<[f32; 4]> {
    let (bw, bh, bpp) = tic.element();
    if (bw, bh) == (1, 1) {
        return linear
            .chunks_exact(bpp as usize)
            .map(|t| swizzle(tic, components(tic, t)))
            .collect();
    }
    let cols = tic.width.div_ceil(bw) as usize;
    let (w, h) = (tic.width as usize, tic.height as usize);
    let mut texels = vec![[0f32; 4]; w * h];
    let block_row = cols * bpp as usize;
    // One block row: its blocks decoded into `out` (bh texel rows).
    let decode_row = |data: &[u8], out: &mut [[f32; 4]]| {
        let mut block = vec![0u32; (bw * bh) as usize];
        let rows = out.len() / w.max(1);
        for (bx, data) in data.chunks_exact(bpp as usize).enumerate() {
            decode_block(tic.format, bw, bh, data, &mut block);
            for (k, &p) in block.iter().enumerate() {
                let (x, y) = (bx * bw as usize + k % bw as usize, k / bw as usize);
                if x < w && y < rows {
                    let [b, gr, r, a] = p.to_le_bytes();
                    let mut c = [r, gr, b, a].map(|v| f32::from(v) / 255.0);
                    if tic.srgb {
                        for v in &mut c[..3] {
                            *v = srgb_to_linear(*v);
                        }
                    }
                    out[y * w + x] = swizzle(tic, c);
                }
            }
        }
    };
    type Row<'a> = (&'a [u8], &'a mut [[f32; 4]]);
    let pairs: Vec<Row> = linear
        .chunks(block_row.max(1))
        .zip(texels.chunks_mut(w * bh as usize))
        .collect();
    // Block-compressed decoding (ASTC especially) is the costly part of
    // a texture upload: large textures use every host core.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if threads > 1 && w * h >= 256 * 256 {
        let per = pairs.len().div_ceil(threads);
        let mut groups: Vec<Vec<Row>> = Vec::new();
        let mut it = pairs.into_iter().peekable();
        while it.peek().is_some() {
            groups.push(it.by_ref().take(per).collect());
        }
        std::thread::scope(|s| {
            for group in groups {
                let decode_row = &decode_row;
                s.spawn(move || {
                    for (data, out) in group {
                        decode_row(data, out);
                    }
                });
            }
        });
    } else {
        for (data, out) in pairs {
            decode_row(data, out);
        }
    }
    texels
}

fn swizzle(tic: &Tic, c: [f32; 4]) -> [f32; 4] {
    tic.swizzle.map(|s| match s {
        2..=5 => c[(s - 2) as usize],
        6 => f32::from_bits(1),
        7 => 1.0,
        _ => 0.0,
    })
}

impl Texture {
    pub fn decode(g: &GuestMemory, tic: &Tic) -> Result<Texture, String> {
        let tic = &tic.level(0);
        // Rows of storage elements (texels, or compressed blocks).
        let (bw, bh, bpp) = tic.element();
        let (cols, rows) = (tic.width.div_ceil(bw), tic.height.div_ceil(bh));
        let row = cols * bpp;
        let layers = tic.layers().clamp(1, 2048);
        let mut texels = Vec::with_capacity((tic.width * tic.height * layers) as usize);
        let read = |at: u64, len: usize| -> Result<Vec<u8>, String> {
            let mut raw = vec![0u8; len];
            g.read(at, &mut raw)
                .map_err(|f| format!("texture: {f:?}"))?;
            Ok(raw)
        };
        if tic.ty == ty::D3 && tic.pitch.is_none() {
            // Block-linear 3D: slices interleave inside blocks.
            let block_rows = 8u32 << tic.block_height_log2;
            let size = (cols * bpp).div_ceil(64) as usize
                * 64
                * rows.next_multiple_of(block_rows) as usize
                * layers.next_multiple_of(1 << tic.block_depth_log2) as usize;
            let raw = read(tic.addr, size)?;
            for z in 0..layers {
                let mut linear = vec![0u8; (row * rows) as usize];
                for y in 0..rows {
                    for x in (0..row.next_multiple_of(16)).step_by(16) {
                        let n = 16.min(row - x) as usize;
                        let from = blocklinear::offset3d(
                            x,
                            y,
                            z,
                            row,
                            rows,
                            tic.block_height_log2,
                            tic.block_depth_log2,
                        );
                        let to = (y * row + x) as usize;
                        if let Some(piece) = raw.get(from..from + n) {
                            linear[to..to + n].copy_from_slice(piece);
                        }
                    }
                }
                texels.extend(convert(tic, &linear));
            }
        } else {
            let stride = if tic.ty == ty::D3 {
                u64::from(tic.pitch.unwrap_or(row)) * u64::from(rows)
            } else {
                tic.layer_stride()
            };
            for layer in 0..layers {
                let at = tic.addr + u64::from(layer) * stride;
                let linear = match tic.pitch {
                    Some(pitch) => {
                        let raw = read(at, (pitch * (rows - 1) + row) as usize)?;
                        let mut out = Vec::with_capacity((row * rows) as usize);
                        for y in 0..rows {
                            let s = (y * pitch) as usize;
                            out.extend_from_slice(&raw[s..s + row as usize]);
                        }
                        out
                    }
                    None => {
                        // Level 0 uses the header's block height.
                        let size = blocklinear::size(row, rows, tic.block_height_log2);
                        let raw = read(at, size)?;
                        blocklinear::deswizzle_width(
                            &raw,
                            row,
                            row.next_multiple_of(16),
                            rows,
                            tic.block_height_log2,
                        )
                    }
                };
                texels.extend(convert(tic, &linear));
            }
        }
        Ok(Texture {
            width: tic.width,
            height: tic.height,
            layers,
            ty: tic.ty,
            texels,
        })
    }

    /// Texel (x, y) of image `layer`, with the sampler's address modes.
    fn texel_in(&self, layer: u32, x: i64, y: i64, s: &Tsc) -> [f32; 4] {
        let base = (layer.min(self.layers - 1) * self.width * self.height) as usize;
        let t = self.texel(x, y, s);
        if layer == 0 {
            return t;
        }
        // Same addressing, offset into the layer.
        let fix = |c: i64, n: u32| c.clamp(0, i64::from(n) - 1) as u32;
        match (s.wrap[0], s.wrap[1]) {
            (wrap::CLAMP_TO_BORDER, _) | (_, wrap::CLAMP_TO_BORDER)
                if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) =>
            {
                s.border
            }
            _ => {
                let wx = match s.wrap[0] {
                    wrap::REPEAT => x.rem_euclid(i64::from(self.width)) as u32,
                    _ => fix(x, self.width),
                };
                let wy = match s.wrap[1] {
                    wrap::REPEAT => y.rem_euclid(i64::from(self.height)) as u32,
                    _ => fix(y, self.height),
                };
                self.texels[base + (wy * self.width + wx) as usize]
            }
        }
    }

    /// Bilinear (or nearest) sample of image `layer` at (u, v).
    pub fn sample_layer(&self, s: &Tsc, u: f32, v: f32, layer: u32) -> [f32; 4] {
        if layer == 0 {
            return self.sample(s, u, v);
        }
        let (x, y) = (u * self.width as f32, v * self.height as f32);
        if !x.is_finite() || !y.is_finite() {
            return [0.0; 4];
        }
        if !s.linear {
            return self.texel_in(layer, x.floor() as i64, y.floor() as i64, s);
        }
        let (x, y) = (x - 0.5, y - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let t = [
            self.texel_in(layer, x0, y0, s),
            self.texel_in(layer, x0 + 1, y0, s),
            self.texel_in(layer, x0, y0 + 1, s),
            self.texel_in(layer, x0 + 1, y0 + 1, s),
        ];
        std::array::from_fn(|c| {
            let top = t[0][c] + (t[1][c] - t[0][c]) * fx;
            let bottom = t[2][c] + (t[3][c] - t[2][c]) * fx;
            top + (bottom - top) * fy
        })
    }

    /// Samples by the texture's own type: `c` holds (u, v, w) for 2D/3D,
    /// a direction for cubes; `layer` the array layer.
    pub fn sample_any(&self, s: &Tsc, c: [f32; 3], layer: f32) -> [f32; 4] {
        let layer_index = |n: u32| (layer.round().max(0.0) as u32).min(n.saturating_sub(1));
        match self.ty {
            ty::D3 => {
                let z = c[2] * self.layers as f32 - 0.5;
                if !z.is_finite() {
                    return [0.0; 4];
                }
                if !s.linear {
                    let z = (c[2] * self.layers as f32)
                        .floor()
                        .clamp(0.0, (self.layers - 1) as f32);
                    return self.sample_layer(s, c[0], c[1], z as u32);
                }
                let z0 = z.floor();
                let f = z - z0;
                let clamp = |z: f32| z.clamp(0.0, (self.layers - 1) as f32) as u32;
                let a = self.sample_layer(s, c[0], c[1], clamp(z0));
                let b = self.sample_layer(s, c[0], c[1], clamp(z0 + 1.0));
                std::array::from_fn(|k| a[k] + (b[k] - a[k]) * f)
            }
            ty::CUBE | ty::CUBE_ARRAY => {
                let (face, u, v) = cube_face(c);
                let cube = layer_index(self.layers / 6);
                let edge = Tsc {
                    wrap: [wrap::CLAMP_TO_EDGE; 3],
                    ..*s
                };
                self.sample_layer(&edge, u, v, cube * 6 + face)
            }
            ty::D1_ARRAY | ty::D2_ARRAY => {
                self.sample_layer(s, c[0], c[1], layer_index(self.layers))
            }
            _ => self.sample(s, c[0], c[1]),
        }
    }

    /// Fetches texel (x, y) of image `layer` (z for 3D), zero outside.
    pub fn fetch_layer(&self, x: i32, y: i32, layer: i32) -> [f32; 4] {
        if x < 0
            || y < 0
            || layer < 0
            || x as u32 >= self.width
            || y as u32 >= self.height
            || layer as u32 >= self.layers
        {
            return [0.0; 4];
        }
        let base = (layer as u32 * self.width * self.height) as usize;
        self.texels[base + (y as u32 * self.width + x as u32) as usize]
    }

    fn texel(&self, x: i64, y: i64, s: &Tsc) -> [f32; 4] {
        let fix = |c: i64, size: u32, mode: u8| -> Option<u32> {
            let n = i64::from(size);
            match mode {
                wrap::REPEAT => Some(c.rem_euclid(n) as u32),
                wrap::MIRRORED_REPEAT => {
                    let m = c.rem_euclid(2 * n);
                    Some(if m < n { m } else { 2 * n - 1 - m } as u32)
                }
                wrap::CLAMP_TO_BORDER => (0..n).contains(&c).then_some(c as u32),
                wrap::CLAMP_TO_EDGE | wrap::CLAMP => Some(c.clamp(0, n - 1) as u32),
                // Mirror-clamp variants: mirror once, then clamp.
                _ => Some(if c < 0 { -c - 1 } else { c }.clamp(0, n - 1) as u32),
            }
        };
        match (
            fix(x, self.width, s.wrap[0]),
            fix(y, self.height, s.wrap[1]),
        ) {
            (Some(x), Some(y)) => self.texels[(y * self.width + x) as usize],
            _ => s.border,
        }
    }

    /// Texel (x, y) of level 0, zero outside the texture.
    pub fn fetch(&self, x: i32, y: i32) -> [f32; 4] {
        if x < 0 || y < 0 || x as u32 >= self.width || y as u32 >= self.height {
            return [0.0; 4];
        }
        self.texels[(y as u32 * self.width + x as u32) as usize]
    }

    /// Gathers component `comp` of the four texels a bilinear sample at
    /// (u, v) reads, in gather order: (x0, y1), (x1, y1), (x1, y0), (x0, y0).
    pub fn gather(&self, s: &Tsc, u: f32, v: f32, comp: u8) -> [f32; 4] {
        let (x, y) = (u * self.width as f32 - 0.5, v * self.height as f32 - 0.5);
        if !x.is_finite() || !y.is_finite() {
            return [0.0; 4];
        }
        let (x0, y0) = (x.floor() as i64, y.floor() as i64);
        let c = usize::from(comp & 3);
        [(x0, y0 + 1), (x0 + 1, y0 + 1), (x0 + 1, y0), (x0, y0)]
            .map(|(x, y)| self.texel(x, y, s)[c])
    }

    /// Samples level 0 at normalized coordinates.
    pub fn sample(&self, s: &Tsc, u: f32, v: f32) -> [f32; 4] {
        let (x, y) = (u * self.width as f32, v * self.height as f32);
        if !x.is_finite() || !y.is_finite() {
            return [0.0; 4];
        }
        if !s.linear {
            return self.texel(x.floor() as i64, y.floor() as i64, s);
        }
        let (x, y) = (x - 0.5, y - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let t = [
            self.texel(x0, y0, s),
            self.texel(x0 + 1, y0, s),
            self.texel(x0, y0 + 1, s),
            self.texel(x0 + 1, y0 + 1, s),
        ];
        std::array::from_fn(|c| {
            let top = t[0][c] + (t[1][c] - t[0][c]) * fx;
            let bottom = t[2][c] + (t[3][c] - t[2][c]) * fx;
            top + (bottom - top) * fy
        })
    }
}

/// The cube face a direction points at and the (u, v) on it.
pub fn cube_face(d: [f32; 3]) -> (u32, f32, f32) {
    let [x, y, z] = d;
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    let (face, sc, tc, ma) = if ax >= ay && ax >= az {
        if x >= 0.0 {
            (0, -z, -y, ax)
        } else {
            (1, z, -y, ax)
        }
    } else if ay >= az {
        if y >= 0.0 {
            (2, x, z, ay)
        } else {
            (3, x, -z, ay)
        }
    } else if z >= 0.0 {
        (4, x, -y, az)
    } else {
        (5, -x, -y, az)
    };
    if ma == 0.0 {
        return (0, 0.5, 0.5);
    }
    (face, (sc / ma + 1.0) / 2.0, (tc / ma + 1.0) / 2.0)
}

/// The texture and sampler a handle word names.
pub fn resolve(g: &GuestMemory, pools: &Pools, handle: u32) -> Result<(Tic, Tsc), String> {
    let (tic_index, tsc_index) = (handle & 0xF_FFFF, handle >> 20);
    if tic_index > pools.tic_max || tsc_index > pools.tsc_max {
        return Err(format!("texture handle {handle:#x} out of the pools"));
    }
    let mut e = [0u8; 32];
    g.read(pools.tic + 32 * u64::from(tic_index), &mut e)
        .map_err(|f| format!("texture header: {f:?}"))?;
    let tic = Tic::parse(&e)?;
    g.read(pools.tsc + 32 * u64::from(tsc_index), &mut e)
        .map_err(|f| format!("sampler: {f:?}"))?;
    Ok((tic, Tsc::parse(&e)))
}

/// Textures a shader stage uses, by handle offset (bindless constant
/// buffer word), decoded once per draw.
#[derive(Default)]
pub struct Bound(pub HashMap<u32, (Texture, Tsc)>);

impl Bound {
    pub fn sample(&self, handle: u32, u: f32, v: f32) -> [f32; 4] {
        self.0
            .get(&handle)
            .map_or([0.0; 4], |(t, s)| t.sample(s, u, v))
    }

    pub fn fetch(&self, handle: u32, x: i32, y: i32) -> [f32; 4] {
        self.0.get(&handle).map_or([0.0; 4], |(t, _)| t.fetch(x, y))
    }

    pub fn sample_any(&self, handle: u32, c: [f32; 3], layer: f32) -> [f32; 4] {
        self.0
            .get(&handle)
            .map_or([0.0; 4], |(t, s)| t.sample_any(s, c, layer))
    }

    pub fn fetch_layer(&self, handle: u32, x: i32, y: i32, layer: i32) -> [f32; 4] {
        self.0
            .get(&handle)
            .map_or([0.0; 4], |(t, _)| t.fetch_layer(x, y, layer))
    }

    pub fn gather(&self, handle: u32, u: f32, v: f32, comp: u8) -> [f32; 4] {
        self.0
            .get(&handle)
            .map_or([0.0; 4], |(t, s)| t.gather(s, u, v, comp))
    }

    /// Width, height, depth and levels (only level 0 is kept).
    pub fn size(&self, handle: u32, lod: u32) -> [u32; 4] {
        match self.0.get(&handle) {
            Some((t, _)) if lod == 0 => {
                let depth = match t.ty {
                    ty::CUBE | ty::CUBE_ARRAY => t.layers / 6,
                    _ => t.layers,
                };
                [t.width, t.height, depth, 1]
            }
            Some(_) => [0, 0, 0, 1],
            None => [0; 4],
        }
    }
}

/// Builds pool entries (for tests and test programs).
pub mod build {
    fn bytes(words: [u32; 8]) -> [u8; 32] {
        let mut e = [0u8; 32];
        for (i, w) in words.iter().enumerate() {
            e[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        e
    }

    /// An RGBA8 unorm 2D texture header; `pitch` Some for pitch layout,
    /// otherwise block-linear with `block_height_log2`.
    pub fn rgba8_tic(
        addr: u64,
        w: u32,
        h: u32,
        pitch: Option<u32>,
        block_height_log2: u32,
    ) -> [u8; 32] {
        let mut words = [0u32; 8];
        // A8R8G8B8, unorm x4, swizzle R G B A.
        words[0] =
            0x08 | 2 << 7 | 2 << 10 | 2 << 13 | 2 << 16 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28;
        words[1] = addr as u32;
        let version = if pitch.is_some() { 2 } else { 3 };
        words[2] = (addr >> 32) as u32 | version << 21;
        words[3] = match pitch {
            Some(p) => p >> 5,
            None => block_height_log2 << 3,
        };
        let ty = if pitch.is_some() { 7 } else { 1 };
        words[4] = (w - 1) | ty << 23;
        words[5] = (h - 1) | 1 << 31;
        bytes(words)
    }

    /// A sampler with the same address mode for u, v and p.
    pub fn tsc(wrap: u8, linear: bool, border: [f32; 4]) -> [u8; 32] {
        let filter = if linear { 2 } else { 1 };
        let mut words = [0u32; 8];
        words[0] = u32::from(wrap) * 0b001_001_001;
        words[1] = filter | filter << 4 | 1 << 6;
        for i in 0..4 {
            words[4 + i] = border[i].to_bits();
        }
        bytes(words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tic_rgba8_pitch(addr: u64, w: u32, h: u32, pitch: u32) -> [u8; 32] {
        build::rgba8_tic(addr, w, h, Some(pitch), 0)
    }

    #[test]
    fn header_fields() {
        let t = Tic::parse(&tic_rgba8_pitch(0x12_3456_7800, 4, 2, 64)).unwrap();
        assert_eq!(
            (t.addr, t.width, t.height, t.pitch),
            (0x12_3456_7800, 4, 2, Some(64))
        );
        assert_eq!(t.swizzle, [2, 3, 4, 5]);
        assert_eq!(t.element(), (1, 1, 4));
        assert_eq!(components(&t, &[255, 0, 51, 255]), [1.0, 0.0, 0.2, 1.0]);
    }

    #[test]
    fn sampler_fields() {
        let s = Tsc::parse(&build::tsc(
            wrap::CLAMP_TO_BORDER,
            true,
            [0.0, 0.0, 0.0, 1.0],
        ));
        assert_eq!(s.wrap, [3, 3, 3]);
        assert!(s.linear);
        assert_eq!(s.border, [0.0, 0.0, 0.0, 1.0]);
        let t = Tic::parse(&build::rgba8_tic(0x1000, 16, 16, None, 2)).unwrap();
        assert_eq!((t.pitch, t.block_height_log2), (None, 2));
    }

    #[test]
    fn mip_level_headers() {
        let mut t = Tic::parse(&build::rgba8_tic(0x10000, 64, 64, None, 4)).unwrap();
        t.levels = 7;
        assert_eq!(t.mip_levels(), 7);
        // Level 0: 256-byte rows, 64 rows: the 16-GOB block shrinks to
        // 8 GOBs, 16 KiB.
        assert_eq!(t.level(0).block_height_log2, 3);
        let l1 = t.level(1);
        assert_eq!((l1.addr, l1.width, l1.height), (0x10000 + 0x4000, 32, 32));
        assert_eq!(l1.block_height_log2, 2);
        assert_eq!(l1.layer_stride(), t.layer_stride());
        assert_eq!(l1.mip_levels(), 1);
    }

    #[test]
    fn packed_float_formats() {
        let mut t = Tic::parse(&build::rgba8_tic(0, 4, 4, Some(64), 0)).unwrap();
        t.format = 0x21;
        t.kinds = [Kind::Float; 4];
        // R = 1.0 (exp 15), G = 2.0 (exp 16), B = 0.5 (exp 14).
        let v: u32 = (15 << 6) | (16 << 6) << 11 | (14 << 5) << 22;
        assert_eq!(t.element(), (1, 1, 4));
        assert_eq!(components(&t, &v.to_le_bytes())[..3], [1.0, 2.0, 0.5]);
        t.format = 0x20;
        // Shared exponent 15+9 = 24 → scale 1: R = 3.
        let v: u32 = 3 | 24 << 27;
        assert_eq!(components(&t, &v.to_le_bytes())[0], 3.0);
    }

    #[test]
    fn bc1_block() {
        // Color 0 red (565 0xF800), color 1 blue (0x001F); indices: row 0
        // all 0 (red), row 1 all 1 (blue), rows 2-3 color 2 (2/3 red).
        let block = [0x00, 0xF8, 0x1F, 0x00, 0x00, 0x55, 0xAA, 0xAA];
        let mut out = [0u32; 16];
        decode_block(0x24, 4, 4, &block, &mut out);
        assert_eq!(out[0].to_le_bytes(), [0, 0, 255, 255]); // B, G, R, A
        assert_eq!(out[4].to_le_bytes(), [255, 0, 0, 255]);
        let mut tic = Tic::parse(&build::rgba8_tic(0, 8, 8, Some(64), 0)).unwrap();
        tic.format = 0x24;
        assert_eq!(tic.element(), (4, 4, 8));
    }

    #[test]
    fn filtering_and_wrapping() {
        let tex = Texture {
            width: 2,
            height: 1,
            layers: 1,
            ty: ty::D2,
            texels: vec![[0.0; 4], [1.0; 4]],
        };
        let mut s = Tsc {
            wrap: [wrap::REPEAT; 3],
            linear: false,
            border: [0.5; 4],
            ..Default::default()
        };
        assert_eq!(tex.sample(&s, 0.25, 0.5)[0], 0.0);
        assert_eq!(tex.sample(&s, 0.75, 0.5)[0], 1.0);
        assert_eq!(tex.sample(&s, 1.25, 0.5)[0], 0.0);
        s.linear = true;
        assert_eq!(tex.sample(&s, 0.5, 0.5)[0], 0.5);
        // Repeat: between the last and the first texel.
        assert_eq!(tex.sample(&s, 1.0, 0.5)[0], 0.5);
        s.wrap = [wrap::CLAMP_TO_EDGE; 3];
        assert_eq!(tex.sample(&s, 1.0, 0.5)[0], 1.0);
        s.wrap = [wrap::CLAMP_TO_BORDER; 3];
        s.linear = false;
        assert_eq!(tex.sample(&s, -0.25, 0.5), [0.5; 4]);
    }

    #[test]
    fn arrays_3d_and_cubes() {
        let s = Tsc {
            wrap: [wrap::CLAMP_TO_EDGE; 3],
            linear: false,
            border: [0.0; 4],
            ..Default::default()
        };
        // Six 1x1 faces (or layers, or slices) holding their index.
        let texels: Vec<[f32; 4]> = (0..6).map(|i| [i as f32; 4]).collect();
        let tex = |ty| Texture {
            width: 1,
            height: 1,
            layers: 6,
            ty,
            texels: texels.clone(),
        };
        assert_eq!(
            tex(ty::D2_ARRAY).sample_any(&s, [0.5, 0.5, 0.0], 3.0)[0],
            3.0
        );
        assert_eq!(tex(ty::D3).sample_any(&s, [0.5, 0.5, 0.9], 0.0)[0], 5.0);
        let cube = tex(ty::CUBE);
        assert_eq!(cube.sample_any(&s, [1.0, 0.0, 0.0], 0.0)[0], 0.0);
        assert_eq!(cube.sample_any(&s, [0.0, -1.0, 0.0], 0.0)[0], 3.0);
        assert_eq!(cube.sample_any(&s, [0.0, 0.0, -1.0], 0.0)[0], 5.0);
        assert_eq!(tex(ty::D2_ARRAY).fetch_layer(0, 0, 4)[0], 4.0);
    }

    #[test]
    fn three_d_block_linear_offsets() {
        // With one slice per block, slices follow each other.
        let plane = crate::blocklinear::size(64, 8, 0);
        assert_eq!(crate::blocklinear::offset3d(0, 0, 1, 64, 8, 0, 0), plane);
        // Two slices per block: slice 1 is the next GOB.
        assert_eq!(crate::blocklinear::offset3d(0, 0, 1, 64, 8, 0, 1), 512);
    }
}
