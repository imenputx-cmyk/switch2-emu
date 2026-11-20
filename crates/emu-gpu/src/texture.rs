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
//! common uncompressed formats, nearest and linear filtering, all
//! address modes (mirror-clamp ones treated as mirror then clamp).

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
        _ => return None,
    })
}

fn bits(v: u32, lo: u32, len: u32) -> u32 {
    (v >> lo) & ((1u64 << len) - 1) as u32
}

impl Tic {
    pub fn parse(e: &[u8; 32]) -> Result<Tic, String> {
        let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().unwrap());
        let format = bits(w(0), 0, 7);
        layout(format).ok_or_else(|| format!("texture format {format:#x}"))?;
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
        if ty != 1 && ty != 7 {
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
        })
    }

    pub fn bytes_per_texel(&self) -> u32 {
        layout(self.format).unwrap().iter().sum::<u32>() / 8
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tsc {
    pub wrap: [u8; 3],
    /// Linear (true) or nearest magnification; minification uses the
    /// same filter since only level 0 is sampled.
    pub linear: bool,
    pub border: [f32; 4],
}

impl Tsc {
    pub fn parse(e: &[u8; 32]) -> Tsc {
        let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().unwrap());
        Tsc {
            wrap: [0, 1, 2].map(|i| bits(w(0), 3 * i, 3) as u8),
            linear: bits(w(1), 0, 2) == 2,
            border: [4, 5, 6, 7].map(|i| f32::from_bits(w(i))),
        }
    }
}

/// Texels of level 0 as RGBA floats (after the header's swizzle).
#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
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
        let bpp = tic.bytes_per_texel();
        let row = tic.width * bpp;
        let linear = match tic.pitch {
            Some(pitch) => {
                let mut raw = vec![0u8; (pitch * (tic.height - 1) + row) as usize];
                g.read(tic.addr, &mut raw)
                    .map_err(|f| format!("texture: {f:?}"))?;
                let mut out = Vec::with_capacity((row * tic.height) as usize);
                for y in 0..tic.height {
                    let s = (y * pitch) as usize;
                    out.extend_from_slice(&raw[s..s + row as usize]);
                }
                out
            }
            None => {
                // Small textures use smaller blocks than the header says
                // only for mip levels; level 0 uses the header's value.
                let size = blocklinear::size(row, tic.height, tic.block_height_log2);
                let mut raw = vec![0u8; size];
                g.read(tic.addr, &mut raw)
                    .map_err(|f| format!("texture: {f:?}"))?;
                blocklinear::deswizzle_width(
                    &raw,
                    row,
                    row.next_multiple_of(16),
                    tic.height,
                    tic.block_height_log2,
                )
            }
        };
        let texels = linear
            .chunks_exact(bpp as usize)
            .map(|t| swizzle(tic, components(tic, t)))
            .collect();
        Ok(Texture {
            width: tic.width,
            height: tic.height,
            texels,
        })
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
        assert_eq!(t.bytes_per_texel(), 4);
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
    fn filtering_and_wrapping() {
        let tex = Texture {
            width: 2,
            height: 1,
            texels: vec![[0.0; 4], [1.0; 4]],
        };
        let mut s = Tsc {
            wrap: [wrap::REPEAT; 3],
            linear: false,
            border: [0.5; 4],
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
}
