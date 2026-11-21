//! Fixed-function state of a draw: the depth buffer and depth test, and
//! color blending.
//!
//! The 3D class takes both Direct3D-style and OpenGL-style enum values
//! for compare functions, blend equations and blend factors (deko3d's
//! `gpu_3d_state.cpp` explains the factor encoding: D3D11_BLEND values,
//! or GL values with bit 14 set); both are accepted here.

use crate::backend::GuestMemory;
use crate::blocklinear;

/// Depth buffer formats (SET_ZT_FORMAT).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthFormat {
    Z32Float,
    Z16Unorm,
    /// Stencil in the low byte, depth in the upper 24 bits.
    S8Z24Unorm,
    /// Depth in the low 24 bits, stencil in the top byte (format 0x16).
    Z24S8Unorm,
    /// 32-bit float depth followed by a 32-bit word with stencil (0x19).
    Z32FloatX24S8,
}

impl DepthFormat {
    pub fn from_raw(v: u32) -> Option<Self> {
        Some(match v {
            0x0A => DepthFormat::Z32Float,
            0x13 => DepthFormat::Z16Unorm,
            0x14 | 0x15 => DepthFormat::S8Z24Unorm,
            0x16 => DepthFormat::Z24S8Unorm,
            0x19 => DepthFormat::Z32FloatX24S8,
            _ => return None,
        })
    }

    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            DepthFormat::Z16Unorm => 2,
            DepthFormat::Z32FloatX24S8 => 8,
            _ => 4,
        }
    }

    pub fn decode(self, p: &[u8]) -> f32 {
        match self {
            DepthFormat::Z32Float | DepthFormat::Z32FloatX24S8 => {
                f32::from_le_bytes(p[..4].try_into().unwrap())
            }
            DepthFormat::Z16Unorm => f32::from(u16::from_le_bytes([p[0], p[1]])) / 65535.0,
            DepthFormat::Z24S8Unorm => {
                (u32::from_le_bytes(p[..4].try_into().unwrap()) & 0xFF_FFFF) as f32 / 16_777_215.0
            }
            DepthFormat::S8Z24Unorm => {
                (u32::from_le_bytes(p[..4].try_into().unwrap()) >> 8) as f32 / 16_777_215.0
            }
        }
    }

    /// The stencil byte of a pixel, if the format has one.
    pub fn stencil_byte(self) -> Option<usize> {
        match self {
            DepthFormat::S8Z24Unorm => Some(0),
            DepthFormat::Z24S8Unorm => Some(3),
            DepthFormat::Z32FloatX24S8 => Some(4),
            _ => None,
        }
    }

    /// Writes depth `z` into the pixel bytes `p` (keeping stencil).
    pub fn encode(self, z: f32, p: &mut [u8]) {
        let z = z.clamp(0.0, 1.0);
        match self {
            DepthFormat::Z32Float | DepthFormat::Z32FloatX24S8 => {
                p[..4].copy_from_slice(&z.to_le_bytes())
            }
            DepthFormat::Z24S8Unorm => {
                let d = (z * 16_777_215.0).round() as u32;
                let v = d | u32::from(p[3]) << 24;
                p[..4].copy_from_slice(&v.to_le_bytes());
            }
            DepthFormat::Z16Unorm => {
                p[..2].copy_from_slice(&((z * 65535.0).round() as u16).to_le_bytes())
            }
            DepthFormat::S8Z24Unorm => {
                let d = (z * 16_777_215.0).round() as u32;
                let v = d << 8 | u32::from(p[0]);
                p[..4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

/// A block-linear depth buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthSurface {
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pub format: DepthFormat,
    pub block_height_log2: u32,
}

impl DepthSurface {
    pub fn row_bytes(&self) -> u32 {
        self.width * self.format.bytes_per_pixel()
    }

    pub fn byte_addr(&self, x: u32, y: u32) -> u64 {
        let bpp = self.format.bytes_per_pixel();
        self.addr + blocklinear::offset(x * bpp, y, self.row_bytes(), self.block_height_log2) as u64
    }

    pub fn size_bytes(&self) -> usize {
        blocklinear::size(self.row_bytes(), self.height, self.block_height_log2)
    }

    /// Reads the surface as tightly packed rows.
    pub fn read_linear(&self, g: &GuestMemory) -> Result<Vec<u8>, String> {
        let mut raw = vec![0u8; self.size_bytes()];
        g.read(self.addr, &mut raw)
            .map_err(|f| format!("depth buffer: {f:?}"))?;
        let row = self.row_bytes();
        Ok(blocklinear::deswizzle_width(
            &raw,
            row,
            row.next_multiple_of(16),
            self.height,
            self.block_height_log2,
        ))
    }

    /// Writes tightly packed rows back.
    pub fn write_linear(&self, g: &mut GuestMemory, data: &[u8]) -> Result<(), String> {
        let mut raw = vec![0u8; self.size_bytes()];
        g.read(self.addr, &mut raw)
            .map_err(|f| format!("depth buffer: {f:?}"))?;
        blocklinear::swizzle_into(
            &mut raw,
            data,
            self.row_bytes(),
            self.height,
            self.block_height_log2,
        );
        g.write(self.addr, &raw)
            .map_err(|f| format!("depth buffer: {f:?}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compare {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl Compare {
    /// D3D style 1..=8 or GL style 0x200..=0x207.
    pub fn from_raw(v: u32) -> Option<Compare> {
        let i = match v {
            1..=8 => v - 1,
            0x200..=0x207 => v - 0x200,
            _ => return None,
        };
        Some(
            [
                Compare::Never,
                Compare::Less,
                Compare::Equal,
                Compare::LessEqual,
                Compare::Greater,
                Compare::NotEqual,
                Compare::GreaterEqual,
                Compare::Always,
            ][i as usize],
        )
    }

    pub fn test(self, a: f32, b: f32) -> bool {
        match self {
            Compare::Never => false,
            Compare::Less => a < b,
            Compare::Equal => a == b,
            Compare::LessEqual => a <= b,
            Compare::Greater => a > b,
            Compare::NotEqual => a != b,
            Compare::GreaterEqual => a >= b,
            Compare::Always => true,
        }
    }
}

/// Stencil operations (D3D 1..=8 or GL values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace,
    IncrSaturate,
    DecrSaturate,
    Invert,
    IncrWrap,
    DecrWrap,
}

impl StencilOp {
    pub fn from_raw(v: u32) -> StencilOp {
        match v {
            2 | 0 => StencilOp::Zero,
            3 | 0x1E01 => StencilOp::Replace,
            4 | 0x1E02 => StencilOp::IncrSaturate,
            5 | 0x1E03 => StencilOp::DecrSaturate,
            6 | 0x150A => StencilOp::Invert,
            7 | 0x8507 => StencilOp::IncrWrap,
            8 | 0x8508 => StencilOp::DecrWrap,
            _ => StencilOp::Keep,
        }
    }

    pub fn apply(self, s: u8, reference: u8) -> u8 {
        match self {
            StencilOp::Keep => s,
            StencilOp::Zero => 0,
            StencilOp::Replace => reference,
            StencilOp::IncrSaturate => s.saturating_add(1),
            StencilOp::DecrSaturate => s.saturating_sub(1),
            StencilOp::Invert => !s,
            StencilOp::IncrWrap => s.wrapping_add(1),
            StencilOp::DecrWrap => s.wrapping_sub(1),
        }
    }
}

/// One face's stencil state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFace {
    pub fail: StencilOp,
    pub zfail: StencilOp,
    pub zpass: StencilOp,
    pub func: Compare,
    pub reference: u8,
    pub func_mask: u8,
    pub write_mask: u8,
}

impl Default for StencilFace {
    fn default() -> Self {
        StencilFace {
            fail: StencilOp::Keep,
            zfail: StencilOp::Keep,
            zpass: StencilOp::Keep,
            func: Compare::Always,
            reference: 0,
            func_mask: 0xFF,
            write_mask: 0xFF,
        }
    }
}

impl StencilFace {
    /// The stencil test against the stored value.
    pub fn test(&self, stored: u8) -> bool {
        let m = self.func_mask;
        self.func
            .test(f32::from(self.reference & m), f32::from(stored & m))
    }

    /// Applies `op` to `stored` through the write mask.
    pub fn update(&self, op: StencilOp, stored: u8) -> u8 {
        let v = op.apply(stored, self.reference);
        (stored & !self.write_mask) | (v & self.write_mask)
    }
}

/// Stencil test state (both faces).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Stencil {
    pub enabled: bool,
    pub front: StencilFace,
    pub back: StencilFace,
}

/// Face culling, in Vulkan's convention (framebuffer coordinates, y
/// down): `front_ccw` says counter-clockwise triangles face forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Cull {
    pub front_ccw: bool,
    pub cull_front: bool,
    pub cull_back: bool,
}

impl Cull {
    /// Whether a triangle with edge-function area `area` (positive:
    /// clockwise in y-down framebuffer space) is culled.
    pub fn culls(&self, area: f32) -> bool {
        let front = if self.front_ccw {
            area < 0.0
        } else {
            area > 0.0
        };
        if front {
            self.cull_front
        } else {
            self.cull_back
        }
    }
}

/// Depth buffer and test state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Depth {
    /// The depth buffer, if one is bound and enabled.
    pub target: Option<DepthSurface>,
    pub test: bool,
    pub write: bool,
    pub func: Compare,
    /// Stencil test (needs a target format with stencil bits).
    pub stencil: Stencil,
}

impl Default for Depth {
    fn default() -> Self {
        Depth {
            target: None,
            test: false,
            write: false,
            stencil: Stencil::default(),
            func: Compare::Always,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl BlendOp {
    pub fn from_raw(v: u32) -> Option<BlendOp> {
        Some(match v {
            1 | 0x8006 => BlendOp::Add,
            2 | 0x800A => BlendOp::Subtract,
            3 | 0x800B => BlendOp::ReverseSubtract,
            4 | 0x8007 => BlendOp::Min,
            5 | 0x8008 => BlendOp::Max,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    DstColor,
    OneMinusDstColor,
    SrcAlphaSaturate,
    ConstColor,
    OneMinusConstColor,
    ConstAlpha,
    OneMinusConstAlpha,
}

impl BlendFactor {
    pub fn from_raw(v: u32) -> Option<BlendFactor> {
        use BlendFactor::*;
        Some(match v {
            1 | 0x4000 => Zero,
            2 | 0x4001 => One,
            3 | 0x4300 => SrcColor,
            4 | 0x4301 => OneMinusSrcColor,
            5 | 0x4302 => SrcAlpha,
            6 | 0x4303 => OneMinusSrcAlpha,
            7 | 0x4304 => DstAlpha,
            8 | 0x4305 => OneMinusDstAlpha,
            9 | 0x4306 => DstColor,
            10 | 0x4307 => OneMinusDstColor,
            11 | 0x4308 => SrcAlphaSaturate,
            14 | 0xC001 => ConstColor,
            15 | 0xC002 => OneMinusConstColor,
            0xC003 => ConstAlpha,
            0xC004 => OneMinusConstAlpha,
            _ => return None,
        })
    }

    /// The factor for component `c` (0..=3).
    fn value(self, c: usize, src: [f32; 4], dst: [f32; 4], k: [f32; 4]) -> f32 {
        use BlendFactor::*;
        match self {
            Zero => 0.0,
            One => 1.0,
            SrcColor => src[c],
            OneMinusSrcColor => 1.0 - src[c],
            SrcAlpha => src[3],
            OneMinusSrcAlpha => 1.0 - src[3],
            DstAlpha => dst[3],
            OneMinusDstAlpha => 1.0 - dst[3],
            DstColor => dst[c],
            OneMinusDstColor => 1.0 - dst[c],
            SrcAlphaSaturate => {
                if c == 3 {
                    1.0
                } else {
                    src[3].min(1.0 - dst[3])
                }
            }
            ConstColor => k[c],
            OneMinusConstColor => 1.0 - k[c],
            ConstAlpha => k[3],
            OneMinusConstAlpha => 1.0 - k[3],
        }
    }
}

/// Blending of color target 0.
#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub struct Blend {
    pub enabled: bool,
    pub op_rgb: BlendOp,
    pub src_rgb: BlendFactor,
    pub dst_rgb: BlendFactor,
    pub op_alpha: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
}

impl Default for Blend {
    fn default() -> Self {
        Blend {
            enabled: false,
            op_rgb: BlendOp::Add,
            src_rgb: BlendFactor::One,
            dst_rgb: BlendFactor::Zero,
            op_alpha: BlendOp::Add,
            src_alpha: BlendFactor::One,
            dst_alpha: BlendFactor::Zero,
        }
    }
}

impl Blend {
    /// Blends `src` (the shader's output) over `dst` (the target's
    /// current color) with blend constant `k`. Unorm targets clamp the
    /// inputs to 0..1 first.
    pub fn apply(&self, src: [f32; 4], dst: [f32; 4], k: [f32; 4], clamp: bool) -> [f32; 4] {
        if !self.enabled {
            return src;
        }
        let cl = |v: [f32; 4]| {
            if clamp {
                v.map(|x| x.clamp(0.0, 1.0))
            } else {
                v
            }
        };
        let (src, dst, k) = (cl(src), cl(dst), cl(k));
        std::array::from_fn(|c| {
            let (op, sf, df) = if c < 3 {
                (self.op_rgb, self.src_rgb, self.dst_rgb)
            } else {
                (self.op_alpha, self.src_alpha, self.dst_alpha)
            };
            let s = src[c] * sf.value(c, src, dst, k);
            let d = dst[c] * df.value(c, src, dst, k);
            match op {
                BlendOp::Add => s + d,
                BlendOp::Subtract => s - d,
                BlendOp::ReverseSubtract => d - s,
                BlendOp::Min => src[c].min(dst[c]),
                BlendOp::Max => src[c].max(dst[c]),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enums_accept_both_styles() {
        assert_eq!(Compare::from_raw(2), Some(Compare::Less));
        assert_eq!(Compare::from_raw(0x203), Some(Compare::LessEqual));
        assert_eq!(
            BlendFactor::from_raw(0x4303),
            Some(BlendFactor::OneMinusSrcAlpha)
        );
        assert_eq!(BlendFactor::from_raw(0xC001), Some(BlendFactor::ConstColor));
        assert_eq!(BlendOp::from_raw(0x800B), Some(BlendOp::ReverseSubtract));
    }

    #[test]
    fn alpha_blending() {
        let b = Blend {
            enabled: true,
            src_rgb: BlendFactor::SrcAlpha,
            dst_rgb: BlendFactor::OneMinusSrcAlpha,
            ..Blend::default()
        };
        let out = b.apply([1.0, 0.0, 0.0, 0.25], [0.0, 0.0, 1.0, 1.0], [0.0; 4], true);
        assert_eq!(out, [0.25, 0.0, 0.75, 0.25]);
    }

    #[test]
    fn stencil_faces() {
        let f = StencilFace {
            fail: StencilOp::Zero,
            zfail: StencilOp::DecrSaturate,
            zpass: StencilOp::Replace,
            func: Compare::Equal,
            reference: 0x35,
            func_mask: 0x0F,
            write_mask: 0xF0,
        };
        assert!(f.test(0x05));
        assert!(!f.test(0x06));
        // Replace through the write mask keeps the low bits.
        assert_eq!(f.update(f.zpass, 0x0A), 0x3A);
        assert_eq!(f.update(f.zfail, 0x00), 0x00);
        assert_eq!(StencilOp::from_raw(0x8507), StencilOp::IncrWrap);
        assert_eq!(StencilOp::IncrWrap.apply(0xFF, 0), 0);
        assert_eq!(DepthFormat::from_raw(0x16).unwrap().stencil_byte(), Some(3));
    }

    #[test]
    fn depth_encodings() {
        let mut p = [0x5Au8, 0, 0, 0];
        DepthFormat::S8Z24Unorm.encode(1.0, &mut p);
        assert_eq!(p, [0x5A, 0xFF, 0xFF, 0xFF]);
        assert_eq!(DepthFormat::S8Z24Unorm.decode(&p), 1.0);
        let mut q = [0u8; 2];
        DepthFormat::Z16Unorm.encode(0.5, &mut q);
        assert!((DepthFormat::Z16Unorm.decode(&q) - 0.5).abs() < 1e-4);
    }
}
