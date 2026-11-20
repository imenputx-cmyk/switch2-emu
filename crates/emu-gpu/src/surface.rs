//! Render target color formats (`SET_COLOR_TARGET_FORMAT` values from
//! the 3D class header) and encoding clear colors into them.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorFormat {
    Rgba32Float,
    Rgba16Unorm,
    /// A8R8G8B8 in NVIDIA's naming: bytes B, G, R, A.
    Bgra8Unorm,
    A2B10G10R10Unorm,
    /// A8B8G8R8: bytes R, G, B, A.
    Rgba8Unorm,
    Rgba8Srgb,
    B5G6R5Unorm,
    /// X8B8G8R8: bytes R, G, B, (unused).
    Rgbx8Unorm,
}

impl ColorFormat {
    pub fn from_raw(v: u32) -> Option<Self> {
        Some(match v {
            0xC0 => ColorFormat::Rgba32Float,
            0xC6 => ColorFormat::Rgba16Unorm,
            0xCF => ColorFormat::Bgra8Unorm,
            0xD1 => ColorFormat::A2B10G10R10Unorm,
            0xD5 => ColorFormat::Rgba8Unorm,
            0xD6 => ColorFormat::Rgba8Srgb,
            0xE8 => ColorFormat::B5G6R5Unorm,
            0xF9 => ColorFormat::Rgbx8Unorm,
            _ => return None,
        })
    }

    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            ColorFormat::Rgba32Float => 16,
            ColorFormat::Rgba16Unorm => 8,
            ColorFormat::B5G6R5Unorm => 2,
            _ => 4,
        }
    }

    /// Encodes an RGBA color (floats, as in SET_COLOR_CLEAR_VALUE).
    pub fn encode(self, c: [f32; 4]) -> Vec<u8> {
        let unorm = |v: f32, max: f32| (v.clamp(0.0, 1.0) * max).round() as u32;
        let u8s = |c: [f32; 4]| c.map(|v| unorm(v, 255.0) as u8);
        match self {
            ColorFormat::Rgba32Float => c.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ColorFormat::Rgba16Unorm => c
                .iter()
                .flat_map(|&v| (unorm(v, 65535.0) as u16).to_le_bytes())
                .collect(),
            ColorFormat::Bgra8Unorm => {
                let [r, g, b, a] = u8s(c);
                vec![b, g, r, a]
            }
            ColorFormat::Rgba8Unorm => u8s(c).to_vec(),
            ColorFormat::Rgba8Srgb => {
                let s = |v: f32| {
                    let v = v.clamp(0.0, 1.0);
                    if v <= 0.003_130_8 {
                        v * 12.92
                    } else {
                        1.055 * v.powf(1.0 / 2.4) - 0.055
                    }
                };
                u8s([s(c[0]), s(c[1]), s(c[2]), c[3]]).to_vec()
            }
            ColorFormat::Rgbx8Unorm => {
                let [r, g, b, _] = u8s(c);
                vec![r, g, b, 0xFF]
            }
            ColorFormat::A2B10G10R10Unorm => {
                let v = unorm(c[0], 1023.0)
                    | unorm(c[1], 1023.0) << 10
                    | unorm(c[2], 1023.0) << 20
                    | unorm(c[3], 3.0) << 30;
                v.to_le_bytes().to_vec()
            }
            ColorFormat::B5G6R5Unorm => {
                let v =
                    (unorm(c[0], 31.0) << 11 | unorm(c[1], 63.0) << 5 | unorm(c[2], 31.0)) as u16;
                v.to_le_bytes().to_vec()
            }
        }
    }

    /// Decodes one pixel to RGBA floats (inverse of `encode`).
    pub fn decode(self, p: &[u8]) -> [f32; 4] {
        let n = |v: u32, max: f32| v as f32 / max;
        let srgb = |c: f32| {
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        match self {
            ColorFormat::Rgba32Float => {
                std::array::from_fn(|i| f32::from_le_bytes(p[4 * i..4 * i + 4].try_into().unwrap()))
            }
            ColorFormat::Rgba16Unorm => std::array::from_fn(|i| {
                n(
                    u32::from(u16::from_le_bytes([p[2 * i], p[2 * i + 1]])),
                    65535.0,
                )
            }),
            ColorFormat::Bgra8Unorm => [p[2], p[1], p[0], p[3]].map(|b| n(u32::from(b), 255.0)),
            ColorFormat::Rgba8Unorm => [p[0], p[1], p[2], p[3]].map(|b| n(u32::from(b), 255.0)),
            ColorFormat::Rgba8Srgb => {
                let c = [p[0], p[1], p[2], p[3]].map(|b| n(u32::from(b), 255.0));
                [srgb(c[0]), srgb(c[1]), srgb(c[2]), c[3]]
            }
            ColorFormat::Rgbx8Unorm => {
                let c = [p[0], p[1], p[2]].map(|b| n(u32::from(b), 255.0));
                [c[0], c[1], c[2], 1.0]
            }
            ColorFormat::A2B10G10R10Unorm => {
                let v = u32::from_le_bytes(p[..4].try_into().unwrap());
                [
                    n(v & 1023, 1023.0),
                    n(v >> 10 & 1023, 1023.0),
                    n(v >> 20 & 1023, 1023.0),
                    n(v >> 30, 3.0),
                ]
            }
            ColorFormat::B5G6R5Unorm => {
                let v = u32::from(u16::from_le_bytes([p[0], p[1]]));
                [n(v >> 11, 31.0), n(v >> 5 & 63, 63.0), n(v & 31, 31.0), 1.0]
            }
        }
    }

    /// Bytes of each pixel that belong to the enabled RGBA components,
    /// for partial (masked) writes; `None` if the format packs channels
    /// below byte granularity.
    pub fn component_bytes(self, mask: [bool; 4]) -> Option<Vec<bool>> {
        let per = |order: [usize; 4], size: usize| {
            order
                .iter()
                .flat_map(|&c| std::iter::repeat_n(mask[c], size))
                .collect()
        };
        Some(match self {
            ColorFormat::Rgba32Float => per([0, 1, 2, 3], 4),
            ColorFormat::Rgba16Unorm => per([0, 1, 2, 3], 2),
            ColorFormat::Rgba8Unorm | ColorFormat::Rgba8Srgb | ColorFormat::Rgbx8Unorm => {
                per([0, 1, 2, 3], 1)
            }
            ColorFormat::Bgra8Unorm => per([2, 1, 0, 3], 1),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings() {
        let c = [1.0, 0.5, 0.0, 1.0];
        assert_eq!(ColorFormat::Rgba8Unorm.encode(c), [255, 128, 0, 255]);
        assert_eq!(ColorFormat::Bgra8Unorm.encode(c), [0, 128, 255, 255]);
        assert_eq!(
            ColorFormat::B5G6R5Unorm.encode([1.0, 0.0, 0.0, 1.0]),
            0xF800u16.to_le_bytes()
        );
        assert_eq!(ColorFormat::Rgba32Float.encode(c).len(), 16);
        assert_eq!(
            ColorFormat::Bgra8Unorm
                .component_bytes([true, false, false, false])
                .unwrap(),
            [false, false, true, false]
        );
    }
}
