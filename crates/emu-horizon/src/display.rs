//! Turning guest framebuffers into host RGBA images: block-linear
//! deswizzling (in `emu_gpu::blocklinear`) and pixel format conversion.

pub use emu_gpu::blocklinear::{deswizzle, size as block_linear_size};

/// Pixel formats (Android `PIXEL_FORMAT_*` values used by the display).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgba8888,
    Rgbx8888,
    Rgb565,
    Bgra8888,
    Rgba4444,
}

impl PixelFormat {
    pub fn from_android(v: u32) -> Option<Self> {
        Some(match v {
            1 => PixelFormat::Rgba8888,
            2 => PixelFormat::Rgbx8888,
            4 => PixelFormat::Rgb565,
            5 => PixelFormat::Bgra8888,
            7 => PixelFormat::Rgba4444,
            _ => return None,
        })
    }

    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            PixelFormat::Rgb565 | PixelFormat::Rgba4444 => 2,
            _ => 4,
        }
    }
}

/// Converts pitch-linear pixels to tightly packed RGBA8.
pub fn to_rgba(src: &[u8], pitch: u32, width: u32, height: u32, format: PixelFormat) -> Vec<u8> {
    let bpp = format.bytes_per_pixel() as usize;
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height as usize {
        let row = &src[y * pitch as usize..];
        for x in 0..width as usize {
            let p = &row[x * bpp..x * bpp + bpp];
            let px = match format {
                PixelFormat::Rgba8888 => [p[0], p[1], p[2], p[3]],
                PixelFormat::Rgbx8888 => [p[0], p[1], p[2], 0xFF],
                PixelFormat::Bgra8888 => [p[2], p[1], p[0], p[3]],
                PixelFormat::Rgb565 => {
                    let v = u16::from_le_bytes([p[0], p[1]]);
                    let (r, g, b) = (v >> 11, (v >> 5) & 0x3F, v & 0x1F);
                    [
                        (r << 3 | r >> 2) as u8,
                        (g << 2 | g >> 4) as u8,
                        (b << 3 | b >> 2) as u8,
                        0xFF,
                    ]
                }
                PixelFormat::Rgba4444 => {
                    let v = u16::from_le_bytes([p[0], p[1]]);
                    let n = |s: u16| (((v >> s) & 0xF) * 0x11) as u8;
                    [n(0), n(4), n(8), n(12)]
                }
            };
            out.extend_from_slice(&px);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_formats() {
        let px = [0x10, 0x20, 0x30, 0x40];
        assert_eq!(
            to_rgba(&px, 4, 1, 1, PixelFormat::Rgba8888),
            [0x10, 0x20, 0x30, 0x40]
        );
        assert_eq!(
            to_rgba(&px, 4, 1, 1, PixelFormat::Bgra8888),
            [0x30, 0x20, 0x10, 0x40]
        );
        assert_eq!(
            to_rgba(&px, 4, 1, 1, PixelFormat::Rgbx8888),
            [0x10, 0x20, 0x30, 0xFF]
        );
        assert_eq!(
            to_rgba(&0xF800u16.to_le_bytes(), 2, 1, 1, PixelFormat::Rgb565),
            [0xFF, 0, 0, 0xFF]
        );
    }
}
