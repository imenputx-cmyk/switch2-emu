//! Turning guest framebuffers into host RGBA images: Tegra block-linear
//! deswizzling and pixel format conversion.
//!
//! Block-linear layout: the image is cut into GOBs of 64 bytes x 8 rows
//! (512 bytes); GOBs are stacked `1 << block_height_log2` high into
//! blocks, and blocks are stored row by row. Inside a GOB, 16-byte
//! pieces are ordered as libnx's `_convertGobTo16Bx2` (framebuffer.c,
//! ISC) describes: byte offset of (x, y) =
//! `(x/32)%2 * 256 + (y/2)%4 * 64 + (x/16)%2 * 32 + y%2 * 16 + x%16`.

/// Byte offset of byte column `x`, row `y` in a block-linear surface
/// whose rows are `pitch` bytes (a multiple of 64).
pub fn block_linear_offset(x: u32, y: u32, pitch: u32, block_height_log2: u32) -> usize {
    let gobs_per_block = 1u32 << block_height_log2;
    let block_rows = 8 * gobs_per_block;
    let blocks_per_row = pitch / 64;
    let block = (y / block_rows) * blocks_per_row + x / 64;
    let gob_in_block = (y % block_rows) / 8;
    let (gx, gy) = (x % 64, y % 8);
    let in_gob = (gx / 32) * 256 + (gy / 2) * 64 + ((gx / 16) % 2) * 32 + (gy % 2) * 16 + gx % 16;
    ((block * gobs_per_block + gob_in_block) * 512 + in_gob) as usize
}

/// Bytes needed for a block-linear surface of `height` rows.
pub fn block_linear_size(pitch: u32, height: u32, block_height_log2: u32) -> usize {
    let block_rows = 8u32 << block_height_log2;
    (pitch * height.div_ceil(block_rows) * block_rows) as usize
}

/// Converts a block-linear surface to pitch-linear rows of `pitch` bytes.
pub fn deswizzle(src: &[u8], pitch: u32, height: u32, block_height_log2: u32) -> Vec<u8> {
    let mut out = vec![0u8; (pitch * height) as usize];
    for y in 0..height {
        // 16-byte runs are contiguous in both layouts.
        for x in (0..pitch).step_by(16) {
            let from = block_linear_offset(x, y, pitch, block_height_log2);
            let to = (y * pitch + x) as usize;
            if let Some(piece) = src.get(from..from + 16) {
                out[to..to + 16].copy_from_slice(piece);
            }
        }
    }
    out
}

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

    /// libnx's `_convertToBlocklinear` (framebuffer.c), ported: the
    /// swizzle the guest applies. Deswizzling must invert it.
    fn libnx_swizzle(input: &[u8], stride: u32, height: u32, bh_log2: u32) -> Vec<u8> {
        let gobs = 1u32 << bh_log2;
        let block_px = 8u32 << bh_log2;
        let width_blocks = stride >> 6;
        let height_blocks = (height + block_px - 1) >> (3 + bh_log2);
        let mut out = vec![0u8; block_linear_size(stride, height, bh_log2)];
        let mut o = 0usize;
        for by in 0..height_blocks {
            for bx in 0..width_blocks {
                for gy in 0..gobs {
                    let (x, y) = (bx * 64, by * block_px + gy * 8);
                    if y < height {
                        for i in 0..32u32 {
                            let yy = ((i >> 1) & 6) | (i & 1);
                            let xx = ((i << 3) & 0x10) | ((i << 1) & 0x20);
                            let from = ((y + yy) * stride + x + xx) as usize;
                            if from + 16 <= input.len() {
                                out[o + 16 * i as usize..o + 16 * i as usize + 16]
                                    .copy_from_slice(&input[from..from + 16]);
                            }
                        }
                    }
                    o += 512;
                }
            }
        }
        out
    }

    #[test]
    fn deswizzle_inverts_libnx_swizzle() {
        for (stride, height, bh) in [(1280 * 4, 720, 4), (64, 8, 0), (320, 100, 2), (5120, 37, 3)] {
            let linear: Vec<u8> = (0..stride * height)
                .map(|i| (i * 7 + i / 251) as u8)
                .collect();
            let swizzled = libnx_swizzle(&linear, stride, height, bh);
            assert_eq!(
                deswizzle(&swizzled, stride, height, bh),
                linear,
                "{stride}x{height}"
            );
        }
    }

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
