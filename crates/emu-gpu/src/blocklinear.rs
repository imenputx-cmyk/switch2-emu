//! Tegra block-linear surface layout.
//!
//! The surface is cut into GOBs of 64 bytes x 8 rows (512 bytes); GOBs
//! are stacked `1 << block_height_log2` high into blocks, and blocks are
//! stored row by row. Inside a GOB, 16-byte pieces are ordered as libnx's
//! `_convertGobTo16Bx2` (framebuffer.c, ISC) describes: byte offset of
//! (x, y) = `(x/32)%2 * 256 + (y/2)%4 * 64 + (x/16)%2 * 32 + y%2 * 16 + x%16`.

/// Byte offset of byte column `x`, row `y` in a block-linear surface
/// whose rows are `width_bytes` wide (rounded up to whole GOBs).
pub fn offset(x: u32, y: u32, width_bytes: u32, block_height_log2: u32) -> usize {
    let gobs_per_block = 1u32 << block_height_log2;
    let block_rows = 8 * gobs_per_block;
    let blocks_per_row = width_bytes.div_ceil(64);
    let block = (y / block_rows) as usize * blocks_per_row as usize + (x / 64) as usize;
    let gob_in_block = ((y % block_rows) / 8) as usize;
    let (gx, gy) = (x % 64, y % 8);
    let in_gob = (gx / 32) * 256 + (gy / 2) * 64 + ((gx / 16) % 2) * 32 + (gy % 2) * 16 + gx % 16;
    (block * gobs_per_block as usize + gob_in_block) * 512 + in_gob as usize
}

/// Bytes covered by a block-linear surface.
pub fn size(width_bytes: u32, height: u32, block_height_log2: u32) -> usize {
    let block_rows = 8u32 << block_height_log2;
    width_bytes.div_ceil(64) as usize
        * 64
        * height.div_ceil(block_rows) as usize
        * block_rows as usize
}

/// Converts a block-linear surface to pitch-linear rows of `pitch` bytes
/// (`pitch` a multiple of 16).
pub fn deswizzle(src: &[u8], pitch: u32, height: u32, block_height_log2: u32) -> Vec<u8> {
    let mut out = vec![0u8; (pitch * height) as usize];
    for y in 0..height {
        // 16-byte runs are contiguous in both layouts.
        for x in (0..pitch).step_by(16) {
            let from = offset(x, y, pitch, block_height_log2);
            let to = (y * pitch + x) as usize;
            if let Some(piece) = src.get(from..from + 16) {
                out[to..to + 16].copy_from_slice(piece);
            }
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
        let mut out = vec![0u8; size(stride, height, bh_log2)];
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
}
