//! The DMA copy engine (class B0B5): copies between pitch-linear and
//! block-linear surfaces, and fills memory through component remapping
//! (constants instead of source data).

use super::{bits, EngineCtx};
use crate::blocklinear;

#[derive(Debug, Default)]
pub struct CopyEngine {
    regs: Vec<u32>,
}

/// Register byte offsets.
mod reg {
    pub const SEMAPHORE_A: u32 = 0x240;
    pub const SEMAPHORE_B: u32 = 0x244;
    pub const SEMAPHORE_PAYLOAD: u32 = 0x248;
    pub const LAUNCH_DMA: u32 = 0x300;
    pub const OFFSET_IN_UPPER: u32 = 0x400;
    pub const OFFSET_IN_LOWER: u32 = 0x404;
    pub const OFFSET_OUT_UPPER: u32 = 0x408;
    pub const OFFSET_OUT_LOWER: u32 = 0x40C;
    pub const PITCH_IN: u32 = 0x410;
    pub const PITCH_OUT: u32 = 0x414;
    pub const LINE_LENGTH_IN: u32 = 0x418;
    pub const LINE_COUNT: u32 = 0x41C;
    pub const REMAP_CONST_A: u32 = 0x700;
    pub const REMAP_CONST_B: u32 = 0x704;
    pub const REMAP_COMPONENTS: u32 = 0x708;
    pub const DST_BLOCK_SIZE: u32 = 0x70C;
    pub const DST_WIDTH: u32 = 0x710;
    pub const DST_ORIGIN: u32 = 0x720;
    pub const SRC_BLOCK_SIZE: u32 = 0x728;
    pub const SRC_WIDTH: u32 = 0x72C;
    pub const SRC_ORIGIN: u32 = 0x73C;
    pub const END: u32 = 0x800;
}

/// One side of a copy.
struct Side {
    base: u64,
    pitch: u32,
    block_linear: bool,
    block_height: u32,
    width_bytes: u32,
    origin_x: u32,
    origin_y: u32,
}

impl Side {
    /// GPU address of byte `x` of line `y`.
    fn addr(&self, x: u32, y: u32) -> u64 {
        if self.block_linear {
            self.base
                + blocklinear::offset(
                    self.origin_x + x,
                    self.origin_y + y,
                    self.width_bytes,
                    self.block_height,
                ) as u64
        } else {
            self.base + u64::from(y) * u64::from(self.pitch) + u64::from(x)
        }
    }
}

impl CopyEngine {
    fn r(&self, offset: u32) -> u32 {
        self.regs.get(offset as usize / 4).copied().unwrap_or(0)
    }

    pub fn write(&mut self, method: u32, value: u32, ctx: &mut EngineCtx) -> bool {
        let offset = method * 4;
        if offset >= reg::END {
            return false;
        }
        if self.regs.is_empty() {
            self.regs = vec![0; reg::END as usize / 4];
        }
        self.regs[method as usize] = value;
        match offset {
            reg::LAUNCH_DMA => self.launch(value, ctx),
            0x100 | 0x140 | 0x260 | 0x264 | 0x254..=0x25C => {}
            reg::SEMAPHORE_A..=reg::SEMAPHORE_PAYLOAD | reg::OFFSET_IN_UPPER..=reg::LINE_COUNT => {}
            reg::REMAP_CONST_A..=0x740 => {}
            _ => return false,
        }
        true
    }

    fn side(&self, input: bool, element: u32) -> Side {
        let (upper, lower, pitch, block, width, origin, layout_bit) = if input {
            (
                reg::OFFSET_IN_UPPER,
                reg::OFFSET_IN_LOWER,
                reg::PITCH_IN,
                reg::SRC_BLOCK_SIZE,
                reg::SRC_WIDTH,
                reg::SRC_ORIGIN,
                7,
            )
        } else {
            (
                reg::OFFSET_OUT_UPPER,
                reg::OFFSET_OUT_LOWER,
                reg::PITCH_OUT,
                reg::DST_BLOCK_SIZE,
                reg::DST_WIDTH,
                reg::DST_ORIGIN,
                8,
            )
        };
        let launch = self.r(reg::LAUNCH_DMA);
        let origin = self.r(origin);
        Side {
            base: u64::from(self.r(upper) & 0xFF) << 32 | u64::from(self.r(lower)),
            pitch: self.r(pitch),
            block_linear: launch & (1 << layout_bit) == 0,
            block_height: bits(self.r(block), 7, 4),
            // Widths and origins count elements when remapping.
            width_bytes: self.r(width) * element,
            origin_x: (origin & 0xFFFF) * element,
            origin_y: origin >> 16,
        }
    }

    fn launch(&mut self, launch: u32, ctx: &mut EngineCtx) {
        let transfer = launch & 3 != 0;
        if transfer {
            let lines = if launch & (1 << 9) != 0 {
                self.r(reg::LINE_COUNT)
            } else {
                1
            };
            let length = self.r(reg::LINE_LENGTH_IN);
            if launch & (1 << 10) != 0 {
                self.remap_copy(length, lines, ctx);
            } else {
                let (src, dst) = (self.side(true, 1), self.side(false, 1));
                let mut line = vec![0u8; length as usize];
                for y in 0..lines {
                    if !src.block_linear && !dst.block_linear {
                        ctx.read(src.addr(0, y), &mut line);
                        ctx.write(dst.addr(0, y), &line);
                        continue;
                    }
                    // 16-byte pieces are contiguous in both layouts.
                    let mut x = 0;
                    while x < length {
                        let n = (16 - (x % 16)).min(length - x) as usize;
                        let mut piece = [0u8; 16];
                        ctx.read(src.addr(x, y), &mut piece[..n]);
                        ctx.write(dst.addr(x, y), &piece[..n]);
                        x += n as u32;
                    }
                }
            }
        }
        // Semaphore release: bits 4:3 (1 = one word, 2 = four words).
        let sem = bits(launch, 4, 3);
        if sem != 0 {
            let va = u64::from(self.r(reg::SEMAPHORE_A) & 0xFF) << 32
                | u64::from(self.r(reg::SEMAPHORE_B));
            let mut data = self.r(reg::SEMAPHORE_PAYLOAD).to_le_bytes().to_vec();
            if sem == 2 {
                data.extend_from_slice(&[0; 12]);
            }
            ctx.write(va, &data);
        }
    }

    /// A copy through the component remapper: each destination component
    /// is a source component or one of two constants.
    fn remap_copy(&mut self, elements: u32, lines: u32, ctx: &mut EngineCtx) {
        let comps = self.r(reg::REMAP_COMPONENTS);
        let size = bits(comps, 17, 16) + 1;
        let num_src = bits(comps, 21, 20) + 1;
        let num_dst = bits(comps, 25, 24) + 1;
        let consts = [self.r(reg::REMAP_CONST_A), self.r(reg::REMAP_CONST_B)];
        let selects: Vec<u32> = (0..num_dst)
            .map(|i| bits(comps, 4 * i + 2, 4 * i))
            .collect();
        let needs_src = selects.iter().any(|&s| s < 4);
        let (src, dst) = (
            self.side(true, size * num_src),
            self.side(false, size * num_dst),
        );
        let (src_el, dst_el) = ((size * num_src) as usize, (size * num_dst) as usize);
        let mut s = vec![0u8; src_el];
        for y in 0..lines {
            for e in 0..elements {
                if needs_src {
                    ctx.read(src.addr(e * src_el as u32, y), &mut s);
                }
                let mut d = vec![0u8; dst_el];
                let mut write = vec![true; dst_el];
                for (i, &sel) in selects.iter().enumerate() {
                    let at = i * size as usize..(i + 1) * size as usize;
                    match sel {
                        0..=3 if (sel as usize) < num_src as usize => {
                            let from = sel as usize * size as usize;
                            d[at].copy_from_slice(&s[from..from + size as usize]);
                        }
                        4 | 5 => {
                            let c = consts[sel as usize - 4].to_le_bytes();
                            d[at].copy_from_slice(&c[..size as usize]);
                        }
                        _ => write[at].fill(false),
                    }
                }
                let base = dst.addr(e * dst_el as u32, y);
                if write.iter().all(|&w| w) {
                    ctx.write(base, &d);
                } else {
                    for (i, &b) in d.iter().enumerate() {
                        if write[i] {
                            ctx.write(base + i as u64, &[b]);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{incr, run, setup};
    use crate::{blocklinear, class};

    #[test]
    fn fill_with_remap_constant() {
        let (mut gpu, mut mem) = setup();
        run(
            &mut gpu,
            &mut mem,
            &[
                incr(4, 0, 1),
                class::DMA_COPY,
                incr(4, 0x408, 2),
                0,
                0x1004_0000,
                incr(4, 0x418, 1),
                16, // elements
                incr(4, 0x700, 1),
                0xDEAD_BEEF,
                incr(4, 0x708, 1),
                4 | (3 << 16), // dst x = const A, 4-byte components, 1 src/1 dst
                // pitch in/out, remap, one line, transfer
                incr(4, 0x300, 1),
                1 | (1 << 7) | (1 << 8) | (1 << 10),
            ],
        );
        let mut b = vec![0u8; 68];
        mem.read(0x104_0000, &mut b).unwrap();
        assert!(b[..64].chunks(4).all(|c| c == 0xDEAD_BEEFu32.to_le_bytes()));
        assert_eq!(&b[64..], [0; 4]);
    }

    #[test]
    fn pitch_to_block_linear_copy() {
        let (mut gpu, mut mem) = setup();
        let data: Vec<u8> = (0..128 * 16).map(|i| (i * 3) as u8).collect();
        mem.write(0x105_0000, &data).unwrap();
        run(
            &mut gpu,
            &mut mem,
            &[
                incr(4, 0, 1),
                class::DMA_COPY,
                incr(4, 0x400, 8),
                0,
                0x1005_0000, // in
                0,
                0x1006_0000, // out
                128,         // pitch in
                0,           // pitch out
                128,         // line length (bytes)
                16,          // lines
                incr(4, 0x70C, 2),
                1 << 4, // dst: 2 GOBs high
                128,    // dst width (bytes)
                incr(4, 0x300, 1),
                1 | (1 << 7) | (1 << 9), // pitch in, block-linear out, multi-line
            ],
        );
        let mut raw = vec![0u8; blocklinear::size(128, 16, 1)];
        mem.read(0x106_0000, &mut raw).unwrap();
        assert_eq!(blocklinear::deswizzle(&raw, 128, 16, 1), data);
    }
}
