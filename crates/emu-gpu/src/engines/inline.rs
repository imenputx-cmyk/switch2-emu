//! Inline-to-memory: data written into the pushbuffer is copied to GPU
//! memory (class A140, and the same methods at 0x180..0x1B4 in the 3D
//! and compute classes). Used to upload small buffers and constants.

use super::{bits, EngineCtx};
use crate::blocklinear;

#[derive(Debug, Default)]
pub struct InlineToMemory {
    line_length: u32,
    line_count: u32,
    offset_out: u64,
    pitch_out: u32,
    dst_block: u32,
    dst_width: u32,
    dst_origin_x: u32,
    dst_origin_y: u32,
    pitch_layout: bool,
    data: Vec<u8>,
    active: bool,
}

impl InlineToMemory {
    /// Handles a method; returns false if it is not an inline method.
    pub fn write(&mut self, method: u32, value: u32, ctx: &mut EngineCtx) -> bool {
        match method * 4 {
            0x180 => self.line_length = value,
            0x184 => self.line_count = value,
            0x188 => {
                self.offset_out = (self.offset_out & 0xFFFF_FFFF) | u64::from(value & 0xFF) << 32
            }
            0x18C => self.offset_out = (self.offset_out & !0xFFFF_FFFF) | u64::from(value),
            0x190 => self.pitch_out = value,
            0x194 => self.dst_block = value,
            0x198 => self.dst_width = value,
            0x19C | 0x1A0 | 0x1A4 => {} // height, depth, layer
            0x1A8 => self.dst_origin_x = value,
            0x1AC => self.dst_origin_y = value,
            0x1B0 => {
                // LAUNCH_DMA: bit 0 = pitch layout.
                self.pitch_layout = value & 1 != 0;
                self.data.clear();
                self.active = self.line_length > 0 && self.line_count > 0;
            }
            0x1B4 => {
                // LOAD_INLINE_DATA
                if !self.active {
                    return true;
                }
                self.data.extend_from_slice(&value.to_le_bytes());
                let total = (self.line_length * self.line_count) as usize;
                if self.data.len() >= total {
                    self.data.truncate(total);
                    self.flush(ctx);
                    self.active = false;
                }
            }
            _ => return false,
        }
        true
    }

    fn flush(&mut self, ctx: &mut EngineCtx) {
        let len = self.line_length as usize;
        let data = std::mem::take(&mut self.data);
        for (y, line) in data.chunks(len).enumerate() {
            let y = y as u32;
            if self.pitch_layout {
                let va = self.offset_out + u64::from(y) * u64::from(self.pitch_out);
                ctx.write(va, line);
            } else {
                let bh = bits(self.dst_block, 7, 4);
                for (x, &b) in line.iter().enumerate() {
                    let off = blocklinear::offset(
                        self.dst_origin_x + x as u32,
                        self.dst_origin_y + y,
                        self.dst_width,
                        bh,
                    );
                    ctx.write(self.offset_out + off as u64, &[b]);
                }
            }
        }
    }
}
