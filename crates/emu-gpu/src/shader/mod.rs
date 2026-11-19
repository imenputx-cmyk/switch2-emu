//! Maxwell shaders: the program header (SPH), the instruction decoder,
//! an interpreter (the reference for translated shaders) and a small
//! assembler for tests.
//!
//! A program is a 0x50-byte header followed by 64-bit instructions in
//! groups of four: a scheduling control word, then three instructions.
//!
//! Instruction encodings follow envytools' Maxwell (gm107) disassembler
//! tables (envydis/gm107.c, MIT); the header follows NVIDIA's Shader
//! Program Header specification (open-gpu-doc, MIT).

pub mod asm;
pub mod decode;
pub mod interp;

/// Bytes of the program header before the first instruction.
pub const HEADER_SIZE: usize = 0x50;

/// Attribute addresses (byte offsets in attribute space).
pub mod attr {
    /// Position (x, y, z, w) as a VS output; fragment coordinates as a
    /// PS input.
    pub const POSITION: u32 = 0x70;
    /// Generic attribute `i` (4 components).
    pub const fn generic(i: u32) -> u32 {
        0x80 + 16 * i
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Pixel,
    Other(u32),
}

/// The fields of the program header this emulator uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub stage: Stage,
    /// Pixel shaders: enabled RGBA components of each color target, in
    /// the order their registers are written (r0, r1, ...).
    pub omap_targets: [u8; 8],
    pub omap_depth: bool,
}

impl Header {
    pub fn parse(h: &[u8]) -> Header {
        let word = |i: usize| {
            h.get(4 * i..4 * i + 4)
                .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
        };
        let stage = match (word(0) >> 10) & 0xF {
            1 => Stage::Vertex,
            5 => Stage::Pixel,
            n => Stage::Other(n),
        };
        let omap = word(0x48 / 4);
        Header {
            stage,
            omap_targets: std::array::from_fn(|i| ((omap >> (4 * i)) & 0xF) as u8),
            omap_depth: word(0x4C / 4) & 2 != 0,
        }
    }
}
