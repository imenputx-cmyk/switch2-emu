// SPDX-License-Identifier: GPL-3.0-or-later
//! NSO: the executable format of Switch system modules and games (the
//! `main`, `rtld`, `sdk`, `subsdk*` files of a game's ExeFS).
//!
//! Ported from Eden's `core/loader/nso.{h,cpp}` (GPL-3.0-or-later, based
//! on yuzu, GPL-2.0-or-later). The 0x100-byte header names three
//! segments (.text, .rodata, .data), each stored at a file offset and
//! loaded at a memory offset, optionally compressed: LZ4 block format,
//! or on Switch 2 ("ZBIC", flag bit 7) zstd frames whose magic is
//! "ZBIC" instead of zstd's.

use crate::LoadError;

pub const HEADER_SIZE: usize = 0x100;
const PAGE: u32 = 0x1000;
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// A segment of a loaded module: offset from the module base and size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub offset: u32,
    pub size: u32,
}

/// A decompressed module, ready to be copied to its load address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    /// .text, .rodata, .data (memory sizes rounded up to pages; .data
    /// includes .bss).
    pub segments: [Segment; 3],
    /// The whole module image (page-aligned size, .bss zeroed).
    pub image: Vec<u8>,
    pub build_id: [u8; 32],
}

fn u32_at(d: &[u8], off: usize) -> Result<u32, LoadError> {
    d.get(off..off + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or(LoadError::Truncated)
}

pub fn is_nso(file: &[u8]) -> bool {
    file.get(..4) == Some(b"NSO0")
}

fn page_align(v: u32) -> u32 {
    v.next_multiple_of(PAGE)
}

/// Decompresses a ZBIC segment: a zstd frame with a different magic.
fn decompress_zbic(src: &[u8], size: usize) -> Result<Vec<u8>, LoadError> {
    let mut frame = src.to_vec();
    if frame.len() < 4 {
        return Err(LoadError::Truncated);
    }
    frame[..4].copy_from_slice(&ZSTD_MAGIC);
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(frame.as_slice())
        .map_err(|_| LoadError::Unsupported("bad ZBIC segment"))?;
    let mut out = Vec::with_capacity(size);
    std::io::Read::read_to_end(&mut decoder, &mut out)
        .map_err(|_| LoadError::Unsupported("bad ZBIC segment"))?;
    if out.len() != size {
        return Err(LoadError::Unsupported("ZBIC segment size mismatch"));
    }
    Ok(out)
}

pub fn parse_nso(file: &[u8]) -> Result<Module, LoadError> {
    if file.len() < HEADER_SIZE {
        return Err(LoadError::Truncated);
    }
    if !is_nso(file) {
        return Err(LoadError::NotElf);
    }
    let flags = u32_at(file, 0xC)?;
    let zbic = flags & (1 << 7) != 0;
    let mut image = Vec::new();
    let mut segments = [Segment { offset: 0, size: 0 }; 3];
    let mut bss = 0;
    for (i, seg) in segments.iter_mut().enumerate() {
        // NSOSegmentHeader: file offset, memory offset, size, then
        // alignment (.bss size for .data).
        let h = 0x10 + 0x10 * i;
        let (file_off, mem_off, size) = (
            u32_at(file, h)? as usize,
            u32_at(file, h + 4)?,
            u32_at(file, h + 8)?,
        );
        if i == 2 {
            bss = u32_at(file, h + 12)?;
        }
        let stored = u32_at(file, 0x60 + 4 * i)? as usize;
        let src = file
            .get(file_off..file_off + stored)
            .ok_or(LoadError::Truncated)?;
        let data = if flags & (1 << i) == 0 {
            src.get(..size as usize)
                .ok_or(LoadError::Truncated)?
                .to_vec()
        } else if zbic {
            decompress_zbic(src, size as usize)?
        } else {
            let mut out = vec![0u8; size as usize];
            let n = lz4_flex::block::decompress_into(src, &mut out)
                .map_err(|_| LoadError::Unsupported("bad LZ4 segment"))?;
            if n != size as usize {
                return Err(LoadError::Unsupported("LZ4 segment size mismatch"));
            }
            out
        };
        let end = mem_off as usize + data.len();
        if image.len() < end {
            image.resize(end, 0);
        }
        image[mem_off as usize..end].copy_from_slice(&data);
        *seg = Segment {
            offset: mem_off,
            size: page_align(size),
        };
    }
    let total = page_align(image.len() as u32 + bss);
    image.resize(total as usize, 0);
    segments[2].size = total - segments[2].offset;
    let mut build_id = [0u8; 32];
    build_id.copy_from_slice(&file[0x40..0x60]);
    Ok(Module {
        segments,
        image,
        build_id,
    })
}

/// Writes an uncompressed NSO with the given segments (each loaded at
/// the next page after the previous one) and .bss size.
pub fn build_uncompressed(segments: [&[u8]; 3], bss: u32) -> Vec<u8> {
    let mut file = vec![0u8; HEADER_SIZE];
    file[..4].copy_from_slice(b"NSO0");
    let mut mem = 0u32;
    for (i, s) in segments.iter().enumerate() {
        let h = 0x10 + 0x10 * i;
        let at = file.len() as u32;
        file[h..h + 4].copy_from_slice(&at.to_le_bytes());
        file[h + 4..h + 8].copy_from_slice(&mem.to_le_bytes());
        file[h + 8..h + 12].copy_from_slice(&(s.len() as u32).to_le_bytes());
        if i == 2 {
            file[h + 12..h + 16].copy_from_slice(&bss.to_le_bytes());
        }
        file[0x60 + 4 * i..0x64 + 4 * i].copy_from_slice(&(s.len() as u32).to_le_bytes());
        file.extend_from_slice(s);
        mem = page_align(mem + s.len() as u32);
    }
    file
}

/// Converts a homebrew NRO to an NSO with the same image (for testing
/// the application boot path with homebrew that supports both).
pub fn from_nro(nro: &[u8]) -> Result<Vec<u8>, LoadError> {
    let n = crate::nro::parse_nro(nro)?;
    let seg = |s: &crate::nro::NroSegment| &nro[s.offset as usize..(s.offset + s.size) as usize];
    if n.ro.offset != page_align(n.text.size)
        || n.data.offset != n.ro.offset + page_align(n.ro.size)
    {
        return Err(LoadError::Unsupported("NRO segments not packed"));
    }
    Ok(build_uncompressed(
        [seg(&n.text), seg(&n.ro), seg(&n.data)],
        n.bss_size,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an NSO with the given segments (stored as-is, or LZ4/ZBIC
    /// compressed when `compress` is 1 or 2).
    pub fn build(segs: [&[u8]; 3], bss: u32, compress: u8) -> Vec<u8> {
        let mut file = vec![0u8; HEADER_SIZE];
        file[..4].copy_from_slice(b"NSO0");
        let mut flags = 0u32;
        let mut mem = 0u32;
        for (i, s) in segs.iter().enumerate() {
            let stored = match compress {
                1 => {
                    flags |= 1 << i;
                    lz4_flex::block::compress(s)
                }
                2 => {
                    flags |= 1 << i | 1 << 7;
                    let mut z = ruzstd::encoding::compress_to_vec(
                        *s,
                        ruzstd::encoding::CompressionLevel::Fastest,
                    );
                    z[..4].copy_from_slice(b"ZBIC");
                    z
                }
                _ => s.to_vec(),
            };
            let h = 0x10 + 0x10 * i;
            let at = file.len() as u32;
            file[h..h + 4].copy_from_slice(&at.to_le_bytes());
            file[h + 4..h + 8].copy_from_slice(&mem.to_le_bytes());
            file[h + 8..h + 12].copy_from_slice(&(s.len() as u32).to_le_bytes());
            if i == 2 {
                file[h + 12..h + 16].copy_from_slice(&bss.to_le_bytes());
            }
            file[0x60 + 4 * i..0x64 + 4 * i].copy_from_slice(&(stored.len() as u32).to_le_bytes());
            file.extend(stored);
            mem = page_align(mem + s.len() as u32);
        }
        file[0xC..0x10].copy_from_slice(&flags.to_le_bytes());
        file
    }

    #[test]
    fn plain_lz4_and_zbic_segments() {
        let text: Vec<u8> = (0..5000u32).map(|i| (i % 7) as u8).collect();
        let ro = b"read only data".repeat(30);
        let data = vec![0xAB; 100];
        for compress in [0, 1, 2] {
            let m = parse_nso(&build([&text, &ro, &data], 0x3000, compress)).unwrap();
            assert_eq!(
                m.segments[0],
                Segment {
                    offset: 0,
                    size: 0x2000
                }
            );
            assert_eq!(
                m.segments[1],
                Segment {
                    offset: 0x2000,
                    size: 0x1000
                }
            );
            assert_eq!(m.segments[2].offset, 0x3000);
            // .data (100 bytes) + .bss (0x3000), page aligned.
            assert_eq!(m.image.len(), 0x7000);
            assert_eq!(&m.image[..5000], &text[..]);
            assert_eq!(&m.image[0x2000..0x2000 + ro.len()], &ro[..]);
            assert_eq!(m.image[0x3000 + 99], 0xAB);
            assert_eq!(m.image[0x3000 + 100], 0);
        }
    }
}
