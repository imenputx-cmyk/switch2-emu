//! NRO: the executable format of Switch homebrew.
//!
//! Layout as documented on switchbrew (https://switchbrew.org/wiki/NRO):
//! the file starts with a 0x10-byte start header (homebrew puts a branch
//! and the MOD0 offset there), followed by the NRO header at 0x10 with
//! magic "NRO0". Segment memory offsets equal file offsets. An optional
//! "ASET" asset section (icon, NACP, RomFS) follows the image.

use crate::LoadError;

pub const NRO_MAGIC_OFFSET: usize = 0x10;
pub const NRO_HEADER_SIZE: usize = 0x80;
const PAGE: u32 = 0x1000;

/// One loadable segment of an NRO.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NroSegment {
    /// Offset from the image base (and from the start of the file).
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nro {
    pub text: NroSegment,
    pub ro: NroSegment,
    pub data: NroSegment,
    pub bss_size: u32,
    /// Offset of the MOD0 header from the image base (from the start header).
    pub mod0_offset: u32,
    /// Size of the image in memory (segments plus .bss, page aligned).
    pub image_size: u64,
    /// The asset section (offset into the file, size), if present.
    pub assets: Option<(usize, usize)>,
}

fn u32_at(d: &[u8], off: usize) -> Result<u32, LoadError> {
    d.get(off..off + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or(LoadError::Truncated)
}

pub fn is_nro(file: &[u8]) -> bool {
    file.get(NRO_MAGIC_OFFSET..NRO_MAGIC_OFFSET + 4) == Some(b"NRO0")
}

pub fn parse_nro(file: &[u8]) -> Result<Nro, LoadError> {
    if file.len() < NRO_HEADER_SIZE {
        return Err(LoadError::Truncated);
    }
    if !is_nro(file) {
        return Err(LoadError::NotElf);
    }
    let total = u32_at(file, 0x18)? as usize;
    if total > file.len() {
        return Err(LoadError::Truncated);
    }
    let flags = u32_at(file, 0x1C)?;
    if flags & 2 != 0 {
        return Err(LoadError::Unsupported("compressed NRO (23.0.0+)"));
    }
    let seg = |i: usize| -> Result<NroSegment, LoadError> {
        let s = NroSegment {
            offset: u32_at(file, 0x20 + i * 8)?,
            size: u32_at(file, 0x24 + i * 8)?,
        };
        if !s.offset.is_multiple_of(PAGE) {
            return Err(LoadError::Unsupported("segment not page aligned"));
        }
        if s.offset as usize + s.size as usize > total {
            return Err(LoadError::Truncated);
        }
        Ok(s)
    };
    let (text, ro, data) = (seg(0)?, seg(1)?, seg(2)?);
    if text.offset > ro.offset || ro.offset > data.offset {
        return Err(LoadError::Unsupported("segments out of order"));
    }
    let bss_size = u32_at(file, 0x38)?;
    let end = u64::from(data.offset) + u64::from(data.size) + u64::from(bss_size);
    let image_size = end.div_ceil(u64::from(PAGE)) * u64::from(PAGE);

    let assets = (file.get(total..total + 4) == Some(b"ASET")).then(|| (total, file.len() - total));
    Ok(Nro {
        text,
        ro,
        data,
        bss_size,
        mod0_offset: u32_at(file, 4)?,
        image_size,
        assets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_nro() -> Vec<u8> {
        let mut f = vec![0u8; 0x3000];
        f[4..8].copy_from_slice(&0x80u32.to_le_bytes()); // MOD0 offset
        f[0x10..0x14].copy_from_slice(b"NRO0");
        f[0x18..0x1C].copy_from_slice(&0x3000u32.to_le_bytes());
        for (i, (off, size)) in [(0u32, 0x1000u32), (0x1000, 0x800), (0x2000, 0x100)]
            .iter()
            .enumerate()
        {
            f[0x20 + i * 8..0x24 + i * 8].copy_from_slice(&off.to_le_bytes());
            f[0x24 + i * 8..0x28 + i * 8].copy_from_slice(&size.to_le_bytes());
        }
        f[0x38..0x3C].copy_from_slice(&0x2F00u32.to_le_bytes()); // bss
        f
    }

    #[test]
    fn parses_header() {
        let nro = parse_nro(&tiny_nro()).unwrap();
        assert_eq!(
            nro.text,
            NroSegment {
                offset: 0,
                size: 0x1000
            }
        );
        assert_eq!(
            nro.ro,
            NroSegment {
                offset: 0x1000,
                size: 0x800
            }
        );
        assert_eq!(
            nro.data,
            NroSegment {
                offset: 0x2000,
                size: 0x100
            }
        );
        assert_eq!(nro.image_size, 0x5000, "data + bss rounded up to a page");
        assert_eq!(nro.mod0_offset, 0x80);
        assert_eq!(nro.assets, None);
    }

    #[test]
    fn detects_assets_and_rejects_bad_files() {
        let mut f = tiny_nro();
        f.extend_from_slice(b"ASET\0\0\0\0");
        assert_eq!(parse_nro(&f).unwrap().assets, Some((0x3000, 8)));
        assert!(parse_nro(&f[..0x40]).is_err());
        let mut bad = tiny_nro();
        bad[0x10] = b'X';
        assert!(!is_nro(&bad));
        let mut unaligned = tiny_nro();
        unaligned[0x28..0x2C].copy_from_slice(&0x1004u32.to_le_bytes());
        assert!(parse_nro(&unaligned).is_err());
    }
}
