//! Loads executable images into guest memory.
//!
//! - ELF (64-bit little-endian AArch64): test programs built with clang.
//! - NRO ([`nro`]): Switch homebrew.
//! - NSO ([`nso`]): system modules and games (LZ4 or Switch 2 ZBIC).

pub mod nca;
pub mod nro;
pub mod nso;

use emu_memory::{Memory, Perm, PAGE_SIZE};
use std::collections::BTreeMap;
use std::fmt;

const PT_LOAD: u32 = 1;
const EM_AARCH64: u16 = 183;
const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    Truncated,
    NotElf,
    Unsupported(&'static str),
    Memory(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Truncated => write!(f, "file is truncated"),
            LoadError::NotElf => write!(f, "not an ELF file"),
            LoadError::Unsupported(what) => write!(f, "unsupported ELF: {what}"),
            LoadError::Memory(e) => write!(f, "memory error: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// One loaded segment, for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub vaddr: u64,
    pub mem_size: u64,
    pub perm: Perm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub entry: u64,
    pub segments: Vec<Segment>,
}

fn u16_at(d: &[u8], off: usize) -> Result<u16, LoadError> {
    d.get(off..off + 2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .ok_or(LoadError::Truncated)
}

fn u32_at(d: &[u8], off: usize) -> Result<u32, LoadError> {
    d.get(off..off + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or(LoadError::Truncated)
}

fn u64_at(d: &[u8], off: usize) -> Result<u64, LoadError> {
    d.get(off..off + 8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .ok_or(LoadError::Truncated)
}

fn union(a: Perm, b: Perm) -> Perm {
    Perm {
        read: a.read || b.read,
        write: a.write || b.write,
        exec: a.exec || b.exec,
    }
}

/// Maps and fills every `PT_LOAD` segment of `elf`. Pages shared by two
/// segments get the union of their permissions.
pub fn load_elf(mem: &mut Memory, elf: &[u8]) -> Result<Image, LoadError> {
    if elf.get(0..4) != Some(b"\x7fELF") {
        return Err(LoadError::NotElf);
    }
    if elf.get(4) != Some(&2) {
        return Err(LoadError::Unsupported("not 64-bit"));
    }
    if elf.get(5) != Some(&1) {
        return Err(LoadError::Unsupported("not little-endian"));
    }
    if u16_at(elf, 18)? != EM_AARCH64 {
        return Err(LoadError::Unsupported("not AArch64"));
    }
    let entry = u64_at(elf, 24)?;
    let phoff = u64_at(elf, 32)? as usize;
    let phentsize = u16_at(elf, 54)? as usize;
    let phnum = u16_at(elf, 56)? as usize;

    struct Load {
        vaddr: u64,
        offset: usize,
        file_size: usize,
        mem_size: u64,
        perm: Perm,
    }
    let mut loads = Vec::new();
    for i in 0..phnum {
        let ph = phoff + i * phentsize;
        if u32_at(elf, ph)? != PT_LOAD {
            continue;
        }
        let flags = u32_at(elf, ph + 4)?;
        let load = Load {
            offset: u64_at(elf, ph + 8)? as usize,
            vaddr: u64_at(elf, ph + 16)?,
            file_size: u64_at(elf, ph + 32)? as usize,
            mem_size: u64_at(elf, ph + 40)?,
            perm: Perm {
                read: flags & PF_R != 0,
                write: flags & PF_W != 0,
                exec: flags & PF_X != 0,
            },
        };
        if load.file_size as u64 > load.mem_size {
            return Err(LoadError::Unsupported(
                "segment file size exceeds memory size",
            ));
        }
        if elf.len() < load.offset + load.file_size {
            return Err(LoadError::Truncated);
        }
        loads.push(load);
    }

    // Work out the permissions of every page first, then map each page once.
    let mut pages: BTreeMap<u64, Perm> = BTreeMap::new();
    for l in &loads {
        if l.mem_size == 0 {
            continue;
        }
        let first = l.vaddr / PAGE_SIZE;
        let last = (l.vaddr + l.mem_size - 1) / PAGE_SIZE;
        for page in first..=last {
            let perm = pages.get(&page).map_or(l.perm, |p| union(*p, l.perm));
            pages.insert(page, perm);
        }
    }
    for (&page, &perm) in &pages {
        mem.map(page * PAGE_SIZE, PAGE_SIZE, perm)
            .map_err(|e| LoadError::Memory(format!("{e:?}")))?;
    }
    for l in &loads {
        // Memory is zeroed when mapped, which also clears .bss.
        mem.write_privileged(l.vaddr, &elf[l.offset..l.offset + l.file_size])
            .map_err(|e| LoadError::Memory(e.to_string()))?;
    }

    Ok(Image {
        entry,
        segments: loads
            .iter()
            .map(|l| Segment {
                vaddr: l.vaddr,
                mem_size: l.mem_size,
                perm: l.perm,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal ELF with one RX segment holding `code`.
    fn tiny_elf(vaddr: u64, code: &[u8], mem_size: u64) -> Vec<u8> {
        let mut e = vec![0u8; 64 + 56];
        e[0..4].copy_from_slice(b"\x7fELF");
        e[4] = 2;
        e[5] = 1;
        e[6] = 1;
        e[16..18].copy_from_slice(&2u16.to_le_bytes());
        e[18..20].copy_from_slice(&EM_AARCH64.to_le_bytes());
        e[24..32].copy_from_slice(&vaddr.to_le_bytes());
        e[32..40].copy_from_slice(&64u64.to_le_bytes());
        e[54..56].copy_from_slice(&56u16.to_le_bytes());
        e[56..58].copy_from_slice(&1u16.to_le_bytes());
        let ph = 64;
        e[ph..ph + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
        e[ph + 4..ph + 8].copy_from_slice(&(PF_R | PF_X).to_le_bytes());
        e[ph + 8..ph + 16].copy_from_slice(&120u64.to_le_bytes());
        e[ph + 16..ph + 24].copy_from_slice(&vaddr.to_le_bytes());
        e[ph + 32..ph + 40].copy_from_slice(&(code.len() as u64).to_le_bytes());
        e[ph + 40..ph + 48].copy_from_slice(&mem_size.to_le_bytes());
        e.extend_from_slice(code);
        e
    }

    #[test]
    fn loads_segment_and_zeroes_bss() {
        let code = 0xD503_201Fu32.to_le_bytes();
        let elf = tiny_elf(0x8000_0000, &code, 0x2000);
        let mut mem = Memory::new();
        let image = load_elf(&mut mem, &elf).unwrap();
        assert_eq!(image.entry, 0x8000_0000);
        assert_eq!(mem.fetch_u32(0x8000_0000).unwrap(), 0xD503_201F);
        assert_eq!(mem.read_u64(0x8000_1FF8).unwrap(), 0, "bss is zero");
        assert!(mem.write_u64(0x8000_0000, 0).is_err(), "text is read-only");
    }

    #[test]
    fn rejects_bad_input() {
        let mut mem = Memory::new();
        assert_eq!(load_elf(&mut mem, b"nope"), Err(LoadError::NotElf));
        let mut elf = tiny_elf(0x1000, &[0; 4], 4);
        elf[18] = 62; // x86-64
        assert_eq!(
            load_elf(&mut mem, &elf),
            Err(LoadError::Unsupported("not AArch64"))
        );
        let elf = tiny_elf(0x1000, &[0; 4], 4);
        assert_eq!(
            load_elf(&mut mem, &elf[..elf.len() - 2]),
            Err(LoadError::Truncated)
        );
    }
}
