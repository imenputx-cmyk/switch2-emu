// SPDX-License-Identifier: GPL-3.0-or-later
//! NPDM (`main.npdm` in an ExeFS): the program's process settings.
//!
//! Layout from Eden's `core/file_sys/program_metadata.h`
//! (GPL-3.0-or-later, based on yuzu): a 0x80-byte "META" header, and an
//! "ACI0" section holding the program ID.

use crate::LoadError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Npdm {
    pub is_64bit: bool,
    pub address_space_type: u8,
    pub main_thread_priority: u8,
    pub main_thread_core: u8,
    pub main_stack_size: u32,
    pub system_resource_size: u32,
    pub program_id: u64,
    pub name: [u8; 16],
}

fn u32_at(d: &[u8], off: usize) -> Result<u32, LoadError> {
    d.get(off..off + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .ok_or(LoadError::Truncated)
}

impl Npdm {
    pub fn parse(d: &[u8]) -> Result<Npdm, LoadError> {
        if d.get(..4) != Some(b"META") {
            return Err(LoadError::Unsupported("not an NPDM"));
        }
        let aci = u32_at(d, 0x70)? as usize;
        if d.get(aci..aci + 4) != Some(b"ACI0") {
            return Err(LoadError::Unsupported("NPDM without ACI0"));
        }
        let program_id = d
            .get(aci + 0x10..aci + 0x18)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .ok_or(LoadError::Truncated)?;
        let mut name = [0u8; 16];
        name.copy_from_slice(&d[0x20..0x30]);
        Ok(Npdm {
            is_64bit: d[0xC] & 1 != 0,
            address_space_type: (d[0xC] >> 1) & 7,
            main_thread_priority: d[0xE],
            main_thread_core: d[0xF],
            main_stack_size: u32_at(d, 0x1C)?,
            system_resource_size: u32_at(d, 0x14)?,
            program_id,
            name,
        })
    }

    pub fn name(&self) -> String {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(16);
        String::from_utf8_lossy(&self.name[..end]).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_program_id() {
        let mut d = vec![0u8; 0x100];
        d[..4].copy_from_slice(b"META");
        d[0xC] = 1 | 3 << 1;
        d[0xE] = 0x2C;
        d[0xF] = 0;
        d[0x1C..0x20].copy_from_slice(&0x10_0000u32.to_le_bytes());
        d[0x20..0x24].copy_from_slice(b"Game");
        d[0x70..0x74].copy_from_slice(&0x80u32.to_le_bytes());
        d[0x80..0x84].copy_from_slice(b"ACI0");
        d[0x90..0x98].copy_from_slice(&0x0100_1234_5678_0000u64.to_le_bytes());
        let n = Npdm::parse(&d).unwrap();
        assert!(n.is_64bit);
        assert_eq!((n.address_space_type, n.main_thread_priority), (3, 0x2C));
        assert_eq!(n.main_stack_size, 0x10_0000);
        assert_eq!(n.program_id, 0x0100_1234_5678_0000);
        assert_eq!(n.name(), "Game");
    }
}
