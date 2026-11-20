// SPDX-License-Identifier: GPL-3.0-or-later
//! Game containers: NCA content archives (encrypted), the PFS0 and HFS0
//! partition file systems, NSP packages and XCI game card images.
//!
//! Structures follow Eden's `core/file_sys` (GPL-3.0-or-later, based on
//! yuzu): `fssystem/fssystem_nca_header.h` (NcaHeader, NcaFsHeader),
//! `fssystem_aes_ctr_storage.cpp` (the CTR IV), `partition_filesystem`
//! and `card_image`.
//!
//! Decryption needs the user's own console keys (a `prod.keys` file as
//! dumped with Lockpick_RCM): `header_key`,
//! `key_area_key_application_XX` and, for titles with a rights ID,
//! `titlekek_XX` with the title key from the package's ticket. No keys
//! are included in this project.

use crate::LoadError;
use aes::cipher::{BlockDecrypt, KeyInit, KeyIvInit, StreamCipher, StreamCipherSeek};
use aes::Aes128;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

type Ctr = ctr::Ctr128BE<Aes128>;

/// Console keys by name (lower case), from a `name = hex` text file.
#[derive(Debug, Clone, Default)]
pub struct Keys(pub HashMap<String, Vec<u8>>);

impl Keys {
    pub fn parse(text: &str) -> Keys {
        let mut map = HashMap::new();
        for line in text.lines() {
            let Some((name, value)) = line.split_once('=') else {
                continue;
            };
            let value = value.trim();
            if value.len() % 2 != 0 {
                continue;
            }
            let bytes: Option<Vec<u8>> = (0..value.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&value[i..i + 2], 16).ok())
                .collect();
            if let Some(b) = bytes {
                map.insert(name.trim().to_ascii_lowercase(), b);
            }
        }
        Keys(map)
    }

    pub fn load(path: &Path) -> Result<Keys, LoadError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| LoadError::Memory(format!("{}: {e}", path.display())))?;
        Ok(Keys::parse(&text))
    }

    fn get<const N: usize>(&self, name: &str) -> Result<[u8; N], LoadError> {
        self.0
            .get(name)
            .and_then(|k| k.as_slice().try_into().ok())
            .ok_or_else(|| LoadError::Memory(format!("missing key {name}")))
    }
}

/// Random-access bytes: a file or a range of another storage.
#[derive(Clone)]
pub enum Storage {
    Memory(Arc<Vec<u8>>),
    File(Arc<File>),
    Slice(Arc<Storage>, u64, u64),
}

impl Storage {
    pub fn open(path: &Path) -> Result<Storage, LoadError> {
        File::open(path)
            .map(|f| Storage::File(Arc::new(f)))
            .map_err(|e| LoadError::Memory(format!("{}: {e}", path.display())))
    }

    pub fn len(&self) -> u64 {
        match self {
            Storage::Memory(v) => v.len() as u64,
            Storage::File(f) => f.metadata().map_or(0, |m| m.len()),
            Storage::Slice(_, _, len) => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), LoadError> {
        if offset + buf.len() as u64 > self.len() {
            return Err(LoadError::Truncated);
        }
        match self {
            Storage::Memory(v) => {
                buf.copy_from_slice(&v[offset as usize..offset as usize + buf.len()]);
                Ok(())
            }
            Storage::File(f) => read_file_at(f, offset, buf),
            Storage::Slice(base, start, _) => base.read_at(start + offset, buf),
        }
    }

    pub fn read_vec(&self, offset: u64, len: usize) -> Result<Vec<u8>, LoadError> {
        let mut v = vec![0u8; len];
        self.read_at(offset, &mut v)?;
        Ok(v)
    }

    pub fn slice(&self, offset: u64, len: u64) -> Result<Storage, LoadError> {
        if offset + len > self.len() {
            return Err(LoadError::Truncated);
        }
        Ok(Storage::Slice(Arc::new(self.clone()), offset, len))
    }
}

#[cfg(unix)]
fn read_file_at(f: &File, offset: u64, buf: &mut [u8]) -> Result<(), LoadError> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, offset)
        .map_err(|e| LoadError::Memory(e.to_string()))
}

#[cfg(windows)]
fn read_file_at(f: &File, mut offset: u64, mut buf: &mut [u8]) -> Result<(), LoadError> {
    while !buf.is_empty() {
        let n = std::os::windows::fs::FileExt::seek_read(f, buf, offset)
            .map_err(|e| LoadError::Memory(e.to_string()))?;
        if n == 0 {
            return Err(LoadError::Truncated);
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

/// XTS tweak of the NCA header: the sector index, big-endian.
fn nintendo_tweak(sector: u128) -> [u8; 16] {
    sector.to_be_bytes()
}

fn u32_at(d: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(d[off..off + 4].try_into().unwrap())
}

fn u64_at(d: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(d[off..off + 8].try_into().unwrap())
}

/// A file of a partition file system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

/// A PFS0 or HFS0 partition: named files in one storage.
#[derive(Clone)]
pub struct Partition {
    pub storage: Storage,
    pub entries: Vec<Entry>,
}

impl Partition {
    /// Parses a PFS0 (entries of 0x18 bytes) or HFS0 (0x40) at the start
    /// of `storage`.
    pub fn parse(storage: Storage) -> Result<Partition, LoadError> {
        let head = storage.read_vec(0, 0x10)?;
        let entry_size = match &head[..4] {
            b"PFS0" => 0x18,
            b"HFS0" => 0x40,
            _ => return Err(LoadError::Unsupported("not a PFS0/HFS0 partition")),
        };
        let (count, strings) = (u32_at(&head, 4) as usize, u32_at(&head, 8) as usize);
        if count > 0x10000 || strings > 0x100_0000 {
            return Err(LoadError::Unsupported("partition header too large"));
        }
        let table = storage.read_vec(0x10, count * entry_size + strings)?;
        let data_start = (0x10 + count * entry_size + strings) as u64;
        let names = &table[count * entry_size..];
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let e = &table[i * entry_size..];
            let name_at = u32_at(e, 0x10) as usize;
            let name = names
                .get(name_at..)
                .and_then(|n| n.split(|&b| b == 0).next())
                .ok_or(LoadError::Truncated)?;
            entries.push(Entry {
                name: String::from_utf8_lossy(name).into_owned(),
                offset: data_start + u64_at(e, 0),
                size: u64_at(e, 8),
            });
        }
        Ok(Partition { storage, entries })
    }

    pub fn open(&self, name: &str) -> Option<Storage> {
        let e = self.entries.iter().find(|e| e.name == name)?;
        self.storage.slice(e.offset, e.size).ok()
    }
}

/// The content types of an NCA.
pub mod content_type {
    pub const PROGRAM: u8 = 0;
    pub const META: u8 = 1;
    pub const CONTROL: u8 = 2;
}

/// A decrypting view of one NCA section (AES-CTR or plain).
#[derive(Clone)]
pub struct Section {
    nca: Storage,
    /// Absolute offset of the section in the NCA.
    start: u64,
    size: u64,
    key: Option<[u8; 16]>,
    upper_iv: u64,
}

impl Section {
    pub fn len(&self) -> u64 {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), LoadError> {
        if offset + buf.len() as u64 > self.size {
            return Err(LoadError::Truncated);
        }
        let abs = self.start + offset;
        let Some(key) = self.key else {
            return self.nca.read_at(abs, buf);
        };
        // CTR: IV = upper (big-endian) || absolute block index.
        let aligned = abs & !0xF;
        let skip = (abs - aligned) as usize;
        let mut data = self
            .nca
            .read_vec(aligned, (skip + buf.len()).next_multiple_of(16))?;
        let mut iv = [0u8; 16];
        iv[..8].copy_from_slice(&self.upper_iv.to_be_bytes());
        iv[8..].copy_from_slice(&(aligned / 16).to_be_bytes());
        let mut c = Ctr::new(&key.into(), &iv.into());
        c.seek(0u64);
        c.apply_keystream(&mut data);
        buf.copy_from_slice(&data[skip..skip + buf.len()]);
        Ok(())
    }

    /// The section's contents as a storage (decrypted in memory).
    pub fn to_storage(&self, offset: u64, len: u64) -> Result<Storage, LoadError> {
        let mut v = vec![0u8; len as usize];
        self.read_at(offset, &mut v)?;
        Ok(Storage::Memory(Arc::new(v)))
    }
}

/// An NCA content archive with its header decrypted.
pub struct Nca {
    pub header: Vec<u8>,
    pub content_type: u8,
    pub program_id: u64,
    pub rights_id: [u8; 16],
    sections: Vec<Option<(Section, Vec<u8>)>>,
}

impl Nca {
    /// Decrypts the header with `header_key` and the key area (or uses
    /// `title_key`, already decrypted, for titles with a rights ID).
    pub fn open(
        storage: Storage,
        keys: &Keys,
        title_key: Option<[u8; 16]>,
    ) -> Result<Nca, LoadError> {
        let header_key: [u8; 32] = keys.get("header_key")?;
        let mut header = storage.read_vec(0, 0xC00)?;
        let xts = xts_mode::Xts128::new(
            Aes128::new(header_key[..16].into()),
            Aes128::new(header_key[16..].into()),
        );
        xts.decrypt_area(&mut header, 0x200, 0, nintendo_tweak);
        if !matches!(&header[0x200..0x204], b"NCA3" | b"NCA2") {
            return Err(LoadError::Unsupported(
                "NCA header does not decrypt (wrong header_key?)",
            ));
        }
        let generation = header[0x206].max(header[0x220]).saturating_sub(1);
        let mut rights_id = [0u8; 16];
        rights_id.copy_from_slice(&header[0x230..0x240]);
        let has_rights = rights_id != [0; 16];
        let ctr_key = if has_rights {
            title_key
        } else {
            let kak_name = match header[0x207] {
                0 => "key_area_key_application",
                1 => "key_area_key_ocean",
                _ => "key_area_key_system",
            };
            let kak: Option<[u8; 16]> = keys.get(&format!("{kak_name}_{generation:02x}")).ok();
            kak.map(|kak| {
                // Key area entry 2 is the AES-CTR key.
                let mut k = [0u8; 16];
                k.copy_from_slice(&header[0x300 + 0x20..0x300 + 0x30]);
                Aes128::new(&kak.into()).decrypt_block((&mut k).into());
                k
            })
        };
        let mut sections = Vec::new();
        for i in 0..4 {
            let info = &header[0x240 + 0x10 * i..];
            let (start, end) = (
                u64::from(u32_at(info, 0)) * 0x200,
                u64::from(u32_at(info, 4)) * 0x200,
            );
            if end <= start {
                sections.push(None);
                continue;
            }
            let fs = header[0x400 + 0x200 * i..0x600 + 0x200 * i].to_vec();
            let key = match fs[4] {
                1 => None,
                3 => Some(ctr_key.ok_or(LoadError::Memory(if has_rights {
                    "missing title key (ticket)".into()
                } else {
                    format!("missing key_area_key for generation {generation:02x}")
                }))?),
                _ => return Err(LoadError::Unsupported("NCA section encryption type")),
            };
            sections.push(Some((
                Section {
                    nca: storage.clone(),
                    start,
                    size: end - start,
                    key,
                    upper_iv: u64_at(&fs, 0x140),
                },
                fs,
            )));
        }
        Ok(Nca {
            content_type: header[0x205],
            program_id: u64_at(&header, 0x210),
            rights_id,
            header,
            sections,
        })
    }

    /// The data region of section `i` (past its hash layers), decrypted.
    pub fn section_data(&self, i: usize) -> Result<(Section, u64, u64), LoadError> {
        let (sec, fs) = self
            .sections
            .get(i)
            .and_then(|s| s.as_ref())
            .ok_or(LoadError::Truncated)?;
        let hash = &fs[8..];
        let (offset, size) = match fs[3] {
            // Hierarchical SHA-256: the last layer region is the data.
            2 => {
                let layers = u32_at(hash, 0x24) as usize;
                let r = 0x28 + 0x10 * (layers.max(1) - 1);
                (u64_at(hash, r), u64_at(hash, r + 8))
            }
            // Hierarchical integrity (IVFC): level max_layers - 2.
            3 => {
                let max_layers = u32_at(hash, 0xC) as usize;
                let l = 0x10 + 0x18 * (max_layers.max(2) - 2);
                (u64_at(hash, l), u64_at(hash, l + 8))
            }
            _ => (0, sec.len()),
        };
        Ok((sec.clone(), offset, size))
    }

    /// Section `i` as a partition file system (the ExeFS of a program).
    pub fn partition(&self, i: usize) -> Result<Partition, LoadError> {
        let (sec, offset, size) = self.section_data(i)?;
        Partition::parse(sec.to_storage(offset, size)?)
    }

    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Is section `i` a partition file system?
    pub fn is_partition(&self, i: usize) -> bool {
        matches!(self.sections.get(i), Some(Some((_, fs))) if fs[2] == 1)
    }
}

/// Decrypted title keys by rights ID.
pub type TitleKeys = HashMap<[u8; 16], [u8; 16]>;

/// Decrypts a title key from a ticket with `titlekek_XX`.
pub fn title_key(ticket: &[u8], keys: &Keys) -> Result<([u8; 16], [u8; 16]), LoadError> {
    if ticket.len() < 0x2C0 {
        return Err(LoadError::Truncated);
    }
    // The signature block's size depends on its type (RSA-4096: 0x240).
    let sig = u32_at(ticket, 0);
    let body = match sig {
        0x10000 | 0x10003 => 0x240,
        0x10001 | 0x10004 => 0x140,
        0x10002 | 0x10005 => 0x80,
        _ => return Err(LoadError::Unsupported("ticket signature type")),
    };
    let t = ticket.get(body..body + 0x180).ok_or(LoadError::Truncated)?;
    let mut key = [0u8; 16];
    key.copy_from_slice(&t[0x40..0x50]);
    let generation = t[0x145].saturating_sub(1);
    let mut rights = [0u8; 16];
    rights.copy_from_slice(&t[0x160..0x170]);
    if t[0x141] != 0 {
        return Err(LoadError::Unsupported("personalized tickets"));
    }
    let kek: [u8; 16] = keys.get(&format!("titlekek_{generation:02x}"))?;
    Aes128::new(&kek.into()).decrypt_block((&mut key).into());
    Ok((rights, key))
}

/// Opens an NSP (PFS0 with NCAs and tickets) or XCI (game card) and
/// returns its NCAs and title keys by rights ID.
pub fn open_package(storage: Storage, keys: &Keys) -> Result<(Vec<Storage>, TitleKeys), LoadError> {
    let head = storage.read_vec(0, 4)?;
    let part = if &head == b"PFS0" {
        Partition::parse(storage)?
    } else {
        // XCI: "HEAD" at 0x100 (0x1100 with a key area in front); the
        // root HFS0's offset at +0x30; NCAs are in its "secure" partition.
        let base = if storage.read_vec(0x100, 4)? == b"HEAD" {
            0
        } else if storage.len() > 0x1104 && storage.read_vec(0x1100, 4)? == b"HEAD" {
            0x1000
        } else {
            return Err(LoadError::Unsupported("not an NSP or XCI"));
        };
        let h = storage.read_vec(base + 0x100, 0x100)?;
        let (root_at, _) = (u64_at(&h, 0x30), u64_at(&h, 0x38));
        let root =
            Partition::parse(storage.slice(base + root_at, storage.len() - base - root_at)?)?;
        let secure = root
            .open("secure")
            .ok_or(LoadError::Unsupported("XCI without a secure partition"))?;
        Partition::parse(secure)?
    };
    let mut ncas = Vec::new();
    let mut title_keys = HashMap::new();
    for e in &part.entries {
        let s = part.storage.slice(e.offset, e.size)?;
        if e.name.ends_with(".nca") {
            ncas.push(s);
        } else if e.name.ends_with(".tik") {
            let ticket = s.read_vec(0, e.size as usize)?;
            if let Ok((rights, key)) = title_key(&ticket, keys) {
                title_keys.insert(rights, key);
            }
        }
    }
    Ok((ncas, title_keys))
}

/// A RomFS image inside an NCA section, decrypted as it is read.
#[derive(Clone)]
pub struct RomFs {
    section: Section,
    offset: u64,
    size: u64,
}

impl std::fmt::Debug for RomFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RomFs({:#x} bytes)", self.size)
    }
}

impl RomFs {
    pub fn len(&self) -> u64 {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Reads up to `buf.len()` bytes at `offset`; returns how many.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize, LoadError> {
        if offset >= self.size {
            return Ok(0);
        }
        let n = (buf.len() as u64).min(self.size - offset) as usize;
        self.section.read_at(self.offset + offset, &mut buf[..n])?;
        Ok(n)
    }
}

/// The program's ExeFS and RomFS (if any) in a package or NCA.
pub struct Program {
    pub exefs: Partition,
    pub romfs: Option<RomFs>,
}

/// Finds the program's ExeFS (the partition with a `main` NSO) in a
/// package or a single NCA.
pub fn find_exefs(path: &Path, keys: &Keys) -> Result<Partition, LoadError> {
    find_program(path, keys).map(|p| p.exefs)
}

/// Finds the program NCA's ExeFS and RomFS.
pub fn find_program(path: &Path, keys: &Keys) -> Result<Program, LoadError> {
    let storage = Storage::open(path)?;
    let is_nca = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("nca"));
    let (ncas, title_keys) = if is_nca {
        (vec![storage], HashMap::new())
    } else {
        open_package(storage, keys)?
    };
    let mut last_error = LoadError::Unsupported("no program NCA");
    for s in ncas {
        let nca = match Nca::open(s.clone(), keys, None) {
            Ok(n) if n.content_type == content_type::PROGRAM => n,
            Ok(_) => continue,
            Err(e) => {
                last_error = e;
                continue;
            }
        };
        let nca = if nca.rights_id != [0; 16] {
            let key = title_keys.get(&nca.rights_id).copied();
            Nca::open(s, keys, key)?
        } else {
            nca
        };
        let mut exefs = None;
        let mut romfs = None;
        for i in 0..nca.section_count() {
            if nca.is_partition(i) {
                match nca.partition(i) {
                    Ok(p) if p.entries.iter().any(|e| e.name == "main") => exefs = Some(p),
                    Ok(_) => {}
                    Err(e) => last_error = e,
                }
            } else if nca.sections.get(i).is_some_and(|s| s.is_some()) && romfs.is_none() {
                if let Ok((section, offset, size)) = nca.section_data(i) {
                    romfs = Some(RomFs {
                        section,
                        offset,
                        size,
                    });
                }
            }
        }
        if let Some(exefs) = exefs {
            return Ok(Program { exefs, romfs });
        }
    }
    Err(last_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes::cipher::BlockEncrypt;

    fn pfs0(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut names: Vec<u8> = Vec::new();
        let mut entries: Vec<u8> = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, content) in files {
            entries.extend((data.len() as u64).to_le_bytes());
            entries.extend((content.len() as u64).to_le_bytes());
            entries.extend((names.len() as u32).to_le_bytes());
            entries.extend(0u32.to_le_bytes());
            names.extend(name.as_bytes());
            names.push(0);
            data.extend(*content);
        }
        names.resize(names.len().next_multiple_of(0x20), 0);
        let mut out = b"PFS0".to_vec();
        out.extend((files.len() as u32).to_le_bytes());
        out.extend((names.len() as u32).to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(entries);
        out.extend(names);
        out.extend(data);
        out
    }

    /// An NCA with one CTR-encrypted PFS0 section (no hash layers).
    fn make_nca(keys: &Keys, exefs: &[u8]) -> Vec<u8> {
        let header_key: [u8; 32] = keys.get("header_key").unwrap();
        let kak: [u8; 16] = keys.get("key_area_key_application_00").unwrap();
        let ctr_key = [0x42u8; 16];
        let mut header = vec![0u8; 0xC00];
        header[0x200..0x204].copy_from_slice(b"NCA3");
        header[0x205] = content_type::PROGRAM;
        header[0x210..0x218].copy_from_slice(&0x0100_0000_0000_1000u64.to_le_bytes());
        let mut k = ctr_key;
        Aes128::new(&kak.into()).encrypt_block((&mut k).into());
        header[0x320..0x330].copy_from_slice(&k);
        // Section 0 from sector 6 (0xC00) on.
        let sectors = exefs.len().div_ceil(0x200) as u32;
        header[0x240..0x244].copy_from_slice(&6u32.to_le_bytes());
        header[0x244..0x248].copy_from_slice(&(6 + sectors).to_le_bytes());
        let fs = &mut header[0x400..0x600];
        fs[2] = 1; // PFS0
        fs[3] = 1; // no hash
        fs[4] = 3; // AES-CTR
        fs[0x140..0x148].copy_from_slice(&0x1234_5678u64.to_le_bytes());
        let xts = xts_mode::Xts128::new(
            Aes128::new(header_key[..16].into()),
            Aes128::new(header_key[16..].into()),
        );
        xts.encrypt_area(&mut header, 0x200, 0, nintendo_tweak);
        let mut body = exefs.to_vec();
        body.resize(sectors as usize * 0x200, 0);
        let mut iv = [0u8; 16];
        iv[..8].copy_from_slice(&0x1234_5678u64.to_be_bytes());
        iv[8..].copy_from_slice(&(0xC00u64 / 16).to_be_bytes());
        Ctr::new(&ctr_key.into(), &iv.into()).apply_keystream(&mut body);
        header.extend(body);
        header
    }

    #[test]
    fn nca_in_nsp() {
        let keys = Keys::parse(
            "header_key = 00112233445566778899aabbccddeeff0123456789abcdef0123456789abcdef\n\
             key_area_key_application_00 = 000102030405060708090a0b0c0d0e0f\n",
        );
        let exefs = pfs0(&[("main", b"NSO0 main module"), ("main.npdm", b"META")]);
        let nca = make_nca(&keys, &exefs);
        let nsp = pfs0(&[("0123.nca", &nca)]);
        let dir = std::env::temp_dir().join(format!("emu-nsp-{}.nsp", std::process::id()));
        std::fs::write(&dir, &nsp).unwrap();
        let part = find_exefs(&dir, &keys).unwrap();
        std::fs::remove_file(&dir).unwrap();
        let main = part.open("main").unwrap();
        assert_eq!(main.read_vec(0, 16).unwrap(), b"NSO0 main module");
        assert_eq!(
            part.open("main.npdm").unwrap().read_vec(0, 4).unwrap(),
            b"META"
        );
    }

    #[test]
    fn keys_file() {
        let k = Keys::parse("; comment\nTitlekek_00 = 0102\nbad = xyz\n");
        assert_eq!(k.0.get("titlekek_00"), Some(&vec![1, 2]));
        assert!(!k.0.contains_key("bad"));
    }
}
