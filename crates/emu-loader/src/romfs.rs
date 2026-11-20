// SPDX-License-Identifier: GPL-3.0-or-later
//! A RomFS image built on the fly from a host directory (an extracted
//! game's `romfs/` folder), so a loose ExeFS + RomFS dump runs like the
//! package it came from.
//!
//! Ported from Lichii's `core/file_sys/fsmitm_romfsbuild.cpp`
//! (GPL-3.0-or-later, from Atmosphère's fs.mitm): the same table layout,
//! path hashes and hash table sizes. Only the header and tables are kept
//! in memory; file data is read from the host files when the game reads
//! it, so multi-gigabyte RomFS folders cost nothing up front.

use crate::LoadError;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const HEADER_SIZE: u64 = 0x50;
const FILE_PARTITION: u64 = 0x200;
const EMPTY: u32 = u32::MAX;

fn path_hash(parent: u32, name: &[u8]) -> u32 {
    let mut hash = parent ^ 123_456_789;
    for &b in name {
        // Names are hashed as C `char`s (signed), like Atmosphère and Eden.
        hash = hash.rotate_right(5) ^ (b as i8 as i32 as u32);
    }
    hash
}

fn hash_table_count(n: usize) -> usize {
    if n < 3 {
        return 3;
    }
    if n < 19 {
        return n | 1;
    }
    let mut count = n;
    while [2, 3, 5, 7, 11, 13, 17].iter().any(|p| count.is_multiple_of(*p)) {
        count += 1;
    }
    count
}

struct Dir {
    /// Full path from the root ("" for the root, "/a/b" below it).
    path: String,
    name_len: usize,
    parent: usize,
    entry: u32,
    child: Option<usize>,
    sibling: Option<usize>,
    file: Option<usize>,
}

struct FileNode {
    path: String,
    name_len: usize,
    parent: usize,
    entry: u32,
    sibling: Option<usize>,
    offset: u64,
    size: u64,
    host: PathBuf,
}

/// A RomFS image backed by a host directory.
pub struct DirRomFs {
    /// Header (at 0) and the four tables (at `meta_offset`).
    header: Vec<u8>,
    meta: Vec<u8>,
    meta_offset: u64,
    /// File data regions: (image offset, size, host file), sorted.
    files: Vec<(u64, u64, PathBuf)>,
    open: Mutex<Option<(usize, File)>>,
    size: u64,
}

impl std::fmt::Debug for DirRomFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DirRomFs({} files, {:#x} bytes)",
            self.files.len(),
            self.size
        )
    }
}

fn walk(
    dir: &Path,
    rel: &str,
    parent: usize,
    dirs: &mut Vec<Dir>,
    files: &mut Vec<FileNode>,
) -> Result<(), LoadError> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| LoadError::Memory(format!("{}: {e}", dir.display())))?
        .filter_map(Result::ok)
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().to_string_lossy().into_owned();
        let path = format!("{rel}/{name}");
        let ty = e
            .file_type()
            .map_err(|e| LoadError::Memory(e.to_string()))?;
        if ty.is_dir() {
            dirs.push(Dir {
                path: path.clone(),
                name_len: name.len(),
                parent,
                entry: 0,
                child: None,
                sibling: None,
                file: None,
            });
            let me = dirs.len() - 1;
            walk(&e.path(), &path, me, dirs, files)?;
        } else {
            let size = e
                .metadata()
                .map_err(|e| LoadError::Memory(e.to_string()))?
                .len();
            files.push(FileNode {
                path,
                name_len: name.len(),
                parent,
                entry: 0,
                sibling: None,
                offset: 0,
                size,
                host: e.path(),
            });
        }
    }
    Ok(())
}

impl DirRomFs {
    pub fn build(root: &Path) -> Result<DirRomFs, LoadError> {
        let mut dirs = vec![Dir {
            path: String::new(),
            name_len: 0,
            parent: 0,
            entry: 0,
            child: None,
            sibling: None,
            file: None,
        }];
        let mut files = Vec::new();
        walk(root, "", 0, &mut dirs, &mut files)?;

        // Sort by path (byte order), keeping indices consistent: sort
        // index lists, then renumber.
        let mut dir_order: Vec<usize> = (0..dirs.len()).collect();
        dir_order.sort_by(|&a, &b| dirs[a].path.as_bytes().cmp(dirs[b].path.as_bytes()));
        let mut file_order: Vec<usize> = (0..files.len()).collect();
        file_order.sort_by(|&a, &b| files[a].path.as_bytes().cmp(files[b].path.as_bytes()));

        // File data offsets and entry offsets.
        let mut data = 0u64;
        let mut entry = 0u32;
        for &i in &file_order {
            data = data.next_multiple_of(16);
            files[i].offset = data;
            data += files[i].size;
            files[i].entry = entry;
            entry += 0x20 + (files[i].name_len as u32).next_multiple_of(4);
        }
        let file_table_size = entry;
        // Link files (reverse order so each directory's list is sorted).
        for &i in file_order.iter().rev() {
            let p = files[i].parent;
            files[i].sibling = dirs[p].file;
            dirs[p].file = Some(i);
        }
        entry = 0;
        for &i in &dir_order {
            dirs[i].entry = entry;
            entry += 0x18 + (dirs[i].name_len as u32).next_multiple_of(4);
        }
        let dir_table_size = entry;
        for &i in dir_order.iter().rev().filter(|&&i| i != 0) {
            let p = dirs[i].parent;
            dirs[i].sibling = dirs[p].child;
            dirs[p].child = Some(i);
        }

        let dir_buckets = hash_table_count(dirs.len());
        let file_buckets = hash_table_count(files.len());
        let mut dir_hash = vec![EMPTY; dir_buckets];
        let mut file_hash = vec![EMPTY; file_buckets];
        let mut dir_table = vec![0u8; dir_table_size as usize];
        let mut file_table = vec![0u8; file_table_size as usize];
        let put = |t: &mut Vec<u8>, at: usize, v: &[u8]| t[at..at + v.len()].copy_from_slice(v);

        for &i in &file_order {
            let f = &files[i];
            let parent_entry = dirs[f.parent].entry;
            let name = &f.path.as_bytes()[f.path.len() - f.name_len..];
            let h = path_hash(parent_entry, name) as usize % file_buckets;
            let at = f.entry as usize;
            put(&mut file_table, at, &parent_entry.to_le_bytes());
            put(
                &mut file_table,
                at + 4,
                &f.sibling.map_or(EMPTY, |s| files[s].entry).to_le_bytes(),
            );
            put(&mut file_table, at + 8, &f.offset.to_le_bytes());
            put(&mut file_table, at + 16, &f.size.to_le_bytes());
            put(&mut file_table, at + 24, &file_hash[h].to_le_bytes());
            put(&mut file_table, at + 28, &(f.name_len as u32).to_le_bytes());
            put(&mut file_table, at + 32, name);
            file_hash[h] = f.entry;
        }
        for &i in &dir_order {
            let d = &dirs[i];
            let parent_entry = if i == 0 { 0 } else { dirs[d.parent].entry };
            let name = &d.path.as_bytes()[d.path.len() - d.name_len..];
            let h = path_hash(parent_entry, name) as usize % dir_buckets;
            let at = d.entry as usize;
            put(&mut dir_table, at, &parent_entry.to_le_bytes());
            put(
                &mut dir_table,
                at + 4,
                &d.sibling.map_or(EMPTY, |s| dirs[s].entry).to_le_bytes(),
            );
            put(
                &mut dir_table,
                at + 8,
                &d.child.map_or(EMPTY, |s| dirs[s].entry).to_le_bytes(),
            );
            put(
                &mut dir_table,
                at + 12,
                &d.file.map_or(EMPTY, |s| files[s].entry).to_le_bytes(),
            );
            put(&mut dir_table, at + 16, &dir_hash[h].to_le_bytes());
            put(&mut dir_table, at + 20, &(d.name_len as u32).to_le_bytes());
            put(&mut dir_table, at + 24, name);
            dir_hash[h] = d.entry;
        }

        let meta_offset = (FILE_PARTITION + data).next_multiple_of(4);
        let dir_hash_size = 4 * dir_buckets as u64;
        let file_hash_size = 4 * file_buckets as u64;
        let fields = [
            HEADER_SIZE,
            meta_offset,
            dir_hash_size,
            meta_offset + dir_hash_size,
            u64::from(dir_table_size),
            meta_offset + dir_hash_size + u64::from(dir_table_size),
            file_hash_size,
            meta_offset + dir_hash_size + u64::from(dir_table_size) + file_hash_size,
            u64::from(file_table_size),
            FILE_PARTITION,
        ];
        let header: Vec<u8> = fields.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut meta: Vec<u8> = dir_hash.iter().flat_map(|v| v.to_le_bytes()).collect();
        meta.extend(dir_table);
        meta.extend(file_hash.iter().flat_map(|v| v.to_le_bytes()));
        meta.extend(file_table);
        let size = meta_offset + meta.len() as u64;
        let mut regions: Vec<(u64, u64, PathBuf)> = file_order
            .iter()
            .map(|&i| {
                (
                    FILE_PARTITION + files[i].offset,
                    files[i].size,
                    files[i].host.clone(),
                )
            })
            .collect();
        regions.sort_by_key(|r| r.0);
        Ok(DirRomFs {
            header,
            meta,
            meta_offset,
            files: regions,
            open: Mutex::new(None),
            size,
        })
    }

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
        buf[..n].fill(0);
        let end = offset + n as u64;
        let copy = |buf: &mut [u8], src: &[u8], src_at: u64| {
            // Overlap of [offset, end) with [src_at, src_at + len).
            let lo = offset.max(src_at);
            let hi = end.min(src_at + src.len() as u64);
            if lo < hi {
                buf[(lo - offset) as usize..(hi - offset) as usize]
                    .copy_from_slice(&src[(lo - src_at) as usize..(hi - src_at) as usize]);
            }
        };
        copy(buf, &self.header, 0);
        copy(buf, &self.meta, self.meta_offset);
        // File data: the regions overlapping the range.
        let first = self.files.partition_point(|r| r.0 + r.1 <= offset);
        for (k, (at, size, path)) in self.files.iter().enumerate().skip(first) {
            if *at >= end {
                break;
            }
            let lo = offset.max(*at);
            let hi = end.min(at + size);
            if lo >= hi {
                continue;
            }
            let mut open = self.open.lock().unwrap();
            if open.as_ref().is_none_or(|(i, _)| *i != k) {
                let f = File::open(path)
                    .map_err(|e| LoadError::Memory(format!("{}: {e}", path.display())))?;
                *open = Some((k, f));
            }
            let f = &open.as_ref().unwrap().1;
            crate::nca::read_file_at(
                f,
                lo - at,
                &mut buf[(lo - offset) as usize..(hi - offset) as usize],
            )?;
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_at(d: &[u8], o: usize) -> u32 {
        u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
    }
    fn u64_at(d: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(d[o..o + 8].try_into().unwrap())
    }

    /// Looks a path up the way games do: hash buckets, then names.
    fn lookup(img: &[u8], path: &str) -> Option<Vec<u8>> {
        let h = |i: usize| u64_at(img, 8 * i) as usize;
        let (dh, dhs, dt, fh, fhs, ft, data) = (h(1), h(2), h(3), h(5), h(6), h(7), h(9));
        let mut parent = 0u32;
        let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
        for (i, part) in parts.iter().enumerate() {
            let last = i == parts.len() - 1;
            let (hash_at, buckets, table) = if last {
                (fh, fhs / 4, ft)
            } else {
                (dh, dhs / 4, dt)
            };
            let mut e = u32_at(
                img,
                hash_at + 4 * (path_hash(parent, part.as_bytes()) as usize % buckets),
            );
            loop {
                if e == EMPTY {
                    return None;
                }
                let at = table + e as usize;
                let (name_at, hash_next, name_len) = if last {
                    (at + 32, at + 24, at + 28)
                } else {
                    (at + 24, at + 16, at + 20)
                };
                let len = u32_at(img, name_len) as usize;
                if u32_at(img, at) == parent && &img[name_at..name_at + len] == part.as_bytes() {
                    if last {
                        let (off, size) =
                            (u64_at(img, at + 8) as usize, u64_at(img, at + 16) as usize);
                        return Some(img[data + off..data + off + size].to_vec());
                    }
                    parent = e;
                    break;
                }
                e = u32_at(img, hash_next);
            }
        }
        None
    }

    #[test]
    fn builds_a_readable_image() {
        let dir = std::env::temp_dir().join(format!("emu-romfs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("data/sub")).unwrap();
        std::fs::write(dir.join("a.txt"), b"alpha").unwrap();
        std::fs::write(dir.join("data/b.bin"), vec![7u8; 100]).unwrap();
        std::fs::write(dir.join("data/sub/c"), b"see").unwrap();
        let r = DirRomFs::build(&dir).unwrap();
        let mut img = vec![0u8; r.len() as usize];
        assert_eq!(r.read_at(0, &mut img).unwrap(), img.len());
        assert_eq!(lookup(&img, "/a.txt").as_deref(), Some(&b"alpha"[..]));
        assert_eq!(lookup(&img, "/data/b.bin"), Some(vec![7u8; 100]));
        assert_eq!(lookup(&img, "/data/sub/c").as_deref(), Some(&b"see"[..]));
        assert_eq!(lookup(&img, "/data/missing"), None);
        // Partial reads across a file boundary match the full image.
        let mut part = vec![0u8; 40];
        r.read_at(0x200, &mut part).unwrap();
        assert_eq!(part, img[0x200..0x228]);
        drop(r);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
