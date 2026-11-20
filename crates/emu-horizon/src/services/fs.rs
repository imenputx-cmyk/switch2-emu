//! `fsp-srv`: the filesystem proxy. The SD card is a host directory
//! (`Config::sdmc_root`). Command ids and argument layouts as used by
//! libnx's `fs.c`; paths arrive in an X buffer, absolute, NUL terminated.
//!
//! Guest paths are confined to the root: `..` components are rejected.

use super::{obj, CallError, Ctx, Service};
use crate::ipc::read_buffer;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// FS results (module 2).
const fn fs_result(description: u32) -> u32 {
    2 | (description << 9)
}
pub const PATH_NOT_FOUND: u32 = fs_result(1);
pub const PATH_ALREADY_EXISTS: u32 = fs_result(2);
pub const SD_CARD_NOT_PRESENT: u32 = fs_result(2001);
pub const INVALID_PATH: u32 = fs_result(6001);
pub const INVALID_ARGUMENT: u32 = fs_result(6060);

const MAX_PATH: usize = 0x301;
const DIR_ENTRY_SIZE: usize = 0x310;

pub struct FileSystemProxy {
    pub sdmc_root: Option<PathBuf>,
}

impl Service for FileSystemProxy {
    fn name(&self) -> &str {
        "fsp-srv"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            1 => {} // SetCurrentProcess
            18 => {
                // OpenSdCardFileSystem
                let root = self
                    .sdmc_root
                    .clone()
                    .ok_or(CallError::Result(SD_CARD_NOT_PRESENT))?;
                ctx.push_object(obj(HostFileSystem { root }));
            }
            // OpenDataStorageByCurrentProcess: the application's RomFS.
            203 => match &ctx.config.romfs {
                Some(r) => ctx.push_object(obj(RomFsStorage(r.clone()))),
                None => return Err(CallError::Result(RESULT_TARGET_NOT_FOUND)),
            },
            1003 => {} // DisableAutoSaveDataCreation
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// An IFileSystem backed by a host directory.
pub struct HostFileSystem {
    root: PathBuf,
}

fn host_error(e: std::io::Error) -> CallError {
    CallError::Result(match e.kind() {
        std::io::ErrorKind::NotFound => PATH_NOT_FOUND,
        std::io::ErrorKind::AlreadyExists => PATH_ALREADY_EXISTS,
        _ => INVALID_ARGUMENT,
    })
}

impl HostFileSystem {
    /// The host path of the guest path in the `index`-th X buffer.
    fn path(&self, ctx: &Ctx, index: usize) -> Result<PathBuf, CallError> {
        let buf = *ctx
            .req
            .x
            .get(index)
            .ok_or(CallError::Result(INVALID_PATH))?;
        let mut b = buf;
        b.size = b.size.min(MAX_PATH as u64);
        let bytes = read_buffer(ctx.mem, b).map_err(|_| CallError::Result(INVALID_PATH))?;
        let end = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
        let guest =
            std::str::from_utf8(&bytes[..end]).map_err(|_| CallError::Result(INVALID_PATH))?;
        resolve(&self.root, guest).ok_or(CallError::Result(INVALID_PATH))
    }
}

/// Joins an absolute guest path to the root, refusing anything that
/// could leave it.
fn resolve(root: &Path, guest: &str) -> Option<PathBuf> {
    let rel = guest.strip_prefix('/')?;
    let mut out = root.to_path_buf();
    for part in rel.split('/').filter(|p| !p.is_empty() && *p != ".") {
        if part == ".." || part.contains('\\') || part.contains(':') {
            return None;
        }
        out.push(part);
    }
    Some(out)
}

impl Service for HostFileSystem {
    fn name(&self) -> &str {
        "fs::IFileSystem"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => {
                // CreateFile(option u32, size u64)
                let path = self.path(ctx, 0)?;
                let size = ctx.in_u64(8);
                let f = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(host_error)?;
                f.set_len(size).map_err(host_error)?;
            }
            1 => {
                let path = self.path(ctx, 0)?;
                if !path.is_file() {
                    return Err(CallError::Result(PATH_NOT_FOUND));
                }
                fs::remove_file(path).map_err(host_error)?;
            }
            2 => fs::create_dir(self.path(ctx, 0)?).map_err(host_error)?,
            3 => fs::remove_dir(self.path(ctx, 0)?).map_err(host_error)?,
            4 => fs::remove_dir_all(self.path(ctx, 0)?).map_err(host_error)?,
            5 | 6 => {
                let (from, to) = (self.path(ctx, 0)?, self.path(ctx, 1)?);
                if to.exists() {
                    return Err(CallError::Result(PATH_ALREADY_EXISTS));
                }
                fs::rename(from, to).map_err(host_error)?;
            }
            7 => {
                // GetEntryType: 0 directory, 1 file.
                let meta = fs::metadata(self.path(ctx, 0)?).map_err(host_error)?;
                ctx.push_u32(if meta.is_dir() { 0 } else { 1 });
            }
            8 => {
                // OpenFile(mode u32)
                let path = self.path(ctx, 0)?;
                let mode = ctx.in_u32(0);
                if !path.is_file() {
                    return Err(CallError::Result(PATH_NOT_FOUND));
                }
                let file = OpenOptions::new()
                    .read(true)
                    .write(mode & 6 != 0)
                    .open(path)
                    .map_err(host_error)?;
                ctx.push_object(obj(HostFile { file, mode }));
            }
            9 => {
                // OpenDirectory(mode u32)
                let path = self.path(ctx, 0)?;
                let mode = ctx.in_u32(0);
                let entries = list_dir(&path, mode)?;
                ctx.push_object(obj(HostDirectory { entries, next: 0 }));
            }
            10 => {}                           // Commit
            11 | 12 => ctx.push_u64(32 << 30), // Free / total space: 32 GiB.
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// Directory entries in the guest format, sorted by name.
fn list_dir(path: &Path, mode: u32) -> Result<Vec<[u8; DIR_ENTRY_SIZE]>, CallError> {
    let mut entries = Vec::new();
    for e in fs::read_dir(path).map_err(host_error)? {
        let e = e.map_err(host_error)?;
        let meta = e.metadata().map_err(host_error)?;
        let is_dir = meta.is_dir();
        if (is_dir && mode & 1 == 0) || (!is_dir && mode & 2 == 0) {
            continue;
        }
        let name = e.file_name();
        let name = name.to_string_lossy();
        let mut entry = [0u8; DIR_ENTRY_SIZE];
        let n = name.len().min(MAX_PATH - 1);
        entry[..n].copy_from_slice(&name.as_bytes()[..n]);
        entry[0x304] = if is_dir { 0 } else { 1 };
        if !is_dir && mode & (1 << 31) == 0 {
            entry[0x308..0x310].copy_from_slice(&meta.len().to_le_bytes());
        }
        entries.push(entry);
    }
    entries.sort();
    Ok(entries)
}

pub struct HostFile {
    file: File,
    mode: u32,
}

impl Service for HostFile {
    fn name(&self) -> &str {
        "fs::IFile"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        let (offset, size) = (ctx.in_u64(8), ctx.in_u64(16));
        match command {
            0 => {
                // Read(option, offset, size) -> bytes read, into a B buffer.
                if self.mode & 1 == 0 {
                    return Err(CallError::Result(INVALID_ARGUMENT));
                }
                let buf = *ctx
                    .req
                    .b
                    .first()
                    .ok_or(CallError::Result(INVALID_ARGUMENT))?;
                let want = size.min(buf.size) as usize;
                self.file
                    .seek(SeekFrom::Start(offset))
                    .map_err(host_error)?;
                let mut data = Vec::with_capacity(want);
                (&mut self.file)
                    .take(want as u64)
                    .read_to_end(&mut data)
                    .map_err(host_error)?;
                ctx.mem
                    .write(buf.addr, &data)
                    .map_err(|_| CallError::Result(crate::result::INVALID_ADDRESS))?;
                ctx.push_u64(data.len() as u64);
            }
            1 => {
                // Write(option, offset, size) from an A buffer.
                if self.mode & 6 == 0 {
                    return Err(CallError::Result(INVALID_ARGUMENT));
                }
                let mut buf = *ctx
                    .req
                    .a
                    .first()
                    .ok_or(CallError::Result(INVALID_ARGUMENT))?;
                buf.size = buf.size.min(size);
                let data = read_buffer(ctx.mem, buf)
                    .map_err(|_| CallError::Result(crate::result::INVALID_ADDRESS))?;
                let end = self.file.metadata().map_err(host_error)?.len();
                if offset + data.len() as u64 > end && self.mode & 4 == 0 {
                    // Growing a file needs append mode.
                    return Err(CallError::Result(INVALID_ARGUMENT));
                }
                self.file
                    .seek(SeekFrom::Start(offset))
                    .map_err(host_error)?;
                self.file.write_all(&data).map_err(host_error)?;
            }
            2 => self.file.flush().map_err(host_error)?,
            3 => self.file.set_len(ctx.in_u64(0)).map_err(host_error)?,
            4 => ctx.push_u64(self.file.metadata().map_err(host_error)?.len()),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct HostDirectory {
    entries: Vec<[u8; DIR_ENTRY_SIZE]>,
    next: usize,
}

impl Service for HostDirectory {
    fn name(&self) -> &str {
        "fs::IDirectory"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => {
                // Read -> entry count, entries into a B buffer.
                let buf = ctx.req.b.first().copied();
                let room = buf.map_or(0, |b| b.size as usize / DIR_ENTRY_SIZE);
                let n = room.min(self.entries.len() - self.next);
                let mut bytes = Vec::with_capacity(n * DIR_ENTRY_SIZE);
                for e in &self.entries[self.next..self.next + n] {
                    bytes.extend_from_slice(e);
                }
                if n > 0 {
                    ctx.mem
                        .write(buf.unwrap().addr, &bytes)
                        .map_err(|_| CallError::Result(crate::result::INVALID_ADDRESS))?;
                }
                self.next += n;
                ctx.push_u64(n as u64);
            }
            1 => ctx.push_u64((self.entries.len() - self.next) as u64),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// fs "target not found" (module 2, description 1001).
const RESULT_TARGET_NOT_FOUND: u32 = 2 | (1001 << 9);

/// `IStorage` over the application's RomFS.
pub struct RomFsStorage(emu_loader::nca::RomFs);

impl Service for RomFsStorage {
    fn name(&self) -> &str {
        "fsp-srv::IStorage (RomFS)"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // Read(offset, size) into the output buffer.
            0 => {
                let (offset, size) = (ctx.in_u64(0), ctx.in_u64(8));
                let mut data = vec![0u8; size.min(64 << 20) as usize];
                let n = self
                    .0
                    .read_at(offset, &mut data)
                    .map_err(|_| CallError::Result(INVALID_ARGUMENT))?;
                ctx.write_out_buffer(0, &data[..n])?;
            }
            4 => ctx.push_u64(self.0.len()), // GetSize
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::resolve;
    use std::path::Path;

    #[test]
    fn guest_paths_stay_inside_the_root() {
        let root = Path::new("/sd");
        assert_eq!(resolve(root, "/a/b.txt").unwrap(), Path::new("/sd/a/b.txt"));
        assert_eq!(resolve(root, "/").unwrap(), Path::new("/sd"));
        assert_eq!(resolve(root, "//a/./b").unwrap(), Path::new("/sd/a/b"));
        assert!(resolve(root, "/../etc/passwd").is_none());
        assert!(resolve(root, "/a/../../x").is_none());
        assert!(resolve(root, "relative").is_none());
    }
}
