//! `fsp-srv`: the filesystem proxy. Only session setup so far; opening
//! the SD card returns a filesystem with no commands implemented yet.

use super::am::Empty;
use super::{obj, CallError, Ctx, Service};

pub struct FileSystemProxy;

impl Service for FileSystemProxy {
    fn name(&self) -> &str {
        "fsp-srv"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            1 => {} // SetCurrentProcess
            18 => ctx.push_object(obj(Empty("fs::IFileSystem(sdmc)"))),
            1003 => {} // DisableAutoSaveDataCreation
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
