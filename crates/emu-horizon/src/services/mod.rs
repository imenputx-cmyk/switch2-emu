//! HLE services, reached over IPC.
//!
//! A service handles CMIF commands by id. Unknown commands are reported
//! (the run stops) instead of being answered with a made-up success, so
//! missing functionality is found early.

pub mod am;
pub mod apm;
pub mod fs;
pub mod hid;
pub mod nv;
pub mod set;
pub mod sm;
pub mod time;
pub mod vi;

use crate::handles::KObject;
use crate::ipc::{Buffer, Request};
use crate::kobjects::KernelObjects;
use crate::result::ResultCode;
use emu_memory::Memory;
use std::cell::RefCell;
use std::rc::Rc;

pub type ServiceRef = Rc<RefCell<dyn Service>>;

/// What a command handler can return besides success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallError {
    /// Answer the client with this result code.
    Result(ResultCode),
    /// The command is not implemented: stop and report it.
    Unimplemented,
}

impl From<ResultCode> for CallError {
    fn from(code: ResultCode) -> Self {
        CallError::Result(code)
    }
}

/// Context of one command: the request, guest memory, and the reply
/// being built.
pub struct Ctx<'a> {
    pub req: &'a Request,
    pub mem: &'a mut Memory,
    /// Output data, after the CMIF header.
    pub out: Vec<u8>,
    /// Interfaces returned to the client (as handles or domain objects).
    pub out_objects: Vec<ServiceRef>,
    /// Kernel objects returned as copy handles.
    pub out_copy: Vec<KObject>,
    pub kernel: &'a mut KernelObjects,
    /// The current system tick (19.2 MHz), for clocks.
    pub tick: u64,
    pub config: &'a Config,
    /// State shared between services (GPU memory, display).
    pub system: &'a mut System,
}

/// A presented frame, RGBA8, `width * height * 4` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Receives presented frames; returns false to stop the emulation.
pub type Presenter = Box<dyn FnMut(&Frame) -> bool>;

/// State shared between services.
#[derive(Default)]
pub struct System {
    /// nvmap objects by handle (== id).
    pub nvmap: std::collections::HashMap<u32, nv::NvMapObject>,
    /// Buffer queues by binder id.
    pub buffer_queues: std::collections::HashMap<u32, vi::BufferQueue>,
    /// Called for each presented frame; without it frames are kept in
    /// `frames`. Returning false asks the emulator to stop (e.g. the
    /// window was closed).
    pub on_present: Option<Presenter>,
    /// The host asked to stop.
    pub quit: bool,
    pub frames: Vec<Frame>,
    pub frames_presented: u64,
}

impl System {
    pub fn present(&mut self, frame: Frame) {
        self.frames_presented += 1;
        match self.on_present.as_mut() {
            Some(f) => {
                if !f(&frame) {
                    self.quit = true;
                }
            }
            None => self.frames.push(frame),
        }
    }
}

/// Host-side settings that services need.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Host directory that backs the SD card. Without one, opening the
    /// SD card fails like a missing card.
    pub sdmc_root: Option<std::path::PathBuf>,
    /// POSIX time at boot for the clocks; `None` uses the host clock.
    pub boot_time: Option<i64>,
    /// Log commands and ioctls services do not implement.
    pub trace_unimplemented: bool,
}

impl Ctx<'_> {
    /// Input data as a little-endian value at `offset`.
    pub fn in_u64(&self, offset: usize) -> u64 {
        let mut b = [0u8; 8];
        let end = (offset + 8).min(self.req.data.len());
        if offset < end {
            b[..end - offset].copy_from_slice(&self.req.data[offset..end]);
        }
        u64::from_le_bytes(b)
    }

    pub fn in_u32(&self, offset: usize) -> u32 {
        self.in_u64(offset) as u32
    }

    pub fn push_u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn push_u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn push_u8(&mut self, v: u8) {
        self.out.push(v);
    }

    pub fn push_bytes(&mut self, v: &[u8]) {
        self.out.extend_from_slice(v);
    }

    pub fn push_object(&mut self, obj: ServiceRef) {
        self.out_objects.push(obj);
    }

    pub fn push_copy(&mut self, obj: KObject) {
        self.out_copy.push(obj);
    }

    /// Returns a lazily created event stored in `slot`, as a copy handle.
    pub fn push_event(&mut self, slot: &mut Option<usize>, signaled: bool) {
        let id = *slot.get_or_insert_with(|| self.kernel.new_event(signaled));
        self.push_copy(KObject::Event { id });
    }

    /// The `index`-th output buffer: a C pointer if the client sent one
    /// with a size, otherwise a B buffer ("auto select" like libnx).
    pub fn out_buffer(&self, index: usize) -> Option<Buffer> {
        self.req
            .c
            .get(index)
            .copied()
            .filter(|b| b.size > 0)
            .or_else(|| self.req.b.get(index).copied().filter(|b| b.size > 0))
    }

    /// The `index`-th input buffer: an X pointer with a size, otherwise
    /// an A buffer ("auto select").
    pub fn in_buffer(&self, index: usize) -> Option<Buffer> {
        self.req
            .x
            .get(index)
            .copied()
            .filter(|b| b.size > 0)
            .or_else(|| self.req.a.get(index).copied().filter(|b| b.size > 0))
    }

    /// Reads the `index`-th input buffer (empty if absent).
    pub fn read_in_buffer(&self, index: usize) -> Result<Vec<u8>, CallError> {
        match self.in_buffer(index) {
            Some(b) => crate::ipc::read_buffer(self.mem, b)
                .map_err(|_| CallError::Result(crate::result::INVALID_ADDRESS)),
            None => Ok(Vec::new()),
        }
    }

    /// Writes `bytes` (truncated to the buffer size) to an output buffer.
    pub fn write_out_buffer(&mut self, index: usize, bytes: &[u8]) -> Result<(), CallError> {
        let b = self
            .out_buffer(index)
            .ok_or(CallError::Result(crate::result::INVALID_ADDRESS))?;
        let n = bytes.len().min(b.size as usize);
        self.mem
            .write(b.addr, &bytes[..n])
            .map_err(|_| CallError::Result(crate::result::INVALID_ADDRESS))
    }
}

/// Wraps a service object for returning it from a command.
pub fn obj<S: Service + 'static>(s: S) -> ServiceRef {
    Rc::new(RefCell::new(s))
}

pub trait Service {
    fn name(&self) -> &str;
    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError>;
}

/// Creates the service registered under `name`, if it is implemented.
pub fn create(name: &str, config: &Config) -> Option<ServiceRef> {
    Some(match name {
        "sm:" => Rc::new(RefCell::new(sm::ServiceManager)),
        "set:sys" => Rc::new(RefCell::new(set::SystemSettings)),
        "apm" | "apm:p" => Rc::new(RefCell::new(apm::Manager)),
        "appletOE" => Rc::new(RefCell::new(am::AppletOE)),
        "vi:u" => Rc::new(RefCell::new(vi::RootService)),
        "nvdrv" | "nvdrv:a" => Rc::new(RefCell::new(nv::NvDrv::default())),
        "hid" => Rc::new(RefCell::new(hid::Hid::default())),
        "time:u" | "time:a" | "time:s" => Rc::new(RefCell::new(match config.boot_time {
            Some(t) => time::StaticService::with_boot_time(t),
            None => time::StaticService::default(),
        })),
        "fsp-srv" => Rc::new(RefCell::new(fs::FileSystemProxy {
            sdmc_root: config.sdmc_root.clone(),
        })),
        _ => return None,
    })
}
