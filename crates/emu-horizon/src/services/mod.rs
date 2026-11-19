//! HLE services, reached over IPC.
//!
//! A service handles CMIF commands by id. Unknown commands are reported
//! (the run stops) instead of being answered with a made-up success, so
//! missing functionality is found early.

pub mod set;
pub mod sm;

use crate::ipc::{Buffer, Request};
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

    pub fn push_object(&mut self, obj: ServiceRef) {
        self.out_objects.push(obj);
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

pub trait Service {
    fn name(&self) -> &str;
    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError>;
}

/// Creates the service registered under `name`, if it is implemented.
pub fn create(name: &str) -> Option<ServiceRef> {
    Some(match name {
        "sm:" => Rc::new(RefCell::new(sm::ServiceManager)),
        "set:sys" => Rc::new(RefCell::new(set::SystemSettings)),
        _ => return None,
    })
}
