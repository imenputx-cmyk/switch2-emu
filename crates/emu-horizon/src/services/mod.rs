//! HLE services, reached over IPC.
//!
//! A service handles CMIF commands by id. Unknown commands are reported
//! (the run stops) instead of being answered with a made-up success, so
//! missing functionality is found early.

pub mod account;
pub mod am;
pub mod apm;
pub mod audfx;
pub mod audio;
pub mod audren;
pub mod fs;
pub mod generic;
pub mod hid;
pub mod nv;
pub mod nvgpu;
pub mod nx2;
pub mod pl;
pub mod set;
pub mod sm;
pub mod stubs;
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
    /// Interfaces returned as new raw sessions (copy handles), even on a
    /// domain session.
    pub out_sessions: Vec<ServiceRef>,
    /// The client's handle table, to look up handles it sent.
    pub handles: &'a crate::handles::HandleTable,
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
pub type Presenter = Box<dyn FnMut(&Frame, &mut hid::InputState) -> bool>;

/// Audio waiting to be played on the host: interleaved stereo f32 at
/// 48 kHz. The front end drains it; a queue longer than a quarter
/// second drops its oldest samples (the guest runs ahead).
#[derive(Clone, Default)]
pub struct AudioOutput(pub std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<f32>>>);

impl AudioOutput {
    /// Host sample rate.
    pub const RATE: u32 = 48_000;
    const MAX: usize = 2 * 12_000;

    /// Queues interleaved stereo `samples` at `rate` Hz (linearly
    /// resampled to 48 kHz).
    pub fn push(&self, samples: &[f32], rate: u32) {
        let Ok(mut q) = self.0.lock() else {
            return;
        };
        if rate == Self::RATE || rate == 0 {
            q.extend(samples);
        } else {
            let frames = samples.len() / 2;
            let out = frames * Self::RATE as usize / rate as usize;
            for i in 0..out {
                let pos = i as f64 * f64::from(rate) / f64::from(Self::RATE);
                let k = pos.floor() as usize;
                let f = (pos - pos.floor()) as f32;
                for c in 0..2 {
                    let a = samples[(2 * k + c).min(samples.len() - 1)];
                    let b = samples[(2 * (k + 1) + c).min(samples.len() - 1)];
                    q.push_back(a + (b - a) * f);
                }
            }
        }
        let excess = q.len().saturating_sub(Self::MAX);
        q.drain(..excess);
    }

    /// Takes up to `n` samples (interleaved).
    pub fn pop(&self, out: &mut [f32]) -> usize {
        let Ok(mut q) = self.0.lock() else {
            return 0;
        };
        let n = out.len().min(q.len());
        for (o, s) in out.iter_mut().zip(q.drain(..n)) {
            *o = s;
        }
        n
    }
}

/// State shared between services.
#[derive(Default)]
pub struct System {
    /// nvmap objects by handle (== id).
    pub nvmap: std::collections::HashMap<u32, nv::NvMapObject>,
    /// The GPU.
    pub gpu: emu_gpu::Gpu,
    /// Buffer queues by binder id.
    pub buffer_queues: std::collections::HashMap<u32, vi::BufferQueue>,
    /// Called for each presented frame; without it frames are kept in
    /// `frames`. Returning false asks the emulator to stop (e.g. the
    /// window was closed).
    pub on_present: Option<Presenter>,
    /// The host asked to stop.
    pub quit: bool,
    /// Controller state from the host, sampled into hid shared memory.
    pub input: hid::InputState,
    /// The hid shared memory object, once created.
    pub hid_shared_memory: Option<usize>,
    pub hid_sampling: u64,
    pub frames: Vec<Frame>,
    pub frames_presented: u64,
    /// GPU errors seen (method faults, untranslatable shaders...), and the
    /// latest one.
    pub gpu_faults: u64,
    pub last_gpu_fault: Option<String>,
    /// Kinds of GPU errors already printed (their text without numbers).
    pub gpu_fault_kinds: std::collections::HashMap<String, u64>,
    /// Events signalled every emulated vblank (60 Hz of guest time).
    pub vblank_events: Vec<usize>,
    /// Events signalled every audio frame (5 ms of guest time): the audio
    /// renderers' system events.
    pub audio_events: Vec<usize>,
    /// Rendered audio for the host: 48 kHz stereo, shared with the
    /// front end's output stream.
    pub audio_out: AudioOutput,
    /// Switch 2 GPU memory (nx2:gpu and its sub-sessions share it).
    pub nx2_gpu: nx2::GpuMemory,
    /// nx2:syncpt slot requests: (syncpoint, value, event) signalled
    /// once the GPU syncpoint reaches the value.
    pub syncpt_waits: Vec<(u32, u32, usize)>,
    /// Who made each event (for diagnostics): event id -> "service cmd".
    pub event_names: std::collections::HashMap<usize, String>,
    /// Switch 2 native memory and images (gsv), shown by vi2 presents.
    pub nx2_native: nx2::NativeObjects,
}

impl System {
    /// Counts (and with `print`, prints) the GPU's new faults.
    pub fn note_gpu_faults(&mut self, print: bool) {
        for f in self.gpu.faults.drain(..) {
            // Each kind of error is printed the first time (and at powers
            // of ten after), so one noisy error does not hide the others.
            let first_line = f.lines().next().unwrap_or("");
            let kind: String = first_line
                .chars()
                .filter(|c| !c.is_ascii_hexdigit() || c.is_alphabetic() && !c.is_ascii_digit())
                .collect();
            let n = self.gpu_fault_kinds.entry(kind).or_insert(0);
            *n += 1;
            if print || *n == 1 {
                eprintln!("[gpu] {f}");
            } else if n.is_power_of_two() && *n >= 16 {
                eprintln!("[gpu] ({}x) {first_line}", *n);
            }
            self.gpu_faults += 1;
            self.last_gpu_fault = Some(f);
        }
    }

    pub fn present(&mut self, frame: Frame) {
        self.frames_presented += 1;
        match self.on_present.as_mut() {
            Some(f) => {
                if !f(&frame, &mut self.input) {
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
    /// How GPU work is carried out.
    pub gpu_backend: GpuBackend,
    /// Answer unknown commands (and services) with an empty success
    /// instead of stopping, logging each one; Lichii's
    /// `EDEN_NX2_AUTO_STUB`, how its notes push Switch 2 titles through
    /// boot.
    pub auto_stub: bool,
    /// The application's RomFS (from its program NCA).
    pub romfs: Option<emu_loader::nca::RomFs>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GpuBackend {
    /// On the CPU, directly in guest memory (the reference).
    #[default]
    Software,
    /// On the host GPU through Vulkan.
    Vulkan,
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

    /// `len` bytes of input data at `offset` (zero past the end).
    pub fn in_bytes(&self, offset: usize, len: usize) -> Vec<u8> {
        (offset..offset + len)
            .map(|i| self.req.data.get(i).copied().unwrap_or(0))
            .collect()
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

    /// Returns `obj` as a new session's handle (a copy handle).
    pub fn push_session(&mut self, obj: ServiceRef) {
        self.out_sessions.push(obj);
    }

    /// The kernel object of the request's `index`-th copy handle.
    pub fn copy_handle(&self, index: usize) -> Option<KObject> {
        let h = *self.req.copy_handles.get(index)?;
        self.handles.get(h).cloned()
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

/// `pctl*`: parental controls, never restricting.
pub struct ParentalControl(String);

impl Service for ParentalControl {
    fn name(&self) -> &str {
        &self.0
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // CreateService, CreateServiceWithoutInitialize.
            0 | 1 => ctx.push_object(obj(ParentalControlService::default())),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `pctl::IParentalControlService`: no restrictions; everything is
/// permitted, queries answer "not restricted".
#[derive(Default)]
pub struct ParentalControlService {
    event: Option<usize>,
}

impl Service for ParentalControlService {
    fn name(&self) -> &str {
        "pctl::IParentalControlService"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // Initialize, Check*/Confirm* permission checks: allowed.
            1 | 1001..=1017 | 1035 | 1046 | 1047 => {}
            // IsRestrictionTemporaryUnlocked, IsRestrictionEnabled, ...
            1031 | 1033 | 1034 | 1201..=1207 | 1403 | 1406 => ctx.push_u8(0),
            // GetSafetyLevel, GetPlayTimerRemainingTime, ...
            1032 | 1039 | 1453 | 1454 | 1455 => ctx.push_u64(0),
            // Get*Event (unlock, play timer, synchronization).
            1404 | 1432 | 1456 | 1457 | 1475 => ctx.push_event(&mut self.event, false),
            _ => {
                if !ctx.config.auto_stub {
                    return Err(CallError::Unimplemented);
                }
                ctx.push_bytes(&[0; 0x10]);
            }
        }
        Ok(())
    }
}

/// `aoc:u`: add-on content (DLC). None installed.
#[derive(Default)]
pub struct AddOnContent {
    event: Option<usize>,
}

impl Service for AddOnContent {
    fn name(&self) -> &str {
        "aoc:u"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // CountAddOnContent(ByApplicationId), ListAddOnContent(...).
            0..=3 => ctx.push_u32(0),
            // GetAddOnContentBaseId(ByApplicationId), PrepareAddOnContent.
            4..=8 => ctx.push_u64(0),
            // GetAddOnContentListChangedEvent(WithProcessId).
            10 | 11 => ctx.push_event(&mut self.event, false),
            // NotifyMountAddOnContent / Unmount, CheckAddOnContentMountStatus.
            12..=13 | 50 => {}
            _ => {
                if !ctx.config.auto_stub {
                    return Err(CallError::Unimplemented);
                }
                ctx.push_bytes(&[0; 0x10]);
            }
        }
        Ok(())
    }
}

/// `ectx:aw` / `ectx:w`: error contexts (nn::err). A registrar is handed
/// out and completes with success; nothing is recorded (as in Lichii's
/// glue/ectx.cpp).
pub struct ErrorContext(pub String);

impl Service for ErrorContext {
    fn name(&self) -> &str {
        &self.0
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_object(Rc::new(RefCell::new(ContextRegistrar))), // CreateContextRegistrar
            1 | 2 => {} // CommitContext, RemoveContext
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `nn::err::context::IContextRegistrar`.
pub struct ContextRegistrar;

impl Service for ContextRegistrar {
    fn name(&self) -> &str {
        "ectx::IContextRegistrar"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_u32(0), // Complete
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `lm`: the log manager. Log messages are dropped.
pub struct LogManager;

impl Service for LogManager {
    fn name(&self) -> &str {
        "lm"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_object(obj(Logger)), // OpenLogger
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// `lm::ILogger`.
pub struct Logger;

impl Service for Logger {
    fn name(&self) -> &str {
        "lm::ILogger"
    }

    fn call(&mut self, command: u32, _ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 | 1 => Ok(()), // Log, SetDestination
            _ => Err(CallError::Unimplemented),
        }
    }
}

/// A service or interface that answers every command with an empty
/// success (`Config::auto_stub`).
pub struct AutoStub(pub String);

impl Service for AutoStub {
    fn name(&self) -> &str {
        &self.0
    }

    fn call(&mut self, _command: u32, _ctx: &mut Ctx) -> Result<(), CallError> {
        Err(CallError::Unimplemented)
    }
}

/// Creates the service registered under `name`: implemented services,
/// and with `auto_stub` any other name.
pub fn create(name: &str, config: &Config) -> Option<ServiceRef> {
    create_known(name, config).or_else(|| {
        config.auto_stub.then(|| {
            eprintln!("[stub] service {name} is not implemented; stubbing it");
            Rc::new(RefCell::new(AutoStub(name.to_string()))) as ServiceRef
        })
    })
}

fn create_known(name: &str, config: &Config) -> Option<ServiceRef> {
    Some(match name {
        "sm:" => Rc::new(RefCell::new(sm::ServiceManager)),
        "set:sys" => Rc::new(RefCell::new(set::SystemSettings)),
        "set" => Rc::new(RefCell::new(set::Settings)),
        "aud:u" => Rc::new(RefCell::new(audio::AudioUserClient)),
        "pl:u" | "pl:s" => Rc::new(RefCell::new(pl::PlatformServiceManager::default())),
        "audren:u" => Rc::new(RefCell::new(audio::AudioRendererManager)),
        "audout:u" => Rc::new(RefCell::new(audio::AudioOutManager)),
        "acc:u0" | "acc:u1" | "acc:su" => {
            Rc::new(RefCell::new(account::AccountService(name.to_string())))
        }
        "lm" => Rc::new(RefCell::new(LogManager)),
        "pctl" | "pctl:s" | "pctl:a" | "pctl:r" => {
            Rc::new(RefCell::new(ParentalControl(name.to_string())))
        }
        "aoc:u" => Rc::new(RefCell::new(AddOnContent::default())),
        "ectx:aw" | "ectx:w" => Rc::new(RefCell::new(ErrorContext(name.to_string()))),
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
        // Services answered from tables (generic.rs).
        other => Rc::new(RefCell::new(generic::TableService::new(
            generic::table_name(other)?,
        )?)),
    })
}
