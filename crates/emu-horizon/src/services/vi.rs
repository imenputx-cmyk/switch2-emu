//! `vi:u`: displays and layers, and the binder buffer queue behind each
//! layer (Android's IGraphicBufferProducer, spoken over
//! IHOSBinderDriver as parcels). Formats as produced/consumed by libnx's
//! `vi.c`, `binder.c`, `parcel.c` and `buffer_producer.c` (ISC).
//!
//! The emulator is the consumer: a queued buffer is converted to RGBA and
//! presented at once, and its slot is free again immediately.

use super::{obj, CallError, Ctx, Frame, Service};
use crate::display::{self, PixelFormat};

/// Layer id handed out by am's CreateManagedDisplayLayer.
pub const MANAGED_LAYER_ID: u64 = 1;
const DISPLAY_ID: u64 = 0;
const DISPLAY_WIDTH: u64 = 1920;
const DISPLAY_HEIGHT: u64 = 1080;
/// Size reported for a newly connected window.
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;

const VI_RESULT_NOT_FOUND: u32 = 114 | (7 << 9);
const VI_RESULT_INVALID_ARGUMENT: u32 = 114 | (1 << 9);

// IGraphicBufferProducer transactions.
const REQUEST_BUFFER: u32 = 1;
const SET_BUFFER_COUNT: u32 = 2;
const DEQUEUE_BUFFER: u32 = 3;
const DETACH_BUFFER: u32 = 4;
const QUEUE_BUFFER: u32 = 7;
const CANCEL_BUFFER: u32 = 8;
const QUERY: u32 = 9;
const CONNECT: u32 = 10;
const DISCONNECT: u32 = 11;
const SET_PREALLOCATED_BUFFER: u32 = 14;
/// Android status codes.
const BAD_VALUE: i32 = -22;
const NO_INIT: i32 = -19;
const WOULD_BLOCK: i32 = -11;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Dequeued,
}

/// The buffer queue of one layer.
#[derive(Debug, Clone)]
pub struct BufferQueue {
    /// Per slot: the flattened GraphicBuffer from SetPreallocatedBuffer.
    buffers: Vec<Option<Vec<u8>>>,
    states: Vec<SlotState>,
    next: usize,
    width: u32,
    height: u32,
}

impl Default for BufferQueue {
    fn default() -> Self {
        BufferQueue {
            buffers: vec![None; 64],
            states: vec![SlotState::Free; 64],
            next: 0,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
        }
    }
}

/// Reads Android parcel payloads.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Parses the parcel header and returns a reader of the payload.
    fn parcel(raw: &'a [u8]) -> Option<Reader<'a>> {
        let w = |o: usize| {
            raw.get(o..o + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        };
        let (size, off) = (w(0)?, w(4)?);
        Some(Reader {
            data: raw.get(off..off.checked_add(size)?)?,
            pos: 0,
        })
    }

    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let b = self.data.get(self.pos..self.pos + n)?;
        self.pos += n.next_multiple_of(4);
        Some(b)
    }

    fn i32(&mut self) -> Option<i32> {
        self.bytes(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
    }

    /// The interface token: strict-mode policy and a UTF-16 name.
    fn interface_token(&mut self) -> Option<()> {
        self.i32()?;
        let len = self.i32()?;
        self.bytes((len as usize + 1) * 2)?;
        Some(())
    }

    fn flattened(&mut self) -> Option<&'a [u8]> {
        let len = self.i32()?;
        let fds = self.i32()?;
        if fds != 0 {
            return None;
        }
        self.bytes(len as usize)
    }
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn i32(&mut self, v: i32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn data(&mut self, b: &[u8]) -> &mut Self {
        self.0.extend_from_slice(b);
        self.0.resize(self.0.len().next_multiple_of(4), 0);
        self
    }

    fn flattened(&mut self, b: &[u8]) -> &mut Self {
        self.i32(b.len() as i32).i32(0).data(b)
    }

    /// Wraps the payload in a parcel header.
    fn finish(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.0.len());
        let size = self.0.len() as u32;
        for v in [size, 0x10, 0, 0x10 + size] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.0);
        out
    }
}

fn u32_at(b: &[u8], off: usize) -> Option<u32> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}

/// Converts the buffer in a flattened GraphicBuffer to a frame. Layout:
/// a 10-word header (magic "GBFR", width, height, stride, format, usage,
/// pid, refcount, numFds, numInts), then the NvGraphicBuffer ints
/// (nvmap id at int 1, plane 0 at int 13).
fn frame_from_buffer(gb: &[u8], ctx: &Ctx) -> Option<Frame> {
    let ints = gb.get(40..)?;
    let nvmap_id = u32_at(ints, 4)?;
    let format = PixelFormat::from_android(u32_at(gb, 16)?)?;
    let plane = ints.get(52..)?;
    let (width, height) = (u32_at(plane, 0)?, u32_at(plane, 4)?);
    let layout = u32_at(plane, 16)?;
    let pitch = u32_at(plane, 20)?;
    let offset = u32_at(plane, 28)?;
    let block_height_log2 = u32_at(plane, 36)?;
    if width == 0
        || height == 0
        || pitch < width * format.bytes_per_pixel()
        || block_height_log2 > 5
    {
        return None;
    }
    let base = ctx.system.nvmap.get(&nvmap_id)?.addr + u64::from(offset);
    let linear = if layout == 3 {
        if pitch % 64 != 0 {
            return None;
        }
        let mut raw = vec![0u8; display::block_linear_size(pitch, height, block_height_log2)];
        ctx.mem.read(base, &mut raw).ok()?;
        display::deswizzle(&raw, pitch, height, block_height_log2)
    } else {
        let mut raw = vec![0u8; (pitch * height) as usize];
        ctx.mem.read(base, &mut raw).ok()?;
        raw
    };
    Some(Frame {
        width,
        height,
        rgba: display::to_rgba(&linear, pitch, width, height, format),
    })
}

/// Runs one IGraphicBufferProducer transaction; returns the reply payload.
fn transact(q: &mut BufferQueue, code: u32, parcel: &[u8], ctx: &mut Ctx) -> Option<Writer> {
    let mut r = Reader::parcel(parcel)?;
    r.interface_token()?;
    let mut w = Writer::default();
    let output = |w: &mut Writer, q: &BufferQueue| {
        // BqBufferOutput: width, height, transform hint, pending buffers.
        w.i32(q.width as i32).i32(q.height as i32).i32(0).i32(0);
    };
    match code {
        CONNECT => {
            output(&mut w, q);
            w.i32(0);
        }
        DISCONNECT | SET_BUFFER_COUNT => {
            w.i32(0);
        }
        SET_PREALLOCATED_BUFFER => {
            let slot = r.i32()? as usize;
            let has = r.i32()?;
            let status = match q.buffers.get_mut(slot) {
                Some(b) => {
                    *b = if has != 0 {
                        Some(r.flattened()?.to_vec())
                    } else {
                        None
                    };
                    q.states[slot] = SlotState::Free;
                    0
                }
                None => BAD_VALUE,
            };
            w.i32(status);
        }
        DEQUEUE_BUFFER => {
            let n = q.buffers.len();
            let slot = (0..n)
                .map(|k| (q.next + k) % n)
                .find(|&s| q.buffers[s].is_some() && q.states[s] == SlotState::Free);
            match slot {
                Some(s) => {
                    q.states[s] = SlotState::Dequeued;
                    q.next = (s + 1) % n;
                    // Slot, then a non-null fence with no fences in it.
                    w.i32(s as i32).i32(1).flattened(&[0u8; 0x24]).i32(0);
                }
                None => {
                    let status = if q.buffers.iter().any(Option::is_some) {
                        WOULD_BLOCK
                    } else {
                        NO_INIT
                    };
                    w.i32(0).i32(0).i32(status);
                }
            }
        }
        REQUEST_BUFFER => {
            let slot = r.i32()? as usize;
            match q.buffers.get(slot).cloned().flatten() {
                Some(b) => {
                    w.i32(1).flattened(&b).i32(0);
                }
                None => {
                    w.i32(0).i32(BAD_VALUE);
                }
            }
        }
        QUEUE_BUFFER => {
            let slot = r.i32()? as usize;
            let status = match q.buffers.get(slot).cloned().flatten() {
                Some(b) if q.states[slot] == SlotState::Dequeued => {
                    q.states[slot] = SlotState::Free;
                    if let Some(frame) = frame_from_buffer(&b, ctx) {
                        ctx.system.present(frame);
                    }
                    0
                }
                _ => BAD_VALUE,
            };
            output(&mut w, q);
            w.i32(status);
        }
        CANCEL_BUFFER | DETACH_BUFFER => {
            let slot = r.i32()? as usize;
            if let Some(s) = q.states.get_mut(slot) {
                *s = SlotState::Free;
            }
            if code == DETACH_BUFFER {
                if let Some(b) = q.buffers.get_mut(slot) {
                    *b = None;
                }
            }
            w.i32(0);
        }
        QUERY => {
            let v = match r.i32()? {
                0 => q.width as i32,  // NATIVE_WINDOW_WIDTH
                1 => q.height as i32, // NATIVE_WINDOW_HEIGHT
                2 => 1,               // NATIVE_WINDOW_FORMAT: RGBA_8888
                3 => 1,               // MIN_UNDEQUEUED_BUFFERS
                _ => {
                    w.i32(0).i32(BAD_VALUE);
                    return Some(w);
                }
            };
            w.i32(v).i32(0);
        }
        _ => return None,
    }
    Some(w)
}

/// The native window parcel returned when opening a layer: a flat
/// binder object whose third word is the binder id, then "dispdrv".
fn native_window(binder_id: u32) -> Vec<u8> {
    let mut w = Writer::default();
    w.i32(2).i32(0).i32(binder_id as i32).i32(0).i32(0).i32(0);
    w.data(b"dispdrv\0").i32(0).i32(0);
    w.finish()
}

pub struct RootService;

impl Service for RootService {
    fn name(&self) -> &str {
        "vi:u"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_object(obj(ApplicationDisplayService::default())), // GetDisplayService
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct ApplicationDisplayService {
    vsync_event: Option<usize>,
}

impl ApplicationDisplayService {
    fn open_layer(&mut self, layer_id: u64, ctx: &mut Ctx) -> Result<(), CallError> {
        let binder_id = layer_id as u32;
        ctx.system.buffer_queues.entry(binder_id).or_default();
        let parcel = native_window(binder_id);
        ctx.write_out_buffer(0, &parcel)?;
        ctx.push_u64(parcel.len() as u64);
        Ok(())
    }
}

impl Service for ApplicationDisplayService {
    fn name(&self) -> &str {
        "vi::IApplicationDisplayService"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            100 | 103 => ctx.push_object(obj(HosBinderDriver::default())),
            101 => ctx.push_object(obj(super::am::Empty("vi::ISystemDisplayService"))),
            102 => ctx.push_object(obj(super::am::Empty("vi::IManagerDisplayService"))),
            1010 => {
                // OpenDisplay(name)
                let name = ctx.req.data.get(..0x40).unwrap_or(&[]);
                let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
                if &name[..end] != b"Default" {
                    return Err(CallError::Result(VI_RESULT_NOT_FOUND));
                }
                ctx.push_u64(DISPLAY_ID);
            }
            1011 => ctx.push_u64(DISPLAY_ID), // OpenDefaultDisplay
            1020 => {}                        // CloseDisplay
            1102 => {
                // GetDisplayResolution
                ctx.push_u64(DISPLAY_WIDTH);
                ctx.push_u64(DISPLAY_HEIGHT);
            }
            2020 => {
                // OpenLayer(display name, layer id, aruid)
                let layer_id = ctx.in_u64(0x40);
                if layer_id == 0 {
                    return Err(CallError::Result(VI_RESULT_INVALID_ARGUMENT));
                }
                self.open_layer(layer_id, ctx)?;
            }
            2021 | 2031 => {
                // CloseLayer / DestroyStrayLayer
                ctx.system.buffer_queues.remove(&(ctx.in_u64(0) as u32));
            }
            2030 => {
                // CreateStrayLayer(flags, display id) -> layer id, size
                let id = 2 + ctx.system.buffer_queues.len() as u64;
                ctx.push_u64(id);
                self.open_layer(id, ctx)?;
            }
            2101 => {}                                           // SetLayerScalingMode
            5202 => ctx.push_event(&mut self.vsync_event, true), // GetDisplayVsyncEvent
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// IHOSBinderDriver: parcels to the binder objects (buffer queues).
#[derive(Default)]
pub struct HosBinderDriver {
    buffer_event: Option<usize>,
}

impl Service for HosBinderDriver {
    fn name(&self) -> &str {
        "vi::IHOSBinderDriver"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 | 3 => {
                // TransactParcel(Auto)(id, code, flags)
                let (id, code) = (ctx.in_u32(0), ctx.in_u32(4));
                let parcel = ctx.read_in_buffer(0)?;
                let Some(mut q) = ctx.system.buffer_queues.remove(&id) else {
                    return Err(CallError::Result(VI_RESULT_NOT_FOUND));
                };
                let reply = transact(&mut q, code, &parcel, ctx);
                ctx.system.buffer_queues.insert(id, q);
                match reply {
                    Some(w) => ctx.write_out_buffer(0, &w.finish())?,
                    None => {
                        if ctx.config.trace_unimplemented {
                            eprintln!("[vi] unhandled binder transaction {code} on {id}");
                        }
                        return Err(CallError::Result(VI_RESULT_INVALID_ARGUMENT));
                    }
                }
            }
            1 => {} // AdjustRefcount
            2 => {
                // GetNativeHandle: the "buffer available" event. Buffers
                // are released at once, so it stays signaled.
                ctx.push_event(&mut self.buffer_event, true);
            }
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
