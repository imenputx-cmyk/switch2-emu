//! `nvdrv`: the NVIDIA driver interface. Clients open device nodes and
//! send Linux-style ioctls (number, type, size and direction packed in
//! the request word). Implemented so far: `/dev/nvmap` (GPU memory
//! objects over guest memory) and the parts of `/dev/nvhost-ctrl` used
//! for fences. Struct layouts follow libnx's `nvidia/ioctl/*.c` (ISC).

use super::{nvgpu, CallError, Ctx, Service};
use std::collections::HashMap;

/// An nvmap object: guest memory the GPU can use.
#[derive(Debug, Clone, Default)]
pub struct NvMapObject {
    pub size: u32,
    /// Guest address, once allocated.
    pub addr: u64,
    pub align: u32,
    pub kind: u8,
}

/// nvdrv error codes (returned in the reply, not as a result).
use nvgpu::{NV_BAD_PARAMETER, NV_NOT_IMPLEMENTED, NV_SUCCESS};
const NV_TIMEOUT: u32 = 5;
const NV_FILE_OPERATION_FAILED: u32 = 0x30003;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    NvMap,
    NvHostCtrl,
    /// A GPU address space (index into `Gpu::address_spaces`).
    AsGpu(usize),
    CtrlGpu,
    /// A GPU channel (index into `Gpu::channels`).
    Gpu(usize),
    /// Switch 2 devices with no emulated state yet (`/dev/nvhost-tsg-gpu`
    /// etc.): every ioctl succeeds.
    Stub,
}

#[derive(Default)]
pub struct NvDrv {
    fds: HashMap<u32, Device>,
    next_fd: u32,
    events: HashMap<u32, usize>,
    /// A Switch 2 NvRm is the client (it asked for the 0xD0-byte
    /// characteristics block). Unknown ioctls then echo their input with
    /// success, as in Lichii (`nx2_stub_device.h`): its NvRm uses ioctls
    /// Switch 1 never had.
    nx2: bool,
    name: &'static str,
}

impl NvDrv {
    /// The `nx2:nvsrv` session gsv 1402 hands out: the same protocol.
    pub fn nvsrv() -> Self {
        NvDrv {
            name: "nx2:nvsrv",
            ..Self::default()
        }
    }
}

/// Fields of an ioctl request word.
struct Ioctl {
    nr: u32,
    kind: u32,
    size: usize,
}

fn decode(request: u32) -> Ioctl {
    Ioctl {
        nr: request & 0xFF,
        kind: (request >> 8) & 0xFF,
        size: ((request >> 16) & 0x3FFF) as usize,
    }
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    b.get(off..off + 4)
        .map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}

fn put_u32(b: &mut [u8], off: usize, v: u32) {
    if let Some(s) = b.get_mut(off..off + 4) {
        s.copy_from_slice(&v.to_le_bytes());
    }
}

impl NvDrv {
    /// Runs one ioctl on `data` in place; returns the nvdrv error code.
    fn ioctl(
        &mut self,
        dev: Device,
        io: &Ioctl,
        data: &mut [u8],
        extra: &[u8],
        inline_out: &mut [u8],
        ctx: &mut Ctx,
    ) -> Result<u32, CallError> {
        let fds = &self.fds;
        let channel_of_fd = |fd: u32| match fds.get(&fd) {
            Some(Device::Gpu(c)) => Some(*c),
            _ => None,
        };
        Ok(match (dev, io.kind, io.nr) {
            (Device::AsGpu(_), 0x41, 0x0B) if self.nx2 => nvgpu::syncpoint_shim(data),
            (Device::AsGpu(a), 0x41, nr) => match nvgpu::as_gpu(a, nr, data, ctx, channel_of_fd) {
                NV_NOT_IMPLEMENTED if self.nx2 => echo(data),
                r => r,
            },
            (Device::CtrlGpu, 0x47, nr) => {
                match nvgpu::ctrl_gpu(nr, data, inline_out, &mut self.nx2, ctx) {
                    NV_NOT_IMPLEMENTED if self.nx2 => echo(data),
                    r => r,
                }
            }
            (Device::Gpu(c), kind @ (0x47 | 0x48), nr) => {
                match nvgpu::channel(c, kind, nr, data, extra, ctx) {
                    NV_NOT_IMPLEMENTED if self.nx2 => echo(data),
                    r => r,
                }
            }
            (Device::Stub, ..) => echo(data),
            // NVMAP_IOC_CREATE {size in, handle out}
            (Device::NvMap, 1, 0x01) => {
                let size = u32_at(data, 0);
                if size == 0 {
                    return Ok(NV_BAD_PARAMETER);
                }
                let handle = ctx.system.nvmap.keys().max().map_or(1, |m| m + 1);
                ctx.system.nvmap.insert(
                    handle,
                    NvMapObject {
                        size,
                        ..Default::default()
                    },
                );
                put_u32(data, 4, handle);
                NV_SUCCESS
            }
            // NVMAP_IOC_FROM_ID {id in, handle out}: ids are handles.
            (Device::NvMap, 1, 0x03) => {
                let id = u32_at(data, 0);
                if !ctx.system.nvmap.contains_key(&id) {
                    return Ok(NV_BAD_PARAMETER);
                }
                put_u32(data, 4, id);
                NV_SUCCESS
            }
            // NVMAP_IOC_ALLOC {handle, heapmask, flags, align, kind, pad, addr}
            (Device::NvMap, 1, 0x04) => {
                let handle = u32_at(data, 0);
                let align = u32_at(data, 12);
                let kind = data.get(16).copied().unwrap_or(0);
                let addr = u64::from(u32_at(data, 24)) | (u64::from(u32_at(data, 28)) << 32);
                let Some(obj) = ctx.system.nvmap.get_mut(&handle) else {
                    return Ok(NV_BAD_PARAMETER);
                };
                obj.addr = addr;
                obj.align = align;
                obj.kind = kind;
                NV_SUCCESS
            }
            // NVMAP_IOC_FREE {handle, pad, refcount out, size out, flags out}
            (Device::NvMap, 1, 0x05) => {
                let handle = u32_at(data, 0);
                let Some(obj) = ctx.system.nvmap.remove(&handle) else {
                    return Ok(NV_BAD_PARAMETER);
                };
                put_u32(data, 8, 0);
                put_u32(data, 12, 0);
                put_u32(data, 16, obj.size);
                put_u32(data, 20, 0);
                NV_SUCCESS
            }
            // NVMAP_IOC_PARAM {handle, param, result out}
            (Device::NvMap, 1, 0x09) => {
                let Some(obj) = ctx.system.nvmap.get(&u32_at(data, 0)) else {
                    return Ok(NV_BAD_PARAMETER);
                };
                let v = match u32_at(data, 4) {
                    1 => obj.size,
                    2 => obj.align,
                    3 => 0x4000_0000, // base: unused
                    4 => 0x4000_0000, // heap
                    5 => u32::from(obj.kind),
                    _ => return Ok(NV_BAD_PARAMETER),
                };
                put_u32(data, 8, v);
                NV_SUCCESS
            }
            // NVMAP_IOC_GET_ID {id out, handle in}
            (Device::NvMap, 1, 0x0E) => {
                let handle = u32_at(data, 4);
                if !ctx.system.nvmap.contains_key(&handle) {
                    return Ok(NV_BAD_PARAMETER);
                }
                put_u32(data, 0, handle);
                NV_SUCCESS
            }
            // NVHOST_IOCTL_CTRL_SYNCPT_READ {id, value out}
            (Device::NvHostCtrl, 0, 0x14) => {
                let v = ctx
                    .system
                    .gpu
                    .syncpoints
                    .get(u32_at(data, 0) as usize)
                    .copied()
                    .unwrap_or(0);
                put_u32(data, 4, v);
                NV_SUCCESS
            }
            // SYNCPT_INCR
            (Device::NvHostCtrl, 0, 0x15) => {
                if let Some(v) = ctx.system.gpu.syncpoints.get_mut(u32_at(data, 0) as usize) {
                    *v = v.wrapping_add(1);
                }
                NV_SUCCESS
            }
            // SYNCPT_WAIT, EVENT_WAIT, EVENT_WAIT_ASYNC: no GPU work is
            // ever pending, so every threshold has been reached.
            (Device::NvHostCtrl, 0, 0x16 | 0x1D | 0x1E) => {
                // SYNCPT_WAIT / EVENT_WAIT / EVENT_WAIT_ASYNC {id, threshold, ...}:
                // GPU work runs at submission, so a fence that has not
                // been reached never will be.
                let (id, threshold) = (u32_at(data, 0) as usize, u32_at(data, 4));
                let cur = ctx.system.gpu.syncpoints.get(id).copied().unwrap_or(0);
                if cur.wrapping_sub(threshold) as i32 >= 0 {
                    NV_SUCCESS
                } else {
                    NV_TIMEOUT
                }
            }
            // EVENT_SIGNAL, EVENT_REGISTER, EVENT_UNREGISTER
            (Device::NvHostCtrl, 0, 0x1C | 0x1F | 0x20) => NV_SUCCESS,
            _ if self.nx2 => {
                if ctx.config.trace_unimplemented {
                    eprintln!(
                        "[nvdrv] Switch 2 ioctl {dev:?} type {:#x} nr {:#x}: echoed",
                        io.kind, io.nr
                    );
                }
                echo(data)
            }
            _ => {
                if ctx.config.trace_unimplemented {
                    eprintln!(
                        "[nvdrv] unimplemented ioctl {dev:?} type {:#x} nr {:#x} size {:#x}",
                        io.kind, io.nr, io.size
                    );
                }
                NV_NOT_IMPLEMENTED
            }
        })
    }
}

/// The reply Lichii gives Switch 2 ioctls it does not know: the input
/// back, with success.
fn echo(_data: &mut [u8]) -> u32 {
    NV_SUCCESS
}

impl Service for NvDrv {
    fn name(&self) -> &str {
        if self.name.is_empty() {
            "nvdrv"
        } else {
            self.name
        }
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => {
                // Open(path in A buffer) -> fd, error
                let path = ctx.read_in_buffer(0)?;
                let end = path.iter().position(|&c| c == 0).unwrap_or(path.len());
                let dev = match &path[..end] {
                    b"/dev/nvmap" => Some(Device::NvMap),
                    b"/dev/nvhost-ctrl" => Some(Device::NvHostCtrl),
                    b"/dev/nvhost-as-gpu" => {
                        Some(Device::AsGpu(ctx.system.gpu.new_address_space()))
                    }
                    b"/dev/nvhost-ctrl-gpu" => Some(Device::CtrlGpu),
                    b"/dev/nvhost-gpu" => Some(Device::Gpu(ctx.system.gpu.new_channel())),
                    // Switch 2 NvRm binds its channels to a TSG.
                    b"/dev/nvhost-tsg-gpu" => Some(Device::Stub),
                    p if self.nx2 && p.starts_with(b"/dev/nvhost-") => Some(Device::Stub),
                    _ => None,
                };
                match dev {
                    Some(dev) => {
                        self.next_fd += 1;
                        self.fds.insert(self.next_fd, dev);
                        ctx.push_u32(self.next_fd);
                        ctx.push_u32(NV_SUCCESS);
                    }
                    None => {
                        if ctx.config.trace_unimplemented {
                            eprintln!(
                                "[nvdrv] unknown device {}",
                                String::from_utf8_lossy(&path[..end])
                            );
                        }
                        ctx.push_u32(0);
                        ctx.push_u32(NV_FILE_OPERATION_FAILED);
                    }
                }
            }
            1 | 11 | 12 => {
                // Ioctl / Ioctl2 (second input buffer) / Ioctl3 (second
                // output buffer)(fd, request) -> error.
                let (fd, request) = (ctx.in_u32(0), ctx.in_u32(4));
                let Some(&dev) = self.fds.get(&fd) else {
                    ctx.push_u32(NV_BAD_PARAMETER);
                    return Ok(());
                };
                let io = decode(request);
                let mut data = ctx.read_in_buffer(0)?;
                data.resize(io.size, 0);
                let extra = if command == 11 {
                    ctx.read_in_buffer(1)?
                } else {
                    Vec::new()
                };
                // Ioctl3's second output buffer ("inline output").
                let mut inline_out = match (command, ctx.out_buffer(1)) {
                    (12, Some(b)) => vec![0u8; b.size as usize],
                    _ => Vec::new(),
                };
                let err = self.ioctl(dev, &io, &mut data, &extra, &mut inline_out, ctx)?;
                if request >> 30 & 2 != 0 && err == NV_SUCCESS {
                    ctx.write_out_buffer(0, &data)?;
                }
                if !inline_out.is_empty() && err == NV_SUCCESS {
                    ctx.write_out_buffer(1, &inline_out)?;
                }
                ctx.push_u32(err);
            }
            2 => {
                // Close(fd) -> error
                let ok = self.fds.remove(&ctx.in_u32(0)).is_some();
                ctx.push_u32(if ok { NV_SUCCESS } else { NV_BAD_PARAMETER });
            }
            3 => ctx.push_u32(NV_SUCCESS), // Initialize(transfer memory)
            4 => {
                // QueryEvent(fd, event id) -> error, copy handle
                let id = ctx.in_u32(4);
                let mut slot = self.events.get(&id).copied();
                ctx.push_event(&mut slot, false);
                self.events.insert(id, slot.unwrap());
                ctx.push_u32(NV_SUCCESS);
            }
            8 => ctx.push_u32(NV_SUCCESS), // SetAruid
            13 => {}                       // SetGraphicsFirmwareMemoryMarginEnabled
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
