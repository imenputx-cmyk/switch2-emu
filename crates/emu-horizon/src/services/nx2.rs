// SPDX-License-Identifier: GPL-3.0-or-later
//! Switch 2 graphics system services, reached through TIPC sessions:
//! `gsv` (from `ISelfController` 2000) and the sessions it opens for
//! syncpoints (`syncpt`), the GPU (`gpu`) and the display (`vi2`).
//!
//! Ported from Lichii's `core/hle/service/nx2_gsv.h` (GPL-3.0-or-later),
//! where these protocols were worked out while booting Switch 2 titles
//! (its NX2_NOTES.md, blockers 17-21, 28, 29, 42-44). Command names
//! marked "guessed" there are guesses here too. Commands not handled
//! get a zeroed reply, as Lichii's auto stub gives `nx2:` sessions.
//!
//! Only the boot-time protocol is here: GPU memory registration, NVN
//! command submission and presentation need the Switch 2 GPU (Ampere
//! classes) and come later.

use super::{obj, CallError, Ctx, Service};
use crate::handles::KObject;
use std::collections::HashMap;

/// GPU virtual address range NVN uses (Lichii `video_core/nx2.h`).
pub const GPU_VA_START: u64 = 0x800_0000;
pub const GPU_VA_END: u64 = 0xC0_0000_0000;

/// Syncpoint page layout of gsv 204 (offsets in the read-only page).
const SYNCPT_VALUES: u32 = 0x40;
const SYNCPT_RO_SUMMARY: u32 = 0x1040;
const SYNCPT_RO_BITS: u32 = 0x1100;
const SYNCPT_RW_SUMMARY: u32 = 0x0;
const SYNCPT_RW_BITS: u32 = 0x100;

fn words(ctx: &mut Ctx, ws: &[u32]) {
    for &w in ws {
        ctx.push_u32(w);
    }
}

fn split(v: u64) -> [u32; 2] {
    [v as u32, (v >> 32) as u32]
}

/// TIPC 0 (u64 token): newer SDKs send it first; the reply echoes it.
fn handshake(ctx: &mut Ctx) {
    let token = ctx.in_u64(0);
    words(ctx, &split(token));
}

/// A zeroed reply for commands nobody has worked out.
fn zeroed(session: &str, command: u32, ctx: &mut Ctx) {
    if ctx.config.trace_unimplemented {
        eprintln!("[nx2] {session} command {command} -> zeroed reply");
    }
    ctx.push_bytes(&[0; 0x20]);
}

fn event(ctx: &mut Ctx, slot: &mut Option<usize>) {
    ctx.push_event(slot, false);
}

/// A Switch 2 session whose commands are not known yet: the TIPC
/// handshake, then zeroed replies.
pub struct Nx2Stub {
    name: &'static str,
}

impl Nx2Stub {
    pub fn new(name: &'static str) -> Self {
        Nx2Stub { name }
    }
}

impl Service for Nx2Stub {
    fn name(&self) -> &str {
        self.name
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => handshake(ctx),
            _ => zeroed(self.name, command, ctx),
        }
        Ok(())
    }
}

/// `ISelfController` 2000: the graphics system server.
#[derive(Default)]
pub struct Gsv {
    syncpt_events: [Option<usize>; 2],
}

impl Service for Gsv {
    fn name(&self) -> &str {
        "nx2:gsv"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => handshake(ctx),
            // Boot and other acknowledgements.
            4 | 5 | 101 | 901 | 903 | 1001 | 601 | 1201 => {}
            202 => ctx.push_session(obj(Syncpt::default())),
            // AttachSyncptMemory(version, max syncpoints; copy handles: a
            // writable and a read-only transfer memory): lay out the
            // read-only page and return the request and notify events.
            204 => {
                let max = ctx.in_u32(4).min((0x1000 - 0x40) / 8);
                if let (
                    Some(KObject::TransferMemory { .. }),
                    Some(KObject::TransferMemory { addr: ro, size }),
                ) = (ctx.copy_handle(0), ctx.copy_handle(1))
                {
                    if size >= 0x2000 {
                        let header = [
                            1,
                            0,
                            max,
                            SYNCPT_VALUES,
                            SYNCPT_RW_SUMMARY,
                            SYNCPT_RO_SUMMARY,
                            SYNCPT_RW_BITS,
                            SYNCPT_RO_BITS,
                        ];
                        let bytes: Vec<u8> = header.iter().flat_map(|w| w.to_le_bytes()).collect();
                        let _ = ctx.mem.write(ro, &bytes);
                    }
                } else if ctx.config.trace_unimplemented {
                    eprintln!("[nx2] gsv 204 without its transfer memories");
                }
                let [mut request, mut notify] = self.syncpt_events;
                event(ctx, &mut request);
                event(ctx, &mut notify);
                self.syncpt_events = [request, notify];
            }
            305 => ctx.push_session(obj(Vi2::default())),
            497 => ctx.push_session(obj(Gpu::default())),
            // Sessions Lichii hands out (the SDK aborts with InvalidHandle,
            // 0xE401, when the reply carries none): GPU time (1121),
            // video decoders (902, 905, 906, 1002), camera (602) and
            // chat (1202). Their commands get zeroed replies for now.
            1121 => ctx.push_session(obj(Nx2Stub::new("nx2:gpu_time"))),
            902 | 905 | 906 | 1002 => ctx.push_session(obj(Nx2Stub::new("nx2:video"))),
            602 => ctx.push_session(obj(Nx2Stub::new("nx2:camera"))),
            1202 => ctx.push_session(obj(Nx2Stub::new("nx2:chat"))),
            // OpenNvsrvSession: NVIDIA's driver server, which speaks the
            // nvdrv protocol (Lichii).
            1402 => ctx.push_session(obj(super::nv::NvDrv::nvsrv())),
            // CalculateRequirements: block-linear size and alignment of a
            // window image, 4 bytes per pixel (Lichii blocker 33). Input:
            // width, height.
            1510 => {
                let (w, h) = (ctx.in_u32(0).max(1), ctx.in_u32(4).max(1));
                let size =
                    u64::from(w.next_multiple_of(16) * 4) * u64::from(h.next_multiple_of(128));
                words(ctx, &split(size));
                words(ctx, &split(0x1_0000));
            }
            _ => zeroed("gsv", command, ctx),
        }
        Ok(())
    }
}

/// gsv 202: syncpoints.
#[derive(Default)]
pub struct Syncpt {
    slot_events: HashMap<u32, Option<usize>>,
}

impl Service for Syncpt {
    fn name(&self) -> &str {
        "nx2:syncpt"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => handshake(ctx),
            1 => words(ctx, &[32]), // GetCount (guessed)
            // GetSlotEvent(slot), RequestSlotEvent(slot, ~syncpoint,
            // value, mode): the slot's event. Waits are answered at once
            // (signalled) until syncpoints are emulated for Switch 2.
            22 | 201 => {
                let slot = ctx.in_u32(0);
                let mut e = self.slot_events.get(&slot).copied().flatten();
                ctx.push_event(&mut e, false);
                if let Some(id) = e {
                    if command == 201 {
                        ctx.kernel.events[id] = true;
                    }
                }
                self.slot_events.insert(slot, e);
            }
            // CancelSlotEvent(slot): the SDK waits on the slot's event
            // after a cancel, so it is signalled (Lichii).
            202 => {
                let slot = ctx.in_u32(0);
                let e = self
                    .slot_events
                    .get(&slot)
                    .copied()
                    .flatten()
                    .unwrap_or_else(|| ctx.kernel.new_event(false));
                ctx.kernel.events[e] = true;
                self.slot_events.insert(slot, Some(e));
            }
            // QueryCounter(~syncpoint): the value (u64) and a zero word.
            103 => words(ctx, &[0, 0, 0]),
            _ => zeroed("syncpt", command, ctx),
        }
        Ok(())
    }
}

/// Switch 2 GPU memory, shared by the GPU session and its sub-sessions.
/// The graphics server maps memory for the GPU itself: memory registered
/// with sub 101 is used at GPU address == CPU address, 211/216 map it
/// elsewhere, and 214 creates GPU-only regions the server backs (Lichii
/// NX2_NOTES rows 27, 30, 40, 42).
#[derive(Default)]
pub struct GpuMemory {
    /// Registered memory by handle: (address, size).
    pub memories: HashMap<u32, (u64, u64)>,
    next_handle: u32,
    /// Job/id events (ids 0..32).
    events: [Option<usize>; GPU_IDS as usize],
    /// GPU-only regions: start -> (backing guest address, size).
    regions: std::collections::BTreeMap<u64, (u64, u64)>,
    /// Next free backing address for GPU-only regions.
    backing_next: u64,
}

/// Ids (job slots) of the GPU session.
const GPU_IDS: u32 = 32;

/// Guest addresses past the 39-bit address space where GPU-only regions
/// get their (lazily allocated) memory; the game never sees them.
const GPU_ONLY_BACKING: u64 = 0x100_0000_0000;

impl GpuMemory {
    fn event(&mut self, id: u32, ctx: &mut Ctx) -> usize {
        let slot = &mut self.events[(id % GPU_IDS) as usize];
        *slot.get_or_insert_with(|| ctx.kernel.new_event(false))
    }

    /// Completes a job: signals its slot's event.
    fn complete(&mut self, slot: u32, ctx: &mut Ctx) {
        let e = self.event(slot, ctx);
        ctx.kernel.events[e] = true;
    }

    /// The backing of the GPU-only region containing `[va, va + size)`.
    fn region_backing(&self, va: u64, size: u64) -> Option<u64> {
        let (&start, &(backing, len)) = self.regions.range(..=va).next_back()?;
        (va + size <= start + len).then_some(backing + (va - start))
    }
}

/// gsv 497: the GPU (nn::gpu::sdk). Sub-sessions (22) are the same
/// object; their state lives in [`GpuMemory`].
#[derive(Default)]
pub struct Gpu {
    devices: [bool; 16],
}

/// A 64-bit value at data word `word` of a TIPC request (word 2 is the
/// first data word, as Lichii numbers them).
fn word64(ctx: &Ctx, word: usize) -> u64 {
    ctx.in_u64((word - 2) * 4)
}

fn word32(ctx: &Ctx, word: usize) -> u32 {
    ctx.in_u32((word - 2) * 4)
}

impl Service for Gpu {
    fn name(&self) -> &str {
        "nx2:gpu"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        let mut m = std::mem::take(&mut ctx.system.nx2_gpu);
        let r = self.dispatch(command, ctx, &mut m);
        ctx.system.nx2_gpu = m;
        r
    }
}

impl Gpu {
    fn dispatch(
        &mut self,
        command: u32,
        ctx: &mut Ctx,
        m: &mut GpuMemory,
    ) -> Result<(), CallError> {
        match command {
            0 => handshake(ctx),
            11 => words(ctx, &[0, GPU_IDS - 1]), // GetIdRange
            12 => {
                // GetIdEvent(id)
                let e = m.event(word32(ctx, 2), ctx);
                ctx.push_copy(KObject::Event { id: e });
            }
            // Start(id): nothing runs here, so it completes at once.
            21 => m.complete(word32(ctx, 2), ctx),
            22 => ctx.push_session(obj(Gpu::default())),
            33 => words(ctx, &[1]),
            41 => {
                // CreateGraphicsDevice(name, flags): a free id (1..).
                let Some(id) = (1..self.devices.len()).find(|&i| !self.devices[i]) else {
                    return Err(CallError::Result(crate::result::OUT_OF_RESOURCE));
                };
                self.devices[id] = true;
                words(ctx, &[id as u32]);
            }
            42 => {
                let id = (word32(ctx, 2) & 0xFF) as usize;
                if !self.devices.get(id).copied().unwrap_or(false) {
                    return Err(CallError::Result(crate::result::NOT_FOUND));
                }
                self.devices[id] = false;
            }
            // User partition [VA start, end); gsv partition empty (min >
            // max); VA range: kind 1, start, size.
            201 => words(
                ctx,
                &[split(GPU_VA_START), split(GPU_VA_END - GPU_VA_START)].concat(),
            ),
            202 => words(ctx, &[u32::MAX, u32::MAX, 0, 0]),
            203 => words(
                ctx,
                &[
                    split(1),
                    split(GPU_VA_START),
                    split(GPU_VA_END - GPU_VA_START),
                ]
                .concat(),
            ),
            // CreateGpuAccessibleMemory(start, last): used at GPU address ==
            // CPU address; the reply's handle names it in later jobs.
            101 => {
                let (start, last) = (word64(ctx, 2), word64(ctx, 4));
                if last > start {
                    ctx.system.gpu.map_shared(start, start, last - start + 1);
                }
                m.next_handle += 1;
                let handle = m.next_handle;
                m.memories
                    .insert(handle, (start, last.wrapping_sub(start).wrapping_add(1)));
                words(ctx, &[handle]);
            }
            // DestroyGpuAccessibleMemory(handle, slot).
            102 => {
                if let Some((addr, size)) = m.memories.remove(&word32(ctx, 2)) {
                    ctx.system.gpu.unmap_shared(addr, size);
                }
                m.complete(word32(ctx, 3), ctx);
            }
            // NormalMapGpuMemory4K: words 3-4 first GPU address, 5-6 last,
            // 7 memory handle, 8-9 offset into it, 17 the job slot.
            211 => {
                let (first, last) = (word64(ctx, 3), word64(ctx, 5));
                let (handle, offset) = (word32(ctx, 7), word64(ctx, 8));
                if let Some(&(addr, _)) = m.memories.get(&handle) {
                    if last > first {
                        ctx.system
                            .gpu
                            .map_shared(first, addr + offset, last - first + 1);
                    }
                } else if ctx.config.trace_unimplemented {
                    eprintln!("[nx2] gpu 211: unknown memory handle {handle}");
                }
                m.complete(word32(ctx, 17), ctx);
            }
            // NormalModifyMemoryKind4K: kinds do not matter here; report
            // the wanted kind (word 7), complete slot (word 10).
            212 => {
                let kind = word32(ctx, 7) & 0xFF;
                m.complete(word32(ctx, 10), ctx);
                words(ctx, &[kind]);
            }
            // NormalUnmapGpuMemory4K: words 3-6 the range, 8 the slot.
            213 => {
                let (first, last) = (word64(ctx, 3), word64(ctx, 5));
                if last > first {
                    unmap_or_back(ctx, m, first, last - first + 1);
                }
                m.complete(word32(ctx, 8), ctx);
            }
            // CreateSparseRegion (GPU-only memory): words 3-6 the range,
            // 11 the slot. Backed by memory of our own, allocated as used.
            214 => {
                let (start, last) = (word64(ctx, 3), word64(ctx, 5));
                if last > start {
                    let size = (last - start + 1).next_multiple_of(0x1000);
                    if m.backing_next == 0 {
                        m.backing_next = GPU_ONLY_BACKING;
                    }
                    let backing = m.backing_next;
                    m.backing_next += size.next_multiple_of(0x20_0000);
                    let _ = ctx.mem.map(backing, size, emu_memory::Perm::RW);
                    if let Some((old, old_size)) = m.regions.insert(start, (backing, size)) {
                        let _ = ctx.mem.unmap(old, old_size);
                    }
                    ctx.system.gpu.map_shared(start, backing, size);
                }
                m.complete(word32(ctx, 11), ctx);
            }
            // DestroySparseRegion: words 3-6 the range, 8 the slot.
            215 => {
                let (first, last) = (word64(ctx, 3), word64(ctx, 5));
                if let Some((backing, size)) = m.regions.remove(&first) {
                    let _ = ctx.mem.unmap(backing, size);
                }
                if last > first {
                    ctx.system.gpu.unmap_shared(first, last - first + 1);
                }
                m.complete(word32(ctx, 8), ctx);
            }
            // ExecuteSparseMapRequests: 40-byte requests in buffer A
            // {GPU address, size, flags, kind, fallback kind, .., memory
            // handle (0 unmaps), offset}; kinds back in buffer B; slot in
            // word 14. Replies the count.
            216 => {
                let data = ctx.read_in_buffer(0)?;
                let mut kinds = Vec::new();
                for r in data.chunks_exact(40) {
                    let u64_at = |o: usize| u64::from_le_bytes(r[o..o + 8].try_into().unwrap());
                    let (va, size) = (u64_at(0), u64_at(8));
                    let kind = r[19];
                    let handle = u32::from_le_bytes(r[24..28].try_into().unwrap());
                    let offset = u64_at(32);
                    kinds.push(kind);
                    if size == 0 {
                        continue;
                    }
                    if handle == 0 {
                        unmap_or_back(ctx, m, va, size);
                    } else if let Some(&(addr, len)) = m.memories.get(&handle) {
                        if offset <= len && size <= len - offset {
                            ctx.system.gpu.map_shared(va, addr + offset, size);
                        }
                    }
                }
                if !kinds.is_empty() && ctx.out_buffer(0).is_some() {
                    ctx.write_out_buffer(0, &kinds)?;
                }
                m.complete(word32(ctx, 14), ctx);
                let n = kinds.len() as u64;
                words(ctx, &split(n));
            }
            _ => zeroed("gpu", command, ctx),
        }
        Ok(())
    }
}

/// Unmaps a GPU range; inside a GPU-only region it shows the region's own
/// memory again.
fn unmap_or_back(ctx: &mut Ctx, m: &GpuMemory, va: u64, size: u64) {
    match m.region_backing(va, size) {
        Some(backing) => ctx.system.gpu.map_shared(va, backing, size),
        None => ctx.system.gpu.unmap_shared(va, size),
    }
}

/// gsv 305: the display (nn::vi2).
#[derive(Default)]
pub struct Vi2 {
    native_events: [Option<usize>; 8],
}

impl Service for Vi2 {
    fn name(&self) -> &str {
        "nx2:vi2"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => handshake(ctx),
            1 | 3 => words(ctx, &[1]), // GetDisplayCount (guessed)
            5 => words(ctx, &[7]),     // GetTimestampRecorderCount (guessed)
            // GetBufferMaxResolution: 2 = 1920x1080 (handheld).
            106 => words(ctx, &[2]),
            11 => {
                let mut e = None;
                event(ctx, &mut e);
            }
            // Acquire/DuplicateNativeEvent(index < 8).
            31 | 32 => {
                let index = ctx.in_u32(0) as usize;
                let Some(slot) = self.native_events.get_mut(index) else {
                    return Err(CallError::Result(crate::result::INVALID_ENUM_VALUE));
                };
                let mut e = *slot;
                ctx.push_event(&mut e, false);
                self.native_events[index] = e;
            }
            _ => zeroed("vi2", command, ctx),
        }
        Ok(())
    }
}
