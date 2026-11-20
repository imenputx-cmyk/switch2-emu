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
    next_memory: u32,
    next_image: u32,
    client_registered: bool,
}

/// Native memory and images, shared by gsv and the display (vi2).
#[derive(Default)]
pub struct NativeObjects {
    /// Native memory by id: (size, bound address).
    pub memories: HashMap<u32, (u64, u64)>,
    /// Native images (pools) by first handle.
    pub images: std::collections::BTreeMap<u32, NativeImage>,
}

impl NativeObjects {
    /// Image `handle`: its pool's preference words and its address (as
    /// bound: a GPU address, equal to the CPU one for registered memory).
    pub fn image(&self, handle: u32) -> Option<([u32; 16], u64)> {
        let (&first, img) = self
            .images
            .range(..=handle)
            .next_back()
            .filter(|(&first, img)| handle - first < img.count)?;
        let base = self.memories.get(&img.memory)?.1;
        Some((img.preference, base + u64::from(handle - first) * img.size))
    }
}

/// A native image pool (gsv 429): `count` images of `size` bytes each in
/// `memory`, described by 16 preference words.
#[derive(Clone, Copy)]
pub struct NativeImage {
    preference: [u32; 16],
    count: u32,
    memory: u32,
    size: u64,
}

/// Layout of a native image from its preference words (Lichii
/// nx2_image_layout.h): width/height in word 2, block height (log2) in
/// word 4, pitch in GOBs in word 5, format in word 6 (5 = RGBA16F, 8 bytes
/// a pixel; others 4). Returns (width, height, pitch, block height log2,
/// size).
pub fn native_image_layout(p: &[u32]) -> (u32, u32, u64, u32, u64) {
    let width = (p[2] & 0xFFFF) + 1;
    let height = (p[2] >> 16) + 1;
    let bpp = if p[6] == 5 { 8 } else { 4 };
    let (pitch, bh) =
        if p[4] <= 8 && u64::from(p[5]) >= (u64::from(width) * bpp).div_ceil(64) && p[5] <= 4096 {
            (u64::from(p[5]) * 64, p[4])
        } else {
            ((u64::from(width) * bpp).next_multiple_of(64), 7)
        };
    let rows = 1u64 << bh;
    (
        width,
        height,
        pitch,
        bh,
        pitch * u64::from(height).next_multiple_of(rows),
    )
}

/// 16 preference words from TIPC data words 2..18.
fn preference(ctx: &Ctx) -> [u32; 16] {
    std::array::from_fn(|i| ctx.in_u32(4 * i))
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
            // RegisterClientHandle: once.
            11 => {
                if std::mem::replace(&mut self.client_registered, true) {
                    return Err(CallError::Result(crate::result::INVALID_STATE));
                }
            }
            // CreateNativeMemory(.., size at words 3-4): an id.
            111 => {
                self.next_memory += 1;
                let id = self.next_memory;
                ctx.system
                    .nx2_native
                    .memories
                    .insert(id, (ctx.in_u64(4), 0));
                words(ctx, &[id]);
            }
            112 => {
                ctx.system.nx2_native.memories.remove(&ctx.in_u32(0));
            }
            // BindNativeMemory / BindNativeGpuMemory(id, address, size, ..).
            114 | 423 => {
                let (id, addr, size) = (ctx.in_u32(0), ctx.in_u64(4), ctx.in_u64(12));
                let m = ctx
                    .system
                    .nx2_native
                    .memories
                    .entry(id)
                    .or_insert((size, 0));
                m.1 = addr;
                if m.0 == 0 {
                    m.0 = size;
                }
            }
            // QueryGrantedFeatures: what was asked; GetSessionFlag: 0.
            115 => words(ctx, &[ctx.in_u32(0), ctx.in_u32(4)]),
            126 => words(ctx, &[0]),
            // Permission checks and attaching address spaces: granted.
            201 | 301 | 401 | 402 | 403 | 434 | 494 | 801 => {}
            // CreateNativeImage (and the feature/chat pools): 16 preference
            // words, count (word 18), memory (word 19); handles are the pool
            // id + index.
            429 | 311 | 321 | 615 | 811 | 1211 => {
                let p = preference(ctx);
                let count = ctx.in_u32(64).max(1);
                let memory = ctx.in_u32(68);
                if self.next_image == 0 {
                    self.next_image = 1;
                }
                let id = self.next_image;
                self.next_image += count;
                let size = native_image_layout(&p).4;
                ctx.system.nx2_native.images.insert(
                    id,
                    NativeImage {
                        preference: p,
                        count,
                        memory,
                        size,
                    },
                );
                words(ctx, &[id]);
            }
            430 | 312 | 612 | 912 => {
                ctx.system.nx2_native.images.remove(&ctx.in_u32(0));
            }
            // Destroy/CreateNativeWindowAliasImagePool, UnmapNativeImage.
            314 | 315 | 427 => {}
            // MapNativeImage (426) / QueryNativeImage (428)(handle): GPU
            // address (426 only), offset in the memory, then a 33-word
            // descriptor (kind 1, preference words 1-2, block height, pitch
            // in GOBs, depth/levels 1, size).
            426 | 428 => {
                let handle = ctx.in_u32(0);
                let found = ctx
                    .system
                    .nx2_native
                    .images
                    .range(..=handle)
                    .next_back()
                    .filter(|(&first, img)| handle - first < img.count)
                    .map(|(&first, img)| (first, *img));
                let (p, offset, base) = match found {
                    Some((first, img)) => (
                        img.preference,
                        u64::from(handle - first) * img.size,
                        ctx.system
                            .nx2_native
                            .memories
                            .get(&img.memory)
                            .map_or(0, |m| m.1),
                    ),
                    None => ([0; 16], 0, 0),
                };
                let (_, _, pitch, bh, size) = native_image_layout(&p);
                let gpu = if command == 426 { base + offset } else { 0 };
                let mut w = [0u32; 37];
                w[..2].copy_from_slice(&split(gpu));
                w[2..4].copy_from_slice(&split(offset));
                let desc = &mut w[4..];
                desc[0] = 1;
                desc[1] = p[1];
                desc[2] = p[2];
                desc[4] = bh;
                desc[5] = (pitch / 64) as u32;
                desc[8] = 1;
                desc[9] = 1;
                desc[11] = size as u32;
                words(ctx, &w);
            }
            // CalculateNativeImageSize(preference, count).
            435 => {
                let size =
                    native_image_layout(&preference(ctx)).4 * u64::from(ctx.in_u32(64).max(1));
                words(ctx, &split(size));
            }
            // Zeroed replies Lichii gives these (no known meaning).
            113
            | 120..=125
            | 131
            | 133
            | 187
            | 211..=215
            | 303
            | 424
            | 425
            | 436
            | 461..=468
            | 493
            | 603
            | 613
            | 614
            | 802
            | 803 => {
                words(ctx, &[0; 8]);
            }
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
            // GetSlotEvent(slot): the slot's event.
            22 => {
                let slot = ctx.in_u32(0);
                let mut e = self.slot_events.get(&slot).copied().flatten();
                ctx.push_event(&mut e, false);
                self.slot_events.insert(slot, e);
            }
            // RequestSlotEvent(slot, ~syncpoint, value (u64), mode): the
            // slot's event is cleared and signalled once the GPU syncpoint
            // reaches the value (Lichii INx2SyncptSession).
            201 => {
                let slot = ctx.in_u32(0);
                let syncpoint = !ctx.in_u32(4);
                let value = ctx.in_u32(8);
                let e = self
                    .slot_events
                    .get(&slot)
                    .copied()
                    .flatten()
                    .unwrap_or_else(|| ctx.kernel.new_event(false));
                self.slot_events.insert(slot, Some(e));
                ctx.system.syncpt_waits.retain(|w| w.2 != e);
                match ctx.system.gpu.syncpoints.get(syncpoint as usize) {
                    Some(&now) if now.wrapping_sub(value) as i32 >= 0 => {
                        ctx.kernel.events[e] = true
                    }
                    Some(_) => {
                        ctx.kernel.events[e] = false;
                        ctx.system.syncpt_waits.push((syncpoint, value, e));
                    }
                    None => ctx.kernel.events[e] = true,
                }
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
            103 => {
                let syncpoint = !ctx.in_u32(0);
                let v = ctx
                    .system
                    .gpu
                    .syncpoints
                    .get(syncpoint as usize)
                    .copied()
                    .unwrap_or(0);
                words(ctx, &[v, 0, 0]);
            }
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

/// The frame of a vi2 present (Lichii INx2Vi2Session::Composite): native
/// image `image` (block linear), cropped to the drawn area (`shown`: width
/// minus one, height minus one, 16 bits each) and to the handheld panel
/// (1080p); flag bit 18 flips it vertically.
fn composite(ctx: &mut Ctx, image: u32, shown: u32, flags: u32) -> Option<super::Frame> {
    let (p, address) = ctx.system.nx2_native.image(image)?;
    let (width, height, pitch, _, _) = native_image_layout(&p);
    // Preference word 4 is log2 of the block height in rows (3 = 1 GOB);
    // out of range means the fallback of 128 rows.
    let gobs_log2 = match p[4] {
        3..=8 if pitch == u64::from(p[5]) * 64 => p[4] - 3,
        _ => 4,
    };
    let address = ctx.system.gpu.shared_translate(address).unwrap_or(address);
    let pitch = u32::try_from(pitch).ok()?;
    let mut raw = vec![0u8; crate::display::block_linear_size(pitch, height, gobs_log2)];
    ctx.mem.read(address, &mut raw).ok()?;
    let linear = crate::display::deswizzle(&raw, pitch, height, gobs_log2);
    let (mut w, mut h) = (width, height);
    let (sw, sh) = ((shown & 0xFFFF) + 1, (shown >> 16) + 1);
    if shown != 0 && sw <= width && sh <= height {
        (w, h) = (sw, sh);
    }
    (w, h) = (w.min(1920), h.min(1080));
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let y = if flags & (1 << 18) != 0 { h - 1 - y } else { y };
        let row = &linear[(y * pitch) as usize..];
        for x in 0..w as usize {
            rgba.extend_from_slice(&match p[6] {
                // RGBA16F, linear: to sRGB.
                5 => {
                    let c = |i: usize| {
                        let h = u16::from_le_bytes([row[x * 8 + 2 * i], row[x * 8 + 2 * i + 1]]);
                        f16_to_f32(h)
                    };
                    [
                        linear_to_srgb(c(0)),
                        linear_to_srgb(c(1)),
                        linear_to_srgb(c(2)),
                        (c(3).clamp(0.0, 1.0) * 255.0).round() as u8,
                    ]
                }
                // RGB10A2.
                4 => {
                    let v = u32::from_le_bytes(row[x * 4..x * 4 + 4].try_into().unwrap());
                    let c = |s: u32| ((v >> s & 0x3FF) >> 2) as u8;
                    [c(0), c(10), c(20), ((v >> 30) * 85) as u8]
                }
                _ => row[x * 4..x * 4 + 4].try_into().unwrap(),
            });
        }
    }
    Some(super::Frame {
        width: w,
        height: h,
        rgba,
    })
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from(h >> 10 & 0x1F);
    let mant = f32::from(h & 0x3FF);
    sign * match exp {
        0 => mant * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn linear_to_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// gsv 305: the display (nn::vi2), from Lichii's INx2Vi2Session.
#[derive(Default)]
pub struct Vi2 {
    native_events: [Option<usize>; 8],
    presents: u64,
}

/// Buffer slots a client may acquire (Lichii: 0x7fff).
const VI2_SLOTS: u64 = 0x7FFF;

impl Vi2 {
    /// The event of native slot `index`, made once and registered as a
    /// vblank event (signalled every frame until bound to something else).
    fn native_event(&mut self, index: usize, ctx: &mut Ctx) -> Result<usize, CallError> {
        let slot = self
            .native_events
            .get_mut(index)
            .ok_or(CallError::Result(crate::result::INVALID_ENUM_VALUE))?;
        Ok(*slot.get_or_insert_with(|| {
            let e = ctx.kernel.new_event(true);
            ctx.system.vblank_events.push(e);
            e
        }))
    }
}

impl Service for Vi2 {
    fn name(&self) -> &str {
        "nx2:vi2"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => handshake(ctx),
            1 | 3 => words(ctx, &[1]), // GetDisplayCount (guessed)
            // GetTimestampRecorderCount: < 8; LayerHistory makes 7.
            5 => words(ctx, &[7]),
            // GetBufferMaxResolution(display name): handheld 1080p (mode 2),
            // as a u64.
            106 => words(ctx, &split(2)),
            11 => {
                let mut e = None;
                event(ctx, &mut e);
            }
            // Acquire/DuplicateNativeEvent(index < 8).
            31 | 32 => {
                let e = self.native_event(ctx.in_u32(0) as usize, ctx)?;
                ctx.push_copy(KObject::Event { id: e });
            }
            // GetRecordedTimestamps: nothing recorded yet, pending.
            53 => words(ctx, &[0, 0, 1]),
            // QueryDisplayRefreshRate: 60 Hz (and a bool).
            103 => words(ctx, &[60, 0]),
            // QueryDisplayHdrStatus: SDR (1), peak luminance 0.
            104 => words(ctx, &[1, 0, 0]),
            // QueryDisplayHdrPaperWhite (nits).
            105 => words(ctx, &[203]),
            // GetLayerBufferBits(layer): every slot ready.
            212 => words(ctx, &split(VI2_SLOTS)),
            // CancelClientBuffer / PresentClientBuffer(token, slot, image,
            // fences): every slot back; four zero words.
            228 | 229 => {
                if command == 229 {
                    self.presents += 1;
                    // Words: token (0-1), slot, image (3), .., shown size
                    // (5), flags (6).
                    let (image, shown, flags) = (ctx.in_u32(12), ctx.in_u32(20), ctx.in_u32(24));
                    match composite(ctx, image, shown, flags) {
                        Some(frame) => {
                            if self.presents <= 3 || self.presents.is_multiple_of(600) {
                                let lit = frame
                                    .rgba
                                    .chunks_exact(4)
                                    .filter(|p| p[0] | p[1] | p[2] != 0)
                                    .count();
                                let total = (frame.width * frame.height).max(1) as usize;
                                eprintln!(
                                    "[nx2] present #{}: image {image}, {}x{}, {}% of pixels not black",
                                    self.presents,
                                    frame.width,
                                    frame.height,
                                    lit * 100 / total
                                );
                            }
                            ctx.system.present(frame)
                        }
                        None if self.presents <= 3 => {
                            eprintln!("[nx2] vi2 present: image {image} not found or unmapped")
                        }
                        None => {}
                    }
                    for e in ctx.system.vblank_events.clone() {
                        ctx.kernel.events[e] = true;
                    }
                }
                let [lo, hi] = split(VI2_SLOTS);
                words(ctx, &[lo, hi, 0, 0, 0, 0]);
            }
            // Bind a native slot (u64 display, u32 index) to vsync or a
            // change source: fire once so the game reads the state.
            261 | 263 | 265 | 267 | 269 | 271 => {
                let e = self.native_event(ctx.in_u32(8) as usize, ctx)?;
                ctx.kernel.events[e] = true;
            }
            262 | 264 | 266 | 268 | 270 | 272 => {}
            _ => zeroed("vi2", command, ctx),
        }
        Ok(())
    }
}

#[cfg(test)]
mod composite_tests {
    use super::*;

    #[test]
    fn half_floats() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(linear_to_srgb(1.0), 255);
        assert_eq!(linear_to_srgb(0.0), 0);
        assert_eq!(linear_to_srgb(0.2159), 128);
    }
}
