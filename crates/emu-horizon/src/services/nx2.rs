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
            202 => {} // CancelSlotEvent
            // QueryCounter(~syncpoint): the value (u64) and a zero word.
            103 => words(ctx, &[0, 0, 0]),
            _ => zeroed("syncpt", command, ctx),
        }
        Ok(())
    }
}

/// gsv 497: the GPU (nn::gpu::sdk).
#[derive(Default)]
pub struct Gpu {
    id_events: HashMap<u32, Option<usize>>,
    next_device: u32,
}

impl Gpu {
    fn id_event(&mut self, id: u32, ctx: &mut Ctx) -> Option<usize> {
        let mut e = self.id_events.get(&id).copied().flatten();
        if e.is_none() {
            e = Some(ctx.kernel.new_event(false));
        }
        self.id_events.insert(id, e);
        e
    }
}

impl Service for Gpu {
    fn name(&self) -> &str {
        "nx2:gpu"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => handshake(ctx),
            11 => words(ctx, &[0, 31]), // GetIdRange (guessed): ids 0..=31
            12 => {
                // GetIdEvent(id)
                let id = ctx.in_u32(0);
                let e = self.id_event(id, ctx).expect("created");
                ctx.push_copy(KObject::Event { id: e });
            }
            21 => {
                // Start(id): completes at once.
                let id = ctx.in_u32(0);
                let e = self.id_event(id, ctx).expect("created");
                ctx.kernel.events[e] = true;
            }
            // OpenSubSession: memory jobs; a session of its own here.
            22 => ctx.push_session(obj(Gpu::default())),
            33 => words(ctx, &[1]),
            41 => {
                // CreateGraphicsDevice(name, flags, label): a non-zero id.
                self.next_device += 1;
                words(ctx, &[self.next_device]);
            }
            42 => {}
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
            _ => zeroed("gpu", command, ctx),
        }
        Ok(())
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
