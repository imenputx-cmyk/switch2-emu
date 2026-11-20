//! `hid`: input. The client reads controller state from a 0x40000-byte
//! shared memory block (layout in libnx's `hidshmem.h`); commands turn
//! input devices on and configure the controllers. The emulator writes
//! the host's controller state (`System::input`) into it periodically.

use super::{obj, CallError, Ctx, Service};
use crate::handles::KObject;

pub const SHARED_MEMORY_SIZE: u64 = 0x40000;

#[derive(Default)]
pub struct Hid {
    shared_memory: Option<usize>,
}

impl Service for Hid {
    fn name(&self) -> &str {
        "hid"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => {
                // CreateAppletResource
                let id = *self.shared_memory.get_or_insert_with(|| {
                    ctx.kernel
                        .new_shared_memory(SHARED_MEMORY_SIZE, initial_shared_memory())
                });
                ctx.system.hid_shared_memory = Some(id);
                ctx.push_object(obj(AppletResource { shared_memory: id }));
            }
            // Activate{DebugPad,TouchScreen,Mouse,Keyboard,Npad},
            // SetSupportedNpadStyleSet, SetSupportedNpadIdType,
            // DeactivateNpad, SetNpadJoyHoldType, ActivateNpadWithRevision:
            // accepted (the input state is not emulated yet).
            1 | 11 | 21 | 31 | 100 | 102 | 103 | 104 | 120 | 109 => {}
            // More of what games call at start (Eden hid_server.cpp):
            // six-axis and gesture activation, joy-con assignment modes,
            // handheld activation, vibration sends, communication mode...
            40 | 41 | 51 | 55 | 66 | 67 | 68 | 69 | 70 | 71 | 72 | 73 | 74 | 75 | 76 | 77 | 78
            | 79 | 80 | 81 | 91 | 105 | 107 | 122 | 123 | 124 | 125 | 126 | 127 | 128 | 130
            | 131 | 133 | 134 | 136 | 201 | 206 | 207 | 208 | 209 | 210 | 211 | 212 | 213 | 214
            | 220 | 300 | 301 | 302 | 303 | 304 | 305 | 306 | 307 | 308 | 309 | 310 | 311 | 400
            | 401 | 402 | 403 | 404 | 405 | 406 | 407 | 408 | 409 | 500 | 501 | 502 | 503 | 504
            | 505 | 510 | 511 | 512 | 513 | 520 | 521 | 522 | 523 | 524 | 525 | 1000 | 1002
            | 1003 | 1004 | 1005 => {}
            // IsSixAxisSensorAtRest, IsFirmwareUpdateAvailableForSixAxisSensor,
            // IsSixAxisSensorFusionEnabled, IsVibrationPermitted,
            // IsVibrationDeviceMounted, IsUsbFullKeyControllerEnabled...
            82 | 83 | 65 | 215 | 216 | 219 => ctx.push_u8(1),
            // GetSupportedNpadStyleSet: full key, handheld, joy-con pair.
            101 => ctx.push_u32(0b111),
            // AcquireNpadStyleSetUpdateEventHandle: signalled once.
            106 => {
                let mut e = None;
                ctx.push_event(&mut e, true);
            }
            // GetPlayerLedPattern(npad): player 1.
            108 => ctx.push_u64(1),
            // GetNpadJoyHoldType: horizontal; GetNpadHandheldActivationMode.
            121 | 129 => ctx.push_u64(0),
            // GetVibrationDeviceInfo(handle): linear resonant, left.
            200 => {
                ctx.push_u32(1);
                ctx.push_u32(1);
            }
            // GetActualVibrationValue: no vibration.
            202 => ctx.push_bytes(&[0; 0x10]),
            // CreateActiveVibrationDeviceList.
            203 => ctx.push_object(obj(ActiveVibrationDeviceList)),
            // GetNpadCommunicationMode.
            1001 => ctx.push_u64(0),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

pub struct AppletResource {
    shared_memory: usize,
}

impl Service for AppletResource {
    fn name(&self) -> &str {
        "hid::IAppletResource"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_copy(KObject::SharedMemory {
                id: self.shared_memory,
            }),
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// Controller state set by the host front end.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InputState {
    /// `HidNpadButton_*` bits.
    pub buttons: u64,
    /// Analog sticks, -32767..=32767 per axis (y up).
    pub stick_l: (i32, i32),
    pub stick_r: (i32, i32),
}

pub mod button {
    pub const A: u64 = 1 << 0;
    pub const B: u64 = 1 << 1;
    pub const X: u64 = 1 << 2;
    pub const Y: u64 = 1 << 3;
    pub const STICK_L: u64 = 1 << 4;
    pub const STICK_R: u64 = 1 << 5;
    pub const L: u64 = 1 << 6;
    pub const R: u64 = 1 << 7;
    pub const ZL: u64 = 1 << 8;
    pub const ZR: u64 = 1 << 9;
    pub const PLUS: u64 = 1 << 10;
    pub const MINUS: u64 = 1 << 11;
    pub const LEFT: u64 = 1 << 12;
    pub const UP: u64 = 1 << 13;
    pub const RIGHT: u64 = 1 << 14;
    pub const DOWN: u64 = 1 << 15;
}

// Shared memory layout (libnx hidshmem.h, ISC; offsets computed from its
// structs for AArch64): npad entries start at 0x9A00, 0x5000 bytes each;
// entry 0 is player 1 (a Pro Controller here), entry 8 handheld mode.
const NPAD: usize = 0x9A00;
const NPAD_ENTRY: usize = 0x5000;
const STYLE_SET: usize = 0x0;
const FULL_KEY_LIFO: usize = 0x28;
const HANDHELD_LIFO: usize = 0x378;
const DEVICE_TYPE: usize = 0x4188;
/// A lifo: header (unused, buffer count, tail, count: u64 each), then
/// 17 entries of 0x30 bytes: sampling number, then the state (sampling
/// number, buttons, left stick, right stick, attributes).
const LIFO_ENTRIES: u64 = 17;
const LIFO_STORAGE: usize = 0x20;
const LIFO_ENTRY: usize = 0x30;

const STYLE_FULL_KEY: u32 = 1 << 0;
const STYLE_HANDHELD: u32 = 1 << 1;
const DEVICE_FULL_KEY: u32 = 1 << 0;
const DEVICE_HANDHELD: u32 = (1 << 2) | (1 << 3);
const ATTR_CONNECTED: u32 = 1 << 0;

/// The initial shared memory: player 1 and handheld connected.
pub fn initial_shared_memory() -> Vec<u8> {
    let mut m = vec![0u8; SHARED_MEMORY_SIZE as usize];
    for (entry, style, device) in [
        (0, STYLE_FULL_KEY, DEVICE_FULL_KEY),
        (8, STYLE_HANDHELD, DEVICE_HANDHELD),
    ] {
        let base = NPAD + entry * NPAD_ENTRY;
        m[base + STYLE_SET..base + STYLE_SET + 4].copy_from_slice(&style.to_le_bytes());
        m[base + DEVICE_TYPE..base + DEVICE_TYPE + 4].copy_from_slice(&device.to_le_bytes());
    }
    m
}

/// Bytes to write for one input sample: (shared memory offset, data).
pub fn sample_writes(sampling: u64, input: &InputState) -> Vec<(usize, Vec<u8>)> {
    let mut state = Vec::with_capacity(LIFO_ENTRY);
    state.extend_from_slice(&sampling.to_le_bytes());
    state.extend_from_slice(&sampling.to_le_bytes());
    state.extend_from_slice(&input.buttons.to_le_bytes());
    for v in [
        input.stick_l.0,
        input.stick_l.1,
        input.stick_r.0,
        input.stick_r.1,
    ] {
        state.extend_from_slice(&v.to_le_bytes());
    }
    state.extend_from_slice(&ATTR_CONNECTED.to_le_bytes());
    state.resize(LIFO_ENTRY, 0);
    let tail = sampling % LIFO_ENTRIES;
    let count = (sampling + 1).min(LIFO_ENTRIES);
    let mut out = Vec::new();
    for lifo in [NPAD + FULL_KEY_LIFO, NPAD + 8 * NPAD_ENTRY + HANDHELD_LIFO] {
        out.push((
            lifo + LIFO_STORAGE + tail as usize * LIFO_ENTRY,
            state.clone(),
        ));
        let mut header = Vec::with_capacity(32);
        for v in [0, LIFO_ENTRIES, tail, count] {
            header.extend_from_slice(&v.to_le_bytes());
        }
        out.push((lifo, header));
    }
    out
}

/// `IActiveVibrationDeviceList`.
pub struct ActiveVibrationDeviceList;

impl Service for ActiveVibrationDeviceList {
    fn name(&self) -> &str {
        "hid::IActiveVibrationDeviceList"
    }

    fn call(&mut self, command: u32, _ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => Ok(()), // ActivateVibrationDevice
            _ => Err(CallError::Unimplemented),
        }
    }
}
