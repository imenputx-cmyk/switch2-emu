//! `set:sys`, system settings. GetFirmwareVersion (command 3) reports the
//! firmware this emulator pretends to run (matches the HosVersion entry
//! given to homebrew).

use super::{CallError, Ctx, Service};

pub const FIRMWARE: (u8, u8, u8) = (21, 0, 0);

pub struct SystemSettings;

/// The 0x100-byte `SetSysFirmwareVersion` structure (layout as in libnx
/// `services/set.h`).
pub fn firmware_version() -> [u8; 0x100] {
    let mut fw = [0u8; 0x100];
    fw[0] = FIRMWARE.0;
    fw[1] = FIRMWARE.1;
    fw[2] = FIRMWARE.2;
    fw[4] = 1; // revision major
    fw[8..8 + 2].copy_from_slice(b"NX");
    let display = format!("{}.{}.{}", FIRMWARE.0, FIRMWARE.1, FIRMWARE.2);
    fw[0x68..0x68 + display.len()].copy_from_slice(display.as_bytes());
    let title = format!("NintendoSDK Firmware for NX {display}-1.0 (switch2-emu)");
    fw[0x80..0x80 + title.len()].copy_from_slice(title.as_bytes());
    fw
}

impl Service for SystemSettings {
    fn name(&self) -> &str {
        "set:sys"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            // GetFirmwareVersion / GetFirmwareVersion2 (out pointer, 0x100).
            3 | 4 => ctx.write_out_buffer(0, &firmware_version()),
            _ => Err(CallError::Unimplemented),
        }
    }
}
