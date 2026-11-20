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
            3 | 4 => ctx.write_out_buffer(0, &firmware_version())?,
            // GetLockScreenFlag, GetQuestFlag, GetAutoUpdateEnableFlag...
            7 | 47 | 95 | 99 | 121 | 136 => ctx.push_u8(0),
            // GetColorSetId: basic white.
            23 => ctx.push_u32(0),
            // GetAccountSettings, GetEulaVersions...
            17 | 21 => ctx.push_u32(0),
            // GetSettingsItemValueSize(class, item) / GetSettingsItemValue:
            // unknown items read as a zero u32 (Eden keeps a table of
            // defaults; none is needed yet).
            37 => ctx.push_u64(4),
            38 => {
                if let Some(b) = ctx.out_buffer(0) {
                    ctx.write_out_buffer(0, &vec![0; b.size.min(8) as usize])?;
                }
                ctx.push_u64(4);
            }
            // GetDeviceNickName (0x80 bytes).
            77 => {
                let mut name = [0u8; 0x80];
                name[..10].copy_from_slice(b"switch2emu");
                ctx.write_out_buffer(0, &name)?;
            }
            // GetPrimaryAlbumStorage: SD card.
            46 => ctx.push_u32(1),
            // GetRegionCode: Europe; GetProductModel: 1.
            60 | 68 => ctx.push_u32(if command == 60 { 2 } else { 1 }),
            // SetXxx: accepted.
            8 | 22 | 24 | 48 | 78 | 96 | 100 | 122 | 137 => {}
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}

/// Language codes in `set` order (Eden `LANGUAGE_CODES`), as the 8-byte
/// NUL-padded values games compare.
const LANGUAGES: [&[u8]; 18] = [
    b"ja", b"en-US", b"fr", b"de", b"it", b"es", b"zh-CN", b"ko", b"nl", b"pt", b"ru", b"zh-TW",
    b"en-GB", b"fr-CA", b"es-419", b"zh-Hans", b"zh-Hant", b"pt-BR",
];

fn language_code(i: usize) -> u64 {
    let mut b = [0u8; 8];
    b[..LANGUAGES[i].len()].copy_from_slice(LANGUAGES[i]);
    u64::from_le_bytes(b)
}

/// `set`: user-facing settings. The console speaks American English.
pub struct Settings;

impl Service for Settings {
    fn name(&self) -> &str {
        "set"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        match command {
            0 => ctx.push_u64(language_code(1)), // GetLanguageCode
            // GetAvailableLanguageCodes(2): codes into the out buffer.
            1 | 5 => {
                let max = if command == 1 { 15 } else { LANGUAGES.len() };
                let codes: Vec<u8> = (0..max)
                    .flat_map(|i| language_code(i).to_le_bytes())
                    .collect();
                ctx.write_out_buffer(0, &codes)?;
                ctx.push_u32(max as u32);
            }
            // MakeLanguageCode(language index).
            2 => {
                let i = ctx.in_u32(0) as usize;
                if i >= LANGUAGES.len() {
                    return Err(CallError::Result(crate::result::INVALID_ENUM_VALUE));
                }
                ctx.push_u64(language_code(i));
            }
            3 => ctx.push_u32(15), // GetAvailableLanguageCodeCount
            6 => ctx.push_u32(LANGUAGES.len() as u32), // GetAvailableLanguageCodeCount2
            4 => ctx.push_u32(2),  // GetRegionCode: Europe
            // GetKeyCodeMap(2): a zeroed map.
            7 | 9 => ctx.write_out_buffer(0, &[0; 0x1000])?,
            8 => ctx.push_u8(0), // GetQuestFlag
            // GetFirmwareVersionForDebug.
            10 => ctx.write_out_buffer(0, &firmware_version())?,
            // GetDeviceNickName.
            11 => {
                let mut name = [0u8; 0x80];
                name[..10].copy_from_slice(b"switch2emu");
                ctx.write_out_buffer(0, &name)?;
            }
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
