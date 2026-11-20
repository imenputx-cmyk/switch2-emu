//! `pl:u`: the system fonts. Their files are Nintendo's, so none ship with
//! the emulator: plain TrueType/OpenType files placed in `user/fonts` next
//! to the executable are served (`FontStandard.ttf`, `FontChineseSimplified.ttf`,
//! `FontExtendedChineseSimplified.ttf`, `FontChineseTraditional.ttf`,
//! `FontKorean.ttf`, `FontNintendoExtended.ttf`, or `0.ttf`..`5.ttf`).
//! Missing fonts are reported as such.
//!
//! Switch 1 titles get the fonts in the BFTTF form through shared memory
//! (commands 0-5, as Eden's IPlatformServiceManager); Switch 2 titles get
//! plain copies through commands 2000/2010 (Lichii,
//! `platform_service_manager.cpp`, NX2_NOTES row 36).

use super::{CallError, Ctx, Service};
use crate::handles::KObject;
use std::path::PathBuf;

/// The Switch 1 font shared memory size.
const SHARED_FONT_SIZE: usize = 0x110_0000;

const NAMES: [&str; 6] = [
    "FontStandard",
    "FontChineseSimplified",
    "FontExtendedChineseSimplified",
    "FontChineseTraditional",
    "FontKorean",
    "FontNintendoExtended",
];

/// BFTTF obfuscation key (Eden `EncryptSharedFont`).
const KEY: u32 = 0x4962_1806;
const BFTTF_MAGIC: u32 = 0x1802_9A7F;

fn font_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf))
        .unwrap_or_default()
        .join("user")
        .join("fonts")
}

fn load_font(index: usize) -> Option<Vec<u8>> {
    let dir = font_dir();
    for name in [NAMES[index].to_string(), index.to_string()] {
        for ext in ["ttf", "otf"] {
            if let Ok(d) = std::fs::read(dir.join(format!("{name}.{ext}"))) {
                return Some(d);
            }
        }
    }
    None
}

/// The fonts that were found: (type, data).
fn fonts() -> Vec<(usize, Vec<u8>)> {
    (0..NAMES.len())
        .filter_map(|i| load_font(i).map(|d| (i, d)))
        .collect()
}

#[derive(Default)]
pub struct PlatformServiceManager {
    /// (offset, size) per type in the Switch 1 memory (BFTTF payload).
    regions: [(u32, u32); 6],
    /// (offset, size) per type in the Switch 2 memory (plain data).
    builtin: [(u32, u32); 6],
    shared: Option<usize>,
    builtin_shared: Option<(usize, u32)>,
    loaded: bool,
}

impl PlatformServiceManager {
    /// Builds both memories once.
    fn load(&mut self, ctx: &mut Ctx) {
        if self.loaded {
            return;
        }
        self.loaded = true;
        let found = fonts();
        if found.is_empty() {
            eprintln!(
                "[pl] no system fonts in {} (text may be missing)",
                font_dir().display()
            );
        }
        // Switch 1: [magic ^ key, size ^ key] then the data XORed by key.
        let mut mem = vec![0u8; SHARED_FONT_SIZE];
        let mut at = 0usize;
        for (i, data) in &found {
            let len = data.len().next_multiple_of(4);
            if at + 8 + len > mem.len() {
                break;
            }
            mem[at..at + 4].copy_from_slice(&(BFTTF_MAGIC ^ KEY).to_le_bytes());
            mem[at + 4..at + 8].copy_from_slice(&(data.len() as u32 ^ KEY).to_le_bytes());
            let mut padded = data.clone();
            padded.resize(len, 0);
            for (k, w) in padded.chunks_exact(4).enumerate() {
                let v = u32::from_le_bytes(w.try_into().unwrap()) ^ KEY;
                mem[at + 8 + 4 * k..at + 12 + 4 * k].copy_from_slice(&v.to_le_bytes());
            }
            self.regions[*i] = ((at + 8) as u32, data.len() as u32);
            at += 8 + len;
        }
        self.shared = Some(ctx.kernel.new_shared_memory(SHARED_FONT_SIZE as u64, mem));
        // Switch 2: page-aligned plain copies.
        let mut plain = Vec::new();
        for (i, data) in &found {
            let off = plain.len();
            plain.extend_from_slice(data);
            plain.resize(plain.len().next_multiple_of(0x1000), 0);
            self.builtin[*i] = (off as u32, data.len() as u32);
        }
        let size = plain.len().max(0x1000) as u32;
        plain.resize(size as usize, 0);
        self.builtin_shared = Some((ctx.kernel.new_shared_memory(u64::from(size), plain), size));
    }
}

impl Service for PlatformServiceManager {
    fn name(&self) -> &str {
        "pl:u"
    }

    fn call(&mut self, command: u32, ctx: &mut Ctx) -> Result<(), CallError> {
        self.load(ctx);
        let font = |ctx: &Ctx| (ctx.in_u32(0) as usize).min(5);
        match command {
            0 => {}               // RequestLoad
            1 => ctx.push_u32(1), // GetLoadState: loaded
            2 => ctx.push_u32(self.regions[font(ctx)].1),
            3 => ctx.push_u32(self.regions[font(ctx)].0),
            4 => ctx.push_copy(KObject::SharedMemory {
                id: self.shared.expect("loaded"),
            }),
            // GetSharedFontInOrderOfPriority(language): types, offsets and
            // sizes into three buffers; loaded flag and count.
            5 | 6 => {
                let present: Vec<usize> = (0..6).filter(|&i| self.regions[i].1 != 0).collect();
                let pack = |f: &dyn Fn(usize) -> u32| -> Vec<u8> {
                    present.iter().flat_map(|&i| f(i).to_le_bytes()).collect()
                };
                for (k, data) in [
                    pack(&|i| i as u32),
                    pack(&|i| self.regions[i].0),
                    pack(&|i| self.regions[i].1),
                ]
                .iter()
                .enumerate()
                {
                    if !data.is_empty() && ctx.out_buffer(k).is_some() {
                        ctx.write_out_buffer(k, data)?;
                    }
                }
                ctx.push_u8(1);
                ctx.push_bytes(&[0; 3]);
                ctx.push_u32(present.len() as u32);
            }
            // GetBuiltInFontSharedMemory (Switch 2): size and the memory.
            2000 => {
                let (id, size) = self.builtin_shared.expect("loaded");
                ctx.push_u32(size);
                ctx.push_copy(KObject::SharedMemory { id });
            }
            // GetBuiltInFontData(type | flag << 8): found, 0, offset, size.
            2010 => {
                let (offset, size) = self.builtin[(ctx.in_u32(0) & 0xFF).min(5) as usize];
                for w in [u32::from(size != 0), 0, offset, size] {
                    ctx.push_u32(w);
                }
            }
            _ => return Err(CallError::Unimplemented),
        }
        Ok(())
    }
}
