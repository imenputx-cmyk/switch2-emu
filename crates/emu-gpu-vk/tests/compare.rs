//! The Vulkan backend must leave guest memory exactly as the software
//! backend does. Runs the same command streams through both. Skipped
//! (with a message) when no Vulkan 1.3 device is available.
//!
//! Float-to-unorm conversion may round differently between Vulkan
//! implementations when a value lands near .5 of a step, so the colors
//! are chosen away from such ties for every format tested.

use emu_gpu::{backend::Backend, class, Gpu};
use emu_gpu_vk::VulkanBackend;
use emu_memory::{Memory, Perm};

const GUEST: u64 = 0x100_0000;
const GPU_VA: u64 = 0x1000_0000;
const SIZE: u64 = 0x40_0000;

fn incr(subch: u32, method: u32, count: u32) -> u32 {
    (1 << 29) | (count << 16) | (subch << 13) | (method / 4)
}

/// Runs `words` (placed at the start of the mapping) and returns the
/// rest of the mapping afterwards.
fn run(backend: Box<dyn Backend>, words: &[u32]) -> Vec<u8> {
    let mut mem = Memory::new();
    mem.map(GUEST, SIZE, Perm::RW).unwrap();
    // Non-zero initial contents: uploads must preserve them.
    let init: Vec<u8> = (0..SIZE as usize - 0x1_0000)
        .map(|i| (i * 7 / 3) as u8)
        .collect();
    mem.write(GUEST + 0x1_0000, &init).unwrap();
    let mut gpu = Gpu {
        backend,
        ..Default::default()
    };
    let a = gpu.new_address_space();
    assert!(gpu.address_spaces[a].map(GPU_VA, GUEST, SIZE));
    let c = gpu.new_channel();
    gpu.channels[c].address_space = Some(a);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    mem.write(GUEST, &bytes).unwrap();
    gpu.submit(c, &[GPU_VA | ((words.len() as u64) << 42)], 0, &mut mem);
    assert!(gpu.faults.is_empty(), "{:?}", gpu.faults);
    let mut out = vec![0u8; SIZE as usize - 0x1_0000];
    mem.read(GUEST + 0x1_0000, &mut out).unwrap();
    out
}

fn vulkan() -> Option<Box<dyn Backend>> {
    match VulkanBackend::new() {
        Ok(b) => {
            eprintln!("Vulkan device: {}", b.device_name);
            Some(Box::new(b))
        }
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

/// Render target 0 at GPU_VA + `offset`.
fn target(offset: u64, width: u32, height: u32, format: u32, memory: u32) -> Vec<u32> {
    let va = GPU_VA + offset;
    vec![
        incr(0, 0, 1),
        class::THREED,
        incr(0, 0x800, 6),
        (va >> 32) as u32,
        va as u32,
        width,
        height,
        format,
        memory,
    ]
}

fn clear(color: [f32; 4], scissor: Option<(u32, u32, u32, u32)>, mask: u32) -> Vec<u32> {
    let mut v = vec![incr(0, 0xD80, 4)];
    v.extend(color.map(f32::to_bits));
    match scissor {
        Some((x0, y0, x1, y1)) => v.extend([incr(0, 0xE00, 3), 1, x0 | x1 << 16, y0 | y1 << 16]),
        None => v.extend([incr(0, 0xE00, 1), 0]),
    }
    v.extend([incr(0, 0x19D0, 1), mask]);
    v
}

fn compare(words: Vec<u32>) {
    let Some(vk) = vulkan() else { return };
    let soft = run(Box::new(emu_gpu::backend::SoftwareBackend), &words);
    let hard = run(vk, &words);
    if let Some(i) = soft.iter().zip(&hard).position(|(a, b)| a != b) {
        panic!(
            "first difference at offset {:#x}: software {:#x}, vulkan {:#x}",
            i + 0x1_0000,
            soft[i],
            hard[i]
        );
    }
}

const RGBA8: u32 = 0xD5;
const BGRA8: u32 = 0xCF;
const PITCH: u32 = 1 << 12;

#[test]
fn pitch_target_with_scissor() {
    let mut w = target(0x1_0000, 64 * 4, 32, RGBA8, PITCH);
    w.extend(clear([1.0, 0.6, 0.2, 1.0], Some((3, 4, 40, 20)), 0x3C));
    compare(w);
}

#[test]
fn block_linear_targets_and_formats() {
    let mut w = Vec::new();
    // 100x50 BGRA, 4 GOBs high, full clear then a scissored one on top.
    w.extend(target(0x2_0000, 100, 50, BGRA8, 2 << 4));
    w.extend(clear([0.2, 0.4, 0.6, 1.0], None, 0x3C));
    w.extend(clear([0.6, 0.2, 1.0, 0.4], Some((10, 7, 60, 33)), 0x3C));
    // Other formats.
    for (i, (fmt, mem)) in [
        (0xE8u32, 1u32 << 4),
        (0xD1, 0),
        (0xC0, 1 << 4),
        (0xC6, PITCH),
        (0xF9, 0),
        (0xD6, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let width = if mem & PITCH != 0 { 64 * 8 } else { 64 };
        w.extend(target(0x10_0000 + 0x4_0000 * i as u64, width, 24, fmt, mem));
        w.extend(clear([0.6, 0.2, 0.4, 1.0], Some((5, 3, 50, 20)), 0x3C));
    }
    compare(w);
}

#[test]
fn masked_clear_falls_back_in_order() {
    let mut w = target(0x3_0000, 64, 16, RGBA8, 1 << 4);
    w.extend(clear([1.0, 1.0, 1.0, 1.0], None, 0x3C));
    w.extend(clear([0.0, 0.6, 0.0, 0.0], Some((8, 2, 30, 12)), 0x08)); // green only
    w.extend(clear([0.4, 0.0, 0.0, 1.0], Some((20, 6, 50, 14)), 0x3C));
    compare(w);
}
