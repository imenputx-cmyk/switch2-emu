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
    run_with(backend, words, |_| {}).0
}

/// Like `run`, with extra guest data written first; also returns the
/// backend.
fn run_with(
    backend: Box<dyn Backend>,
    words: &[u32],
    setup: impl Fn(&mut Memory),
) -> (Vec<u8>, Box<dyn Backend>) {
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
    setup(&mut mem);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    mem.write(GUEST, &bytes).unwrap();
    gpu.submit(c, &[GPU_VA | ((words.len() as u64) << 42)], 0, &mut mem);
    assert!(gpu.faults.is_empty(), "{:?}", gpu.faults);
    let mut out = vec![0u8; SIZE as usize - 0x1_0000];
    mem.read(GUEST + 0x1_0000, &mut out).unwrap();
    (out, gpu.backend)
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

#[test]
fn draws_with_translated_shaders() {
    use emu_gpu::shader::{asm, attr};
    // Shaders at +0x30000 (vertex) and +0x30200 (pixel); vertices at
    // +0x31000: position, then color, 32 bytes each.
    let vs = asm::program(
        &asm::vertex_header(),
        &[
            asm::ald(0, attr::generic(0), 4),
            asm::ald(4, attr::generic(1), 4),
            asm::fmul32i(4, 4, 0.75),
            asm::ast(0, attr::POSITION, 4),
            asm::ast(4, attr::generic(0), 4),
            asm::exit(),
        ],
    );
    let ps = asm::program(
        &asm::pixel_header(),
        &[
            asm::ipa(0, 0x80),
            asm::ipa(1, 0x84),
            asm::ipa(2, 0x88),
            asm::mov32i(3, 0x3F80_0000),
            asm::fadd_imm(1, 1, 0.25),
            asm::exit(),
        ],
    );
    shaded_draw(vs, ps, 8);
}

#[test]
fn control_flow_shaders() {
    use emu_gpu::shader::{asm, attr};
    let vs = asm::program(
        &asm::vertex_header(),
        &[
            asm::ald(0, attr::generic(0), 4),
            asm::ald(4, attr::generic(1), 4),
            asm::ast(0, attr::POSITION, 4),
            asm::ast(4, attr::generic(0), 4),
            asm::exit(),
        ],
    );
    shaded_draw(vs, asm::control_flow_sample(), 16);
}

/// Draws a triangle and a strip with the given shaders in software and
/// on Vulkan and compares the results.
fn shaded_draw(vs: Vec<u8>, ps: Vec<u8>, max_differing: usize) {
    let Some(vk) = vulkan() else { return };
    let verts: Vec<f32> = vec![
        -0.9, -0.8, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, //
        0.8, -0.9, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, // w = 2: perspective
        0.1, 0.9, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, -0.5, 0.95, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
    ];
    let setup = move |mem: &mut Memory| {
        mem.write(GUEST + 0x3_0000, &vs).unwrap();
        mem.write(GUEST + 0x3_0200, &ps).unwrap();
        let bytes: Vec<u8> = verts.iter().flat_map(|f| f.to_le_bytes()).collect();
        mem.write(GUEST + 0x3_1000, &bytes).unwrap();
    };
    let code = GPU_VA + 0x3_0000;
    let vtx = GPU_VA + 0x3_1000;
    let mut w = target(0x4_0000, 96, 64, RGBA8, 2 << 4);
    w.extend(clear([0.2, 0.2, 0.2, 1.0], None, 0x3C));
    w.extend([
        incr(0, 0x1608, 2),
        (code >> 32) as u32,
        code as u32,
        incr(0, 0x2000 + 64, 2),
        1 | 1 << 4,
        0,
        incr(0, 0x2000 + 5 * 64, 2),
        1 | 5 << 4,
        0x200,
        incr(0, 0x1160, 2),
        1 << 21 | 7 << 27,
        16 << 7 | 1 << 21 | 7 << 27,
        incr(0, 0x1C00, 3),
        32 | 1 << 12,
        (vtx >> 32) as u32,
        vtx as u32,
        incr(0, 0xA00, 6),
        48.0f32.to_bits(),
        (-32.0f32).to_bits(),
        1.0f32.to_bits(),
        48.0f32.to_bits(),
        32.0f32.to_bits(),
        0,
        incr(0, 0x121C, 1),
        1,
        incr(0, 0xE00, 3),
        1,
        4 | 90 << 16,
        2 | 60 << 16, // scissor
        // A triangle, then a strip over vertices 1..4.
        incr(0, 0x1618, 1),
        4,
        incr(0, 0xD74, 2),
        0,
        3,
        incr(0, 0x1614, 1),
        0,
        incr(0, 0x1214, 1),
        1 | 3 << 16 | 5 << 28,
    ]);
    let (soft, _) = run_with(
        Box::new(emu_gpu::backend::SoftwareBackend),
        &w,
        setup.clone(),
    );
    let (hard, backend) = run_with(vk, &w, setup);
    let vkb = backend.as_any().downcast_ref::<VulkanBackend>().unwrap();
    assert_eq!(
        (vkb.gpu_draws, vkb.fallback_draws),
        (2, 0),
        "draws must run on the GPU"
    );
    // Rasterization and interpolation may differ slightly between
    // implementations: compare the render target with a tolerance.
    let rt = 0x4_0000 - 0x1_0000;
    let size = emu_gpu::blocklinear::size(96 * 4, 64, 2);
    let (a, b) = (&soft[rt..rt + size], &hard[rt..rt + size]);
    let worst = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
    let differing = a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| x != y).count();
    eprintln!("software vs Vulkan: {differing} pixels differ, by at most {worst}");
    assert!(
        worst <= 2 || differing <= max_differing,
        "worst {worst}, {differing} pixels differ"
    );
    assert!(
        a.chunks(4).filter(|p| p[..3] != [51, 51, 51]).count() > 500,
        "the draw covered little"
    );
}

#[test]
fn indexed_draws_with_constant_buffers() {
    use emu_gpu::shader::{asm, attr};
    let Some(vk) = vulkan() else { return };
    // VS: position.xy = attr0.xy * c0[0].xy + c0[0].zw; color = attr1.
    // PS: color * c3[1] (a tint).
    let vs = asm::program(
        &asm::vertex_header(),
        &[
            asm::ald(0, attr::generic(0), 4),
            asm::ald(4, attr::generic(1), 4),
            asm::mov_cbuf(8, 0, 8),
            asm::mov_cbuf(9, 0, 12),
            asm::ffma_cbuf(0, 0, 0, 0, 8),
            asm::ffma_cbuf(1, 1, 0, 4, 9),
            asm::ast(0, attr::POSITION, 4),
            asm::ast(4, attr::generic(0), 4),
            asm::exit(),
        ],
    );
    let ps = asm::program(
        &asm::pixel_header(),
        &[
            asm::ipa(0, 0x80),
            asm::ipa(1, 0x84),
            asm::ipa(2, 0x88),
            asm::ipa(3, 0x8C),
            asm::fmul_cbuf(0, 0, 3, 0x10),
            asm::fmul_cbuf(1, 1, 3, 0x14),
            asm::fmul_cbuf(2, 2, 3, 0x18),
            asm::exit(),
        ],
    );
    // 2 dummy vertices, then a quad's 4 corners (base vertex 2).
    let verts: Vec<f32> = vec![
        9.0, 9.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, //
        9.0, 9.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, //
        -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, //
        1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, //
        1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, //
        -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let indices: [u16; 8] = [7, 7, 0, 1, 2, 0, 2, 3]; // first = 2
    let setup = move |mem: &mut Memory| {
        mem.write(GUEST + 0x3_0000, &vs).unwrap();
        mem.write(GUEST + 0x3_0200, &ps).unwrap();
        let bytes: Vec<u8> = verts.iter().flat_map(|f| f.to_le_bytes()).collect();
        mem.write(GUEST + 0x3_1000, &bytes).unwrap();
        let ib: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        mem.write(GUEST + 0x3_2000, &ib).unwrap();
    };
    let (code, vtx, ib, cb0, cb3) = (
        GPU_VA + 0x3_0000,
        GPU_VA + 0x3_1000,
        GPU_VA + 0x3_2000,
        GPU_VA + 0x3_3000,
        GPU_VA + 0x3_4000,
    );
    let f = |v: f32| v.to_bits();
    let mut w = target(0x4_0000, 96, 64, RGBA8, 2 << 4);
    w.extend(clear([0.2, 0.2, 0.2, 1.0], None, 0x3C));
    w.extend([
        incr(0, 0x1608, 2),
        (code >> 32) as u32,
        code as u32,
        incr(0, 0x2000 + 64, 2),
        1 | 1 << 4,
        0,
        incr(0, 0x2000 + 5 * 64, 2),
        1 | 5 << 4,
        0x200,
        incr(0, 0x1160, 2),
        1 << 21 | 7 << 27,
        16 << 7 | 1 << 21 | 7 << 27,
        incr(0, 0x1C00, 3),
        32 | 1 << 12,
        (vtx >> 32) as u32,
        vtx as u32,
        incr(0, 0xA00, 6),
        f(48.0),
        f(-32.0),
        f(1.0),
        f(48.0),
        f(32.0),
        0,
        incr(0, 0x121C, 1),
        1,
        incr(0, 0xE00, 1),
        0,
        // Constant buffer 0 for the VS: scale (0.5, 0.75), offset (0.2, -0.1),
        // uploaded through LOAD_CONSTANT_BUFFER.
        incr(0, 0x2380, 4),
        256,
        (cb0 >> 32) as u32,
        cb0 as u32,
        0,
        incr(0, 0x2390, 4),
        f(0.5),
        f(0.75),
        f(0.2),
        f(-0.1),
        incr(0, 0x2410, 1),
        1, // stage 0 (VS), slot 0, valid
        // Constant buffer 3 for the PS: tint at offset 0x10.
        incr(0, 0x2380, 4),
        256,
        (cb3 >> 32) as u32,
        cb3 as u32,
        0x10,
        incr(0, 0x2390, 3),
        f(1.0),
        f(0.5),
        f(0.25),
        incr(0, 0x2410 + 4 * 32, 1),
        1 | 3 << 4, // stage 4 (PS), slot 3
        // Index buffer: 16-bit at ib, base vertex 2; draw 6 indices from 2.
        incr(0, 0x17C8, 2),
        (ib >> 32) as u32,
        ib as u32,
        incr(0, 0x17D8, 2),
        1,
        2,
        incr(0, 0x1434, 1),
        2,
        incr(0, 0x1618, 1),
        4,
        incr(0, 0x17E0, 1),
        6,
        incr(0, 0x1614, 1),
        0,
    ]);
    let (soft, _) = run_with(
        Box::new(emu_gpu::backend::SoftwareBackend),
        &w,
        setup.clone(),
    );
    let (hard, backend) = run_with(vk, &w, setup);
    let vkb = backend.as_any().downcast_ref::<VulkanBackend>().unwrap();
    assert_eq!(
        (vkb.gpu_draws, vkb.fallback_draws),
        (1, 0),
        "the draw must run on the GPU"
    );
    let rt = 0x4_0000 - 0x1_0000;
    let size = emu_gpu::blocklinear::size(96 * 4, 64, 2);
    let (a, b) = (&soft[rt..rt + size], &hard[rt..rt + size]);
    let worst = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
    let differing = a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| x != y).count();
    eprintln!("software vs Vulkan: {differing} pixels differ, by at most {worst}");
    assert!(
        worst <= 2 || differing <= 8,
        "worst {worst}, {differing} pixels differ"
    );
    // The quad covers x 33.6..81.6 and y 11.2..59.2, and is
    // tinted: red scaled by 1, green by 0.5, blue by 0.25.
    let lin = emu_gpu::blocklinear::deswizzle_width(a, 96 * 4, 96 * 4, 64, 2);
    let px = |x: usize, y: usize| &lin[(y * 96 + x) * 4..(y * 96 + x) * 4 + 4];
    assert_eq!(px(20, 32), [51, 51, 51, 255]);
    let corner = px(35, 57); // near the (-1,-1) vertex (33.6, 59.2): red
    assert!(corner[0] > 200 && corner[1] < 30, "{corner:?}");
    let white = px(35, 13); // near the (-1, 1) vertex (33.6, 11.2): white, tinted
    assert!(
        white[0] > 230 && (115..140).contains(&white[1]) && (55..75).contains(&white[2]),
        "{white:?}"
    );
}
