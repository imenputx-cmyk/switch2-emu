//! Runs the compiled test programs in tests/programs and checks their output.

use std::path::PathBuf;

fn program(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/programs")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn hello_program_runs_to_completion() {
    let result = emu_run::run_elf(&program("hello.elf"), 10_000_000).expect("program runs");
    // Expected values checked independently (Python) and by running the
    // same ELF in Unicorn, which produced identical output.
    let expected = "\
Hello from the guest CPU!
fib(20) = 6765
factorial(20) = 2432902008176640000
primes below 256 = 54
sorted = -123456789012 -7 0 5 5 42 99 1000000007
fnv1a(\"switch2-emu\") = 0xca4822b387ad65f5
apply(sub, 10, 52) = -42
apply(divide, -100, 7) = -14
divide by zero = 0
";
    assert_eq!(String::from_utf8(result.stdout).unwrap(), expected);
    assert_eq!(result.exit_code, 7);
}

#[test]
fn float_program_runs_to_completion() {
    let result = emu_run::run_elf(&program("float.elf"), 10_000_000).expect("program runs");
    let out = String::from_utf8(result.stdout).unwrap();
    // Output is identical to a full run in Unicorn (tools/run_unicorn.py);
    // the physics loop was also recomputed in Python.
    let expected_tail = "\
sqrt(2): fsqrt=1.414214 newton=1.414214
matmul row 2: 3.500000 2.000000 -9.000000
ball lands after 488 steps at v=-37.895735
conversions: -3 9223372036854775807 0 0.100000
special: 1/0 > 1e308 is true, 0/0 is nan
";
    assert!(out.ends_with(expected_tail), "unexpected output:\n{out}");
    // The Mandelbrot set is drawn with its widest row in the middle.
    let rows: Vec<&str> = out.lines().take(12).collect();
    assert_eq!(rows[6].trim_start().chars().next(), Some('@'), "{out}");
    assert_eq!(result.exit_code, 0);
}

#[test]
fn simd_program_runs_to_completion() {
    // NEON intrinsics and auto-vectorized loops built for the Cortex-A78C.
    // Output is identical to a full run in Unicorn; sum_bytes, the
    // saturating add, the table lookup and the dot product were also
    // checked by hand, and the half-precision results with numpy.float16.
    let result = emu_run::run_elf(&program("simd.elf"), 10_000_000).expect("program runs");
    let expected = "\
saxpy checksum   = 0xdacce6e5ee0dfec5
sum_bytes        = 127572
grayscale        = 0xf7a16afd6521da21
vqaddq_s16       = 32767 -32768 200 -200 32767 -32768 0 2
normalize4       = 0xe836c9497104989e
first normal x*1000 = 599
tbl              = svool_zlyoxnqk_t
vdotq_u32        = 1030 1078 1126 1174
harmonic(20) in fp16 = 3.595703
fp16 fma        = 0.149902 1501.000000 -inf 0.000000
";
    assert_eq!(String::from_utf8(result.stdout).unwrap(), expected);
    assert_eq!(result.exit_code, 0);
}

#[test]
fn crypto_program_matches_published_test_vectors() {
    // AES-128: FIPS-197 Appendix C.1. SHA-256/SHA-1 of "abc": FIPS 180
    // examples (also Python's hashlib). Implemented with the AESE/AESD/
    // AESMC/AESIMC, SHA1*, SHA256* and PMULL instructions.
    let result = emu_run::run_elf(&program("crypto.elf"), 10_000_000).expect("program runs");
    let expected = "\
aes128 encrypt = 69c4e0d86a7b0430d8cdb78070b4c55a
aes128 decrypt = 00112233445566778899aabbccddeeff
sha256(abc)    = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
sha1(abc)      = a9993e364706816aba3e25717850c26c9cd0d89d
pmull64        = 0x000000000000007d 0x000000000000007d
";
    assert_eq!(String::from_utf8(result.stdout).unwrap(), expected);
}

#[test]
fn homebrew_nro_boots_through_the_homebrew_abi() {
    // tests/programs/homebrew/hb_hello.c, converted with tools/elf2nro.py.
    // Result codes follow libnx (module 1, description << 9): 0xF001
    // InvalidEnumValue, 0xE401 InvalidHandle, 0xCA01 InvalidSize. The heap
    // checksum was recomputed in Python.
    let nro = program("hb_hello.nro");
    let r = emu_run::run_nro(&nro, 10_000_000, false).expect("boots");
    assert_eq!(r.stop, emu_horizon::Stop::ReturnedToLoader(42));
    let expected = "\
Hello from homebrew!
relocated pointers work
entry x1 = 0xffffffffffffffff
config MainThreadHandle = 0x8000
config AppletType = 0x0
config HosVersion = 0x150000
loader: switch2-emu
heap region = 0x1000000000
heap region size = 0x180000000
aslr region = 0x8000000
GetInfo(bad type) = 0xf001
GetInfo(bad handle) = 0xe401
SetHeapSize(0x123) = 0xca01
SetHeapSize(4 MiB) = 0x0
heap = 0x1000000000
heap checksum = 0x2ff40000
code:  type 0x3 perm 0x5
stack: type 0xb perm 0x3 size 0x100000
heap:  type 0x5 size 0x400000
MapMemory = 0x0
source while mapped: perm 0x0 attr 0x1
UnmapMemory = 0x0
value seen through heap = 0x1234567890abcdef
tick advanced = 0x1
Returning 42 to the loader
";
    assert_eq!(String::from_utf8(r.debug_output).unwrap(), expected);
}

#[test]
fn homebrew_ipc_with_libnx_client_code() {
    // tests/programs/homebrew/hb_ipc.c uses libnx's own header-only IPC
    // code (service.h/cmif.h/hipc.h), so these are the exact messages real
    // homebrew sends: sm: RegisterClient (with PID), GetServiceHandle (move
    // handle), set:sys GetFirmwareVersion2 (fixed-size C buffer), domain
    // conversion and domain messages, close. Error codes: 0xF201 kernel
    // NotFound, 0xE15 sm NotRegistered, 0xE401 kernel InvalidHandle.
    let r = emu_run::run_nro(&program("hb_ipc.nro"), 10_000_000, false).expect("boots");
    assert_eq!(r.stop, emu_horizon::Stop::ReturnedToLoader(0));
    let expected = "\
ConnectToNamedPort(sm:) = 0x0
ConnectToNamedPort(nope) = 0xf201
sm pointer buffer size = 0x8000
sm RegisterClient = 0x0
sm GetServiceHandle(set:sys) = 0x0
sm GetServiceHandle(nosuch) = 0xe15
set:sys GetFirmwareVersion2 = 0x0
firmware major = 0x15
display version = 21.0.0
display title = NintendoSDK Firmware for NX 21.0.0-1.0 (switch2-emu)
ConvertToDomain = 0x0
domain object id = 0x1
domain GetFirmwareVersion = 0x0
domain display version = 21.0.0
CloseHandle(closed handle) = 0xe401
IPC test done
";
    assert_eq!(String::from_utf8(r.debug_output).unwrap(), expected);
}

#[test]
fn libnx_application_startup_and_sd_card() {
    // tests/programs/libnx_compat/hb_libnx.c links libnx's own service
    // clients (built from source), so this is the full IPC sequence of a
    // libnx application's startup: apm, appletOE and its proxies (events,
    // focus state), hid (shared memory), time:u (clocks read from shared
    // memory), fsp-srv (domain + cloned sessions). Then it uses the SD
    // card, which is a host directory. FS results: 0x402 PathAlreadyExists,
    // 0x202 PathNotFound, 0x2EE202 InvalidPath.
    let sd = std::env::temp_dir().join(format!("switch2-emu-sd-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&sd);
    std::fs::create_dir_all(sd.join("games")).unwrap();
    std::fs::write(sd.join("hello.txt"), "hello from the host SD card\n").unwrap();
    let config = emu_horizon::services::Config {
        sdmc_root: Some(sd.clone()),
        boot_time: Some(1_800_000_000),
        ..Default::default()
    };
    let r =
        emu_run::run_nro_with(&program("hb_libnx.nro"), 10_000_000, false, config).expect("boots");
    let written = std::fs::read_to_string(sd.join("save/data.bin"));
    std::fs::remove_dir_all(&sd).unwrap();
    assert_eq!(r.stop, emu_horizon::Stop::ReturnedToLoader(0));
    let expected = "\
hosversion = 0x150000
svcSetHeapSize = 0x0
smInitialize = 0x0
appletInitialize = 0x0
hidInitialize = 0x0
timeInitialize = 0x0
fsInitialize = 0x0
applet type = 0x0
timeGetCurrentTime = 0x0
posix time = 0x6b49d200
fsOpenSdCardFileSystem = 0x0
OpenFile(/hello.txt) = 0x0
Read = 0x0
contents: hello from the host SD card
CreateDirectory = 0x0
CreateFile = 0x0
CreateFile again = 0x402
OpenFile(rw+append) = 0x0
Write = 0x0
GetSize = 0x0
size = 0x10
Read back = 0x0
tail: by libnx
OpenDirectory = 0x0
ReadDirectory = 0x0
  dir  games
  file hello.txt
  dir  save
OpenDirectory(/../escape) = 0x2ee202
OpenFile(/missing.txt) = 0x202
";
    assert_eq!(String::from_utf8(r.debug_output).unwrap(), expected);
    assert_eq!(written.unwrap(), "written by libnx");
}

#[test]
fn threads_and_synchronization() {
    // tests/programs/libnx_compat/hb_threads.c: four guest threads using
    // libnx's mutex (ArbitrateLock/Unlock), condvar (process-wide keys),
    // semaphore and light event (address arbiter), joined with
    // WaitSynchronization. 0xEA01 (59905) is TimedOut.
    let r = emu_run::run_nro(&program("hb_threads.nro"), 50_000_000, false).expect("boots");
    assert_eq!(r.stop, emu_horizon::Stop::ReturnedToLoader(0));
    assert_eq!(
        String::from_utf8(r.debug_output).unwrap(),
        "\
workers past the event before signal: 0
workers past the event: 4
counter (expect 2000): 2000
consumed sum (expect 5050): 5050
join rc: 0
slept at least 5 ms: 1
condvar timeout rc: 59905
mutex held again: 1
"
    );
}

#[test]
fn framebuffer_reaches_the_display() {
    // tests/programs/libnx_compat/hb_gfx.c draws three 1280x720 frames
    // with libnx's framebuffer API: vi (display, managed layer), the
    // binder buffer queue, nvdrv/nvmap memory, and libnx's block-linear
    // swizzle, which the emulator undoes when presenting.
    let r = emu_run::run_nro(&program("hb_gfx.nro"), 1_000_000_000, false).expect("boots");
    assert_eq!(r.stop, emu_horizon::Stop::ReturnedToLoader(0));
    let out = String::from_utf8(r.debug_output).unwrap();
    assert!(out.ends_with("presented frame 0x2\n"), "{out}");
    assert!(
        !out.lines()
            .any(|l| l.contains("= 0x") && !l.ends_with("= 0x0")),
        "{out}"
    );
    assert_eq!(r.frames.len(), 3);
    let (w, h) = (1280u32, 720u32);
    for (frame, f) in r.frames.iter().enumerate() {
        let frame = frame as u32;
        assert_eq!((f.width, f.height), (w, h));
        for y in 0..h {
            for x in 0..w {
                // The pattern hb_gfx.c draws.
                let mut b = frame * 100;
                if ((x + frame * 64) / 80 + y / 80).is_multiple_of(2) {
                    b = 255;
                }
                let expected = [(x * 255 / w) as u8, (y * 255 / h) as u8, b as u8, 0xFF];
                let at = ((y * w + x) * 4) as usize;
                assert_eq!(f.rgba[at..at + 4], expected, "frame {frame} at ({x}, {y})");
            }
        }
    }
}

#[test]
fn breakout_game_with_input() {
    // tests/programs/libnx_compat/hb_breakout.c: a small game on the
    // display path and hid input (libnx pad API). The test is the player:
    // it holds Right for a while (which also ends the demo mode), then
    // presses Plus to quit, and checks that the paddle followed.
    use emu_horizon::services::hid::{button, InputState};
    use emu_horizon::services::Frame;
    use std::cell::RefCell;
    use std::rc::Rc;

    const W: usize = 320;
    const PADDLE_Y: usize = 180 - 180 / 12;
    // Centre of the white paddle on its row.
    fn paddle_x(f: &Frame) -> f64 {
        let row = &f.rgba[PADDLE_Y * W * 4..(PADDLE_Y + 1) * W * 4];
        let xs: Vec<usize> = (0..W)
            .filter(|&x| row[x * 4] >= 230 && row[x * 4 + 2] >= 230)
            .collect();
        assert!(!xs.is_empty(), "no paddle");
        xs.iter().sum::<usize>() as f64 / xs.len() as f64
    }
    let positions = Rc::new(RefCell::new(Vec::new()));
    let seen = positions.clone();
    let player = Box::new(move |f: &Frame, input: &mut InputState| {
        assert_eq!((f.width, f.height), (320, 180));
        let mut seen = seen.borrow_mut();
        seen.push(paddle_x(f));
        *input = InputState {
            buttons: match seen.len() {
                20..=79 => button::RIGHT,
                120.. => button::PLUS,
                _ => 0,
            },
            ..Default::default()
        };
        true
    });
    let r = emu_run::run_nro_presenting(
        &program("hb_breakout.nro"),
        500_000_000,
        false,
        Default::default(),
        Some(player),
    )
    .expect("boots");
    assert_eq!(
        r.stop,
        emu_horizon::Stop::ReturnedToLoader(0),
        "{:?}",
        String::from_utf8_lossy(&r.debug_output)
    );
    let out = String::from_utf8(r.debug_output).unwrap();
    assert!(out.starts_with("breakout: score "), "{out}");
    let p = positions.borrow();
    assert!(
        (120..130).contains(&p.len()),
        "quit after {} frames",
        p.len()
    );
    // Holding Right drives the paddle to the right wall.
    let right_wall = (W - 320 / 7 / 2) as f64;
    assert!(p[18] < right_wall - 40.0, "{}", p[18]);
    assert!((p[85] - right_wall).abs() < 2.0, "{}", p[85]);
}

#[test]
fn gpu_commands_and_rendered_frames() {
    // tests/programs/libnx_compat/hb_gpu.c submits pushbuffers through
    // libnx's nvidia stack (nvhost-as-gpu, nvhost-gpu, syncpoint fences):
    // copy engine fill and block-linear round trip, inline uploads and a
    // semaphore, then frames made of 3D-engine clears into the display's
    // framebuffers.
    let r = emu_run::run_nro(&program("hb_gpu.nro"), 200_000_000, false).expect("boots");
    assert_eq!(r.stop, emu_horizon::Stop::ReturnedToLoader(0));
    let out = String::from_utf8(r.debug_output).unwrap();
    let expected = "\
framebufferCreate = 0x0
nvGpuInit = 0x0
3D class = 0xb197
nvAddressSpaceCreate = 0x0
nvGpuChannelCreate = 0x0
map cmd = 0x0
map scratch = 0x0
map framebuffer = 0x0
submit = 0x0
fill word 0 = 0x12345678
fill word 63 = 0x12345678
fill word 64 = 0x0
inline 0 = 0xaaaa0001
inline 2 = 0xcccc0003
block-linear round trip = 0x1
semaphore = 0x5e3a
gpu frame 0x0
gpu frame 0x1
gpu frame 0x2
";
    assert_eq!(out, expected);
    assert_eq!(r.frames.len(), 3);
    let px = |f: &emu_horizon::services::Frame, x: u32, y: u32| {
        let at = ((y * f.width + x) * 4) as usize;
        [f.rgba[at], f.rgba[at + 1], f.rgba[at + 2], f.rgba[at + 3]]
    };
    let unorm = |v: f32| (v * 255.0).round() as u8;
    for (i, f) in r.frames.iter().enumerate() {
        let i = i as u32;
        assert_eq!((f.width, f.height), (1280, 720));
        assert_eq!(px(f, 5, 5), [unorm(0.05), unorm(0.05), unorm(0.15), 255]);
        assert_eq!(px(f, 640, 200), [unorm(0.9), unorm(0.2), unorm(0.2), 255]);
        assert_eq!(px(f, 1279, 559), [unorm(0.2), unorm(0.4), unorm(0.95), 255]);
        // The yellow square moves 300 pixels per frame.
        assert_eq!(
            px(f, 100 + i * 300, 260),
            [255, unorm(0.9), unorm(0.2), 255]
        );
        assert_eq!(
            px(f, 300 + i * 300, 260),
            [unorm(0.05), unorm(0.05), unorm(0.15), 255]
        );
    }
}

#[test]
fn gpu_frames_match_on_vulkan() {
    // The same program on the Vulkan backend: frames must match the
    // software backend's (within 1 per channel: float-to-unorm rounding
    // of clear colors may differ between Vulkan implementations).
    // Skipped when there is no Vulkan 1.3 device.
    if let Err(e) = emu_gpu_vk::VulkanBackend::new() {
        eprintln!("skipping: {e}");
        return;
    }
    for name in ["hb_gpu.nro", "hb_triangle.nro"] {
        let nro = program(name);
        let soft = emu_run::run_nro(&nro, 200_000_000, false).expect("boots");
        let config = emu_horizon::services::Config {
            gpu_backend: emu_horizon::services::GpuBackend::Vulkan,
            ..Default::default()
        };
        let vk = emu_run::run_nro_with(&nro, 200_000_000, false, config).expect("boots");
        assert_eq!(vk.stop, emu_horizon::Stop::ReturnedToLoader(0), "{name}");
        assert_eq!(vk.debug_output, soft.debug_output, "{name}");
        assert_eq!(vk.frames.len(), soft.frames.len(), "{name}");
        for (a, b) in soft.frames.iter().zip(&vk.frames) {
            assert_eq!((a.width, a.height), (b.width, b.height));
            let worst = a
                .rgba
                .iter()
                .zip(&b.rgba)
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap();
            assert!(worst <= 1, "{name}: frames differ by {worst}");
        }
    }
}

#[test]
fn shaded_triangles() {
    // tests/programs/libnx_compat/hb_triangle.c draws with Maxwell vertex
    // and pixel shaders (triangle_shaders.h) from a vertex buffer: a
    // triangle with red/green/blue corners, then (frame 1) moved right
    // plus a strip quad drawn with the single-method draw.
    let r = emu_run::run_nro(&program("hb_triangle.nro"), 200_000_000, false).expect("boots");
    assert_eq!(r.stop, emu_horizon::Stop::ReturnedToLoader(0));
    assert!(String::from_utf8(r.debug_output)
        .unwrap()
        .ends_with("triangle frame 0x1\n"));
    assert_eq!(r.frames.len(), 2);
    let px = |f: &emu_horizon::services::Frame, x: u32, y: u32| {
        let at = ((y * f.width + x) * 4) as usize;
        [f.rgba[at], f.rgba[at + 1], f.rgba[at + 2], f.rgba[at + 3]]
    };
    let near = |got: [u8; 4], want: [u8; 4], tol: u8| {
        got.iter().zip(want).all(|(g, w)| g.abs_diff(w) <= tol)
    };
    let bg = [26, 26, 31, 255];
    for (frame, dx) in [(0usize, 0u32), (1, 160)] {
        let f = &r.frames[frame];
        assert_eq!(px(f, 5, 700), bg);
        // Centroid: an even mix of the three corners.
        let c = px(f, 640 + dx, 408);
        assert!(
            near(c, [85, 85, 85, 255], 2),
            "frame {frame} centroid {c:?}"
        );
        // Near the corners: mostly red, green, blue.
        let top = px(f, 640 + dx, 80);
        assert!(top[0] > 240 && top[1] < 15 && top[2] < 15, "{top:?}");
        let left = px(f, 200 + dx, 572);
        assert!(left[1] > 200 && left[0] < 40, "{left:?}");
        let right = px(f, 1080 + dx, 572);
        assert!(right[2] > 200 && right[0] < 40, "{right:?}");
        // Outside the triangle, beside its apex.
        assert_eq!(px(f, 560 + dx, 80), bg);
    }
    // The strip quad (frame 1 only): yellow on the left, magenta on the right.
    assert_eq!(px(&r.frames[0], 40, 70), bg);
    assert!(near(px(&r.frames[1], 33, 70), [255, 255, 0, 255], 3));
    assert!(near(px(&r.frames[1], 286, 70), [255, 0, 255, 255], 3));
}
