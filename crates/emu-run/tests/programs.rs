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
