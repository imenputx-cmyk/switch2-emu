# switch2-emu

A learning-first, from-scratch emulator project for the Nintendo Switch 2 hardware architecture, written in Rust.

This project does not include, distribute, or require any copyrighted Nintendo firmware, keys, or games. The goal is the engineering of the core components: CPU, memory, GPU, and the OS service layer.

## Roadmap

1. **Foundation:** Cargo workspace, CI, and test setup. *(done)*
2. **Guest CPU state:** AArch64 register file and NZCV flags. *(done)*
3. **Decoder and interpreter:** Decode A64 instructions against the ARM Architecture Reference Manual, one instruction class at a time, cross-checked against Unicorn.
   - Done: the integer instruction set (including LSE atomics, CAS/CASP, LDXP/STXP, LDAPR, CRC32); scalar single/double floating point with ARM NaN rules; SIMD&FP loads and stores including LD1-LD4/ST1-ST4; Advanced SIMD integer and single/double floating-point arithmetic, widening/narrowing, saturating (with FPSR.QC), shifts, permutes, TBL/TBX, reductions, by-element forms, dot product, and the ARM reciprocal (square root) estimates.
   - Half precision (FP16) in scalar and vector forms, and the Cryptographic Extension (AES, SHA-1, SHA-256, 64-bit PMULL), checked against the FIPS test vectors. System instructions used from user code: DC ZVA, cache maintenance by address, IC IVAU, and the EL0 counters (CNTVCT/CNTPCT).
   - To do: FPCR rounding modes other than round-to-nearest, flush-to-zero, and the FPSR exception flags. Not targeted, because the Cortex-A78C does not have them: FCMLA/FCADD, FRINT32/64, BF16, LSE2.
4. **Memory and loader:** Guest virtual memory map and loading of homebrew/test ROMs. *(paged memory, ELF and NRO loaders done; NSO to come)*
5. **Kernel and services (HLE):** Emulation of Horizon's SVCs and IPC services. *(in progress: homebrew boots through the homebrew ABI; memory, info, debug output and single-thread synchronization SVCs work; CMIF IPC with domains works, with the `sm:` and `set:sys` services)*
6. **JIT:** Dynamic translation from AArch64 to x86-64 after the interpreter is correct.
7. **GPU:** Parse the command stream and translate it to Vulkan.
8. **Audio, input, and performance.**

Accuracy comes before speed. The interpreter is the reference that the JIT and GPU work will be tested against.

## Layout

```
crates/
  emu-memory/  Guest virtual memory: 4 KiB pages, permissions, faults
  emu-cpu/     AArch64 CPU: register state, decoder, reference interpreter, FP rules
  emu-loader/  ELF and NRO loaders (NSO later)
  emu-horizon/ Horizon kernel HLE: memory state map, handles, SVCs, homebrew boot,
               HIPC/CMIF IPC (sessions, domains) and HLE services
  emu-run/     Runs freestanding test programs (CLI: emu-run file.elf)
tests/programs/  C test programs, built with clang (ELF files are committed)
tools/
  asm.py       Assemble instructions with llvm-mc and print encodings
  difftest.py  Differential test against Unicorn on random instructions
  run_unicorn.py  Run a test program in Unicorn, to compare whole-program output
  elf2nro.py   Convert a position-independent ELF to an NRO (homebrew format)
```

## Build and test

```sh
cargo test
cargo run --release -p emu-run -- tests/programs/hello.elf
cargo run --release -p emu-run -- tests/programs/float.elf
cargo run --release -p emu-run -- tests/programs/simd.elf
cargo run --release -p emu-run -- tests/programs/crypto.elf
cargo run --release -p emu-run -- tests/programs/hb_hello.nro   # homebrew, real Switch SVCs
cargo run --release -p emu-run -- tests/programs/hb_ipc.nro     # IPC with libnx's client code
```

### How correctness is checked

1. **Unit tests** use encodings produced by `llvm-mc`, not written by hand.
2. **Differential testing** (`tools/difftest.py`) runs random instructions from every implemented encoding class in both this interpreter and [Unicorn](https://www.unicorn-engine.org/) (QEMU's ARM64 core), from the same random state, and compares registers, flags, PC and memory. Unicorn is a test-time tool only and is not part of the emulator.

   ```sh
   pip install unicorn
   cargo build --release --example difftest_runner
   python3 tools/difftest.py --cases 20000
   ```
   The target CPU is the Switch 2's Cortex-A78C. A census of 60,000 random encodings found no instruction that LLVM's Cortex-A78C model accepts and this interpreter rejects; the remaining encodings Unicorn executes are extensions the A78C lacks (FCMLA/FCADD, FRINT32/64, FMLAL, FJCVTZS) or invalid ones. Note that Unicorn runs code at EL1, so it also allows EL1 system registers that user code can never read. Where Unicorn's "max" model differs (it has FEAT_LSE2), where the architecture leaves a choice to the implementation, or where Unicorn has a known bug (signed atomic min/max on narrow sizes, checked against a reference model instead), the harness reports the case in its own named category instead of hiding it.

   Floating-point registers start from a pool of hard cases (signed zeros, infinities, quiet and signalling NaNs with payloads, denormals, boundary values), with FPCR.DN on in a quarter of the cases.
3. **Whole programs:** C programs in `tests/programs` are compiled with clang, run to completion, and their output is compared with expected values and with a full run in Unicorn (`tools/run_unicorn.py`).

## References

- ARM Architecture Reference Manual for A-profile (ARM DDI 0487): instruction encodings and pseudocode.
- [switchbrew wiki](https://switchbrew.org/wiki/): the NRO format, the homebrew ABI, SVC numbers, arguments, `GetInfo` types, `MemoryInfo` layout and memory states.
- [libnx](https://github.com/switchbrew/libnx) (ISC): read to learn what homebrew expects (config entries, region queries, TLS layout, IPC message layout, result codes). No libnx code is copied into the emulator. The IPC test homebrew (`tests/programs/hb_ipc.nro`) is compiled against libnx's header-only IPC code on purpose, so the emulator is tested with the exact messages real homebrew sends; its license is in `tests/programs/LICENSE-libnx.md`.
- Ryujinx (MIT) and yuzu: overall structure of a kernel HLE. Only used as background; no code is copied.
- [oboromi](https://github.com/0xNikilite/oboromi) (GPL-3.0): used only as an architectural reference for how a Switch 2 emulator can be organized (multi-core CPU manager, HIPC/service layer, SM86 GPU decoding). No code is copied from it, so this project keeps its own MIT/Apache-2.0 license.
