# switch2-emu

A learning-first, from-scratch emulator project for the Nintendo Switch 2 hardware architecture, written in Rust.

This project does not include, distribute, or require any copyrighted Nintendo firmware, keys, or games. The goal is the engineering of the core components: CPU, memory, GPU, and the OS service layer.

## Roadmap

1. **Foundation:** Cargo workspace, CI, and test setup. *(done)*
2. **Guest CPU state:** AArch64 register file and NZCV flags. *(done)*
3. **Decoder and interpreter:** Decode A64 instructions against the ARM Architecture Reference Manual, one instruction class at a time, cross-checked against Unicorn.
   - Done: integer instructions; scalar single/double floating point with ARM NaN rules; SIMD&FP loads and stores; common Advanced SIMD moves, logical ops and ADD/SUB.
   - To do: most Advanced SIMD arithmetic, half precision (FP16), LSE atomics, CRC32. Also FPCR rounding modes other than round-to-nearest, flush-to-zero, and FPSR exception flags.
4. **Memory and loader:** Guest virtual memory map and loading of homebrew/test ROMs. *(paged memory and ELF loader done; NSO/NRO to come)*
5. **Service layer (HLE):** Simple emulation of OS calls.
6. **JIT:** Dynamic translation from AArch64 to x86-64 after the interpreter is correct.
7. **GPU:** Parse the command stream and translate it to Vulkan.
8. **Audio, input, and performance.**

Accuracy comes before speed. The interpreter is the reference that the JIT and GPU work will be tested against.

## Layout

```
crates/
  emu-memory/  Guest virtual memory: 4 KiB pages, permissions, faults
  emu-cpu/     AArch64 CPU: register state, decoder, reference interpreter, FP rules
  emu-loader/  ELF loader (NSO/NRO later)
  emu-run/     Runs freestanding test programs (CLI: emu-run file.elf)
tests/programs/  C test programs, built with clang (ELF files are committed)
tools/
  asm.py       Assemble instructions with llvm-mc and print encodings
  difftest.py  Differential test against Unicorn on random instructions
  run_unicorn.py  Run a test program in Unicorn, to compare whole-program output
```

## Build and test

```sh
cargo test
cargo run --release -p emu-run -- tests/programs/hello.elf
cargo run --release -p emu-run -- tests/programs/float.elf
```

### How correctness is checked

1. **Unit tests** use encodings produced by `llvm-mc`, not written by hand.
2. **Differential testing** (`tools/difftest.py`) runs random instructions from every implemented encoding class in both this interpreter and [Unicorn](https://www.unicorn-engine.org/) (QEMU's ARM64 core), from the same random state, and compares registers, flags, PC and memory. Unicorn is a test-time tool only and is not part of the emulator.

   ```sh
   pip install unicorn
   cargo build --release --example difftest_runner
   python3 tools/difftest.py --cases 20000
   ```
   Floating-point registers start from a pool of hard cases (signed zeros, infinities, quiet and signalling NaNs with payloads, denormals, boundary values), with FPCR.DN on in a quarter of the cases.
3. **Whole programs:** C programs in `tests/programs` are compiled with clang, run to completion, and their output is compared with expected values and with a full run in Unicorn (`tools/run_unicorn.py`).

## References

- ARM Architecture Reference Manual for A-profile (ARM DDI 0487): instruction encodings and pseudocode.
- [oboromi](https://github.com/0xNikilite/oboromi) (GPL-3.0): used only as an architectural reference for how a Switch 2 emulator can be organized (multi-core CPU manager, HIPC/service layer, SM86 GPU decoding). No code is copied from it, so this project keeps its own MIT/Apache-2.0 license.
