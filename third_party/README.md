# Third-party code

| Directory | Project | License | Notes |
|---|---|---|---|
| `dynarmic/` | [dynarmic](https://github.com/azahar-emu/dynarmic) (Azahar fork) | 0BSD | AArch64 → x86-64 / arm64 JIT used by `emu-jit`. Bundled externals: fmt (MIT), mcl (MIT), robin-map (MIT), xbyak (BSD-3-Clause). |
| `boost/` | [Boost](https://www.boost.org) 1.88.0 | BSL-1.0 | Only the header subset dynarmic includes. |

Local changes to dynarmic:

- Zydis is not vendored; `src/dynarmic/common/x64_disassemble.cpp` is a stub,
  so dumping the emitted x86-64 code prints nothing useful.
- Tests, docs and unused externals (catch, oaknut, biscuit, zydis, zycore)
  were left out.

All of these licenses are compatible with the project's GPL-3.0-or-later.
