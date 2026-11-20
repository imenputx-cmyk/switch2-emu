/* This file is part of the dynarmic project.
 * Copyright (c) 2021 MerryMage
 * SPDX-License-Identifier: 0BSD
 *
 * switch2-emu: Zydis is not vendored, so disassembly of emitted x64 code is
 * not available. These stubs keep the interface intact.
 */

#include "dynarmic/common/x64_disassemble.h"

#include <cstdio>

namespace Dynarmic::Common {

void DumpDisassembledX64(const void* ptr, size_t size) {
    std::printf("(x64 disassembly unavailable: %p, %zu bytes)\n", ptr, size);
}

std::vector<std::string> DisassembleX64(const void*, size_t) {
    return {};
}

}  // namespace Dynarmic::Common
