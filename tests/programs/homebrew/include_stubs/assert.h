/* Freestanding stand-in for <assert.h>, needed by libnx's sf headers. */
#pragma once
#define assert(x) ((void)0)
#define static_assert(...) _Static_assert(__VA_ARGS__, "")
