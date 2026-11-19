/* Stand-in for devkitPro newlib's <sys/lock.h>: the lock types libnx uses. */
#pragma once
#include <stdint.h>
typedef uint32_t _LOCK_T;
typedef struct { _LOCK_T lock; uint32_t thread_tag; uint32_t counter; } _LOCK_RECURSIVE_T;
