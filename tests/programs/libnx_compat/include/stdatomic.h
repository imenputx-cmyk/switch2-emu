/* GCC lets the C11 atomic generics work on plain (non-_Atomic) objects,
 * and libnx relies on that; clang does not. Map them to the __atomic
 * builtins, which accept plain pointers. */
#pragma once
typedef enum {
    memory_order_relaxed = __ATOMIC_RELAXED,
    memory_order_consume = __ATOMIC_CONSUME,
    memory_order_acquire = __ATOMIC_ACQUIRE,
    memory_order_release = __ATOMIC_RELEASE,
    memory_order_acq_rel = __ATOMIC_ACQ_REL,
    memory_order_seq_cst = __ATOMIC_SEQ_CST,
} memory_order;
#define atomic_load_explicit(p, o) \
    ({ __typeof__(*(p)) _v; __atomic_load((p), &_v, (o)); _v; })
#define atomic_load(p) __atomic_load_n((p), __ATOMIC_SEQ_CST)
#define atomic_store_explicit(p, v, o) __atomic_store_n((p), (v), (o))
#define atomic_store(p, v) __atomic_store_n((p), (v), __ATOMIC_SEQ_CST)
#define atomic_fetch_add_explicit(p, v, o) __atomic_fetch_add((p), (v), (o))
#define atomic_fetch_add(p, v) __atomic_fetch_add((p), (v), __ATOMIC_SEQ_CST)
#define atomic_fetch_sub_explicit(p, v, o) __atomic_fetch_sub((p), (v), (o))
#define atomic_fetch_sub(p, v) __atomic_fetch_sub((p), (v), __ATOMIC_SEQ_CST)
#define atomic_exchange_explicit(p, v, o) __atomic_exchange_n((p), (v), (o))
#define atomic_exchange(p, v) __atomic_exchange_n((p), (v), __ATOMIC_SEQ_CST)
#define atomic_compare_exchange_strong(p, e, v) \
    __atomic_compare_exchange_n((p), (e), (v), 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define atomic_thread_fence(o) __atomic_thread_fence(o)
