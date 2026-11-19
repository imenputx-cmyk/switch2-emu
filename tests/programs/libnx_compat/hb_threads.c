/* Threads on the emulated kernel, using libnx's own (ISC) mutex, condvar,
 * semaphore and light-event code from source: svcCreateThread/Start/Exit,
 * ArbitrateLock/Unlock, WaitProcessWideKeyAtomic/SignalProcessWideKey,
 * WaitForAddress/SignalToAddress, WaitSynchronization on threads and
 * SleepThread. libnx's thread.c needs newlib, so threads are created
 * with raw SVCs and their ThreadVars are filled in here. */
#include <string.h>
#include "switch/types.h"
#include "switch/result.h"
#include "switch/arm/tls.h"
#include "switch/arm/counter.h"
#include "switch/kernel/svc.h"
#include "switch/kernel/mutex.h"
#include "switch/kernel/condvar.h"
#include "switch/kernel/semaphore.h"
#include "switch/kernel/levent.h"

void glue_setup_tls(void);

static char out[512];
static u64 out_len;
static void put(const char *s) { while (*s) out[out_len++] = *s++; }
static void put_dec(u64 v) {
    char b[24]; int n = 0;
    do { b[n++] = '0' + v % 10; v /= 10; } while (v);
    while (n) out[out_len++] = b[--n];
}
static void line(const char *label, u64 v) {
    put(label); put_dec(v); put("\n");
    svcOutputDebugString(out, out_len);
    out_len = 0;
}

/* libnx's per-thread variables at the end of the TLS area. */
typedef struct { u32 magic; Handle handle; void *thread_ptr, *reent, *tls_tp; } ThreadVars;
static ThreadVars *thread_vars(void) { return (ThreadVars *)((u8 *)armGetTls() + 0x200 - sizeof(ThreadVars)); }

#define WORKERS 4
static Handle handles[WORKERS];
static _Alignas(16) u8 stacks[WORKERS][0x4000];

/* 1: a counter incremented under a contended mutex. */
static Mutex counter_lock;
static u64 counter;

/* 2: a bounded queue with two condition variables. */
static Mutex q_lock;
static CondVar not_empty, not_full;
static u32 queue[4], q_head, q_len;
static u64 consumed_sum;

/* 3: a semaphore and a light event (address arbiter). */
static Semaphore sem;
static LEvent go;
static u32 go_seen;

static void worker(void *arg) {
    u64 i = (u64)arg;
    thread_vars()->handle = handles[i];

    leventWait(&go, UINT64_MAX);
    mutexLock(&counter_lock);
    go_seen++;
    mutexUnlock(&counter_lock);

    for (int k = 0; k < 500; k++) {
        mutexLock(&counter_lock);
        u64 v = counter;
        if (k % 7 == 0)
            svcSleepThread(0); /* yield while holding the lock */
        counter = v + 1;
        mutexUnlock(&counter_lock);
    }

    if (i == 0) {
        /* Producer. */
        for (u32 n = 1; n <= 100; n++) {
            mutexLock(&q_lock);
            while (q_len == 4)
                condvarWait(&not_full, &q_lock);
            queue[(q_head + q_len++) % 4] = n;
            condvarWakeOne(&not_empty);
            mutexUnlock(&q_lock);
        }
    } else if (i == 1) {
        /* Consumer. */
        for (int n = 0; n < 100; n++) {
            mutexLock(&q_lock);
            while (q_len == 0)
                condvarWait(&not_empty, &q_lock);
            consumed_sum += queue[q_head];
            q_head = (q_head + 1) % 4;
            q_len--;
            condvarWakeOne(&not_full);
            mutexUnlock(&q_lock);
        }
    }
    semaphoreSignal(&sem);
    svcExitThread();
}

int hb_main(void *ctx, u64 x1, void *saved_lr) {
    (void)x1; (void)saved_lr;
    glue_setup_tls();
    /* The homebrew ABI's config entries: key 1 is the main thread handle. */
    for (const u64 *e = ctx; e[0] != 0; e += 3)
        if ((u32)e[0] == 1)
            thread_vars()->handle = (Handle)e[1];

    semaphoreInit(&sem, 0);
    leventInit(&go, false, false);
    for (u64 i = 0; i < WORKERS; i++) {
        Result rc = svcCreateThread(&handles[i], (void *)worker, (void *)i,
                                    stacks[i] + sizeof(stacks[i]), 0x2C, -2);
        if (rc) { line("svcCreateThread failed: ", rc); return 1; }
    }
    for (int i = 0; i < WORKERS; i++)
        svcStartThread(handles[i]);

    /* Let the workers block on the event, then release them all. */
    svcSleepThread(1000000);
    line("workers past the event before signal: ", go_seen);
    leventSignal(&go);

    for (int i = 0; i < WORKERS; i++)
        semaphoreWait(&sem);
    line("workers past the event: ", go_seen);
    line("counter (expect 2000): ", counter);
    line("consumed sum (expect 5050): ", consumed_sum);

    s32 idx = -1;
    Result rc = svcWaitSynchronization(&idx, handles, WORKERS, UINT64_MAX);
    line("join rc: ", rc);
    for (int i = 0; i < WORKERS; i++) svcCloseHandle(handles[i]);

    /* Sleeping advances the clock; a timed wait times out. */
    u64 t0 = armGetSystemTick();
    svcSleepThread(5000000);
    u64 dt = armTicksToNs(armGetSystemTick() - t0);
    line("slept at least 5 ms: ", dt >= 5000000);
    mutexLock(&q_lock);
    rc = condvarWaitTimeout(&not_empty, &q_lock, 1000000);
    line("condvar timeout rc: ", rc);
    line("mutex held again: ", mutexIsLockedByCurrentThread(&q_lock));
    mutexUnlock(&q_lock);
    return 0;
}
