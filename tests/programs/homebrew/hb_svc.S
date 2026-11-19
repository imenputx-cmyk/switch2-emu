// SVC wrappers with the signatures declared in libnx's kernel/svc.h,
// written for this project (argument registers from switchbrew's SVC page).
    .text
    .global svcConnectToNamedPort
svcConnectToNamedPort:          // Result (Handle* out, const char* name)
    str     x0, [sp, #-16]!
    mov     x1, x1
    svc     #0x1F
    ldr     x2, [sp], #16
    str     w1, [x2]
    ret

    .global svcSendSyncRequest
svcSendSyncRequest:             // Result (Handle session)
    svc     #0x21
    ret

    .global svcCloseHandle
svcCloseHandle:                 // Result (Handle)
    svc     #0x16
    ret

    .global svcOutputDebugString
svcOutputDebugString:           // Result (const char*, u64)
    svc     #0x27
    ret
