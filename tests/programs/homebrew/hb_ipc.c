/* IPC test homebrew. The client side is libnx's own header-only IPC code
 * (sf/service.h, sf/cmif.h, sf/hipc.h; ISC license, see LICENSE-libnx.md),
 * so the emulator is exercised with exactly the messages real homebrew
 * sends. The test logic itself is written for this project. */
#include <switch/types.h>
#include <switch/result.h>
#include <switch/kernel/svc.h>
#include <switch/sf/service.h>
#include <switch/services/sm.h>

/* Same layout as libnx's SetSysFirmwareVersion (services/set.h, which
 * needs newlib headers to include). */
typedef struct {
    u8 major, minor, micro, padding1;
    u8 revision_major, revision_minor, padding2, padding3;
    char platform[0x20];
    char version_hash[0x40];
    char display_version[0x18];
    char display_title[0x80];
} SetSysFirmwareVersion;

void hb_relocate(u64 base, const void *dyn);

/* -fno-builtin turns __builtin_strlen in smEncodeName into a call. */
size_t strlen(const char *s) {
    size_t n = 0;
    while (s[n]) n++;
    return n;
}

static char out[512];
static u64 out_len;

static void put(const char *s) { while (*s) out[out_len++] = *s++; }
static void put_hex(u64 v) {
    static const char d[] = "0123456789abcdef";
    put("0x");
    int started = 0;
    for (int i = 60; i >= 0; i -= 4) {
        int n = (v >> i) & 15;
        if (n || started || i == 0) { out[out_len++] = d[n]; started = 1; }
    }
}
static void line(const char *label, u64 v) {
    put(label); put_hex(v); put("\n");
    svcOutputDebugString(out, out_len);
    out_len = 0;
}
static void text(const char *label, const char *s) {
    put(label); put(s); put("\n");
    svcOutputDebugString(out, out_len);
    out_len = 0;
}

int hb_main(void *cfg, u64 x1) {
    (void)cfg; (void)x1;
    Handle h;
    Result rc = svcConnectToNamedPort(&h, "sm:");
    line("ConnectToNamedPort(sm:) = ", rc);
    line("ConnectToNamedPort(nope) = ", svcConnectToNamedPort(&(Handle){0}, "nope"));

    Service sm;
    serviceCreate(&sm, h); /* also asks for the pointer buffer size */
    line("sm pointer buffer size = ", sm.pointer_buffer_size);

    u64 pid_placeholder = 0;
    rc = serviceDispatchIn(&sm, 0, pid_placeholder, .in_send_pid = true);
    line("sm RegisterClient = ", rc);

    Handle set_h = 0;
    SmServiceName name = smEncodeName("set:sys");
    rc = serviceDispatchIn(&sm, 1, name, .out_handle_attrs = {SfOutHandleAttr_HipcMove}, .out_handles = &set_h);
    line("sm GetServiceHandle(set:sys) = ", rc);

    Handle bad_h = 0;
    SmServiceName bad = smEncodeName("nosuch");
    rc = serviceDispatchIn(&sm, 1, bad, .out_handle_attrs = {SfOutHandleAttr_HipcMove}, .out_handles = &bad_h);
    line("sm GetServiceHandle(nosuch) = ", rc);

    Service setsys;
    serviceCreate(&setsys, set_h);
    SetSysFirmwareVersion fw = {0};
    rc = serviceDispatch(&setsys, 4,
        .buffer_attrs = {SfBufferAttr_FixedSize | SfBufferAttr_HipcPointer | SfBufferAttr_Out},
        .buffers = {{&fw, sizeof(fw)}});
    line("set:sys GetFirmwareVersion2 = ", rc);
    line("firmware major = ", fw.major);
    text("display version = ", fw.display_version);
    text("display title = ", fw.display_title);

    /* The same call through a domain object. */
    rc = serviceConvertToDomain(&setsys);
    line("ConvertToDomain = ", rc);
    line("domain object id = ", setsys.object_id);
    SetSysFirmwareVersion fw2 = {0};
    rc = serviceDispatch(&setsys, 3,
        .buffer_attrs = {SfBufferAttr_FixedSize | SfBufferAttr_HipcPointer | SfBufferAttr_Out},
        .buffers = {{&fw2, sizeof(fw2)}});
    line("domain GetFirmwareVersion = ", rc);
    text("domain display version = ", fw2.display_version);

    serviceClose(&setsys);
    serviceClose(&sm);
    line("CloseHandle(closed handle) = ", svcCloseHandle(h));
    text("IPC test done", "");
    return 0;
}
