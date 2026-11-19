/* Runs libnx's real startup and service-client code (compiled from libnx
 * sources; ISC license, see ../LICENSE-libnx.md) on the emulator: the
 * same steps as libnx's __libnx_init/__appInit, then a few service calls.
 * The test logic is written for this project. */
#include <string.h>
#include "switch/types.h"
#include "switch/result.h"
#include "switch/kernel/svc.h"
#include "switch/kernel/virtmem.h"
#include "switch/runtime/env.h"
#include "switch/runtime/hosversion.h"
#include "switch/services/sm.h"
#include "switch/services/applet.h"
#include "switch/services/hid.h"
#include "switch/services/time.h"
#include "switch/services/fs.h"
#include "switch/services/set.h"

void glue_set_heap(void *base, size_t size);
void glue_setup_tls(void);
void virtmemSetup(void); /* libnx internal, called by __libnx_init */

static char out[4096];
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

int hb_main(void *ctx, u64 x1, void *saved_lr) {
    (void)x1;
    /* __libnx_init: environment, virtual memory regions, heap. */
    glue_setup_tls();
    envSetup(ctx, INVALID_HANDLE, (LoaderReturnFn)saved_lr);
    line("hosversion = ", hosversionGet());
    virtmemSetup();
    u64 total = 0, used = 0, heap = 0;
    svcGetInfo(&total, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0);
    svcGetInfo(&used, InfoType_UsedMemorySize, CUR_PROCESS_HANDLE, 0);
    u64 size = 0x4000000; /* 64 MiB is plenty for this test */
    line("svcSetHeapSize = ", svcSetHeapSize((void **)&heap, size));
    glue_set_heap((void *)heap, size);

    /* __appInit: the default services, in libnx's order. */
    line("smInitialize = ", smInitialize());
    line("appletInitialize = ", appletInitialize());
    line("hidInitialize = ", hidInitialize());
    line("timeInitialize = ", timeInitialize());
    line("fsInitialize = ", fsInitialize());

    line("applet type = ", appletGetAppletType());
    u64 now = 0;
    line("timeGetCurrentTime = ", timeGetCurrentTime(TimeType_Default, &now));
    line("posix time = ", now);
    FsFileSystem sd;
    Result rc = fsOpenSdCardFileSystem(&sd);
    line("fsOpenSdCardFileSystem = ", rc);
    if (R_FAILED(rc))
        return 0;

    /* Read a file the host put on the SD card. */
    FsFile f;
    static char path[FS_MAX_PATH], buf[64];
    strcpy(path, "/hello.txt");
    line("OpenFile(/hello.txt) = ", fsFsOpenFile(&sd, path, FsOpenMode_Read, &f));
    u64 n = 0;
    line("Read = ", fsFileRead(&f, 0, buf, sizeof(buf) - 1, FsReadOption_None, &n));
    buf[n] = 0;
    put("contents: ");
    put(buf);
    fsFileClose(&f);

    /* Create, write and read back a file in a new directory. */
    strcpy(path, "/save");
    line("CreateDirectory = ", fsFsCreateDirectory(&sd, path));
    strcpy(path, "/save/data.bin");
    line("CreateFile = ", fsFsCreateFile(&sd, path, 0, 0));
    line("CreateFile again = ", fsFsCreateFile(&sd, path, 0, 0));
    line("OpenFile(rw+append) = ",
         fsFsOpenFile(&sd, path, FsOpenMode_Read | FsOpenMode_Write | FsOpenMode_Append, &f));
    static const char msg[] = "written by libnx";
    line("Write = ", fsFileWrite(&f, 0, msg, sizeof(msg) - 1, FsWriteOption_Flush));
    s64 fsize = 0;
    line("GetSize = ", fsFileGetSize(&f, &fsize));
    line("size = ", (u64)fsize);
    memset(buf, 0, sizeof(buf));
    line("Read back = ", fsFileRead(&f, 8, buf, 8, FsReadOption_None, &n));
    put("tail: ");
    put(buf);
    put("\n");
    fsFileClose(&f);

    /* List the root directory. */
    FsDir d;
    strcpy(path, "/");
    line("OpenDirectory = ", fsFsOpenDirectory(&sd, path, FsDirOpenMode_ReadDirs | FsDirOpenMode_ReadFiles, &d));
    static FsDirectoryEntry entries[8];
    s64 count = 0;
    line("ReadDirectory = ", fsDirRead(&d, &count, 8, entries));
    for (s64 i = 0; i < count; i++) {
        put(entries[i].type == FsDirEntryType_Dir ? "  dir  " : "  file ");
        put(entries[i].name);
        put("\n");
    }
    fsDirClose(&d);

    strcpy(path, "/../escape");
    line("OpenDirectory(/../escape) = ", fsFsOpenDirectory(&sd, path, FsDirOpenMode_ReadDirs, &d));
    strcpy(path, "/missing.txt");
    line("OpenFile(/missing.txt) = ", fsFsOpenFile(&sd, path, FsOpenMode_Read, &f));
    fsFsClose(&sd);
    return 0;
}
