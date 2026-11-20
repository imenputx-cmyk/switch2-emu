//! nvdrv's GPU devices: `/dev/nvhost-as-gpu` (GPU address spaces),
//! `/dev/nvhost-ctrl-gpu` (GPU information) and `/dev/nvhost-gpu`
//! (channels: GPFIFO submission). Ioctl numbers and struct layouts as in
//! libnx's `nvidia/ioctl/*.c` (ISC); the characteristics are the values
//! libnx documents for the Switch's GM20B.

use super::Ctx;
use emu_gpu::mmu::BIG_PAGE;

pub const NV_SUCCESS: u32 = 0;
pub const NV_NOT_IMPLEMENTED: u32 = 1;
pub const NV_BAD_PARAMETER: u32 = 4;

fn u32_at(b: &[u8], off: usize) -> u32 {
    b.get(off..off + 4)
        .map_or(0, |s| u32::from_le_bytes(s.try_into().unwrap()))
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    b.get(off..off + 8)
        .map_or(0, |s| u64::from_le_bytes(s.try_into().unwrap()))
}

fn put(b: &mut [u8], off: usize, v: &[u8]) {
    if let Some(s) = b.get_mut(off..off + v.len()) {
        s.copy_from_slice(v);
    }
}

/// `/dev/nvhost-as-gpu` ioctls (type 0x41) on address space `asid`.
pub fn as_gpu(
    asid: usize,
    nr: u32,
    data: &mut [u8],
    ctx: &mut Ctx,
    channel_of_fd: impl Fn(u32) -> Option<usize>,
) -> u32 {
    let gpu = &mut ctx.system.gpu;
    match nr {
        // BIND_CHANNEL {channel fd}
        0x01 => match channel_of_fd(u32_at(data, 0)) {
            Some(c) => {
                gpu.channels[c].address_space = Some(asid);
                NV_SUCCESS
            }
            None => NV_BAD_PARAMETER,
        },
        // ALLOC_SPACE {pages, page_size, flags (bit 0 fixed), pad, offset/align}
        0x02 => {
            let size = u64::from(u32_at(data, 0)) * u64::from(u32_at(data, 4));
            let fixed = u32_at(data, 8) & 1 != 0;
            let aspace = &mut gpu.address_spaces[asid];
            let va = if fixed {
                let va = u64_at(data, 16);
                aspace.reserve_fixed(va, size).then_some(va)
            } else {
                aspace.reserve(size, u64_at(data, 16))
            };
            match va {
                Some(va) => {
                    put(data, 16, &va.to_le_bytes());
                    NV_SUCCESS
                }
                None => NV_BAD_PARAMETER,
            }
        }
        // FREE_SPACE {offset, pages, page_size}
        0x03 => {
            if gpu.address_spaces[asid].free(u64_at(data, 0)) {
                NV_SUCCESS
            } else {
                NV_BAD_PARAMETER
            }
        }
        // UNMAP_BUFFER {offset}
        0x05 => {
            gpu.address_spaces[asid].unmap(u64_at(data, 0));
            NV_SUCCESS
        }
        // MAP_BUFFER_EX {flags, kind, nvmap handle, page size, buffer
        // offset, mapping size, offset}
        0x06 => {
            let flags = u32_at(data, 0);
            let (handle, buffer_offset, mut size, va) = (
                u32_at(data, 8),
                u64_at(data, 16),
                u64_at(data, 24),
                u64_at(data, 32),
            );
            if flags & 0x100 != 0 {
                // Modify: changes the kind of an existing mapping.
                return if gpu.address_spaces[asid].find(va).is_some() {
                    NV_SUCCESS
                } else {
                    NV_BAD_PARAMETER
                };
            }
            let Some(obj) = ctx.system.nvmap.get(&handle) else {
                return NV_BAD_PARAMETER;
            };
            if obj.addr == 0 || buffer_offset > u64::from(obj.size) {
                return NV_BAD_PARAMETER;
            }
            if size == 0 {
                size = u64::from(obj.size) - buffer_offset;
            }
            let cpu = obj.addr + buffer_offset;
            let aspace = &mut ctx.system.gpu.address_spaces[asid];
            let mapped = if flags & 1 != 0 {
                aspace.map(va, cpu, size).then_some(va)
            } else {
                aspace.map_anywhere(cpu, size, BIG_PAGE)
            };
            match mapped {
                Some(va) => {
                    if u32_at(data, 12) == 0 {
                        put(data, 12, &(BIG_PAGE as u32).to_le_bytes());
                    }
                    put(data, 32, &va.to_le_bytes());
                    NV_SUCCESS
                }
                None => NV_BAD_PARAMETER,
            }
        }
        // GET_VA_REGIONS {not used, bufsize, pad, 2 x {offset, page size, pad, pages}}
        0x08 => {
            put(data, 8, &0x30u32.to_le_bytes());
            for (i, (offset, page, pages)) in [
                (0x400_0000u64, 0x1000u32, 0x3F_BFFFu64),
                (0x4_0000_0000, 0x1_0000, 0x1B_FFFF),
            ]
            .into_iter()
            .enumerate()
            {
                let at = 16 + 24 * i;
                put(data, at, &offset.to_le_bytes());
                put(data, at + 8, &page.to_le_bytes());
                put(data, at + 16, &pages.to_le_bytes());
            }
            NV_SUCCESS
        }
        0x09 => NV_SUCCESS, // INITIALIZE_EX
        _ => NV_NOT_IMPLEMENTED,
    }
}

/// `/dev/nvhost-ctrl-gpu` ioctls (type 0x47).
pub fn ctrl_gpu(
    nr: u32,
    data: &mut [u8],
    inline_out: &mut [u8],
    nx2: &mut bool,
    ctx: &mut Ctx,
) -> u32 {
    match nr {
        0x01 => put(data, 0, &0x8000u32.to_le_bytes()), // ZCULL_GET_CTX_SIZE
        0x02 => {
            // ZCULL_GET_INFO
            for (i, v) in [
                0x20u32, 0x20, 0x400, 0x800, 0x20, 0x20, 0xC0, 0x20, 0x40, 0x10,
            ]
            .iter()
            .enumerate()
            {
                put(data, 4 * i, &v.to_le_bytes());
            }
        }
        0x03 | 0x04 => {} // ZBC_SET_TABLE / ZBC_QUERY_TABLE
        0x05 => {
            // GET_CHARACTERISTICS {buf size, buf addr, characteristics}.
            // A Switch 2 NvRm asks for 0xD0 bytes and needs the Ampere chip.
            if u64_at(data, 0) >= NX2_CHARACTERISTICS_SIZE as u64 {
                *nx2 = true;
                let block = nx2_characteristics();
                put(data, 0, &(NX2_CHARACTERISTICS_SIZE as u64).to_le_bytes());
                let room = data.len().saturating_sub(16).min(block.len());
                put(data, 16, &block[..room]);
                let n = inline_out.len().min(block.len());
                inline_out[..n].copy_from_slice(&block[..n]);
                return NV_SUCCESS;
            }
            put(data, 0, &0xA0u64.to_le_bytes());
            let mut c = Vec::with_capacity(0xA0);
            let w32 = |c: &mut Vec<u8>, v: u32| c.extend_from_slice(&v.to_le_bytes());
            let w64 = |c: &mut Vec<u8>, v: u64| c.extend_from_slice(&v.to_le_bytes());
            for v in [0x120, 0xB, 0xA1, 1] {
                w32(&mut c, v); // arch, impl, rev, num_gpc
            }
            w64(&mut c, 0x4_0000); // L2 size
            w64(&mut c, 0); // video memory
            for v in [
                2, 0x20, 0x2_0000, 0x2_0000, 0x1B, 0x3_0000, 1, 0x503, 0x503, 0x80, 0x28, 0,
            ] {
                w32(&mut c, v);
            }
            w64(&mut c, 0x55); // flags
            for v in [
                0x902D, 0xB197, 0xB1C0, 0xB06F, 0xA140, 0xB0B5, 1, 0, 2, 1, 0, 1, 0x2_1D70, 0,
            ] {
                w32(&mut c, v);
            }
            w64(&mut c, 0x62_3032_6D67); // "gm20b"
            w64(&mut c, 0);
            put(data, 16, &c);
        }
        0x06 => put(data, 16, &3u32.to_le_bytes()), // GET_TPC_MASKS: 2 TPCs
        0x14 => {
            // ZBC_GET_ACTIVE_SLOT_MASK {slot, mask}
            put(data, 0, &7u32.to_le_bytes());
            put(data, 4, &0u32.to_le_bytes());
        }
        // GET_ENGINE_INFO (Switch 2) {buf size, reserved, buf addr}: items
        // {engine id, instance, runlist id, reserved} in the inline output.
        // NvRm picks channel runlists from it: graphics (runlist 0) and an
        // async copy engine (runlist 1), as Lichii reports.
        0x1F if *nx2 => {
            let items: [[u32; 4]; 2] = [[0, 0, 0, 0], [2, 0, 1, 0]];
            let needed = 32u32;
            if u32_at(data, 0) >= needed && inline_out.len() >= needed as usize {
                for (i, w) in items.iter().flatten().enumerate() {
                    put(inline_out, 4 * i, &w.to_le_bytes());
                }
            }
            put(data, 0, &needed.to_le_bytes());
        }
        0x1C => {
            // GET_GPU_TIME {timestamp (ns), reserved}
            let ns = ctx.tick.saturating_mul(625) / 12;
            put(data, 0, &ns.to_le_bytes());
        }
        _ => return NV_NOT_IMPLEMENTED,
    }
    NV_SUCCESS
}

/// Size of the Switch 2 characteristics block (Switch 1: 0xA0).
const NX2_CHARACTERISTICS_SIZE: usize = 0xD0;

/// The characteristics of Switch 2's GPU as its NvRm and NVN2 require
/// them (Lichii `nvhost_ctrl_gpu.cpp` ApplyNx2Characteristics; values are
/// partly best guesses there too).
fn nx2_characteristics() -> Vec<u8> {
    let mut c = vec![0u8; NX2_CHARACTERISTICS_SIZE];
    let w32 = |c: &mut Vec<u8>, at: usize, v: u32| c[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let w64 = |c: &mut Vec<u8>, at: usize, v: u64| c[at..at + 8].copy_from_slice(&v.to_le_bytes());
    w32(&mut c, 0x00, 0x170); // arch: Ampere
    w32(&mut c, 0x04, 0xE); // impl: chip 0x17E
    w32(&mut c, 0x08, 0xA1); // rev
    w32(&mut c, 0x0C, 1); // GPCs
    w64(&mut c, 0x10, 0x4_0000); // L2 size
    w32(&mut c, 0x20, 6); // TPCs per GPC
    w32(&mut c, 0x24, 0x20); // bus type
    w32(&mut c, 0x28, 0x2_0000); // big page size
    w32(&mut c, 0x2C, 0x2_0000); // compression page size
    w32(&mut c, 0x30, 0x1B); // PDE coverage bits
    w32(&mut c, 0x34, 0x3_0000); // available big page sizes
    w32(&mut c, 0x38, 1); // GPC mask
    w32(&mut c, 0x3C, 0x808); // SM version: NVN2 only accepts 0x808
    w32(&mut c, 0x40, 0x808); // SPA version
    w32(&mut c, 0x44, 0x80); // warp count
    w32(&mut c, 0x48, 0x28); // GPU VA bits
                             // Flags: TSG (8), TSG subcontexts (22), syncpoint address (26),
                             // mapping modify (34), remap (35), 2D/3D/compute/I2M/ZBC (40-44).
    let flags = 0x55u64 | 1 << 8 | 1 << 22 | 1 << 26 | 1 << 34 | 1 << 35 | 0x1F << 40;
    w64(&mut c, 0x50, flags);
    for (i, class) in [0x902D, 0xC797, 0xC7C0, 0xC56F, 0xA140, 0xC7B5]
        .into_iter()
        .enumerate()
    {
        w32(&mut c, 0x58 + 4 * i, class);
    }
    w32(&mut c, 0x84, 1); // max GPC count
    w32(&mut c, 0x88, 0x2_1D70); // ROP/L2 mask
                                 // Nine s16 interface versions at 0x70..=0x80; NvRm needs e.g. >= 6.
    for at in (0x70..=0x80).step_by(2) {
        c[at..at + 2].copy_from_slice(&0x40i16.to_le_bytes());
    }
    // Chip name: Lichii's value, which reads "ga01f" in memory (its
    // comment says "ga10f"); kept as is since NVN2 accepts it there.
    w64(&mut c, 0x90, 0x66_3130_6167);
    // Subcontexts (VEIDs) per TSG; without them a second NVN queue gets
    // no command memory (Lichii row 45).
    w32(&mut c, 0xB4, 64);
    c
}

/// Where the Switch 2 read-only syncpoint page sits in every GPU address
/// space (the top of the VA range NVN uses), and how many it holds.
pub const SYNCPOINT_SHIM_VA: u64 = 0xC0_0000_0000 - 0x1000;
pub const SYNCPOINT_SHIM_COUNT: u32 = 0x1000 / 4;

/// nvhost-as-gpu 0x0B (Switch 2): where the read-only syncpoint page is:
/// {address, stride, count}. NVN asserts without one (Lichii row 46).
pub fn syncpoint_shim(data: &mut [u8]) -> u32 {
    put(data, 0, &SYNCPOINT_SHIM_VA.to_le_bytes());
    put(data, 8, &4u32.to_le_bytes());
    put(data, 12, &SYNCPOINT_SHIM_COUNT.to_le_bytes());
    NV_SUCCESS
}

/// `/dev/nvhost-gpu` ioctls (types 0x48 and 0x47) on a channel.
/// `extra` holds the GPFIFO entries of KICKOFF_PB (Ioctl2's second input).
pub fn channel(ch: usize, kind: u32, nr: u32, data: &mut [u8], extra: &[u8], ctx: &mut Ctx) -> u32 {
    match (kind, nr) {
        // SET_NVMAP_FD, SET_TIMEOUT, ZCULL_BIND, SET_ERROR_NOTIFIER,
        // SET_PRIORITY, SET_TIMESLICE, SET_USER_DATA
        (0x48, 0x01 | 0x03 | 0x0B | 0x0C | 0x0D | 0x1D) | (0x47, 0x14) => NV_SUCCESS,
        // GET_ERROR_INFO, GET_ERROR_NOTIFICATION: no errors.
        (0x48, 0x16 | 0x17) => {
            data.fill(0);
            NV_SUCCESS
        }
        // ALLOC_OBJ_CTX {class, flags, obj id out}
        (0x48, 0x09) => {
            put(data, 8, &u64::from(u32_at(data, 0)).to_le_bytes());
            NV_SUCCESS
        }
        // ALLOC_GPFIFO_EX2 {entries, flags, unk, fence out {id, value}, ...}
        (0x48, 0x1A) => {
            let gpu = &ctx.system.gpu;
            let sp = gpu.channels[ch].syncpoint;
            put(data, 12, &sp.to_le_bytes());
            put(data, 16, &gpu.syncpoints[sp as usize].to_le_bytes());
            NV_SUCCESS
        }
        // SUBMIT_GPFIFO {gpfifo, count, flags, fence, entries...} and
        // KICKOFF_PB (entries in the second input buffer).
        (0x48, 0x08 | 0x1B) => {
            let count = u32_at(data, 8) as usize;
            let flags = u32_at(data, 12);
            let src = if nr == 0x08 {
                data.get(24..).unwrap_or(&[])
            } else {
                extra
            };
            if src.len() < count * 8 {
                return NV_BAD_PARAMETER;
            }
            let entries: Vec<u64> = (0..count).map(|i| u64_at(src, 8 * i)).collect();
            ctx.system.gpu.trace = ctx.config.trace_unimplemented;
            // INCREMENT_VALUE (bit 8, Switch 2): the syncpoint rises by the
            // given fence value once the work is done. NVN2 signals
            // completion only through semaphores, so the increments are
            // ours to do (Lichii row 32).
            let increment = if flags & 0x100 != 0 {
                u32_at(data, 20)
            } else {
                0
            };
            let (id, value) = ctx
                .system
                .gpu
                .submit_with_increment(ch, &entries, flags, increment, ctx.mem);
            for f in ctx.system.gpu.faults.drain(..) {
                if ctx.config.trace_unimplemented {
                    eprintln!("[gpu] fault: {f}");
                }
            }
            put(data, 16, &id.to_le_bytes());
            put(data, 20, &value.to_le_bytes());
            NV_SUCCESS
        }
        _ => NV_NOT_IMPLEMENTED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch2_characteristics_block() {
        let c = nx2_characteristics();
        assert_eq!(c.len(), 0xD0);
        assert_eq!(u32_at(&c, 0), 0x170);
        assert_eq!(u32_at(&c, 4), 0xE);
        assert_eq!(u32_at(&c, 0x3C), 0x808);
        assert_eq!(u32_at(&c, 0x5C), 0xC797); // 3D class
        assert_eq!(u32_at(&c, 0x6C), 0xC7B5); // copy class
        assert_ne!(u64_at(&c, 0x50) & (1 << 26), 0); // syncpoint address
        assert_eq!(i16::from_le_bytes([c[0x78], c[0x79]]), 0x40);
        assert_eq!(&c[0x90..0x95], b"ga01f");
        assert_eq!(u32_at(&c, 0xB4), 64);
    }
}
