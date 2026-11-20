//! The kernel's view of the address space: which ranges are mapped, with
//! what state, permission and attributes. `QueryMemory` reads this.
//!
//! The memory state values are the low byte that `svcQueryMemory` reports
//! (names follow libnx's `MemoryType`); see https://switchbrew.org/wiki/SVC.

use std::collections::BTreeMap;

/// Size of the 39-bit address space used by Switch applications.
pub const ADDRESS_SPACE_END: u64 = 1 << 39;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MemState {
    Free = 0x00,
    /// Text and read-only data of a loaded module.
    CodeStatic = 0x03,
    /// Data and .bss of a loaded module.
    CodeMutable = 0x04,
    /// Mapped with SetHeapSize.
    Heap = 0x05,
    /// Shared memory mapped with MapSharedMemory.
    Shared = 0x06,
    /// Alias of heap memory made with MapMemory (thread stacks).
    Stack = 0x0B,
    ThreadLocal = 0x0C,
    Inaccessible = 0x10,
}

pub const PERM_R: u32 = 1;
pub const PERM_W: u32 = 2;
pub const PERM_X: u32 = 4;

/// MemoryAttribute bits.
pub const ATTR_LOCKED: u32 = 1;
pub const ATTR_UNCACHED: u32 = 8;
/// Switch 2: memory the GPU may access (svc 0x80).
pub const ATTR_GPU_ACCESSIBLE: u32 = 0x20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub size: u64,
    pub state: MemState,
    pub perm: u32,
    pub attr: u32,
}

/// What `svcQueryMemory` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInfo {
    pub base: u64,
    pub size: u64,
    pub state: MemState,
    pub attr: u32,
    pub perm: u32,
}

impl MemoryInfo {
    /// The 0x28-byte `MemoryInfo` structure written to guest memory.
    pub fn to_bytes(self) -> [u8; 0x28] {
        let mut b = [0u8; 0x28];
        b[0..8].copy_from_slice(&self.base.to_le_bytes());
        b[8..16].copy_from_slice(&self.size.to_le_bytes());
        b[16..20].copy_from_slice(&(self.state as u32).to_le_bytes());
        b[20..24].copy_from_slice(&self.attr.to_le_bytes());
        b[24..28].copy_from_slice(&self.perm.to_le_bytes());
        // IPC and device reference counts, padding: zero.
        b
    }
}

#[derive(Debug, Default)]
pub struct MemoryMap {
    /// Mapped blocks by start address. Gaps are free.
    blocks: BTreeMap<u64, Block>,
}

impl MemoryMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// True if no block overlaps `[addr, addr + size)`.
    pub fn is_free(&self, addr: u64, size: u64) -> bool {
        let end = addr + size;
        if let Some((&start, b)) = self.blocks.range(..end).next_back() {
            if start + b.size > addr {
                return false;
            }
        }
        true
    }

    pub fn insert(&mut self, addr: u64, block: Block) {
        debug_assert!(self.is_free(addr, block.size));
        self.blocks.insert(addr, block);
    }

    /// Splits blocks so that `addr` is a block boundary.
    fn split_at(&mut self, addr: u64) {
        let Some((&start, &b)) = self.blocks.range(..=addr).next_back() else {
            return;
        };
        if start < addr && addr < start + b.size {
            let first = addr - start;
            self.blocks.insert(start, Block { size: first, ..b });
            self.blocks.insert(
                addr,
                Block {
                    size: b.size - first,
                    ..b
                },
            );
        }
    }

    /// Calls `f` on every block inside `[addr, addr + size)`, after
    /// splitting the blocks at the range boundaries.
    pub fn update(&mut self, addr: u64, size: u64, mut f: impl FnMut(&mut Block)) {
        self.split_at(addr);
        self.split_at(addr + size);
        for (_, b) in self.blocks.range_mut(addr..addr + size) {
            f(b);
        }
    }

    /// Removes every block inside `[addr, addr + size)`.
    pub fn remove(&mut self, addr: u64, size: u64) {
        self.split_at(addr);
        self.split_at(addr + size);
        let keys: Vec<u64> = self
            .blocks
            .range(addr..addr + size)
            .map(|(k, _)| *k)
            .collect();
        for k in keys {
            self.blocks.remove(&k);
        }
    }

    /// True if the whole range is covered by blocks for which `pred` holds.
    pub fn all(&self, addr: u64, size: u64, pred: impl Fn(&Block) -> bool) -> bool {
        let mut cursor = addr;
        let end = addr + size;
        while cursor < end {
            let Some((&start, b)) = self.blocks.range(..=cursor).next_back() else {
                return false;
            };
            if start + b.size <= cursor || !pred(b) {
                return false;
            }
            cursor = start + b.size;
        }
        true
    }

    /// Total size of all mapped blocks.
    pub fn used(&self) -> u64 {
        self.blocks.values().map(|b| b.size).sum()
    }

    /// `svcQueryMemory`: the block containing `addr`, merged with
    /// neighbours that have the same state, permission and attributes.
    pub fn query(&self, addr: u64) -> MemoryInfo {
        if addr >= ADDRESS_SPACE_END {
            return MemoryInfo {
                base: ADDRESS_SPACE_END,
                size: ADDRESS_SPACE_END.wrapping_neg(),
                state: MemState::Inaccessible,
                attr: 0,
                perm: 0,
            };
        }
        if let Some((&start, &b)) = self.blocks.range(..=addr).next_back() {
            if addr < start + b.size {
                let same = |o: &Block| o.state == b.state && o.perm == b.perm && o.attr == b.attr;
                let mut base = start;
                while let Some((&s, p)) = self.blocks.range(..base).next_back() {
                    if s + p.size == base && same(p) {
                        base = s;
                    } else {
                        break;
                    }
                }
                let mut end = start + b.size;
                while let Some(n) = self.blocks.get(&end) {
                    if same(n) {
                        end += n.size;
                    } else {
                        break;
                    }
                }
                return MemoryInfo {
                    base,
                    size: end - base,
                    state: b.state,
                    attr: b.attr,
                    perm: b.perm,
                };
            }
        }
        // Free gap between the previous and the next block.
        let base = self
            .blocks
            .range(..=addr)
            .next_back()
            .map_or(0, |(&s, b)| s + b.size);
        let end = self
            .blocks
            .range(addr..)
            .next()
            .map_or(ADDRESS_SPACE_END, |(&s, _)| s);
        MemoryInfo {
            base,
            size: end - base,
            state: MemState::Free,
            attr: 0,
            perm: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap(size: u64) -> Block {
        Block {
            size,
            state: MemState::Heap,
            perm: PERM_R | PERM_W,
            attr: 0,
        }
    }

    #[test]
    fn query_free_mapped_and_outside() {
        let mut m = MemoryMap::new();
        m.insert(0x10000, heap(0x2000));
        let free = m.query(0x5000);
        assert_eq!(
            (free.base, free.size, free.state),
            (0, 0x10000, MemState::Free)
        );
        let h = m.query(0x11000);
        assert_eq!((h.base, h.size, h.state), (0x10000, 0x2000, MemState::Heap));
        let after = m.query(0x20000);
        assert_eq!(
            (after.base, after.size),
            (0x12000, ADDRESS_SPACE_END - 0x12000)
        );
        assert_eq!(m.query(ADDRESS_SPACE_END).state, MemState::Inaccessible);
    }

    #[test]
    fn neighbours_merge_and_updates_split() {
        let mut m = MemoryMap::new();
        m.insert(0x10000, heap(0x1000));
        m.insert(0x11000, heap(0x1000));
        assert_eq!(m.query(0x10000).size, 0x2000, "identical neighbours merge");
        m.update(0x10800 & !0xFFF, 0x1000, |b| b.perm = 0);
        let q = m.query(0x10000);
        assert_eq!((q.size, q.perm), (0x1000, 0));
        assert_eq!(m.query(0x11000).perm, PERM_R | PERM_W);
        assert!(m.all(0x10000, 0x2000, |b| b.state == MemState::Heap));
        m.remove(0x10000, 0x1000);
        assert!(m.is_free(0x10000, 0x1000) && !m.is_free(0x10000, 0x1001));
        assert_eq!(m.used(), 0x1000);
    }
}
