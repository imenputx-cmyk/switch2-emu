//! A GPU address space: GPU virtual addresses mapped onto guest memory
//! (the Tegra GPU shares memory with the CPU, so a mapping is just a
//! window into the process's address space).

use emu_memory::Memory;
use std::collections::BTreeMap;

/// GPU page size used for allocations ("big pages").
pub const BIG_PAGE: u64 = 0x2_0000;
/// Allocations without a fixed address start here (below is left to
/// fixed allocations) and must stay under 1 << 40.
const ALLOC_BASE: u64 = 0x4_0000_0000;
const VA_LIMIT: u64 = 1 << 40;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mapping {
    pub size: u64,
    /// Guest (CPU) address the GPU address maps to.
    pub cpu: u64,
}

#[derive(Debug, Default)]
pub struct AddressSpace {
    /// Reserved ranges (AllocSpace): start -> size.
    reserved: BTreeMap<u64, u64>,
    /// Mappings: GPU start -> mapping.
    mappings: BTreeMap<u64, Mapping>,
    next: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFault {
    /// No mapping covers this GPU address.
    Unmapped(u64),
    /// The mapping points at guest memory that is not there.
    Guest(u64),
}

impl AddressSpace {
    pub fn new() -> Self {
        AddressSpace {
            next: ALLOC_BASE,
            ..Default::default()
        }
    }

    fn overlaps_reserved(&self, va: u64, size: u64) -> bool {
        self.reserved
            .range(..va + size)
            .next_back()
            .is_some_and(|(&s, &l)| s + l > va)
    }

    /// Reserves `size` bytes at a free address aligned to `align`.
    pub fn reserve(&mut self, size: u64, align: u64) -> Option<u64> {
        let align = align.max(BIG_PAGE);
        let mut va = self.next.next_multiple_of(align);
        while self.overlaps_reserved(va, size) {
            let (&s, &l) = self.reserved.range(..va + size).next_back()?;
            va = (s + l).next_multiple_of(align);
        }
        if va + size > VA_LIMIT {
            return None;
        }
        self.reserved.insert(va, size);
        self.next = va + size;
        Some(va)
    }

    /// Reserves a range at a fixed address.
    pub fn reserve_fixed(&mut self, va: u64, size: u64) -> bool {
        if va + size > VA_LIMIT || self.overlaps_reserved(va, size) {
            return false;
        }
        self.reserved.insert(va, size);
        true
    }

    pub fn free(&mut self, va: u64) -> bool {
        let Some(size) = self.reserved.remove(&va) else {
            return false;
        };
        let inside: Vec<u64> = self
            .mappings
            .range(va..va + size)
            .map(|(&k, _)| k)
            .collect();
        for k in inside {
            self.mappings.remove(&k);
        }
        true
    }

    /// Maps `size` bytes at `va` (which must not overlap a mapping) to
    /// guest address `cpu`. Mapping inside a reserved range is the usual
    /// case; outside one the range is reserved too.
    pub fn map(&mut self, va: u64, cpu: u64, size: u64) -> bool {
        if size == 0 || va + size > VA_LIMIT {
            return false;
        }
        if self
            .mappings
            .range(..va + size)
            .next_back()
            .is_some_and(|(&s, m)| s + m.size > va)
        {
            return false;
        }
        if !self
            .reserved
            .iter()
            .any(|(&s, &l)| s <= va && va + size <= s + l)
        {
            if self.overlaps_reserved(va, size) {
                return false;
            }
            self.reserved.insert(va, size);
        }
        self.mappings.insert(va, Mapping { size, cpu });
        true
    }

    /// Maps somewhere free; returns the GPU address.
    pub fn map_anywhere(&mut self, cpu: u64, size: u64, align: u64) -> Option<u64> {
        let va = self.reserve(size.next_multiple_of(BIG_PAGE), align)?;
        self.mappings.insert(va, Mapping { size, cpu });
        Some(va)
    }

    /// Removes the mapping that starts at `va`.
    /// Removes whatever is mapped in `[va, va + size)`, splitting mappings
    /// that cross its ends.
    pub fn unmap_range(&mut self, va: u64, size: u64) {
        let end = va.saturating_add(size);
        let hit: Vec<(u64, Mapping)> = self
            .mappings
            .range(..end)
            .rev()
            .take_while(|(&s, m)| s + m.size > va)
            .map(|(&s, &m)| (s, m))
            .collect();
        for (s, m) in hit {
            self.mappings.remove(&s);
            if s < va {
                self.mappings.insert(
                    s,
                    Mapping {
                        size: va - s,
                        cpu: m.cpu,
                    },
                );
            }
            let m_end = s + m.size;
            if m_end > end {
                self.mappings.insert(
                    end,
                    Mapping {
                        size: m_end - end,
                        cpu: m.cpu + (end - s),
                    },
                );
            }
        }
    }

    /// Maps `[va, va + size)` to guest `cpu`, replacing what was there
    /// (Switch 2 maps through the graphics server, not reservations).
    pub fn map_over(&mut self, va: u64, cpu: u64, size: u64) -> bool {
        if size == 0 || va.saturating_add(size) > VA_LIMIT {
            return false;
        }
        self.unmap_range(va, size);
        self.mappings.insert(va, Mapping { size, cpu });
        true
    }

    pub fn unmap(&mut self, va: u64) -> Option<Mapping> {
        self.mappings.remove(&va)
    }

    /// The mapping containing `va`, with its start.
    pub fn find(&self, va: u64) -> Option<(u64, Mapping)> {
        let (&s, &m) = self.mappings.range(..=va).next_back()?;
        (va < s + m.size).then_some((s, m))
    }

    pub fn translate(&self, va: u64) -> Option<u64> {
        self.find(va).map(|(s, m)| m.cpu + (va - s))
    }

    /// Reads GPU memory, possibly across mappings.
    pub fn read(&self, mem: &Memory, mut va: u64, buf: &mut [u8]) -> Result<(), GpuFault> {
        let mut done = 0;
        while done < buf.len() {
            let (s, m) = self.find(va).ok_or(GpuFault::Unmapped(va))?;
            let n = ((s + m.size - va) as usize).min(buf.len() - done);
            let cpu = m.cpu + (va - s);
            mem.read(cpu, &mut buf[done..done + n])
                .map_err(|_| GpuFault::Guest(cpu))?;
            done += n;
            va += n as u64;
        }
        Ok(())
    }

    /// Writes GPU memory (ignoring CPU-side page permissions, like the
    /// GPU does).
    pub fn write(&self, mem: &mut Memory, mut va: u64, data: &[u8]) -> Result<(), GpuFault> {
        let mut done = 0;
        while done < data.len() {
            let (s, m) = self.find(va).ok_or(GpuFault::Unmapped(va))?;
            let n = ((s + m.size - va) as usize).min(data.len() - done);
            let cpu = m.cpu + (va - s);
            mem.write_privileged(cpu, &data[done..done + n])
                .map_err(|_| GpuFault::Guest(cpu))?;
            done += n;
            va += n as u64;
        }
        Ok(())
    }

    pub fn read_u32(&self, mem: &Memory, va: u64) -> Result<u32, GpuFault> {
        let mut b = [0u8; 4];
        self.read(mem, va, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_map_translate() {
        let mut a = AddressSpace::new();
        let va = a.reserve(0x10_0000, 0).unwrap();
        assert_eq!(va % BIG_PAGE, 0);
        assert!(a.map(va + 0x1000, 0x8000_0000, 0x2000));
        assert!(!a.map(va + 0x2000, 0x9000_0000, 0x1000), "overlap");
        assert_eq!(a.translate(va + 0x1004), Some(0x8000_0004));
        assert_eq!(a.translate(va + 0x3000), None);
        let other = a.reserve(0x1000, 0).unwrap();
        assert!(other >= va + 0x10_0000);
        assert!(a.unmap(va + 0x1000).is_some());
        assert_eq!(a.translate(va + 0x1004), None);
        assert!(a.free(va));
    }
}
