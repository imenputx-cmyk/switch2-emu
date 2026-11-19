//! Guest virtual memory.
//!
//! Memory is split into 4 KiB pages. Each page is mapped on demand with a
//! set of permissions. All accesses are little-endian, like the guest.
//!
//! This is the simple, always-correct version: a hash map from page number
//! to page. A faster flat mapping can replace it later behind the same API.

use std::collections::HashMap;
use std::fmt;

pub const PAGE_BITS: u32 = 12;
pub const PAGE_SIZE: u64 = 1 << PAGE_BITS;
const PAGE_MASK: u64 = PAGE_SIZE - 1;

/// Access permissions of a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Perm {
    pub read: bool,
    pub write: bool,
    pub exec: bool,
}

impl Perm {
    pub const R: Perm = Perm {
        read: true,
        write: false,
        exec: false,
    };
    pub const RW: Perm = Perm {
        read: true,
        write: true,
        exec: false,
    };
    pub const RX: Perm = Perm {
        read: true,
        write: false,
        exec: true,
    };
    pub const RWX: Perm = Perm {
        read: true,
        write: true,
        exec: true,
    };
}

/// Why a memory access failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    Unmapped,
    NotReadable,
    NotWritable,
    NotExecutable,
}

/// A failed guest memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    pub addr: u64,
    pub kind: FaultKind,
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} at {:#x}", self.kind, self.addr)
    }
}

impl std::error::Error for Fault {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapError {
    /// Address or size is not a multiple of the page size.
    Unaligned,
    /// The range wraps past the end of the address space.
    Overflow,
}

struct Page {
    data: Box<[u8; PAGE_SIZE as usize]>,
    perm: Perm,
}

#[derive(Clone, Copy)]
enum Access {
    /// Mapped, permissions ignored (loader writes).
    Any,
    Read,
    Write,
    Fetch,
}

#[derive(Default)]
pub struct Memory {
    pages: HashMap<u64, Page>,
}

impl Memory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Maps `size` bytes at `addr` as zeroed memory. Both must be
    /// page-aligned. Mapping over an existing page replaces it.
    pub fn map(&mut self, addr: u64, size: u64, perm: Perm) -> Result<(), MapError> {
        if addr & PAGE_MASK != 0 || size & PAGE_MASK != 0 {
            return Err(MapError::Unaligned);
        }
        if size > 0 {
            // The range may end exactly at the top of the address space.
            addr.checked_add(size - 1).ok_or(MapError::Overflow)?;
        }
        let first = addr >> PAGE_BITS;
        for i in 0..(size >> PAGE_BITS) {
            self.pages.insert(
                first + i,
                Page {
                    data: Box::new([0; PAGE_SIZE as usize]),
                    perm,
                },
            );
        }
        Ok(())
    }

    /// Changes the permissions of already mapped pages.
    pub fn protect(&mut self, addr: u64, size: u64, perm: Perm) -> Result<(), Fault> {
        let first = addr >> PAGE_BITS;
        let last = addr.saturating_add(size.max(1) - 1) >> PAGE_BITS;
        for page in first..=last {
            match self.pages.get_mut(&page) {
                Some(p) => p.perm = perm,
                None => {
                    return Err(Fault {
                        addr: page << PAGE_BITS,
                        kind: FaultKind::Unmapped,
                    })
                }
            }
        }
        Ok(())
    }

    pub fn is_mapped(&self, addr: u64) -> bool {
        self.pages.contains_key(&(addr >> PAGE_BITS))
    }

    fn check(&self, addr: u64, access: Access) -> Result<&Page, Fault> {
        let page = self.pages.get(&(addr >> PAGE_BITS)).ok_or(Fault {
            addr,
            kind: FaultKind::Unmapped,
        })?;
        let (ok, kind) = match access {
            Access::Any => (true, FaultKind::Unmapped),
            Access::Read => (page.perm.read, FaultKind::NotReadable),
            Access::Write => (page.perm.write, FaultKind::NotWritable),
            Access::Fetch => (page.perm.exec, FaultKind::NotExecutable),
        };
        if ok {
            Ok(page)
        } else {
            Err(Fault { addr, kind })
        }
    }

    /// Checks every page in the range first, so a faulting access has no
    /// partial effect.
    fn check_range(&self, addr: u64, len: usize, access: Access) -> Result<(), Fault> {
        if len == 0 {
            return Ok(());
        }
        let end = addr.wrapping_add(len as u64 - 1);
        let mut page_addr = addr;
        loop {
            self.check(page_addr, access)?;
            let next = (page_addr | PAGE_MASK).wrapping_add(1);
            // Stop when the next page starts after the end (handles wrap).
            if next.wrapping_sub(addr) > end.wrapping_sub(addr) {
                return Ok(());
            }
            page_addr = next;
        }
    }

    fn copy_out(&self, addr: u64, buf: &mut [u8]) {
        let mut done = 0;
        while done < buf.len() {
            let a = addr.wrapping_add(done as u64);
            let off = (a & PAGE_MASK) as usize;
            let n = (PAGE_SIZE as usize - off).min(buf.len() - done);
            let page = &self.pages[&(a >> PAGE_BITS)];
            buf[done..done + n].copy_from_slice(&page.data[off..off + n]);
            done += n;
        }
    }

    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.check_range(addr, buf.len(), Access::Read)?;
        self.copy_out(addr, buf);
        Ok(())
    }

    pub fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault> {
        self.check_range(addr, data.len(), Access::Write)?;
        self.copy_in(addr, data);
        Ok(())
    }

    /// Writes ignoring page permissions, for loaders. Pages must be mapped.
    pub fn write_privileged(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault> {
        self.check_range(addr, data.len(), Access::Any)?;
        self.copy_in(addr, data);
        Ok(())
    }

    fn copy_in(&mut self, addr: u64, data: &[u8]) {
        let mut done = 0;
        while done < data.len() {
            let a = addr.wrapping_add(done as u64);
            let off = (a & PAGE_MASK) as usize;
            let n = (PAGE_SIZE as usize - off).min(data.len() - done);
            let page = self.pages.get_mut(&(a >> PAGE_BITS)).expect("checked");
            page.data[off..off + n].copy_from_slice(&data[done..done + n]);
            done += n;
        }
    }

    /// Fetches one instruction word. Requires execute permission.
    pub fn fetch_u32(&self, addr: u64) -> Result<u32, Fault> {
        self.check_range(addr, 4, Access::Fetch)?;
        let mut b = [0; 4];
        self.copy_out(addr, &mut b);
        Ok(u32::from_le_bytes(b))
    }

    /// Reads `size` bytes (1, 2, 4 or 8), zero-extended to 64 bits.
    pub fn read_uint(&self, addr: u64, size: usize) -> Result<u64, Fault> {
        debug_assert!(matches!(size, 1 | 2 | 4 | 8));
        let mut b = [0u8; 8];
        self.read(addr, &mut b[..size])?;
        Ok(u64::from_le_bytes(b))
    }

    /// Writes the low `size` bytes (1, 2, 4 or 8) of `value`.
    pub fn write_uint(&mut self, addr: u64, size: usize, value: u64) -> Result<(), Fault> {
        debug_assert!(matches!(size, 1 | 2 | 4 | 8));
        self.write(addr, &value.to_le_bytes()[..size])
    }

    pub fn read_u64(&self, addr: u64) -> Result<u64, Fault> {
        self.read_uint(addr, 8)
    }

    pub fn write_u64(&mut self, addr: u64, value: u64) -> Result<(), Fault> {
        self.write_uint(addr, 8, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_requires_alignment() {
        let mut m = Memory::new();
        assert_eq!(m.map(0x1001, PAGE_SIZE, Perm::RW), Err(MapError::Unaligned));
        assert_eq!(m.map(0x1000, 100, Perm::RW), Err(MapError::Unaligned));
        assert_eq!(
            m.map(u64::MAX - PAGE_SIZE + 1, PAGE_SIZE * 2, Perm::RW),
            Err(MapError::Overflow)
        );
    }

    #[test]
    fn mapped_memory_is_zeroed_and_roundtrips() {
        let mut m = Memory::new();
        m.map(0x1000, PAGE_SIZE, Perm::RW).unwrap();
        assert_eq!(m.read_u64(0x1000).unwrap(), 0);
        m.write_u64(0x1008, 0x1122_3344_5566_7788).unwrap();
        assert_eq!(m.read_u64(0x1008).unwrap(), 0x1122_3344_5566_7788);
        assert_eq!(m.read_uint(0x1008, 1).unwrap(), 0x88, "little-endian");
        assert_eq!(m.read_uint(0x100C, 4).unwrap(), 0x1122_3344);
    }

    #[test]
    fn access_across_page_boundary() {
        let mut m = Memory::new();
        m.map(0x1000, PAGE_SIZE * 2, Perm::RW).unwrap();
        m.write_u64(0x1FFC, 0xAABB_CCDD_EEFF_0011).unwrap();
        assert_eq!(m.read_u64(0x1FFC).unwrap(), 0xAABB_CCDD_EEFF_0011);
        assert_eq!(m.read_uint(0x2000, 4).unwrap(), 0xAABB_CCDD);
    }

    #[test]
    fn faults_have_no_partial_effect() {
        let mut m = Memory::new();
        m.map(0x1000, PAGE_SIZE, Perm::RW).unwrap();
        // Second half of the write falls into an unmapped page.
        let err = m.write_u64(0x1FFC, u64::MAX).unwrap_err();
        assert_eq!(
            err,
            Fault {
                addr: 0x2000,
                kind: FaultKind::Unmapped
            }
        );
        assert_eq!(m.read_uint(0x1FFC, 4).unwrap(), 0, "nothing was written");
    }

    #[test]
    fn permissions_are_enforced() {
        let mut m = Memory::new();
        m.map(0x1000, PAGE_SIZE, Perm::RX).unwrap();
        m.map(0x2000, PAGE_SIZE, Perm::RW).unwrap();
        assert_eq!(
            m.write_u64(0x1000, 1).unwrap_err().kind,
            FaultKind::NotWritable
        );
        assert_eq!(
            m.fetch_u32(0x2000).unwrap_err().kind,
            FaultKind::NotExecutable
        );
        assert_eq!(m.read_u64(0x5000).unwrap_err().kind, FaultKind::Unmapped);

        m.write_privileged(0x1000, &0xD503_201Fu32.to_le_bytes())
            .unwrap();
        assert_eq!(m.fetch_u32(0x1000).unwrap(), 0xD503_201F);

        m.protect(0x1000, PAGE_SIZE, Perm::R).unwrap();
        assert_eq!(
            m.fetch_u32(0x1000).unwrap_err().kind,
            FaultKind::NotExecutable
        );
    }

    #[test]
    fn access_at_top_of_address_space() {
        let mut m = Memory::new();
        m.map(u64::MAX - PAGE_SIZE + 1, PAGE_SIZE, Perm::RW)
            .unwrap();
        m.write_u64(u64::MAX - 7, 42).unwrap();
        assert_eq!(m.read_u64(u64::MAX - 7).unwrap(), 42);
        // Wrapping past the top hits unmapped page 0.
        assert_eq!(
            m.read_u64(u64::MAX - 3).unwrap_err().kind,
            FaultKind::Unmapped
        );
    }
}
