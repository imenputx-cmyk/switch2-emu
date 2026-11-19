//! Kernel object handles.
//!
//! A handle packs a table index in its low 15 bits and a counter above
//! them, so stale handles are detected. Two pseudo-handles always refer to
//! the calling thread and process.

pub const CURRENT_THREAD: u32 = 0xFFFF_8000;
pub const CURRENT_PROCESS: u32 = 0xFFFF_8001;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KObject {
    Thread {
        id: u64,
    },
    Process,
    /// A client session to an IPC service (index into the kernel's
    /// session table).
    Session {
        id: usize,
    },
    /// The readable side of an event (index into `KernelObjects::events`).
    Event {
        id: usize,
    },
    SharedMemory {
        id: usize,
    },
}

#[derive(Debug, Default)]
pub struct HandleTable {
    slots: Vec<Option<(u16, KObject)>>,
    counter: u16,
}

impl HandleTable {
    /// Horizon's default handle table size.
    pub const CAPACITY: usize = 1024;

    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, obj: KObject) -> Option<u32> {
        self.counter = self.counter.wrapping_add(1).max(1) & 0x7FFF;
        let counter = self.counter.max(1);
        let index = match self.slots.iter().position(Option::is_none) {
            Some(i) => i,
            None if self.slots.len() < Self::CAPACITY => {
                self.slots.push(None);
                self.slots.len() - 1
            }
            None => return None,
        };
        self.slots[index] = Some((counter, obj));
        Some((u32::from(counter) << 15) | index as u32)
    }

    pub fn get(&self, handle: u32) -> Option<&KObject> {
        let index = (handle & 0x7FFF) as usize;
        let counter = ((handle >> 15) & 0xFFFF) as u16;
        match self.slots.get(index) {
            Some(Some((c, obj))) if *c == counter => Some(obj),
            _ => None,
        }
    }

    pub fn close(&mut self, handle: u32) -> bool {
        if self.get(handle).is_none() {
            return false;
        }
        self.slots[(handle & 0x7FFF) as usize] = None;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_close_and_stale_handles() {
        let mut t = HandleTable::new();
        let a = t.insert(KObject::Process).unwrap();
        assert_eq!(t.get(a), Some(&KObject::Process));
        assert!(t.close(a));
        assert!(!t.close(a), "double close fails");
        let b = t.insert(KObject::Thread { id: 1 }).unwrap();
        assert_eq!(a & 0x7FFF, b & 0x7FFF, "slot is reused");
        assert_ne!(a, b, "but the counter makes the old handle stale");
        assert_eq!(t.get(a), None);
    }
}
