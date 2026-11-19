//! Kernel objects that services hand out to clients: events and shared
//! memory. Handles refer to them by index (`KObject::Event { id }`, ...).

/// A shared memory block. Services fill `init`; when the client maps it
/// the contents are copied in, and `mapped_at` lets the service update
/// the mapping later (e.g. input state every frame).
#[derive(Debug, Clone)]
pub struct SharedMemory {
    pub size: u64,
    pub init: Vec<u8>,
    pub mapped_at: Option<u64>,
}

#[derive(Debug, Default)]
pub struct KernelObjects {
    /// Signaled state of each event.
    pub events: Vec<bool>,
    pub shared_memory: Vec<SharedMemory>,
}

impl KernelObjects {
    pub fn new_event(&mut self, signaled: bool) -> usize {
        self.events.push(signaled);
        self.events.len() - 1
    }

    pub fn new_shared_memory(&mut self, size: u64, init: Vec<u8>) -> usize {
        self.shared_memory.push(SharedMemory {
            size,
            init,
            mapped_at: None,
        });
        self.shared_memory.len() - 1
    }
}
