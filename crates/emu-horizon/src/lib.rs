//! High-level emulation (HLE) of Horizon, the Switch operating system
//! kernel: the process address space, kernel objects and SVCs, and booting
//! homebrew through the homebrew ABI.
//!
//! References used (documentation and permissively licensed code, nothing
//! copied): the switchbrew wiki pages on SVCs, NRO and the homebrew ABI,
//! and libnx (ISC) for what homebrew expects at startup. Ryujinx (MIT) and
//! yuzu informed the overall structure.
//!
//! Only one thread is emulated so far, so synchronization SVCs are
//! simplified (documented at each SVC).

pub mod handles;
pub mod ipc;
pub mod kobjects;
pub mod memmap;
pub mod result;
pub mod services;

use emu_cpu::{Cpu, Exit};
use emu_loader::nro::parse_nro;
use emu_loader::LoadError;
use emu_memory::{Memory, Perm};
use handles::{HandleTable, KObject, CURRENT_PROCESS, CURRENT_THREAD};
use memmap::{Block, MemState, MemoryMap, ATTR_LOCKED, ATTR_UNCACHED, PERM_R, PERM_W, PERM_X};
use result::*;

const PAGE: u64 = 0x1000;
/// SetHeapSize works in 2 MiB units.
const HEAP_ALIGN: u64 = 0x20_0000;

/// Address space layout of the emulated process. Homebrew discovers it
/// with `svcGetInfo`, so the exact values only need to be consistent.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub aslr: (u64, u64),
    pub heap: (u64, u64),
    pub alias: (u64, u64),
    pub stack: (u64, u64),
    /// Where the homebrew image is loaded.
    pub image_base: u64,
    /// The loader's page: config entries and the return stub.
    pub loader_page: u64,
    pub tls_page: u64,
    pub main_stack_size: u64,
    /// Total memory the application may use.
    pub total_memory: u64,
}

impl Default for Layout {
    fn default() -> Self {
        Layout {
            aslr: (0x0800_0000, memmap::ADDRESS_SPACE_END - 0x0800_0000),
            heap: (0x10_0000_0000, 0x1_8000_0000),
            alias: (0x20_0000_0000, 0x10_0000_0000),
            stack: (0x40_0000_0000, 0x8000_0000),
            image_base: 0x0800_0000,
            loader_page: 0x0700_0000,
            tls_page: 0x0600_0000,
            main_stack_size: 0x10_0000,
            total_memory: 0x1_0000_0000,
        }
    }
}

/// Why the emulated process stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// svcExitProcess.
    ExitProcess,
    /// The homebrew returned to the loader with this value in X0.
    ReturnedToLoader(u64),
    /// svcBreak.
    Break {
        reason: u64,
        arg: u64,
        info: u64,
    },
    /// An SVC that is not implemented yet.
    UnimplementedSvc {
        id: u16,
        pc: u64,
    },
    /// An IPC command that is not implemented yet.
    UnimplementedIpc {
        service: String,
        command: u32,
        control: bool,
    },
    /// A malformed IPC message.
    BadIpc {
        pc: u64,
        error: ipc::ParseError,
    },
    /// The only thread waits forever on objects nothing can signal.
    Deadlock {
        pc: u64,
    },
    /// The CPU stopped (fault, undefined instruction, BRK, ...).
    Cpu(Exit),
    StepLimit,
}

/// Values of the homebrew loader config (https://switchbrew.org/wiki/Homebrew_ABI).
mod config_key {
    pub const END_OF_LIST: u32 = 0;
    pub const MAIN_THREAD_HANDLE: u32 = 1;
    pub const SYSCALL_AVAILABLE_HINT: u32 = 6;
    pub const APPLET_TYPE: u32 = 7;
    pub const RANDOM_SEED: u32 = 14;
    pub const HOS_VERSION: u32 = 16;
    pub const SYSCALL_AVAILABLE_HINT2: u32 = 17;
}
const LOADER_INFO: &str = "switch2-emu";

/// An IPC session: the service object behind it and, once converted to
/// a domain, its table of objects (index = object id, 0 unused).
type Domain = std::rc::Rc<std::cell::RefCell<Vec<Option<services::ServiceRef>>>>;

struct Session {
    object: services::ServiceRef,
    /// Shared by clones of the session (CloneCurrentObject).
    domain: Option<Domain>,
    closed: bool,
}

pub struct Horizon {
    pub mem: Memory,
    pub cpu: Cpu,
    pub map: MemoryMap,
    pub handles: HandleTable,
    pub layout: Layout,
    heap_size: u64,
    main_thread_handle: u32,
    thread_id: u64,
    process_id: u64,
    /// MapMemory aliases: destination -> (source, size).
    aliases: Vec<(u64, u64, u64)>,
    /// Text written with svcOutputDebugString.
    pub debug_output: Vec<u8>,
    /// IPC sessions, indexed by `KObject::Session::id`.
    sessions: Vec<Session>,
    /// Events and shared memory created by services.
    pub kobjects: kobjects::KernelObjects,
    /// Log of SVC calls, for diagnostics.
    pub trace_svcs: bool,
}

impl Default for Horizon {
    fn default() -> Self {
        Self::new()
    }
}

impl Horizon {
    pub fn new() -> Self {
        Horizon {
            mem: Memory::new(),
            cpu: Cpu::new(),
            map: MemoryMap::new(),
            handles: HandleTable::new(),
            layout: Layout::default(),
            heap_size: 0,
            main_thread_handle: 0,
            thread_id: 0x51,
            process_id: 0x50,
            aliases: Vec::new(),
            debug_output: Vec::new(),
            sessions: Vec::new(),
            kobjects: kobjects::KernelObjects::default(),
            trace_svcs: false,
        }
    }

    /// Maps a block in both the kernel's map and guest memory.
    fn map_block(&mut self, addr: u64, size: u64, state: MemState, perm: u32) {
        let host = Perm {
            read: perm & PERM_R != 0,
            write: perm & PERM_W != 0,
            exec: perm & PERM_X != 0,
        };
        self.mem.map(addr, size, host).expect("aligned mapping");
        self.map.insert(
            addr,
            Block {
                size,
                state,
                perm,
                attr: 0,
            },
        );
    }

    fn set_host_perm(&mut self, addr: u64, size: u64, perm: u32) {
        let host = Perm {
            read: perm & PERM_R != 0,
            write: perm & PERM_W != 0,
            exec: perm & PERM_X != 0,
        };
        self.mem.protect(addr, size, host).expect("mapped");
    }

    /// Loads an NRO and prepares the main thread to enter it the way the
    /// homebrew loader does: X0 = config entries, X1 = -1, LR = return to
    /// the loader.
    pub fn boot_homebrew(&mut self, nro_file: &[u8]) -> Result<(), LoadError> {
        let nro = parse_nro(nro_file)?;
        let l = self.layout;
        let base = l.image_base;

        // Text and RO are code, data and .bss are mutable.
        let text_end = u64::from(nro.ro.offset);
        let ro_end = u64::from(nro.data.offset);
        self.map_block(base, text_end, MemState::CodeStatic, PERM_R | PERM_X);
        if ro_end > text_end {
            self.map_block(
                base + text_end,
                ro_end - text_end,
                MemState::CodeStatic,
                PERM_R,
            );
        }
        self.map_block(
            base + ro_end,
            nro.image_size - ro_end,
            MemState::CodeMutable,
            PERM_R | PERM_W,
        );
        let file_end = (u64::from(nro.data.offset) + u64::from(nro.data.size)) as usize;
        self.mem
            .write_privileged(base, &nro_file[..file_end])
            .expect("mapped above");

        // Main thread: stack, TLS and handle.
        let stack_top = l.stack.0 + l.main_stack_size;
        self.map_block(
            l.stack.0,
            l.main_stack_size,
            MemState::Stack,
            PERM_R | PERM_W,
        );
        self.map_block(l.tls_page, PAGE, MemState::ThreadLocal, PERM_R | PERM_W);
        self.main_thread_handle = self
            .handles
            .insert(KObject::Thread { id: self.thread_id })
            .expect("empty table");

        // Loader page: return stub (BRK, recognised by `run`) and config.
        self.map_block(l.loader_page, PAGE, MemState::CodeStatic, PERM_R | PERM_X);
        let brk_loader: u32 = 0xD420_0000 | (0x4C44 << 5); // brk #0x4c44 ("LD")
        self.mem
            .write_privileged(l.loader_page, &brk_loader.to_le_bytes())
            .unwrap();
        let info_addr = l.loader_page + 0x40;
        self.mem
            .write_privileged(info_addr, LOADER_INFO.as_bytes())
            .unwrap();
        let config_addr = l.loader_page + 0x100;
        let entries: [(u32, u32, u64, u64); 7] = [
            (
                config_key::MAIN_THREAD_HANDLE,
                1,
                u64::from(self.main_thread_handle),
                0,
            ),
            (config_key::APPLET_TYPE, 1, 0, 0),
            (config_key::SYSCALL_AVAILABLE_HINT, 0, u64::MAX, u64::MAX),
            (config_key::SYSCALL_AVAILABLE_HINT2, 0, u64::MAX, 0),
            (config_key::HOS_VERSION, 0, 21 << 16, 0),
            (
                config_key::RANDOM_SEED,
                0,
                0x5357_4954_4348_3245,
                0x4D55_4C41_544F_5221,
            ),
            (
                config_key::END_OF_LIST,
                0,
                info_addr,
                LOADER_INFO.len() as u64,
            ),
        ];
        for (i, (key, flags, v0, v1)) in entries.iter().enumerate() {
            let mut e = [0u8; 24];
            e[0..4].copy_from_slice(&key.to_le_bytes());
            e[4..8].copy_from_slice(&flags.to_le_bytes());
            e[8..16].copy_from_slice(&v0.to_le_bytes());
            e[16..24].copy_from_slice(&v1.to_le_bytes());
            self.mem
                .write_privileged(config_addr + 24 * i as u64, &e)
                .unwrap();
        }

        self.cpu = Cpu::new();
        self.cpu.set_pc(base);
        self.cpu.write_sp(31, stack_top);
        self.cpu.write_xzr(0, config_addr);
        self.cpu.write_xzr(1, u64::MAX);
        self.cpu.write_xzr(30, l.loader_page);
        self.cpu.tpidrro_el0 = l.tls_page;
        Ok(())
    }

    /// Runs until the process stops or `max_steps` instructions ran.
    pub fn run(&mut self, max_steps: u64) -> Stop {
        let start = self.cpu.counter;
        loop {
            let budget = max_steps.saturating_sub(self.cpu.counter - start);
            if budget == 0 {
                return Stop::StepLimit;
            }
            match self.cpu.run(&mut self.mem, budget) {
                None => return Stop::StepLimit,
                Some(Exit::Svc(id)) => {
                    if let Err(stop) = self.svc(id) {
                        return stop;
                    }
                }
                Some(Exit::Brk(0x4C44)) if self.cpu.pc() == self.layout.loader_page => {
                    return Stop::ReturnedToLoader(self.cpu.read_xzr(0));
                }
                Some(exit) => return Stop::Cpu(exit),
            }
        }
    }

    fn x(&self, n: u8) -> u64 {
        self.cpu.read_xzr(n)
    }

    fn set_x(&mut self, n: u8, v: u64) {
        self.cpu.write_xzr(n, v);
    }

    fn ret(&mut self, code: ResultCode) {
        self.set_x(0, u64::from(code));
    }

    fn read_cstr(&self, addr: u64, max: usize) -> Option<String> {
        let mut out = Vec::new();
        for i in 0..max as u64 {
            let b = self.mem.read_uint(addr + i, 1).ok()? as u8;
            if b == 0 {
                break;
            }
            out.push(b);
        }
        String::from_utf8(out).ok()
    }

    /// Dispatches one SVC. Arguments and results follow
    /// https://switchbrew.org/wiki/SVC (W0 holds the result code).
    fn svc(&mut self, id: u16) -> Result<(), Stop> {
        if self.trace_svcs {
            eprintln!(
                "[svc] 0x{id:02X} x0={:#x} x1={:#x} x2={:#x}",
                self.x(0),
                self.x(1),
                self.x(2)
            );
        }
        let pc = self.cpu.pc() - 4;
        match id {
            0x01 => self.svc_set_heap_size(),
            0x02 => self.svc_set_memory_permission(),
            0x03 => self.svc_set_memory_attribute(),
            0x04 => self.svc_map_memory(),
            0x05 => self.svc_unmap_memory(),
            0x06 => self.svc_query_memory(),
            0x07 => return Err(Stop::ExitProcess),
            0x0B => {
                // SleepThread: with a single thread there is nothing to
                // yield to. Let time pass so polling loops make progress.
                let ns = self.x(0) as i64;
                if ns > 0 {
                    self.cpu.counter += (ns as u64).saturating_mul(192) / 10_000;
                }
            }
            0x0C => {
                // GetThreadPriority: the main thread runs at 44 (0x2C).
                if !self.is_thread(self.x(1) as u32) {
                    self.ret(INVALID_HANDLE);
                    return Ok(());
                }
                self.ret(SUCCESS);
                self.set_x(1, 0x2C);
            }
            0x0D | 0x0F => self.ret(SUCCESS), // Set priority / core mask: accepted.
            0x0E => {
                // GetThreadCoreMask: core 0, affinity mask core 0.
                self.ret(SUCCESS);
                self.set_x(1, 0);
                self.set_x(2, 1);
            }
            0x10 => self.set_x(0, 0), // GetCurrentProcessorNumber.
            0x12 | 0x17 => {
                // ClearEvent / ResetSignal.
                match self.handles.get(self.x(0) as u32) {
                    Some(&KObject::Event { id: event }) => {
                        let was = std::mem::replace(&mut self.kobjects.events[event], false);
                        self.ret(if id == 0x17 && !was {
                            INVALID_STATE
                        } else {
                            SUCCESS
                        });
                    }
                    _ => self.ret(INVALID_HANDLE),
                }
            }
            0x13 => self.svc_map_shared_memory(),
            0x14 => self.svc_unmap_shared_memory(),
            0x18 => return self.svc_wait_synchronization(pc),
            0x16 => {
                let ok = self.handles.close(self.x(0) as u32);
                self.ret(if ok { SUCCESS } else { INVALID_HANDLE });
            }
            0x1A => self.ret(SUCCESS), // ArbitrateLock: only one thread, never contended.
            0x1B => {
                // ArbitrateUnlock: no waiters, so the mutex becomes free.
                let addr = self.x(0);
                match self.mem.write_uint(addr, 4, 0) {
                    Ok(()) => self.ret(SUCCESS),
                    Err(_) => self.ret(INVALID_ADDRESS),
                }
            }
            0x1C => {
                // WaitProcessWideKeyAtomic: no other thread can signal, so
                // this times out; the mutex is held again on return.
                self.ret(TIMED_OUT);
            }
            0x1D => self.ret(SUCCESS), // SignalProcessWideKey: no waiters.
            0x1E => {
                let tick = self.cpu.counter;
                self.set_x(0, tick);
            }
            0x1F => self.svc_connect_to_named_port(),
            0x21 => return self.svc_send_sync_request(pc),
            0x24 => {
                self.ret(SUCCESS);
                self.set_x(1, self.process_id);
            }
            0x25 => {
                if !self.is_thread(self.x(1) as u32) {
                    self.ret(INVALID_HANDLE);
                    return Ok(());
                }
                self.ret(SUCCESS);
                self.set_x(1, self.thread_id);
            }
            0x26 => {
                return Err(Stop::Break {
                    reason: self.x(0),
                    arg: self.x(1),
                    info: self.x(2),
                })
            }
            0x27 => {
                let (addr, len) = (self.x(0), self.x(1) as usize);
                let mut buf = vec![0u8; len.min(0x10000)];
                match self.mem.read(addr, &mut buf) {
                    Ok(()) => {
                        self.debug_output.extend_from_slice(&buf);
                        self.ret(SUCCESS);
                    }
                    Err(_) => self.ret(INVALID_ADDRESS),
                }
            }
            0x29 => self.svc_get_info(),
            0x2A | 0x2B => self.ret(SUCCESS), // Cache flushes: memory is coherent.
            _ => return Err(Stop::UnimplementedSvc { id, pc }),
        }
        Ok(())
    }

    fn is_thread(&self, handle: u32) -> bool {
        handle == CURRENT_THREAD || matches!(self.handles.get(handle), Some(KObject::Thread { .. }))
    }

    fn is_process(&self, handle: u32) -> bool {
        handle == CURRENT_PROCESS || matches!(self.handles.get(handle), Some(KObject::Process))
    }

    fn svc_set_heap_size(&mut self) {
        let size = self.x(1);
        if !size.is_multiple_of(HEAP_ALIGN) {
            return self.ret(INVALID_SIZE);
        }
        let (base, region_size) = self.layout.heap;
        if size > region_size {
            return self.ret(OUT_OF_MEMORY);
        }
        if size > self.heap_size {
            let grow = size - self.heap_size;
            if self.map.used() + grow > self.layout.total_memory {
                return self.ret(OUT_OF_MEMORY);
            }
            self.map_block(base + self.heap_size, grow, MemState::Heap, PERM_R | PERM_W);
        } else if size < self.heap_size {
            let (start, len) = (base + size, self.heap_size - size);
            if !self
                .map
                .all(start, len, |b| b.state == MemState::Heap && b.attr == 0)
            {
                return self.ret(INVALID_MEMORY_STATE);
            }
            self.map.remove(start, len);
            self.mem.unmap(start, len).unwrap();
        }
        self.heap_size = size;
        self.ret(SUCCESS);
        self.set_x(1, base);
    }

    fn page_range_ok(addr: u64, size: u64) -> Option<ResultCode> {
        if !addr.is_multiple_of(PAGE) {
            return Some(INVALID_ADDRESS);
        }
        if size == 0 || !size.is_multiple_of(PAGE) {
            return Some(INVALID_SIZE);
        }
        if addr
            .checked_add(size)
            .is_none_or(|end| end > memmap::ADDRESS_SPACE_END)
        {
            return Some(INVALID_MEMORY_RANGE);
        }
        None
    }

    fn svc_set_memory_permission(&mut self) {
        let (addr, size, perm) = (self.x(0), self.x(1), self.x(2) as u32);
        if let Some(err) = Self::page_range_ok(addr, size) {
            return self.ret(err);
        }
        // Only none, R and RW may be set this way.
        if !matches!(perm, 0 | PERM_R | 3) {
            return self.ret(INVALID_MEMORY_PERMISSIONS);
        }
        if !self
            .map
            .all(addr, size, |b| b.state == MemState::Heap && b.attr == 0)
        {
            return self.ret(INVALID_MEMORY_STATE);
        }
        self.map.update(addr, size, |b| b.perm = perm);
        self.set_host_perm(addr, size, perm);
        self.ret(SUCCESS);
    }

    fn svc_set_memory_attribute(&mut self) {
        let (addr, size, mask, value) = (self.x(0), self.x(1), self.x(2) as u32, self.x(3) as u32);
        if let Some(err) = Self::page_range_ok(addr, size) {
            return self.ret(err);
        }
        // Only the Uncached attribute can be changed from user mode.
        if (mask | value) & !ATTR_UNCACHED != 0 || value & !mask != 0 {
            return self.ret(INVALID_COMBINATION);
        }
        if !self.map.all(addr, size, |b| b.state == MemState::Heap) {
            return self.ret(INVALID_MEMORY_STATE);
        }
        self.map
            .update(addr, size, |b| b.attr = (b.attr & !mask) | value);
        self.ret(SUCCESS);
    }

    /// MapMemory(dst, src, size): aliases heap memory into the stack
    /// region (libnx uses this for thread stacks). The source becomes
    /// inaccessible and locked until UnmapMemory.
    fn svc_map_memory(&mut self) {
        let (dst, src, size) = (self.x(0), self.x(1), self.x(2));
        if let Some(err) = Self::page_range_ok(dst, size).or(Self::page_range_ok(src, size)) {
            return self.ret(err);
        }
        let (sb, ss) = self.layout.stack;
        if dst < sb || dst + size > sb + ss {
            return self.ret(INVALID_MEMORY_RANGE);
        }
        if !self.map.all(src, size, |b| {
            b.state == MemState::Heap && b.perm == 3 && b.attr == 0
        }) {
            return self.ret(INVALID_MEMORY_STATE);
        }
        if !self.map.is_free(dst, size) {
            return self.ret(INVALID_MEMORY_STATE);
        }
        self.mem.move_pages(src, dst, size).expect("checked");
        // Keep inaccessible placeholder pages at the source.
        self.mem.map(src, size, Perm::NONE).unwrap();
        self.map.update(src, size, |b| {
            b.perm = 0;
            b.attr |= ATTR_LOCKED;
        });
        self.map.insert(
            dst,
            Block {
                size,
                state: MemState::Stack,
                perm: 3,
                attr: 0,
            },
        );
        self.aliases.push((dst, src, size));
        self.ret(SUCCESS);
    }

    fn svc_unmap_memory(&mut self) {
        let (dst, src, size) = (self.x(0), self.x(1), self.x(2));
        if let Some(err) = Self::page_range_ok(dst, size).or(Self::page_range_ok(src, size)) {
            return self.ret(err);
        }
        let Some(i) = self.aliases.iter().position(|a| *a == (dst, src, size)) else {
            return self.ret(INVALID_MEMORY_STATE);
        };
        self.aliases.remove(i);
        self.mem.unmap(src, size).unwrap();
        self.mem
            .move_pages(dst, src, size)
            .expect("alias pages exist");
        self.map.remove(dst, size);
        self.map.update(src, size, |b| {
            b.perm = 3;
            b.attr &= !ATTR_LOCKED;
        });
        self.ret(SUCCESS);
    }

    fn svc_map_shared_memory(&mut self) {
        let (handle, addr, size, perm) = (self.x(0) as u32, self.x(1), self.x(2), self.x(3) as u32);
        let Some(&KObject::SharedMemory { id }) = self.handles.get(handle) else {
            return self.ret(INVALID_HANDLE);
        };
        if let Some(err) = Self::page_range_ok(addr, size) {
            return self.ret(err);
        }
        if perm == 0 || perm & !(PERM_R | PERM_W) != 0 {
            return self.ret(INVALID_MEMORY_PERMISSIONS);
        }
        let shm = &self.kobjects.shared_memory[id];
        if shm.size != size {
            return self.ret(INVALID_SIZE);
        }
        if shm.mapped_at.is_some() || !self.map.is_free(addr, size) {
            return self.ret(INVALID_MEMORY_STATE);
        }
        let init = shm.init.clone();
        self.map_block(addr, size, MemState::Shared, perm);
        self.mem.write_privileged(addr, &init).expect("just mapped");
        self.kobjects.shared_memory[id].mapped_at = Some(addr);
        self.ret(SUCCESS);
    }

    fn svc_unmap_shared_memory(&mut self) {
        let (handle, addr, size) = (self.x(0) as u32, self.x(1), self.x(2));
        let Some(&KObject::SharedMemory { id }) = self.handles.get(handle) else {
            return self.ret(INVALID_HANDLE);
        };
        let shm = &self.kobjects.shared_memory[id];
        if shm.mapped_at != Some(addr) || shm.size != size {
            return self.ret(INVALID_MEMORY_STATE);
        }
        self.mem.unmap(addr, size).expect("mapped");
        self.map.remove(addr, size);
        self.kobjects.shared_memory[id].mapped_at = None;
        self.ret(SUCCESS);
    }

    /// WaitSynchronization: returns the index of a signaled object. With
    /// a single thread nothing can signal while it waits, so a finite
    /// timeout just lets the time pass and an infinite one is a deadlock.
    fn svc_wait_synchronization(&mut self, pc: u64) -> Result<(), Stop> {
        let (list, count, timeout) = (self.x(1), self.x(2), self.x(3) as i64);
        if count > 0x40 {
            self.ret(OUT_OF_RANGE);
            return Ok(());
        }
        for i in 0..count {
            let Ok(h) = self.mem.read_uint(list + 4 * i, 4) else {
                self.ret(INVALID_ADDRESS);
                return Ok(());
            };
            match self.handles.get(h as u32) {
                Some(&KObject::Event { id }) => {
                    if self.kobjects.events[id] {
                        self.ret(SUCCESS);
                        self.set_x(1, i);
                        return Ok(());
                    }
                }
                Some(KObject::Thread { .. } | KObject::Process) | Some(KObject::Session { .. }) => {
                }
                _ => {
                    self.ret(INVALID_HANDLE);
                    return Ok(());
                }
            }
        }
        if timeout < 0 {
            return Err(Stop::Deadlock { pc });
        }
        self.cpu.counter += (timeout as u64).saturating_mul(192) / 10_000;
        self.ret(TIMED_OUT);
        Ok(())
    }

    fn svc_query_memory(&mut self) {
        let (out, addr) = (self.x(0), self.x(2));
        let info = self.map.query(addr);
        match self.mem.write(out, &info.to_bytes()) {
            Ok(()) => {
                self.ret(SUCCESS);
                self.set_x(1, 0); // PageInfo
            }
            Err(_) => self.ret(INVALID_ADDRESS),
        }
    }

    /// ConnectToNamedPort: only the service manager has a named port.
    fn svc_connect_to_named_port(&mut self) {
        let Some(name) = self.read_cstr(self.x(1), 12) else {
            return self.ret(INVALID_ADDRESS);
        };
        if name != "sm:" {
            return self.ret(NOT_FOUND);
        }
        let service = services::create(&name).expect("sm: exists");
        match self.new_session(service) {
            Some(h) => {
                self.ret(SUCCESS);
                self.set_x(1, u64::from(h));
            }
            None => self.ret(OUT_OF_HANDLES),
        }
    }

    fn new_session(&mut self, object: services::ServiceRef) -> Option<u32> {
        let id = self.sessions.len();
        self.sessions.push(Session {
            object,
            domain: None,
            closed: false,
        });
        self.handles.insert(KObject::Session { id })
    }

    /// SendSyncRequest: handles one CMIF request from the TLS command
    /// buffer and writes the reply there.
    fn svc_send_sync_request(&mut self, pc: u64) -> Result<(), Stop> {
        let Some(&KObject::Session { id }) = self.handles.get(self.x(0) as u32) else {
            self.ret(INVALID_HANDLE);
            return Ok(());
        };
        let tls = self.cpu.tpidrro_el0;
        let mut msg = [0u8; ipc::MESSAGE_SIZE];
        if self.mem.read(tls, &mut msg).is_err() {
            self.ret(INVALID_ADDRESS);
            return Ok(());
        }
        let is_domain = self.sessions[id].domain.is_some();
        let req =
            ipc::parse_request(&msg, is_domain).map_err(|error| Stop::BadIpc { pc, error })?;

        let reply = match req.kind {
            ipc::TYPE_CLOSE => {
                self.sessions[id].closed = true;
                self.ret(SUCCESS);
                return Ok(());
            }
            ipc::TYPE_CONTROL | ipc::TYPE_CONTROL_WITH_CONTEXT => self.control(id, &req)?,
            ipc::TYPE_REQUEST | ipc::TYPE_REQUEST_WITH_CONTEXT => self.request(id, &req)?,
            _ => {
                let service = self.sessions[id].object.borrow().name().to_string();
                return Err(Stop::UnimplementedIpc {
                    service,
                    command: u32::from(req.kind),
                    control: true,
                });
            }
        };
        let out = ipc::write_reply(&reply, is_domain && req.kind != ipc::TYPE_CONTROL);
        self.mem.write(tls, &out).expect("TLS is mapped");
        self.ret(SUCCESS);
        Ok(())
    }

    /// Control requests (sent on the session itself, never as domain
    /// messages).
    fn control(&mut self, id: usize, req: &ipc::Request) -> Result<ipc::Reply, Stop> {
        let mut reply = ipc::Reply::default();
        match req.command {
            0 => {
                // ConvertCurrentObjectToDomain: the object becomes id 1.
                let object = self.sessions[id].object.clone();
                self.sessions[id].domain = Some(Domain::new(vec![None, Some(object)].into()));
                reply.data.extend_from_slice(&1u32.to_le_bytes());
            }
            2 | 4 => {
                // CloneCurrentObject(Ex): a new session to the same object.
                let object = self.sessions[id].object.clone();
                let domain = self.sessions[id].domain.clone();
                match self.new_session(object) {
                    Some(h) => {
                        let Some(KObject::Session { id: new }) = self.handles.get(h).cloned()
                        else {
                            unreachable!()
                        };
                        self.sessions[new].domain = domain;
                        reply.move_handles.push(h);
                    }
                    None => reply.result = OUT_OF_HANDLES,
                }
            }
            3 => {
                // QueryPointerBufferSize.
                reply.data.extend_from_slice(&0x8000u16.to_le_bytes());
            }
            command => {
                let service = self.sessions[id].object.borrow().name().to_string();
                return Err(Stop::UnimplementedIpc {
                    service,
                    command,
                    control: true,
                });
            }
        }
        Ok(reply)
    }

    fn request(&mut self, id: usize, req: &ipc::Request) -> Result<ipc::Reply, Stop> {
        let object = match &req.domain {
            None => self.sessions[id].object.clone(),
            Some(d) => {
                let domain = self.sessions[id].domain.clone().expect("domain session");
                let mut table = domain.borrow_mut();
                if d.kind == ipc::DOMAIN_CLOSE {
                    if let Some(slot) = table.get_mut(d.object_id as usize) {
                        *slot = None;
                    }
                    return Ok(ipc::Reply::default());
                }
                match table.get(d.object_id as usize).cloned().flatten() {
                    Some(o) => o,
                    // Unknown object id: HIPC "target not found".
                    None => {
                        return Ok(ipc::Reply {
                            result: 10 | (410 << 9),
                            ..Default::default()
                        })
                    }
                }
            }
        };
        let mut ctx = services::Ctx {
            req,
            mem: &mut self.mem,
            out: Vec::new(),
            out_objects: Vec::new(),
            out_copy: Vec::new(),
            kernel: &mut self.kobjects,
            tick: self.cpu.counter,
        };
        let outcome = object.borrow_mut().call(req.command, &mut ctx);
        let services::Ctx {
            out,
            out_objects,
            out_copy,
            ..
        } = ctx;
        let mut reply = ipc::Reply::default();
        match outcome {
            Ok(()) => {
                reply.data = out;
                for kobj in out_copy {
                    match self.handles.insert(kobj) {
                        Some(h) => reply.copy_handles.push(h),
                        None => reply.result = OUT_OF_HANDLES,
                    }
                }
                for obj in out_objects {
                    if let Some(domain) = self.sessions[id].domain.clone() {
                        let mut table = domain.borrow_mut();
                        table.push(Some(obj));
                        reply.objects.push((table.len() - 1) as u32);
                    } else {
                        match self.new_session(obj) {
                            Some(h) => reply.move_handles.push(h),
                            None => reply.result = OUT_OF_HANDLES,
                        }
                    }
                }
            }
            Err(services::CallError::Result(code)) => reply.result = code,
            Err(services::CallError::Unimplemented) => {
                let service = object.borrow().name().to_string();
                return Err(Stop::UnimplementedIpc {
                    service,
                    command: req.command,
                    control: false,
                });
            }
        }
        Ok(reply)
    }

    fn svc_get_info(&mut self) {
        let (kind, handle, sub) = (self.x(1) as u32, self.x(2) as u32, self.x(3));
        let l = self.layout;
        let process_info = |v: u64| Some(v);
        let value = match kind {
            0..=7 | 12..=18 | 21..=24 | 28 => {
                if !self.is_process(handle) {
                    return self.ret(INVALID_HANDLE);
                }
                if sub != 0 {
                    return self.ret(INVALID_COMBINATION);
                }
                match kind {
                    0 => process_info(0xF),                   // CoreMask
                    1 => process_info(0xFFFF_FFFF_FFFF_FFF0), // PriorityMask
                    2 => process_info(l.alias.0),
                    3 => process_info(l.alias.1),
                    4 => process_info(l.heap.0),
                    5 => process_info(l.heap.1),
                    6 | 21 => process_info(l.total_memory),
                    7 | 22 => process_info(self.map.used()),
                    12 => process_info(l.aslr.0),
                    13 => process_info(l.aslr.1),
                    14 => process_info(l.stack.0),
                    15 => process_info(l.stack.1),
                    16 | 17 => process_info(0), // System resource size.
                    18 => process_info(0x0100_0000_0000_1000), // ProgramId
                    23 => process_info(1),      // IsApplication
                    24 => process_info(64),     // FreeThreadCount
                    _ => process_info(0),       // AliasRegionExtraSize
                }
            }
            8 => Some(0), // DebuggerAttached
            9 => Some(0), // ResourceLimit: none.
            11 => {
                // RandomEntropy: fixed, so runs are reproducible.
                if sub > 3 {
                    return self.ret(INVALID_COMBINATION);
                }
                Some(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(sub + 1))
            }
            _ => None,
        };
        match value {
            Some(v) => {
                self.ret(SUCCESS);
                self.set_x(1, v);
            }
            None => self.ret(INVALID_ENUM_VALUE),
        }
    }
}
