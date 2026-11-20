//! Runs programs on the emulated CPU.
//!
//! - ELF test programs talk to the host through a tiny test ABI, `svc #0`
//!   with the call number in x8 (see `tests/programs/rt.h`).
//! - NRO homebrew boots through the homebrew ABI on the Horizon HLE kernel
//!   (`emu-horizon`) and uses real Switch SVCs.

use emu_cpu::{Cpu, Exit};
use emu_loader::load_elf;
use emu_memory::{Memory, Perm};
use std::fmt;

pub const STACK_TOP: u64 = 0x7FFF_0000_0000;
pub const STACK_SIZE: u64 = 0x10_0000;

const SYS_WRITE: u64 = 1;
const SYS_EXIT: u64 = 2;

#[derive(Debug)]
pub struct RunResult {
    pub exit_code: u64,
    pub stdout: Vec<u8>,
    pub instructions: u64,
}

#[derive(Debug)]
pub enum RunError {
    Load(emu_loader::LoadError),
    /// The CPU stopped for a reason the test ABI does not handle.
    Stopped {
        exit: Exit,
        cpu: Box<Cpu>,
    },
    BadSyscall {
        number: u64,
        pc: u64,
    },
    StepLimit,
    /// The requested GPU backend could not be created.
    Gpu(String),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Load(e) => write!(f, "load failed: {e}"),
            RunError::Gpu(e) => write!(f, "GPU backend: {e}"),
            RunError::Stopped { exit, cpu } => {
                write!(
                    f,
                    "guest stopped: {exit:?} (pc={:#x}, sp={:#x})",
                    cpu.pc(),
                    cpu.sp()
                )
            }
            RunError::BadSyscall { number, pc } => {
                write!(f, "unknown test syscall {number} at {pc:#x}")
            }
            RunError::StepLimit => write!(f, "step limit reached"),
        }
    }
}

impl std::error::Error for RunError {}

pub fn run_elf(elf: &[u8], max_steps: u64) -> Result<RunResult, RunError> {
    let mut mem = Memory::new();
    let image = load_elf(&mut mem, elf).map_err(RunError::Load)?;
    mem.map(STACK_TOP - STACK_SIZE, STACK_SIZE, Perm::RW)
        .expect("stack mapping");

    let mut cpu = Cpu::new();
    cpu.set_pc(image.entry);
    cpu.write_sp(31, STACK_TOP);

    let mut stdout = Vec::new();
    let mut steps = 0u64;
    loop {
        let budget = max_steps.saturating_sub(steps);
        if budget == 0 {
            return Err(RunError::StepLimit);
        }
        let before = cpu.counter;
        let exit = cpu.run(&mut mem, budget);
        steps += cpu.counter - before;
        match exit {
            None => return Err(RunError::StepLimit),
            Some(Exit::Svc(0)) => match cpu.read_xzr(8) {
                SYS_WRITE => {
                    let mut buf = vec![0u8; cpu.read_xzr(1) as usize];
                    if let Err(fault) = mem.read(cpu.read_xzr(0), &mut buf) {
                        let pc = cpu.pc() - 4;
                        return Err(RunError::Stopped {
                            exit: Exit::Fault { pc, fault },
                            cpu: Box::new(cpu),
                        });
                    }
                    stdout.extend_from_slice(&buf);
                    cpu.write_xzr(0, buf.len() as u64);
                }
                SYS_EXIT => {
                    return Ok(RunResult {
                        exit_code: cpu.read_xzr(0),
                        stdout,
                        instructions: steps,
                    })
                }
                number => {
                    return Err(RunError::BadSyscall {
                        number,
                        pc: cpu.pc() - 4,
                    })
                }
            },
            Some(exit) => {
                return Err(RunError::Stopped {
                    exit,
                    cpu: Box::new(cpu),
                })
            }
        }
    }
}

/// Result of running homebrew on the Horizon HLE kernel.
#[derive(Debug)]
pub struct HomebrewResult {
    pub stop: emu_horizon::Stop,
    /// Text the program wrote with svcOutputDebugString.
    pub debug_output: Vec<u8>,
    pub instructions: u64,
    /// Frames the program presented on the display.
    pub frames: Vec<emu_horizon::services::Frame>,
}

#[cfg(feature = "window")]
pub mod window;

pub fn run_nro(nro: &[u8], max_steps: u64, trace_svcs: bool) -> Result<HomebrewResult, RunError> {
    run_nro_with(
        nro,
        max_steps,
        trace_svcs,
        emu_horizon::services::Config::default(),
    )
}

/// Like [`run_nro`], with host settings (SD card directory, clock).
pub fn run_nro_with(
    nro: &[u8],
    max_steps: u64,
    trace_svcs: bool,
    config: emu_horizon::services::Config,
) -> Result<HomebrewResult, RunError> {
    run_nro_presenting(nro, max_steps, trace_svcs, config, None)
}

/// What to boot on the Horizon HLE kernel.
pub enum Program {
    /// A homebrew NRO.
    Homebrew(Vec<u8>),
    /// An application's NSO modules, in ExeFS order, and its NPDM.
    Application(Vec<emu_loader::nso::Module>, Option<emu_loader::npdm::Npdm>),
}

impl Program {
    /// The ExeFS module names, in load order.
    fn module_names() -> Vec<String> {
        let mut names = vec!["rtld".to_string(), "main".to_string()];
        names.extend((0..10).map(|i| format!("subsdk{i}")));
        names.push("sdk".into());
        names
    }

    /// Reads an NRO or NSO file, an ExeFS directory (rtld, main,
    /// subsdk0-9, sdk), or a game package (NSP, XCI, or a program NCA),
    /// which needs the user's console keys (`keys`, by default
    /// `~/.switch/prod.keys`).
    pub fn open(path: &std::path::Path, keys: Option<&std::path::Path>) -> Result<Program, String> {
        let parse = |data: &[u8], name: &str| {
            emu_loader::nso::parse_nso(data).map_err(|e| format!("{name}: {e}"))
        };
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if matches!(ext.as_str(), "nsp" | "xci" | "nca") {
            let keys_path = match keys {
                Some(k) => k.to_path_buf(),
                None => std::env::var_os("HOME")
                    .or_else(|| std::env::var_os("USERPROFILE"))
                    .map(|h| std::path::Path::new(&h).join(".switch/prod.keys"))
                    .ok_or("no --keys and no home directory")?,
            };
            let keys = emu_loader::nca::Keys::load(&keys_path).map_err(|e| format!("keys: {e}"))?;
            let exefs = emu_loader::nca::find_exefs(path, &keys).map_err(|e| e.to_string())?;
            let read = |name: &str| {
                exefs
                    .open(name)
                    .map(|s| s.read_vec(0, s.len() as usize).map_err(|e| e.to_string()))
                    .transpose()
            };
            let mut modules = Vec::new();
            for name in Program::module_names() {
                if let Some(data) = read(&name)? {
                    modules.push(parse(&data, &name)?);
                }
            }
            let npdm = read("main.npdm")?
                .map(|d| emu_loader::npdm::Npdm::parse(&d).map_err(|e| format!("main.npdm: {e}")))
                .transpose()?;
            return Ok(Program::Application(modules, npdm));
        }
        if path.is_dir() {
            let mut modules = Vec::new();
            for name in Program::module_names() {
                let p = path.join(&name);
                if let Ok(data) = std::fs::read(&p) {
                    modules.push(parse(&data, &name)?);
                }
            }
            if modules.is_empty() {
                return Err(format!("{}: no NSO modules", path.display()));
            }
            let npdm = match std::fs::read(path.join("main.npdm")) {
                Ok(d) => {
                    Some(emu_loader::npdm::Npdm::parse(&d).map_err(|e| format!("main.npdm: {e}"))?)
                }
                Err(_) => None,
            };
            return Ok(Program::Application(modules, npdm));
        }
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if emu_loader::nro::is_nro(&data) {
            Ok(Program::Homebrew(data))
        } else if emu_loader::nso::is_nso(&data) {
            Ok(Program::Application(vec![parse(&data, "NSO")?], None))
        } else {
            Err(format!(
                "{}: not an NRO, NSO or ExeFS directory",
                path.display()
            ))
        }
    }
}

/// Like [`run_nro_with`], handing each presented frame to `on_present`
/// (e.g. a window) instead of collecting it.
pub fn run_nro_presenting(
    nro: &[u8],
    max_steps: u64,
    trace_svcs: bool,
    config: emu_horizon::services::Config,
    on_present: Option<emu_horizon::services::Presenter>,
) -> Result<HomebrewResult, RunError> {
    run_program_presenting(
        &Program::Homebrew(nro.to_vec()),
        max_steps,
        trace_svcs,
        config,
        on_present,
    )
}

/// Boots a homebrew program or an application and runs it.
pub fn run_program_presenting(
    program: &Program,
    max_steps: u64,
    trace_svcs: bool,
    config: emu_horizon::services::Config,
    on_present: Option<emu_horizon::services::Presenter>,
) -> Result<HomebrewResult, RunError> {
    let mut hos = emu_horizon::Horizon::new();
    hos.system.on_present = on_present;
    if config.gpu_backend == emu_horizon::services::GpuBackend::Vulkan {
        let vk = emu_gpu_vk::VulkanBackend::new().map_err(RunError::Gpu)?;
        eprintln!("[emu-run] Vulkan device: {}", vk.device_name);
        hos.system.gpu.backend = Box::new(vk);
    }
    hos.trace_svcs = trace_svcs;
    hos.config = config;
    match program {
        Program::Homebrew(nro) => hos.boot_homebrew(nro),
        Program::Application(modules, npdm) => hos.boot_application(modules, npdm.as_ref()),
    }
    .map_err(RunError::Load)?;
    let stop = hos.run(max_steps);
    Ok(HomebrewResult {
        stop,
        debug_output: hos.debug_output,
        instructions: hos.cpu.counter,
        frames: std::mem::take(&mut hos.system.frames),
    })
}

/// Writes a frame as an RGBA PNG.
pub fn write_png(
    frame: &emu_horizon::services::Frame,
    path: &std::path::Path,
) -> std::io::Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, frame.width, frame.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(std::io::Error::other)?;
    w.write_image_data(&frame.rgba)
        .map_err(std::io::Error::other)
}
