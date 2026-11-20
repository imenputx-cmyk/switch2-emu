use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    // Ctrl+C stops the emulation cleanly, reporting where it was.
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _ = emu_run::INTERRUPT.set(flag.clone());
    let _ = ctrlc::set_handler(move || {
        if flag.swap(true, std::sync::atomic::Ordering::Relaxed) {
            std::process::exit(130); // a second Ctrl+C: quit at once
        }
    });
    // Run on a thread with a large stack (Windows gives the main thread
    // 1 MiB), and report how it ended even if it panics.
    let worker = std::thread::Builder::new()
        .name("emu".into())
        .stack_size(256 << 20)
        .spawn(real_main)
        .expect("thread");
    match worker.join() {
        Ok(code) => code,
        Err(_) => {
            eprintln!("[emu-run] the emulator crashed (panic above)");
            ExitCode::from(101)
        }
    }
}

fn real_main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let trace = args.iter().any(|a| a == "--trace-svc");
    if args.iter().any(|a| a == "--cpu=interp") {
        emu_run::USE_JIT.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    // --debug: a debugger prompt in the terminal when the program stops
    // (errors, breakpoints, Ctrl+C); --debug=start also before it runs.
    if let Some(a) = args
        .iter()
        .find(|a| *a == "--debug" || a.starts_with("--debug="))
    {
        let mode = if a == "--debug=start" { 2 } else { 1 };
        emu_run::DEBUG_CONSOLE.store(mode, std::sync::atomic::Ordering::Relaxed);
    }
    // --gdb or --gdb=PORT: wait for a debugger (GDB remote protocol).
    if let Some(a) = args
        .iter()
        .find(|a| *a == "--gdb" || a.starts_with("--gdb="))
    {
        let port = a
            .strip_prefix("--gdb=")
            .and_then(|p| p.parse().ok())
            .unwrap_or(6543u32);
        emu_run::GDB_PORT.store(port, std::sync::atomic::Ordering::Relaxed);
    }
    let Some(path) = args.iter().find(|a| !a.starts_with("--")).cloned() else {
        eprintln!(
            "usage: emu-run [--trace-svc] [--sdmc=DIR] [--frames=DIR] [--steps=N] [--gpu=vulkan|software] [--keys=prod.keys] [--auto-stub] [--cpu=interp] [--gdb[=PORT]] [--debug[=start]] <program.elf | homebrew.nro | game.nso | exefs-dir | game.nsp | game.xci>\n\
             (built with --features window, frames are shown in a window unless --frames is given)"
        );
        return ExitCode::from(2);
    };
    let start = Instant::now();
    let p = std::path::Path::new(&path);
    let is_elf = !p.is_dir() && std::fs::read(p).is_ok_and(|d| d.starts_with(b"\x7fELF"));
    if !is_elf {
        let keys = args
            .iter()
            .find_map(|a| a.strip_prefix("--keys="))
            .map(std::path::Path::new);
        let program = match emu_run::Program::open(p, keys) {
            Ok(program) => program,
            Err(e) => {
                eprintln!("[emu-run] {e}");
                return ExitCode::from(2);
            }
        };
        let config = emu_horizon::services::Config {
            sdmc_root: args
                .iter()
                .find_map(|a| a.strip_prefix("--sdmc="))
                .map(Into::into),
            boot_time: None,
            trace_unimplemented: trace,
            auto_stub: args.iter().any(|a| a == "--auto-stub"),
            romfs: None,
            // Vulkan by default (falling back to software rendering when
            // no device is found); --gpu=software forces the latter.
            gpu_backend: if args.iter().any(|a| a == "--gpu=software") {
                emu_horizon::services::GpuBackend::Software
            } else {
                emu_horizon::services::GpuBackend::Vulkan
            },
        };
        let dump_frames = args.iter().any(|a| a.starts_with("--frames="));
        #[cfg(feature = "window")]
        let (presenter, shown) = match (!dump_frames).then(|| emu_run::window::presenter(&path)) {
            Some((p, s)) => (Some(p), Some(s)),
            None => (None, None),
        };
        #[cfg(not(feature = "window"))]
        let (presenter, shown): (_, Option<()>) = (None, None);
        let _ = dump_frames;
        // With a window the program runs until it exits or the window
        // closes; otherwise stop after 10^10 instructions.
        let limit = args
            .iter()
            .find_map(|a| a.strip_prefix("--steps=")?.parse().ok())
            .unwrap_or(if presenter.is_some() {
                u64::MAX
            } else {
                10_000_000_000
            });
        let result = emu_run::run_program_presenting(&program, limit, trace, config, presenter);
        return match result {
            Ok(r) => {
                std::io::stdout().write_all(&r.debug_output).unwrap();
                if let Some(dir) = args.iter().find_map(|a| a.strip_prefix("--frames=")) {
                    let dir = std::path::Path::new(dir);
                    std::fs::create_dir_all(dir).ok();
                    for (i, f) in r.frames.iter().enumerate() {
                        let path = dir.join(format!("frame{i:04}.png"));
                        if let Err(e) = emu_run::write_png(f, &path) {
                            eprintln!("[emu-run] cannot write {}: {e}", path.display());
                        }
                    }
                    eprintln!(
                        "[emu-run] wrote {} frames to {}",
                        r.frames.len(),
                        dir.display()
                    );
                }
                let secs = start.elapsed().as_secs_f64();
                eprintln!(
                    "[emu-run] program stopped: {:?}, {} instructions in {:.3}s",
                    r.stop, r.instructions, secs
                );
                #[cfg(feature = "window")]
                if let Some(shown) = &shown {
                    shown.wait_until_closed();
                }
                let _ = &shown;
                match r.stop {
                    emu_horizon::Stop::ReturnedToLoader(code) => ExitCode::from(code as u8),
                    emu_horizon::Stop::ExitProcess => ExitCode::SUCCESS,
                    _ => ExitCode::from(1),
                }
            }
            Err(e) => {
                eprintln!("[emu-run] {e}");
                ExitCode::from(1)
            }
        };
    }
    let elf = std::fs::read(p).expect("read above");
    match emu_run::run_elf(&elf, 1_000_000_000) {
        Ok(result) => {
            std::io::stdout().write_all(&result.stdout).unwrap();
            let secs = start.elapsed().as_secs_f64();
            eprintln!(
                "[emu-run] exit code {}, {} instructions in {:.3}s ({:.1} MIPS)",
                result.exit_code,
                result.instructions,
                secs,
                result.instructions as f64 / secs / 1e6
            );
            ExitCode::from(result.exit_code as u8)
        }
        Err(e) => {
            eprintln!("[emu-run] {e}");
            ExitCode::from(1)
        }
    }
}
