use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let trace = args.iter().any(|a| a == "--trace-svc");
    let Some(path) = args.iter().find(|a| !a.starts_with("--")).cloned() else {
        eprintln!(
            "usage: emu-run [--trace-svc] [--sdmc=DIR] [--frames=DIR] [--steps=N] [--gpu=vulkan] [--keys=prod.keys] <program.elf | homebrew.nro | game.nso | exefs-dir | game.nsp | game.xci>\n\
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
            gpu_backend: if args.iter().any(|a| a == "--gpu=vulkan") {
                emu_horizon::services::GpuBackend::Vulkan
            } else {
                emu_horizon::services::GpuBackend::Software
            },
        };
        let dump_frames = args.iter().any(|a| a.starts_with("--frames="));
        #[cfg(feature = "window")]
        let presenter = (!dump_frames).then(|| emu_run::window::presenter(&path));
        #[cfg(not(feature = "window"))]
        let presenter = None;
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
        return match emu_run::run_program_presenting(&program, limit, trace, config, presenter) {
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
                    "[emu-run] homebrew stopped: {:?}, {} instructions in {:.3}s",
                    r.stop, r.instructions, secs
                );
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
