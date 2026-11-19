use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let trace = args.iter().any(|a| a == "--trace-svc");
    let Some(path) = args.iter().find(|a| !a.starts_with("--")).cloned() else {
        eprintln!("usage: emu-run [--trace-svc] <program.elf | homebrew.nro>");
        return ExitCode::from(2);
    };
    let elf = match std::fs::read(&path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let start = Instant::now();
    if emu_loader::nro::is_nro(&elf) {
        return match emu_run::run_nro(&elf, 1_000_000_000, trace) {
            Ok(r) => {
                std::io::stdout().write_all(&r.debug_output).unwrap();
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
