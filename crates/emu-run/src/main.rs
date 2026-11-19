use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: emu-run <program.elf>");
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
