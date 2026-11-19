//! Runs the compiled test programs in tests/programs and checks their output.

use std::path::PathBuf;

fn program(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/programs")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn hello_program_runs_to_completion() {
    let result = emu_run::run_elf(&program("hello.elf"), 10_000_000).expect("program runs");
    // Expected values checked independently (Python) and by running the
    // same ELF in Unicorn, which produced identical output.
    let expected = "\
Hello from the guest CPU!
fib(20) = 6765
factorial(20) = 2432902008176640000
primes below 256 = 54
sorted = -123456789012 -7 0 5 5 42 99 1000000007
fnv1a(\"switch2-emu\") = 0xca4822b387ad65f5
apply(sub, 10, 52) = -42
apply(divide, -100, 7) = -14
divide by zero = 0
";
    assert_eq!(String::from_utf8(result.stdout).unwrap(), expected);
    assert_eq!(result.exit_code, 7);
}
