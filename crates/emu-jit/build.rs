// Builds the vendored dynarmic (third_party/dynarmic) with CMake and the
// small C++ shim in shim/ that exposes it to Rust.

use std::path::{Path, PathBuf};

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../..")
        .canonicalize()
        .unwrap();
    let dynarmic_src = root.join("third_party/dynarmic");
    let boost = root.join("third_party/boost");

    println!("cargo:rerun-if-changed=shim/shim.cpp");
    println!(
        "cargo:rerun-if-changed={}",
        dynarmic_src.join("src").display()
    );

    // dynarmic is always built optimized: a debug JIT is far too slow, and
    // on MSVC a Debug build would also link the wrong C runtime.
    let dst = cmake::Config::new(&dynarmic_src)
        .profile("Release")
        .define("DYNARMIC_TESTS", "OFF")
        .define("DYNARMIC_FRONTENDS", "A64")
        .define("DYNARMIC_WARNINGS_AS_ERRORS", "OFF")
        .define("DYNARMIC_USE_BUNDLED_EXTERNALS", "ON")
        .define("DYNARMIC_USE_PRECOMPILED_HEADERS", "OFF")
        .define("Boost_INCLUDE_DIR", &boost)
        .define("CMAKE_POLICY_DEFAULT_CMP0167", "OLD")
        .build_target("dynarmic")
        .build();
    let build = dst.join("build");

    for lib in ["dynarmic", "fmt", "mcl"] {
        let dir = find_lib(&build, lib)
            .unwrap_or_else(|| panic!("built library {lib} not found under {}", build.display()));
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static=dynarmic");
    println!("cargo:rustc-link-lib=static=mcl");
    let fmt = if find_named(&build, "fmtd").is_some() {
        "fmtd"
    } else {
        "fmt"
    };
    println!("cargo:rustc-link-lib=static={fmt}");

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file("shim/shim.cpp")
        .include(dynarmic_src.join("src"))
        .include(dynarmic_src.join("externals/mcl/include"))
        .include(&boost)
        .flag_if_supported("/EHsc")
        .flag_if_supported("/Zc:__cplusplus")
        .compile("emu_jit_shim");
}

/// Finds the directory holding the static library `name` (libNAME.a or
/// NAME.lib, possibly under a Release/ configuration folder).
fn find_lib(dir: &Path, name: &str) -> Option<PathBuf> {
    find_named(dir, name).or_else(|| find_named(dir, &format!("{name}d")))
}

fn find_named(dir: &Path, name: &str) -> Option<PathBuf> {
    let unix = format!("lib{name}.a");
    let msvc = format!("{name}.lib");
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_named(&path, name) {
                return Some(found);
            }
        } else if let Some(file) = path.file_name().and_then(|f| f.to_str()) {
            if file == unix || file == msvc {
                return Some(dir.to_path_buf());
            }
        }
    }
    None
}
