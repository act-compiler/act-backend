use std::{env, fs, path::PathBuf, process::Command};

fn main() {
  // Rerun when the script, the allocator's sources or the cost model change
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=cpp/malloc/CMakeLists.txt");
  println!("cargo:rerun-if-changed=cpp/malloc/include");
  println!("cargo:rerun-if-changed=cpp/malloc/src");

  println!("cargo:rustc-link-lib=dylib=stdc++");

  // OR-Tools, in the directory ORTOOLS_DIR names
  println!("cargo:rerun-if-env-changed=ORTOOLS_DIR");
  let ortools_dir = env::var("ORTOOLS_DIR").unwrap_or("/opt/ortools".to_string());
  // A run path is split at colons
  if ortools_dir.contains(':') {
    panic!(
      "ORTOOLS_DIR holds a colon, which no run path can carry: {}",
      ortools_dir
    );
  }
  // Its libraries lie under lib or lib64
  let ortools_lib = ["lib", "lib64"]
    .iter()
    .map(|lib| PathBuf::from(&ortools_dir).join(lib))
    .find(|lib| lib.join("libortools.so").exists())
    .unwrap_or_else(|| {
      panic!(
        "no libortools.so under lib or lib64 of {}: ORTOOLS_DIR names where OR-Tools is installed",
        ortools_dir
      )
    });
  println!("cargo:rustc-link-search=native={}", ortools_lib.display());
  println!("cargo:rustc-link-lib=dylib=ortools");
  println!("cargo:rustc-link-lib=dylib=absl_base");
  println!("cargo:rustc-link-lib=dylib=absl_log_internal_check_op");
  println!("cargo:rustc-link-lib=dylib=absl_log_internal_message");
  // The run path finds OR-Tools again at each run
  for arg in [
    "-Xlinker",
    "-rpath",
    "-Xlinker",
    ortools_lib.to_str().unwrap(),
  ] {
    println!("cargo:rustc-link-arg={}", arg);
  }

  // The JSON headers, which the allocator's build reads from NLOHMANN_JSON_DIR
  println!("cargo:rerun-if-env-changed=NLOHMANN_JSON_DIR");

  // Create cpp/malloc/build, run `cmake ..` inside it, then `make -j4`
  let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
  let cpp_build_dir = manifest.join("cpp").join("malloc").join("build");
  fs::create_dir_all(&cpp_build_dir).expect("failed to create cpp/malloc/build");

  let status = Command::new("cmake")
    .arg("..")
    .current_dir(&cpp_build_dir)
    .status()
    .expect("failed to run cmake");
  if !status.success() {
    panic!("cmake failed");
  }

  let status = Command::new("make")
    .arg("-j4")
    .current_dir(&cpp_build_dir)
    .status()
    .expect("failed to run make");
  if !status.success() {
    panic!("make failed");
  }

  println!("cargo:rustc-link-search=native={}", cpp_build_dir.display());
  println!("cargo:rustc-link-lib=static=malloc_static");

  // The cost model: the default, or the library the backend names
  // {{COST_MODEL}}
}
