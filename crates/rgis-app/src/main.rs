#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(all(
    not(target_arch = "wasm32"),
    any(target_os = "linux", target_os = "freebsd")
))]
mod native;

#[cfg(all(
    not(target_arch = "wasm32"),
    any(target_os = "linux", target_os = "freebsd")
))]
fn main() {
    let startup_paths: Vec<std::path::PathBuf> = std::env::args_os()
        .skip(1)
        .map(std::path::PathBuf::from)
        .collect();
    native::run(startup_paths);
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(any(target_os = "linux", target_os = "freebsd"))
))]
fn main() {
    eprintln!("rgis native GPUI rendering is currently supported on Linux and FreeBSD");
}
