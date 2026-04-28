# rgis

A GIS desktop application written in Rust, built on GTK4 / libadwaita with OpenGL-based rendering.

## Features

- Read GeoJSON, Shapefile, and FlatGeobuf data
- OpenGL rendering via `glow` and `lyon_tessellation`
- Raster tile background layers
- Coordinate reproduction via `proj4rs`

## Building on Ubuntu

Tested on Ubuntu 24.04. Earlier versions need GTK4 ≥ 4.14 and libadwaita ≥ 1.5 from a backport or PPA.

### 1. Install a Rust toolchain

The pinned toolchain is declared in [rust-toolchain.toml](rust-toolchain.toml) and will be installed automatically by `rustup` on first build.

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Install system dependencies

```sh
sudo apt update
sudo apt install \
    build-essential \
    pkg-config \
    libgtk-4-dev \
    libadwaita-1-dev \
    libglib2.0-dev \
    libgdk-pixbuf-2.0-dev \
    libgraphene-1.0-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libepoxy-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev
```

The GTK4 / libadwaita stack pulls in everything needed for `gtk4-sys`, `libadwaita-sys`, `gdk4-sys`, `gsk4-sys`, `graphene-sys`, `gio-sys`, `glib-sys`, and `gobject-sys`.

OpenGL is used directly by `rgis-render` via [`glow`](https://crates.io/crates/glow). At runtime we resolve GL entry points with `dlsym(RTLD_DEFAULT, ...)` after promoting `libGL.so.1` and `libEGL.so.1` to the global symbol scope, so both must be installed and loadable:

- `libepoxy-dev` — required by GTK4's `GtkGLArea`.
- `libgl1-mesa-dev` — provides `libGL.so.1` (X11 / GLX backends).
- `libegl1-mesa-dev` — provides `libEGL.so.1` (Wayland / EGL backends).

On hardware with proprietary drivers (NVIDIA, AMDGPU PRO) the vendor packages provide equivalent `libGL.so.1` / `libEGL.so.1` and the Mesa `-dev` packages can be omitted.

`reqwest` is configured with `rustls-tls`, so no system OpenSSL (`libssl-dev`) is required.

### 3. Build and run

```sh
cargo run --release
```

For a faster iteration cycle during development:

```sh
cargo run
```

## Project layout

| Crate | Purpose |
| --- | --- |
| [crates/rgis-app](crates/rgis-app) | GTK4 / libadwaita application shell and `rgis` binary |
| [crates/rgis-core](crates/rgis-core) | Shared types and core abstractions |
| [crates/rgis-io](crates/rgis-io) | GeoJSON / Shapefile / FlatGeobuf readers |
| [crates/rgis-render](crates/rgis-render) | OpenGL rendering backend |
| [crates/rgis-tiles](crates/rgis-tiles) | Raster tile fetching and caching |

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
