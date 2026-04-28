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
    libssl-dev
```

The GTK4 / libadwaita stack pulls in everything needed for `gtk4-sys`, `libadwaita-sys`, `gdk4-sys`, `gsk4-sys`, `graphene-sys`, `gio-sys`, `glib-sys`, and `gobject-sys`. `libepoxy-dev` is required for the OpenGL context used by `GtkGLArea` / `glow`. `libssl-dev` is only needed if you swap `reqwest`'s `rustls-tls` feature for `native-tls`; it can be omitted otherwise.

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
