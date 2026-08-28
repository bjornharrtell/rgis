# rgis

A Rust GIS desktop application built on [egui](https://crates.io/crates/egui)/[eframe](https://crates.io/crates/eframe) for UI and [wgpu](https://crates.io/crates/wgpu) (with [earcut](https://crates.io/crates/earcut) tessellation) for map rendering. The same app runs natively and compiles to `wasm32-unknown-unknown` for the browser.

## Features

- Read GeoJSON, Shapefile, and FlatGeobuf data, via a file-picker dialog or startup CLI arguments
- Reproject GeoJSON layers declared in a non-WGS-84 CRS (e.g. `EPSG:25832`) using [proj4rs](https://crates.io/crates/proj4rs)
- Ships with a bundled demo dataset ([crates/rgis-app/assets/sample.geojson](crates/rgis-app/assets/sample.geojson)), loaded when the native app is started with no file arguments and always in the browser build
- Render vector layers as tessellated triangle meshes uploaded to the GPU via `wgpu`
- Show OpenStreetMap raster tiles behind vector data, with an on-disk cache on native
- Layer panel to toggle visibility and remove layers
- Keep map state in Web Mercator while reporting cursor coordinates and scale in the status bar

## Building on Ubuntu

The pinned toolchain is declared in [rust-toolchain.toml](rust-toolchain.toml) and will be installed automatically by `rustup` on first build.

### 1. Install Rust

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Install system dependencies

`eframe`/`winit` on Linux needs Wayland/Vulkan/font/text/input libraries. This workspace also uses image loading and HTTP tile fetching.

```sh
sudo apt update
sudo apt install \
    build-essential \
    pkg-config \
    libwayland-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libvulkan-dev \
    libxcb1-dev \
    libxcb-xkb-dev \
    libxrandr-dev \
    libxi-dev \
    libx11-dev \
    libasound2-dev
```

Tile fetching uses [`ehttp`](https://crates.io/crates/ehttp), so no OpenSSL development headers are required.

### 3. Build and run

```sh
cargo run --release
```

For a faster edit/build cycle:

```sh
cargo run
```

## Building for the browser (wasm)

[crates/rgis-web](crates/rgis-web) boots the same `rgis_app::RgisApp` (used by
the native binary) into an HTML5 `<canvas>` via `eframe::WebRunner`, using
whichever `wgpu` backend (WebGPU or WebGL2) the browser supports. It always
loads the bundled demo dataset so the map has something to show without using
the file picker.

```sh
rustup target add wasm32-unknown-unknown
cargo install trunk
cd crates/rgis-web
trunk serve
```

Then open <http://127.0.0.1:8080/rgis/> in a browser.

A `trunk build --release` produces static assets in `crates/rgis-web/dist`
which are what the [`pages` workflow](.github/workflows/pages.yml)
publishes to GitHub Pages on every push to `main`.

## Project layout

| Crate | Purpose |
| --- | --- |
| [crates/rgis-app](crates/rgis-app) | Shared `eframe::App` (UI, layer panel, status bar, pan/zoom) and the `rgis` native binary |
| [crates/rgis-core](crates/rgis-core) | Core GIS types, styling, projection helpers, project/viewport state |
| [crates/rgis-io](crates/rgis-io) | GeoJSON / Shapefile / FlatGeobuf readers and CRS reprojection to Web Mercator |
| [crates/rgis-render](crates/rgis-render) | Tessellates layer geometry into `wgpu` vertex/index buffers and draws them via an `egui-wgpu` paint callback |
| [crates/rgis-tiles](crates/rgis-tiles) | Raster tile fetching and caching |
| [crates/rgis-web](crates/rgis-web) | wasm/browser entry point that boots `rgis-app` via `eframe::WebRunner` |

## Architecture notes

- `rgis-core`, `rgis-io`, and `rgis-tiles` stay backend-agnostic.
- `rgis-render` tessellates `rgis-core::Layer` geometry (via `earcut` for fills, with hand-extruded stroke/point quads) into a `SceneMesh`, then draws it each frame from an `egui-wgpu` `MapCallback` paint callback.
- `rgis-app` owns the `eframe::App` implementation: the layer panel, status bar, tile image cache, and pan/zoom interactions, shared verbatim between the native binary and the wasm build.

## Development

CI runs `cargo fmt --all -- --check`, so run `cargo fmt --all` before
committing. To catch this locally, enable the repo's git hook once per clone:

```sh
git config core.hooksPath .githooks
```

This runs `cargo fmt --all -- --check` on every commit and rejects it if
formatting is needed.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
