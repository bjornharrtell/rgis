# rgis

A Rust GIS desktop application built on [gpui](https://crates.io/crates/gpui) for both UI and rendering.

## Features

- Read GeoJSON, Shapefile, and FlatGeobuf data
- Render vector layers as gpui paths (fills, strokes, point markers)
- Show OpenStreetMap raster tiles behind vector data
- Keep map state in Web Mercator while reporting cursor coordinates in WGS-84

## Building on Ubuntu

The pinned toolchain is declared in [rust-toolchain.toml](rust-toolchain.toml) and will be installed automatically by `rustup` on first build.

### 1. Install Rust

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Install system dependencies

`gpui` on Linux needs Wayland/Vulkan/font/text/input libraries. This workspace also uses image loading, HTTP tile fetching, and local disk cache storage.

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

`reqwest` is configured with `rustls-tls`, so OpenSSL development headers are not required for the current dependency set.

### 3. Build and run

```sh
cargo run --release
```

For a faster edit/build cycle:

```sh
cargo run
```

## Building for the browser (wasm)

[crates/rgis-web](crates/rgis-web) is a `wasm32-unknown-unknown` build that
renders GeoJSON on an HTML5 canvas, reusing `rgis-core` for viewport math and
Web Mercator projection. It supports pan (click-drag) and zoom (scroll wheel).

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
| [crates/rgis-app](crates/rgis-app) | gpui application shell and `rgis` binary |
| [crates/rgis-core](crates/rgis-core) | Core GIS types, styling, projection helpers, project/viewport state |
| [crates/rgis-io](crates/rgis-io) | GeoJSON / Shapefile / FlatGeobuf readers |
| [crates/rgis-render](crates/rgis-render) | gpui-based map rendering helpers that build screen-space `Path<Pixels>` values |
| [crates/rgis-tiles](crates/rgis-tiles) | Raster tile fetching and caching |
| [crates/rgis-web](crates/rgis-web) | wasm/browser build rendering GeoJSON on an HTML5 canvas |

## Architecture notes

- `rgis-core`, `rgis-io`, and `rgis-tiles` stay backend-agnostic.
- `rgis-render` converts `rgis-core::Layer` geometry into gpui `PathBuilder` output each frame in screen space.
- `rgis-app` owns the gpui window, sidebar, status bar, tile image cache, and pan/zoom interactions.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
