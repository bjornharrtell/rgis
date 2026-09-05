//! Visual-parity test: renders rgis's own MapLibre-style pipeline for a
//! handful of real viewports over the OpenFreeMap "liberty" style and
//! compares the pixels against pre-generated maplibre-gl-js golden
//! screenshots (see `tools/parity/README.md` for how those are produced).
//!
//! Scope (see `tools/parity/generate-goldens.mjs` for the matching JS-side
//! note): symbol layers (text labels + icons) and fill-extrusion layers
//! (3D buildings) are intentionally left out of both sides of the
//! comparison. Font shaping/hinting differs meaningfully between a browser
//! and rgis's own SDF glyph pipeline regardless of whether the underlying
//! style evaluation is correct, so comparing label pixels would produce
//! noise unrelated to real parity bugs. Likewise, rgis deliberately does
//! not replicate MapLibre's perspective-lit 3D extrusion walls (it draws
//! extrusion footprints as flat fills), so that's not something this test
//! should ever flag as a "bug". This test therefore focuses on fill / line
//! / background / raster rendering.
//!
//! This test is **not** run as part of the normal `cargo test`: it needs a
//! real GPU adapter and live network access (to fetch the same vector/
//! raster tiles maplibre-gl-js fetched when the goldens were generated), so
//! it's marked `#[ignore]`. Run it explicitly with:
//!
//! ```sh
//! cargo test -p rgis-render --test parity -- --ignored --nocapture
//! ```
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use image::RgbaImage;
use rgis_core::{Viewport, lonlat_to_mercator};
use rgis_render::{
    BasemapTileDraw, EvalContext, MapCallback, MapRenderResources, StyleSheet, TileDraw,
    build_background_mesh, build_scene_mesh, build_tile_mesh, tile_screen_transform,
};
use rgis_tiles::{
    OPENFREEMAP_MAX_ZOOM, StyleRasterSource, TileCoord, TileFetcher, VectorTile, VectorTileFetcher,
    visible_tiles_for_zoom,
};

const LIBERTY_STYLE_JSON: &str = include_str!("../../rgis-style/fixtures/liberty.json");
const VIEWPORTS_JSON: &str = include_str!("../../../tools/parity/viewports.json");
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(serde::Deserialize)]
struct ViewportSpec {
    name: String,
    lon: f64,
    lat: f64,
    zoom: f64,
    width: u32,
    height: u32,
}

/// Polls `fetcher`'s decoded/raw channels until every requested coord has
/// resolved (decoding raw bytes inline as they arrive) or `timeout` elapses.
fn fetch_vector_tiles_blocking(
    fetcher: &VectorTileFetcher,
    coords: &[TileCoord],
    timeout: Duration,
) -> HashMap<TileCoord, Arc<VectorTile>> {
    let mut result = HashMap::new();
    for &coord in coords {
        fetcher.request(coord);
    }
    let deadline = Instant::now() + timeout;
    while result.len() < coords.len() && Instant::now() < deadline {
        let mut progressed = false;
        while let Ok(ready) = fetcher.receiver.try_recv() {
            result.insert(ready.coord, ready.tile);
            progressed = true;
        }
        while let Ok(fetched) = fetcher.raw_receiver.try_recv() {
            if let Ok(tile) = fetcher.decode_and_cache(fetched.coord, &fetched.bytes) {
                result.insert(fetched.coord, tile);
            }
            progressed = true;
        }
        if !progressed {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    result
}

fn fetch_raster_tiles_blocking(
    fetcher: &TileFetcher,
    coords: &[TileCoord],
    timeout: Duration,
) -> HashMap<TileCoord, Arc<RgbaImage>> {
    let mut result = HashMap::new();
    for &coord in coords {
        fetcher.request(coord);
    }
    let deadline = Instant::now() + timeout;
    while result.len() < coords.len() && Instant::now() < deadline {
        let mut progressed = false;
        while let Ok(ready) = fetcher.receiver.try_recv() {
            result.insert(ready.coord, ready.image);
            progressed = true;
        }
        if !progressed {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    result
}

/// Mirrors `rgis-app`'s (private) `tile_draw_key` -- only the *raster
/// source id* needs to be distinct per key here since this test only ever
/// draws each raster tile once (no icons share this key space).
fn tile_draw_key(source_id: &str, coord: TileCoord) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source_id.hash(&mut hasher);
    coord.z.hash(&mut hasher);
    coord.x.hash(&mut hasher);
    coord.y.hash(&mut hasher);
    hasher.finish()
}

/// Builds the `MapCallback` for one viewport: fetches every visible vector
/// tile (and any visible raster-source tiles), tessellates them under
/// `style`, and assembles the same background/basemap/raster draw lists
/// `rgis-app`'s render loop builds every frame -- see
/// `rgis-app/src/lib.rs`'s `render_map`/`collect_raster_tile_draws` for the
/// production equivalent this mirrors.
fn build_frame(style: &StyleSheet, viewport: &Viewport) -> MapCallback {
    let vector_fetcher = VectorTileFetcher::new_openfreemap();
    let coords = visible_tiles_for_zoom(viewport, OPENFREEMAP_MAX_ZOOM);
    let tiles = fetch_vector_tiles_blocking(&vector_fetcher, &coords, FETCH_TIMEOUT);
    assert_eq!(
        tiles.len(),
        coords.len(),
        "failed to fetch every visible vector tile before the timeout"
    );

    let basemap_tiles: Vec<BasemapTileDraw> = coords
        .iter()
        .map(|&coord| {
            let tile = &tiles[&coord];
            let mesh = Arc::new(build_tile_mesh(tile, coord, style));
            let transform = tile_screen_transform(coord, viewport);
            BasemapTileDraw {
                coord,
                mesh,
                offset: transform.offset,
                scale: transform.scale,
                width_scale: transform.width_scale,
                size: transform.size,
            }
        })
        .collect();

    let mut mesh = build_background_mesh(viewport, style);
    let background_index_count = mesh.indices.len() as u32;
    mesh.extend(build_scene_mesh(&[], viewport));

    // Raster-source layers (e.g. `liberty`'s low-zoom `ne2_shaded` Natural
    // Earth relief background) -- see `rgis-app::raster_fetchers_for_style`/
    // `collect_raster_tile_draws`, replicated here against the same style.
    let mut raster_tiles: Vec<TileDraw> = Vec::new();
    let eval_ctx = EvalContext::new(viewport.zoom);
    for layer in style.layers_of_kind("raster") {
        if !layer.matches_zoom(viewport.zoom) {
            continue;
        }
        let Some(source_id) = &layer.source else {
            continue;
        };
        let Some(source) = style.sources.get(source_id) else {
            continue;
        };
        let Some(template) = source.tiles.as_ref().and_then(|t| t.first()) else {
            continue;
        };
        let max_zoom = source.maxzoom.unwrap_or(22.0) as u8;
        let tile_size = source.tile_size.unwrap_or(256);
        let raster_source = StyleRasterSource::new(template.clone(), max_zoom, tile_size);
        let fetcher = TileFetcher::new(raster_source);
        let coords = visible_tiles_for_zoom(viewport, fetcher.max_zoom());
        let images = fetch_raster_tiles_blocking(&fetcher, &coords, FETCH_TIMEOUT);
        let opacity = layer.paint("raster-opacity").eval_f64(&eval_ctx, 1.0) as f32;
        for coord in coords {
            let Some(image) = images.get(&coord) else {
                continue;
            };
            let transform = tile_screen_transform(coord, viewport);
            raster_tiles.push(TileDraw {
                key: tile_draw_key(source_id, coord),
                rect: [
                    transform.offset[0],
                    transform.offset[1],
                    transform.size,
                    transform.size,
                ],
                rgba: Arc::clone(image),
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                opacity,
            });
        }
    }
    let raster_tile_count = raster_tiles.len() as u32;

    MapCallback {
        mesh,
        background_index_count,
        basemap_tiles,
        tiles: raster_tiles,
        raster_tile_count,
        vector_tile_count: 0,
        labels: Vec::new(),
        glyph_bitmaps: HashMap::new(),
        width: viewport.width_px as f32,
        height: viewport.height_px as f32,
    }
}

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Renders `callback` into an offscreen `width`x`height` RGBA image, doing
/// exactly what `egui_wgpu` would do for a `MapCallback` inside a real
/// paint pass, but manually: a real `eframe`/`egui` window isn't needed for
/// `MapCallback::prepare`/`paint` since both are plain `CallbackTrait`
/// methods taking a device/queue/render-pass and a type-erased resource
/// map, all of which are constructible directly.
fn render_offscreen(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    callback: &MapCallback,
    width: u32,
    height: u32,
) -> RgbaImage {
    let mut resources = egui_wgpu::CallbackResources::default();
    resources.insert(MapRenderResources::new(device, TARGET_FORMAT));

    let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("parity-msaa-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: rgis_render::MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let resolve_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("parity-resolve-color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let resolve_view = resolve_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut prepare_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parity-prepare-encoder"),
    });
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [width, height],
        pixels_per_point: 1.0,
    };
    let extra_buffers = {
        use egui_wgpu::CallbackTrait;
        callback.prepare(
            device,
            queue,
            &screen_descriptor,
            &mut prepare_encoder,
            &mut resources,
        )
    };
    let mut command_buffers = vec![prepare_encoder.finish()];
    command_buffers.extend(extra_buffers);

    let mut paint_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parity-paint-encoder"),
    });
    {
        let render_pass = paint_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("parity-render-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &msaa_view,
                resolve_target: Some(&resolve_view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let mut render_pass = render_pass.forget_lifetime();
        let info = epaint::PaintCallbackInfo {
            viewport: epaint::Rect::from_min_size(
                epaint::Pos2::ZERO,
                epaint::vec2(width as f32, height as f32),
            ),
            clip_rect: epaint::Rect::from_min_size(
                epaint::Pos2::ZERO,
                epaint::vec2(width as f32, height as f32),
            ),
            pixels_per_point: 1.0,
            screen_size_px: [width, height],
        };
        use egui_wgpu::CallbackTrait;
        callback.paint(info, &mut render_pass, &resources);
    }
    command_buffers.push(paint_encoder.finish());

    // Read the resolved (single-sample) texture back into an `RgbaImage`,
    // padding each row up to wgpu's 256-byte copy alignment then trimming
    // the padding back off once mapped.
    let unpadded_bytes_per_row = width * 4;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(256) * 256;
    let buffer_size = (padded_bytes_per_row * height) as u64;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity-readback-buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("parity-copy-encoder"),
    });
    copy_encoder.copy_texture_to_buffer(
        resolve_texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    command_buffers.push(copy_encoder.finish());
    queue.submit(command_buffers);

    let slice = readback_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("device poll failed");
    rx.recv()
        .expect("map_async callback never fired")
        .expect("failed to map readback buffer");

    let data = slice
        .get_mapped_range()
        .expect("failed to get mapped range");
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + unpadded_bytes_per_row as usize;
        pixels.extend_from_slice(&data[start..end]);
    }
    drop(data);
    readback_buffer.unmap();

    RgbaImage::from_raw(width, height, pixels).expect("readback buffer size mismatch")
}

/// Loose-tolerance pixel comparison: reports the mean per-channel absolute
/// difference and the fraction of "grossly different" pixels (channel
/// diff > 60/255 anywhere in the pixel). See the module doc comment for
/// why this is loose -- e.g. anti-aliasing edges, MSAA vs. browser
/// rasterization, and any minor filter/timing differences in tile fetches.
struct DiffStats {
    mean_abs_diff: f64,
    gross_diff_fraction: f64,
}

fn compare_images(actual: &RgbaImage, golden: &RgbaImage) -> DiffStats {
    assert_eq!(
        actual.dimensions(),
        golden.dimensions(),
        "rendered image and golden must be the same size"
    );
    let mut total_diff: u64 = 0;
    let mut gross_diff_pixels: u64 = 0;
    let pixel_count = (actual.width() * actual.height()) as u64;
    for (a, g) in actual.pixels().zip(golden.pixels()) {
        let mut pixel_max_diff = 0u8;
        for c in 0..3 {
            let diff = a[c].abs_diff(g[c]);
            total_diff += diff as u64;
            pixel_max_diff = pixel_max_diff.max(diff);
        }
        if pixel_max_diff > 60 {
            gross_diff_pixels += 1;
        }
    }
    DiffStats {
        mean_abs_diff: total_diff as f64 / (pixel_count * 3) as f64,
        gross_diff_fraction: gross_diff_pixels as f64 / pixel_count as f64,
    }
}

#[test]
#[ignore = "needs a real GPU adapter and live network access; run with `--ignored`"]
fn renders_match_maplibre_gl_js_within_loose_tolerance() {
    let mut style =
        StyleSheet::parse(LIBERTY_STYLE_JSON).expect("bundled liberty.json should parse");
    // `fill-extrusion` (3D buildings) is intentionally excluded from this
    // comparison -- see the module doc comment and generate-goldens.mjs's
    // matching filter. MapLibre renders extrusions with perspective-lit
    // walls even at pitch 0, which rgis doesn't attempt to replicate (it
    // draws extrusion footprints as flat fills), so including them would
    // compare two deliberately different renderings rather than a parity
    // bug.
    style.layers.retain(|l| l.kind != "fill-extrusion");
    let specs: Vec<ViewportSpec> =
        serde_json::from_str(VIEWPORTS_JSON).expect("tools/parity/viewports.json should parse");
    assert!(!specs.is_empty(), "expected at least one viewport");

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no wgpu adapter available -- this test needs a real or software GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("parity-test-device"),
        ..Default::default()
    }))
    .expect("failed to create wgpu device");

    let goldens_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/parity/goldens");
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/parity-out");
    std::fs::create_dir_all(&out_dir).ok();

    let mut failures = Vec::new();
    for spec in &specs {
        let viewport = Viewport {
            center: lonlat_to_mercator(spec.lon, spec.lat),
            zoom: spec.zoom,
            width_px: spec.width,
            height_px: spec.height,
        };
        let callback = build_frame(&style, &viewport);
        let actual = render_offscreen(&device, &queue, &callback, spec.width, spec.height);

        let golden_path = goldens_dir.join(format!("{}.png", spec.name));
        let golden = image::open(&golden_path)
            .unwrap_or_else(|e| panic!("failed to load golden {golden_path:?}: {e}"))
            .into_rgba8();

        let stats = compare_images(&actual, &golden);
        println!(
            "[{}] mean_abs_diff={:.2}/255 gross_diff_fraction={:.2}%",
            spec.name,
            stats.mean_abs_diff,
            stats.gross_diff_fraction * 100.0
        );

        let actual_path = out_dir.join(format!("{}-actual.png", spec.name));
        actual.save(&actual_path).ok();

        // Loose tolerance, per the agreed plan: this is meant to catch
        // gross regressions (wrong colors, missing layer types, badly
        // broken filters/zoom interpolation), not pixel-perfect equality.
        // The `region` viewport in particular carries extra baseline noise
        // from the `ne2_shaded` Natural Earth raster background (rgis and
        // maplibre-gl-js resample/blend that raw imagery slightly
        // differently), very-sub-pixel-width road lines at zoom 6 (browsers
        // vs. this app's analytic-AA line shader don't feather sub-1px
        // strokes identically), and low-opacity many-cornered landcover
        // fills (forests/coastlines): this renderer's fill antialiasing
        // draws a same-color outline as a chain of overlapping stroke
        // quads (see `eval_fill_outline_paint`), and at concave vertices
        // those quads overlap slightly, so a <1.0 `fill-opacity` gets
        // alpha-blended more than once right at the boundary -- a
        // real but architectural (not style-evaluation) difference from
        // MapLibre's coverage-based antialiasing, which has no such
        // double-blend. None of these is a style-evaluation bug, so the
        // thresholds below are calibrated with headroom for them.
        //
        // These thresholds were tightened after fixing a real bug in
        // `rgis_core::Viewport::resolution` (it assumed a 256px tile, but
        // maplibre-gl-js's own camera zoom convention uses 512px --
        // `MAPLIBRE_TILE_SIZE`/`viewport_matches_maplibre_gl_js_bounds_convention`
        // in `rgis-core`), which had been showing 4x the real geographic
        // area for the same nominal zoom and dominated every viewport's
        // diff (typical `gross_diff_fraction` dropped from ~20% to ~1-4%
        // once fixed).
        const MAX_MEAN_ABS_DIFF: f64 = 15.0;
        const MAX_GROSS_DIFF_FRACTION: f64 = 0.10;
        if stats.mean_abs_diff > MAX_MEAN_ABS_DIFF
            || stats.gross_diff_fraction > MAX_GROSS_DIFF_FRACTION
        {
            failures.push(format!(
                "viewport \"{}\": mean_abs_diff={:.2} (max {MAX_MEAN_ABS_DIFF}), \
                 gross_diff_fraction={:.2}% (max {:.2}%) -- see {actual_path:?}",
                spec.name,
                stats.mean_abs_diff,
                stats.gross_diff_fraction * 100.0,
                MAX_GROSS_DIFF_FRACTION * 100.0,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "one or more viewports exceeded the loose parity tolerance:\n{}",
        failures.join("\n")
    );
}
