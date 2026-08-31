//! Shared `eframe::App` implementation for `rgis`, used by both the native
//! binary ([`crate`] via `main.rs`) and the browser build (`rgis-web`, via
//! `eframe::WebRunner`).

use std::num::NonZeroUsize;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
use std::sync::Arc;

use lru::LruCache;
use poll_promise::Promise;
use rgis_core::{Layer, LayerId, Project, mercator_to_lonlat};
use rgis_io::{IoError, LoadedLayer};
use rgis_render::{GlyphBitmapRanges, LabelGlyphInstance, SceneMesh, TileMesh};
use rgis_tiles::{
    GLYPH_BUFFER, GLYPH_PIXELS_PER_EM, GlyphFetcher, OPENFREEMAP_MAX_ZOOM, StyleRasterSource,
    TileCoord, TileFetcher, VectorTileFetcher, glyph_range_start, visible_tiles_for_zoom,
};

mod status_bar;
#[cfg(target_arch = "wasm32")]
mod tile_worker_pool;

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// Debug-only hook so automated browser testing can jump the viewport
    /// directly to a `(lon, lat, zoom)` instead of simulating imprecise
    /// wheel/drag input -- see `rgis-web`'s `start`, which exposes this to
    /// the page as `window.debugJumpViewport(lon, lat, zoom)`.
    static DEBUG_VIEWPORT_JUMP: std::cell::Cell<Option<(f64, f64, f64)>> =
        const { std::cell::Cell::new(None) };
}
#[cfg(target_arch = "wasm32")]
pub fn debug_jump_viewport(lon: f64, lat: f64, zoom: f64) {
    DEBUG_VIEWPORT_JUMP.set(Some((lon, lat, zoom)));
}

/// Debug-only hook so automated browser testing can read the main thread's
/// current wasm linear memory size on demand (bytes), instead of scraping
/// periodic console logging -- see `rgis-web`'s `start`, which exposes this
/// as `window.debugMemBytes()`.
#[cfg(target_arch = "wasm32")]
pub fn debug_wasm_memory_bytes() -> u32 {
    wasm_memory_bytes()
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// Every distinct tile coord ever successfully tessellated this session,
    /// regardless of later LRU eviction from `RgisApp::tile_meshes` --
    /// diagnostic-only, to distinguish "memory grows because an ever-larger
    /// number of distinct tiles has been visited" (expected, not a leak)
    /// from "memory keeps growing even though few/no new tiles are being
    /// seen" (an actual leak) -- see repo memory notes on the ongoing OOM
    /// investigation.
    static DISTINCT_TILES_SEEN: std::cell::RefCell<std::collections::HashSet<TileCoord>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

/// Debug-only hook so automated browser testing can read the cumulative
/// count of distinct tiles ever tessellated this session, via
/// `window.debugDistinctTileCount()`.
#[cfg(target_arch = "wasm32")]
pub fn debug_distinct_tile_count() -> u32 {
    DISTINCT_TILES_SEEN.with(|s| s.borrow().len() as u32)
}

/// A demo dataset (a few simple shapes over Europe) bundled with the binary
/// so both the native app and the browser build have something to show
/// without requiring a file picker.
pub const SAMPLE_GEOJSON: &[u8] = include_bytes!("../assets/sample.geojson");

/// Max number of tessellated tile meshes kept in `RgisApp::tile_meshes`.
/// Without a bound this cache grows forever as the user pans/zooms over new
/// areas (unlike the GPU-side buffers, which only ever hold the currently
/// visible tiles) -- matches the order of magnitude of the other tile caches
/// in this codebase (`rgis_tiles::TileCache`'s 256, `VectorTileFetcher`'s
/// 128 decoded-tile cache).
const TILE_MESH_CACHE_SIZE: usize = 256;
/// Per raster style source (e.g. `natural_earth`) -- see `raster_tile_cache`.
const RASTER_TILE_CACHE_SIZE: usize = 128;
/// Fixed `TileDraw::key` shared by every icon quad, since they all sample
/// the one fetched sprite atlas texture (as opposed to raster tiles, which
/// each get their own key/texture) -- see `collect_label_draws`.
const SPRITE_ATLAS_TILE_KEY: u64 = u64::MAX;

/// Max newly-arrived tiles handed off to tessellation per `drain_ready_tiles`
/// call (once per frame) -- see the comment on its call site for why this
/// matters most on wasm, where tessellation isn't actually backgrounded.
const MAX_TESSELLATIONS_PER_FRAME: usize = 3;

/// Extra vertical padding added above/below the sidebar tree row content.
const ROW_VPAD: f32 = 5.0;
/// Background fill painted behind a hovered sidebar tree row.
const ROW_HOVER_FILL: egui::Color32 = egui::Color32::from_gray(40);

/// Screen width (logical points) below which the layer list switches from a
/// docked side panel to a floating overlay toggled by a button, so the map
/// gets the full viewport width on phones/narrow windows.
const MOBILE_WIDTH_THRESHOLD: f32 = 600.0;

/// One or more layers finished (or failed) loading, each tagged with a
/// display name for error messages.
type LoadResults = Vec<(String, Result<LoadedLayer, IoError>)>;

/// The default style, bundled into the binary (rather than fetched over the
/// network at startup) so the basemap renders immediately -- see
/// `RgisApp::set_style` for switching to a different style document live.
const DEFAULT_STYLE_JSON: &str = include_str!("../../rgis-style/fixtures/liberty.json");

pub struct RgisApp {
    project: Project,
    vector_tile_fetcher: Arc<VectorTileFetcher>,
    glyph_fetcher: Arc<GlyphFetcher>,
    /// The currently active MapLibre/Mapbox style document, driving every
    /// basemap tile's fill/line/label styling (see
    /// `rgis_render::build_tile_mesh`) -- swappable at runtime via
    /// [`RgisApp::set_style`], which also invalidates `tile_meshes` so
    /// already-tessellated tiles get rebuilt under the new style.
    style: Arc<rgis_render::StyleSheet>,
    /// Raw JSON of `style`, kept alongside it only so the wasm worker pool
    /// (a separate wasm instance per worker, with no shared memory) can
    /// ship the current style to a worker as a plain string -- see
    /// `tile_worker_pool`.
    #[cfg(target_arch = "wasm32")]
    style_json: Rc<str>,
    /// One [`TileFetcher`] per `"type": "raster"` style source (e.g.
    /// `natural_earth`), created from `style.sources` -- see
    /// `raster_fetchers_for_style`. Rebuilt whenever the style changes
    /// (`set_style`), since a different style may use different raster
    /// sources entirely.
    raster_fetchers: std::collections::HashMap<String, Arc<TileFetcher>>,
    /// Decoded raster tile bitmaps, per source id, drained each frame from
    /// each fetcher's `receiver` in `drain_ready_tiles` -- see
    /// `raster_fetchers`.
    raster_tile_cache:
        std::collections::HashMap<String, LruCache<TileCoord, Arc<image::RgbaImage>>>,
    /// Fetcher for the current style's sprite atlas (`style.sprite`), if
    /// any; recreated by `set_style`. `None` once the style has no
    /// `sprite` field, or before that fetch has resolved.
    sprite_fetcher: Option<Arc<rgis_tiles::SpriteFetcher>>,
    /// The decoded sprite atlas, once `sprite_fetcher`'s single fetch
    /// resolves -- see `drain_ready_tiles`/`collect_label_draws`.
    sprite_atlas: Option<Arc<rgis_tiles::SpriteAtlas>>,
    /// Tessellated mesh per tile, in tile-local metres (viewport-independent
    /// — see `rgis_render::build_tile_mesh`) — built once per tile and
    /// reused across every pan/zoom, unlike the final screen-space mesh
    /// which is cheap to recompute every frame from these. Bounded (LRU) so
    /// panning/zooming over a large area doesn't grow memory forever.
    tile_meshes: LruCache<TileCoord, Arc<TileMesh>>,
    pending_tiles: std::collections::HashSet<TileCoord>,
    /// Decode-and-tessellation jobs in flight: a tile arriving from
    /// `vector_tile_fetcher` (either already decoded, from its cache, or as
    /// raw bytes needing MVT decode first) is processed on a background
    /// thread (native) / spawned task (wasm) rather than inline in
    /// `drain_ready_tiles`, since decoding and tessellating a complex tile
    /// can take long enough to visibly stall input handling (e.g.
    /// zoom-wheel events) if done synchronously in the UI update loop. The
    /// inner `Option` is `None` when decoding raw bytes failed.
    pending_tile_meshes: Vec<Promise<(TileCoord, Option<TileMesh>)>>,
    /// wasm-only pool of dedicated Web Workers used to decode+tessellate
    /// tiles arriving as raw network bytes off the main thread -- see
    /// `tile_worker_pool` module docs.
    #[cfg(target_arch = "wasm32")]
    tile_worker_pool: Rc<tile_worker_pool::TileWorkerPool>,
    pending_loads: Vec<Promise<LoadResults>>,
    cursor_lonlat: Option<(f64, f64)>,
    last_error: Option<String>,
    layers_expanded: bool,
    /// Screen-space start position of an in-progress shift-drag
    /// bounding-box zoom gesture (see `render_map`) -- `None` when no such
    /// gesture is active, in which case a plain drag pans the map instead.
    bbox_zoom_start: Option<egui::Pos2>,
    /// Whether the floating layer-list overlay is shown on narrow/mobile
    /// viewports (see `MOBILE_WIDTH_THRESHOLD`) — irrelevant on desktop,
    /// where the layer list is always visible as a docked side panel.
    mobile_layers_open: bool,
    /// e.g. "Vulkan"/"Metal" (native) or "BrowserWebGpu"/"Gl" (web) — shown
    /// in the status bar since the web build silently falls back to WebGL2
    /// (much higher per-draw-call overhead) when WebGPU isn't available.
    gpu_backend_label: String,
}

impl RgisApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be configured with the wgpu renderer");
        render_state.renderer.write().callback_resources.insert(
            rgis_render::MapRenderResources::new(&render_state.device, render_state.target_format),
        );
        let gpu_backend_label = format!("{:?}", render_state.adapter.get_info().backend);
        let style = rgis_render::StyleSheet::parse(DEFAULT_STYLE_JSON)
            .expect("bundled default style JSON should parse");
        let raster_fetchers = raster_fetchers_for_style(&style);
        let sprite_fetcher = style.sprite.as_deref().map(rgis_tiles::SpriteFetcher::new);

        Self {
            project: Project::default(),
            vector_tile_fetcher: VectorTileFetcher::new_openfreemap(),
            glyph_fetcher: GlyphFetcher::new(),
            style: Arc::new(style),
            #[cfg(target_arch = "wasm32")]
            style_json: Rc::from(DEFAULT_STYLE_JSON),
            raster_fetchers,
            raster_tile_cache: std::collections::HashMap::new(),
            sprite_fetcher,
            sprite_atlas: None,
            tile_meshes: LruCache::new(NonZeroUsize::new(TILE_MESH_CACHE_SIZE).unwrap()),
            pending_tiles: std::collections::HashSet::new(),
            pending_tile_meshes: Vec::new(),
            #[cfg(target_arch = "wasm32")]
            tile_worker_pool: Rc::new(tile_worker_pool::TileWorkerPool::new()),
            pending_loads: Vec::new(),
            cursor_lonlat: None,
            last_error: None,
            layers_expanded: true,
            bbox_zoom_start: None,
            mobile_layers_open: false,
            gpu_backend_label,
        }
    }

    /// Parses `style_json` as a MapLibre/Mapbox style document and switches
    /// the basemap to it live: every currently cached tile mesh is dropped
    /// (see `tile_meshes`) so visible tiles retessellate under the new
    /// style on the next frame (re-decoding is *not* needed --
    /// `vector_tile_fetcher`'s own decoded-tile cache is untouched), without
    /// restarting the app. Returns an error (leaving the current style
    /// active) if `style_json` doesn't parse.
    pub fn set_style(&mut self, style_json: &str) -> Result<(), String> {
        let parsed = rgis_render::StyleSheet::parse(style_json).map_err(|e| e.to_string())?;
        self.raster_fetchers = raster_fetchers_for_style(&parsed);
        self.raster_tile_cache.clear();
        self.sprite_fetcher = parsed.sprite.as_deref().map(rgis_tiles::SpriteFetcher::new);
        self.sprite_atlas = None;
        self.style = Arc::new(parsed);
        #[cfg(target_arch = "wasm32")]
        {
            self.style_json = Rc::from(style_json);
        }
        self.tile_meshes.clear();
        self.pending_tiles.clear();
        self.pending_tile_meshes.clear();
        Ok(())
    }

    /// Queue loading files already on disk (native only — used for files
    /// passed on the command line at startup).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn queue_load_paths(&mut self, paths: Vec<std::path::PathBuf>) {
        if paths.is_empty() {
            return;
        }
        self.pending_loads.push(Promise::spawn_thread(
            "load-layers",
            move || -> LoadResults {
                paths
                    .into_iter()
                    .map(|path| {
                        let name = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("layer")
                            .to_string();
                        let result = rgis_io::load_path(&path);
                        (name, result)
                    })
                    .collect()
            },
        ));
    }

    /// Prompt the user for one or more layer files and queue them for
    /// loading, via the platform-appropriate file dialog + IO.
    fn queue_pick_files(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pending_loads.push(Promise::spawn_thread(
                "pick-and-load-layers",
                move || -> LoadResults {
                    let Some(paths) = rfd::FileDialog::new().pick_files() else {
                        return Vec::new();
                    };
                    paths
                        .into_iter()
                        .map(|path| {
                            let name = path
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("layer")
                                .to_string();
                            let result = rgis_io::load_path(&path);
                            (name, result)
                        })
                        .collect()
                },
            ));
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.pending_loads.push(Promise::spawn_local(async move {
                let Some(handles) = rfd::AsyncFileDialog::new().pick_files().await else {
                    return Vec::new();
                };
                let mut results = Vec::new();
                for handle in handles {
                    let name = handle.file_name();
                    let bytes = handle.read().await;
                    let result = rgis_io::load_bytes(&name, &bytes);
                    results.push((name, result));
                }
                results
            }));
        }
    }

    fn poll_pending_loads(&mut self) {
        let pending = std::mem::take(&mut self.pending_loads);
        let mut still_pending = Vec::new();
        for promise in pending {
            match promise.try_take() {
                Ok(results) => {
                    for (name, result) in results {
                        self.apply_load_result(name, result);
                    }
                }
                Err(promise) => still_pending.push(promise),
            }
        }
        self.pending_loads = still_pending;
    }

    /// Load an in-memory byte buffer as a layer. Works on any target;
    /// useful for a bundled demo dataset in the browser build.
    pub fn queue_load_bytes(&mut self, name: String, bytes: Vec<u8>) {
        let result = rgis_io::load_bytes(&name, &bytes);
        self.apply_load_result(name, result);
    }

    /// Load the bundled [`SAMPLE_GEOJSON`] demo dataset as a layer.
    pub fn queue_load_sample(&mut self) {
        self.queue_load_bytes("sample.geojson".to_string(), SAMPLE_GEOJSON.to_vec());
    }

    fn apply_load_result(&mut self, name: String, result: Result<LoadedLayer, IoError>) {
        match result {
            Ok(loaded) => {
                let id = self.project.next_layer_id();
                let layer = Layer::new(id, loaded.name, loaded.features);
                let bounds = layer.bounds;
                self.project.add_layer(layer);
                if let Some(bounds) = bounds {
                    self.project.viewport.fit_bounds(&bounds);
                }
                self.last_error = None;
            }
            Err(error) => {
                self.last_error = Some(format!("Failed to load {name}: {error}"));
            }
        }
    }

    fn drain_ready_tiles(&mut self) {
        for (source_id, fetcher) in &self.raster_fetchers {
            let cache = self
                .raster_tile_cache
                .entry(source_id.clone())
                .or_insert_with(|| {
                    LruCache::new(NonZeroUsize::new(RASTER_TILE_CACHE_SIZE).unwrap())
                });
            while let Ok(ready) = fetcher.receiver.try_recv() {
                cache.put(ready.coord, ready.image);
            }
        }
        if let Some(fetcher) = &self.sprite_fetcher
            && let Ok(ready) = fetcher.receiver.try_recv()
        {
            self.sprite_atlas = Some(ready.atlas);
        }

        // Capped per frame: on wasm, `Promise::spawn_local` has no real
        // background thread to run on -- its future body runs synchronously
        // as a microtask, and ALL microtasks queued in one `update()` call
        // run back-to-back before the browser can paint or handle input.
        // Spawning a big burst of decode+tessellation jobs at once (e.g.
        // many tiles arriving together after a fast zoom) previously froze
        // the tab for as long as all of them together took; capping how
        // many start per frame spreads that cost across frames instead,
        // keeping the UI responsive at the cost of tiles finishing a bit
        // later.
        for _ in 0..MAX_TESSELLATIONS_PER_FRAME {
            if let Ok(ready) = self.vector_tile_fetcher.receiver.try_recv() {
                let coord = ready.coord;
                let tile = ready.tile;
                let style = Arc::clone(&self.style);
                #[cfg(not(target_arch = "wasm32"))]
                let promise = Promise::spawn_thread("tessellate-tile", move || {
                    (
                        coord,
                        Some(rgis_render::build_tile_mesh(&tile, coord, &style)),
                    )
                });
                #[cfg(target_arch = "wasm32")]
                let promise = Promise::spawn_local(async move {
                    (
                        coord,
                        Some(rgis_render::build_tile_mesh(&tile, coord, &style)),
                    )
                });
                self.pending_tile_meshes.push(promise);
            } else if let Ok(fetched) = self.vector_tile_fetcher.raw_receiver.try_recv() {
                // Raw network-fetched bytes: MVT decode (parsing every
                // feature's geometry/properties out of the protobuf) is
                // real CPU work too, so it's handled together with
                // tessellation rather than running unconditionally in the
                // network response callback (see
                // `VectorTileFetcher::fetch_url`). On wasm this whole
                // decode+tessellate step runs on a pooled Web Worker (see
                // `tile_worker_pool`) instead of the main thread; note this
                // bypasses `VectorTileFetcher`'s decoded-tile cache (the
                // worker never reports the decoded `VectorTile` back, only
                // the final mesh) -- an accepted trade-off, since that
                // cache only matters for the rare case of the final mesh
                // cache having evicted a tile the fetcher's own cache still
                // holds (see the `tile_worker_pool` module docs).
                let coord = fetched.coord;
                let bytes = fetched.bytes;
                #[cfg(not(target_arch = "wasm32"))]
                let promise = {
                    let fetcher = Arc::clone(&self.vector_tile_fetcher);
                    let style = Arc::clone(&self.style);
                    Promise::spawn_thread("decode-tessellate-tile", move || {
                        match fetcher.decode_and_cache(coord, &bytes) {
                            Ok(tile) => (
                                coord,
                                Some(rgis_render::build_tile_mesh(&tile, coord, &style)),
                            ),
                            Err(_) => (coord, None),
                        }
                    })
                };
                #[cfg(target_arch = "wasm32")]
                let promise = {
                    let pool = Rc::clone(&self.tile_worker_pool);
                    let style_json = Rc::clone(&self.style_json);
                    Promise::spawn_local(async move {
                        let mesh = pool.tessellate(coord, bytes, style_json).await;
                        (coord, mesh)
                    })
                };
                self.pending_tile_meshes.push(promise);
            } else {
                break;
            }
        }

        let pending = std::mem::take(&mut self.pending_tile_meshes);
        let mut still_pending = Vec::new();
        for promise in pending {
            match promise.try_take() {
                Ok((coord, Some(mesh))) => {
                    self.pending_tiles.remove(&coord);
                    #[cfg(target_arch = "wasm32")]
                    DISTINCT_TILES_SEEN.with(|s| s.borrow_mut().insert(coord));
                    self.tile_meshes.put(coord, Arc::new(mesh));
                }
                Ok((coord, None)) => {
                    self.pending_tiles.remove(&coord);
                }
                Err(promise) => still_pending.push(promise),
            }
        }
        self.pending_tile_meshes = still_pending;
    }

    fn drain_ready_glyphs(&mut self) -> bool {
        let mut drained_any = false;
        while self.glyph_fetcher.receiver.try_recv().is_ok() {
            drained_any = true;
        }
        drained_any
    }

    /// Walks up from `coord` to find the closest already-cached ancestor
    /// tile, used as a placeholder (drawn scaled up to cover the same area)
    /// while the actual tile is still loading, so zooming in shows the old
    /// detail enlarged instead of a blank gap.
    fn nearest_cached_ancestor(&self, coord: TileCoord) -> Option<TileCoord> {
        let (mut z, mut x, mut y) = (coord.z, coord.x, coord.y);
        while z > 0 {
            z -= 1;
            x /= 2;
            y /= 2;
            let ancestor = TileCoord { z, x, y };
            if self.tile_meshes.contains(&ancestor) {
                return Some(ancestor);
            }
        }
        None
    }

    /// For every `raster` style layer (e.g. `natural_earth`), requests
    /// currently-visible tiles from its source's [`TileFetcher`] and builds
    /// screen-space [`rgis_render::TileDraw`]s for whatever's already
    /// cached/decoded -- tiles still in flight simply don't draw this frame
    /// (no lower-zoom placeholder fallback, unlike `basemap_tiles`, since
    /// raster imagery is a coarse background layer where a brief blank gap
    /// while tiles arrive is much less noticeable than for line/label
    /// detail).
    fn collect_raster_tile_draws(&mut self) -> Vec<rgis_render::TileDraw> {
        let mut draws = Vec::new();
        for layer in self.style.layers_of_kind("raster") {
            // Respect the layer's own `minzoom`/`maxzoom` visibility range
            // (distinct from the source's `maxzoom`, which only clamps
            // which tiles get *fetched*/overzoomed). Without this check a
            // layer like `natural_earth` (`maxzoom: 7`) keeps drawing its
            // stretched, overzoomed low-res tile past its intended cutoff,
            // which is especially visible for tilesets that bake labels
            // into the raster imagery itself (e.g. Natural Earth's
            // shaded-relief tiles include country-name labels) -- those
            // then show up as ghostly duplicate text behind the real
            // vector labels.
            if !layer.matches_zoom(self.project.viewport.zoom) {
                continue;
            }
            let Some(source_id) = &layer.source else {
                continue;
            };
            let Some(fetcher) = self.raster_fetchers.get(source_id) else {
                continue;
            };
            let coords = visible_tiles_for_zoom(&self.project.viewport, fetcher.max_zoom());
            let ctx = rgis_render::EvalContext::new(self.project.viewport.zoom);
            let opacity = layer.paint("raster-opacity").eval_f64(&ctx, 1.0) as f32;
            let cache = self
                .raster_tile_cache
                .entry(source_id.clone())
                .or_insert_with(|| {
                    LruCache::new(NonZeroUsize::new(RASTER_TILE_CACHE_SIZE).unwrap())
                });
            for coord in coords {
                if let Some(image) = cache.get(&coord) {
                    let transform =
                        rgis_render::tile_screen_transform(coord, &self.project.viewport);
                    draws.push(rgis_render::TileDraw {
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
                } else {
                    fetcher.request(coord);
                }
            }
        }
        draws
    }

    /// The root "LAYERS" tree entry: click to expand/collapse, hover to
    /// reveal the "add layer" action — mirrors VS Code's explorer pane.
    fn render_layers_root_row(&mut self, ui: &mut egui::Ui) {
        let row_height = ui.spacing().interact_size.y + ROW_VPAD * 2.0;
        let rect = egui::Rect::from_min_size(
            ui.cursor().min,
            egui::vec2(ui.available_width(), row_height),
        );
        // A passive geometric check (not a registered widget) so it doesn't
        // compete with the add-button's own click hit-testing.
        let hovered = ui.rect_contains_pointer(rect);
        if hovered {
            ui.painter().rect_filled(rect, 2.0, ROW_HOVER_FILL);
        }

        let mut add_clicked = false;
        ui.horizontal(|ui| {
            // Force the row to its full padded height so content is
            // vertically centered within it (plain `ui.horizontal` only
            // assumes `interact_size.y` and top-aligns any extra space).
            ui.set_min_height(row_height);
            let icon_size = ui.spacing().icon_width.max(10.0);
            let (icon_rect, icon_response) =
                ui.allocate_exact_size(egui::vec2(icon_size, icon_size), egui::Sense::click());
            if icon_response.clicked() {
                self.layers_expanded = !self.layers_expanded;
            }
            let openness = if self.layers_expanded { 1.0 } else { 0.0 };
            egui::containers::collapsing_header::paint_default_icon(
                ui,
                openness,
                &icon_response.with_new_rect(icon_rect),
            );
            ui.label(egui::RichText::new("LAYERS").small().strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(4.0);
                if hovered && icon_button(ui, "➕", "Add Layer…").clicked() {
                    add_clicked = true;
                }
            });
        });

        if add_clicked {
            self.queue_pick_files();
        }
    }

    /// The layer tree + error message, shared between the desktop docked
    /// panel and the mobile floating overlay (see `render_sidebar`).
    fn render_layers_content(&mut self, ui: &mut egui::Ui) {
        let mut to_toggle: Option<LayerId> = None;
        let mut to_remove: Option<LayerId> = None;
        let mut show_tiles = self.project.show_tiles;

        egui::ScrollArea::vertical().show(ui, |ui| {
            self.render_layers_root_row(ui);

            if self.layers_expanded {
                // Listed top-to-bottom in draw order (topmost drawn
                // layer first), with the OSM basemap last since it's
                // always the bottom of the stack.
                for layer in self.project.layers.iter().rev() {
                    let mut visible = layer.visible;
                    let (toggled, removed) = tree_row(ui, &mut visible, &layer.name, true);
                    if toggled {
                        to_toggle = Some(layer.id);
                    }
                    if removed {
                        to_remove = Some(layer.id);
                    }
                }

                tree_row(ui, &mut show_tiles, "OpenFreeMap Background", false);
            }
        });

        self.project.show_tiles = show_tiles;
        if let Some(id) = to_toggle
            && let Some(layer) = self.project.get_layer_mut(id)
        {
            layer.visible = !layer.visible;
        }
        if let Some(id) = to_remove {
            self.project.remove_layer(id);
        }

        if let Some(error) = &self.last_error {
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(0xe0, 0x6c, 0x75), error);
        }
    }

    /// On desktop/wide viewports, renders the layer list as the usual
    /// docked left panel. On narrow/mobile viewports it instead renders a
    /// small floating toggle button plus (when opened) a floating overlay
    /// window, so the map can use the full screen width -- see
    /// `MOBILE_WIDTH_THRESHOLD`.
    fn render_sidebar(&mut self, ui: &mut egui::Ui) {
        let is_mobile = ui.ctx().input(|i| i.viewport_rect()).width() < MOBILE_WIDTH_THRESHOLD;

        if !is_mobile {
            egui::Panel::left("sidebar")
                .default_size(280.0)
                .show(ui, |ui| self.render_layers_content(ui));
            return;
        }

        egui::Area::new(egui::Id::new("mobile_layers_toggle"))
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(8.0, 8.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    let label = if self.mobile_layers_open {
                        "✕"
                    } else {
                        "☰"
                    };
                    if ui.button(label).clicked() {
                        self.mobile_layers_open = !self.mobile_layers_open;
                    }
                });
            });

        if self.mobile_layers_open {
            egui::Window::new("Layers")
                .id(egui::Id::new("mobile_layers_window"))
                .default_pos(egui::pos2(8.0, 48.0))
                .default_width(260.0)
                .max_height(ui.ctx().input(|i| i.viewport_rect()).height() - 96.0)
                .collapsible(false)
                .resizable(true)
                .show(ui.ctx(), |ui| self.render_layers_content(ui));
        }
    }

    fn render_status_bar(&self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status_bar").show(ui, |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label("EPSG:4326");
                ui.separator();
                ui.label(&self.gpu_backend_label);
                ui.separator();
                ui.label(status_bar::format_scale(self.project.viewport.resolution()));
                ui.separator();
                let coords = self
                    .cursor_lonlat
                    .map(|(lon, lat)| status_bar::format_coordinates(lon, lat))
                    .unwrap_or_else(|| "—".to_string());
                ui.label(coords);
            });
        });
    }

    fn render_map(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

                self.project.viewport.width_px = rect.width().max(1.0).round() as u32;
                self.project.viewport.height_px = rect.height().max(1.0).round() as u32;

                // Shift + left-click-drag draws a bounding box to zoom to,
                // instead of panning -- the mode is decided once, at drag
                // start, so releasing shift mid-drag doesn't switch modes.
                if response.drag_started() && ui.input(|i| i.modifiers.shift) {
                    self.bbox_zoom_start = response.interact_pointer_pos();
                }

                // Skip single-finger-drag panning while a bbox-zoom drag or
                // a pinch/two-finger gesture is active -- otherwise the
                // emulated primary pointer (which tracks one of the
                // touches) would apply a second, conflicting pan on top of
                // the two-finger translation handled below.
                let multi_touch_active = ui.ctx().multi_touch().is_some();
                if response.dragged() {
                    if self.bbox_zoom_start.is_none() && !multi_touch_active {
                        let delta = response.drag_delta();
                        self.project.viewport.pan(delta.x, delta.y);
                    }
                } else if response.drag_stopped()
                    && let Some(start) = self.bbox_zoom_start.take()
                {
                    let end = response
                        .interact_pointer_pos()
                        .unwrap_or(ui.ctx().pointer_latest_pos().unwrap_or(start));
                    let min = start.min(end);
                    let max = start.max(end);
                    // Ignore near-zero-area boxes (e.g. a shift-click with
                    // no real drag) rather than zooming to nothing.
                    if (max.x - min.x) > 4.0 && (max.y - min.y) > 4.0 {
                        let world_a = self
                            .project
                            .viewport
                            .screen_to_world([min.x - rect.min.x, min.y - rect.min.y]);
                        let world_b = self
                            .project
                            .viewport
                            .screen_to_world([max.x - rect.min.x, max.y - rect.min.y]);
                        let bounds = rgis_core::Bounds {
                            min_x: world_a.x.min(world_b.x),
                            min_y: world_a.y.min(world_b.y),
                            max_x: world_a.x.max(world_b.x),
                            max_y: world_a.y.max(world_b.y),
                        };
                        self.project.viewport.fit_bounds(&bounds);
                    }
                }

                if response.hovered() {
                    let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > f32::EPSILON
                        && let Some(pos) = response.hover_pos()
                    {
                        let local = pos - rect.min;
                        let zoom_delta = if scroll > 0.0 { 0.125 } else { -0.125 };
                        self.project
                            .viewport
                            .zoom_toward([local.x, local.y], zoom_delta);
                    }
                }

                // Two-finger pinch-to-zoom (touch devices). `multi_touch` is
                // a global gesture (not tied to a specific widget response),
                // so it's gated on the gesture's center falling within the
                // map rect -- otherwise a pinch elsewhere (e.g. over the
                // mobile floating layer list) would also zoom the map.
                if let Some(touch) = ui.ctx().multi_touch()
                    && rect.contains(touch.center_pos)
                {
                    let local = touch.center_pos - rect.min;
                    if (touch.zoom_delta - 1.0).abs() > f32::EPSILON {
                        let zoom_delta = (touch.zoom_delta as f64).log2();
                        self.project
                            .viewport
                            .zoom_toward([local.x, local.y], zoom_delta);
                    }
                    // Two-finger drag also pans, in addition to pinch-zoom.
                    if touch.translation_delta != egui::Vec2::ZERO {
                        self.project
                            .viewport
                            .pan(touch.translation_delta.x, touch.translation_delta.y);
                    }
                }

                self.cursor_lonlat = response.hover_pos().map(|pos| {
                    let local = pos - rect.min;
                    let world = self.project.viewport.screen_to_world([local.x, local.y]);
                    mercator_to_lonlat(world.x, world.y)
                });

                let mut mesh = SceneMesh::default();
                let mut background_index_count = 0;
                let mut basemap_tiles = Vec::new();
                let mut fallback_tile_count = 0;
                if self.project.show_tiles {
                    let mut current_draws = Vec::new();
                    let mut fallback_coords = std::collections::HashSet::new();
                    for coord in
                        visible_tiles_for_zoom(&self.project.viewport, OPENFREEMAP_MAX_ZOOM)
                    {
                        if let Some(tile_mesh) = self.tile_meshes.get(&coord) {
                            let transform =
                                rgis_render::tile_screen_transform(coord, &self.project.viewport);
                            current_draws.push(rgis_render::BasemapTileDraw {
                                coord,
                                mesh: Arc::clone(tile_mesh),
                                offset: transform.offset,
                                scale: transform.scale,
                                width_scale: transform.width_scale,
                                size: transform.size,
                            });
                        } else {
                            if self.pending_tiles.insert(coord) {
                                self.vector_tile_fetcher.request(coord);
                            }
                            if let Some(ancestor) = self.nearest_cached_ancestor(coord) {
                                fallback_coords.insert(ancestor);
                            }
                        }
                    }
                    // Draw already-cached lower-zoom tiles first (scaled up
                    // to cover the same area) so still-loading tiles don't
                    // leave a blank gap; matching current-zoom tiles then
                    // draw on top once they arrive.
                    for coord in fallback_coords {
                        if let Some(tile_mesh) = self.tile_meshes.get(&coord) {
                            let transform =
                                rgis_render::tile_screen_transform(coord, &self.project.viewport);
                            basemap_tiles.push(rgis_render::BasemapTileDraw {
                                coord,
                                mesh: Arc::clone(tile_mesh),
                                offset: transform.offset,
                                scale: transform.scale,
                                width_scale: transform.width_scale,
                                size: transform.size,
                            });
                        }
                    }
                    fallback_tile_count = basemap_tiles.len();
                    basemap_tiles.extend(current_draws);
                    mesh = rgis_render::build_background_mesh(&self.project.viewport, &self.style);
                    background_index_count = mesh.indices.len() as u32;
                }
                let mut raster_tiles = if self.project.show_tiles {
                    self.collect_raster_tile_draws()
                } else {
                    Vec::new()
                };
                let raster_tile_count = raster_tiles.len() as u32;
                mesh.extend(rgis_render::build_scene_mesh(
                    &self.project.layers,
                    &self.project.viewport,
                ));

                // Screen-space label glyph quads must be collected before
                // `basemap_tiles` moves into the paint callback below. Only
                // the true current-zoom tiles (not the lower-zoom
                // `fallback_tile_count` placeholder tiles prepended above)
                // contribute labels: a fallback tile covers its *entire*
                // area even when only some of its children are still
                // loading, so it typically overlaps already-loaded
                // current-zoom tiles too. Extracting labels from it as well
                // would duplicate every already-visible label, at the
                // wrong (overzoomed) size and, since lower-zoom vector
                // tiles can carry less complete attribute data (e.g. a
                // missing `name:en` that falls back to the local-language
                // name), sometimes with different text entirely -- exactly
                // the "ghost" duplicate label symptom this avoids. Missing
                // a label for one frame while its tile finishes loading is
                // far less jarring than a wrong/duplicate one.
                let (label_glyphs, glyph_bitmaps, pending_glyphs, icon_draws) =
                    self.collect_label_draws(&basemap_tiles[fallback_tile_count..], rect);
                raster_tiles.extend(icon_draws);

                let callback = rgis_render::MapCallback {
                    mesh,
                    background_index_count,
                    basemap_tiles,
                    tiles: raster_tiles,
                    raster_tile_count,
                    labels: label_glyphs,
                    glyph_bitmaps,
                    width: rect.width(),
                    height: rect.height(),
                };
                ui.painter()
                    .add(egui_wgpu::Callback::new_paint_callback(rect, callback));
                if pending_glyphs {
                    ui.ctx().request_repaint();
                }

                // Draw the in-progress bbox-zoom selection rectangle, if
                // any, on top of the map.
                if let Some(start) = self.bbox_zoom_start
                    && let Some(current) = response.interact_pointer_pos()
                {
                    let drag_rect = egui::Rect::from_two_pos(start, current);
                    ui.painter().rect_stroke(
                        drag_rect,
                        0.0,
                        egui::Stroke::new(1.5, egui::Color32::WHITE),
                        egui::StrokeKind::Outside,
                    );
                    ui.painter().rect_filled(
                        drag_rect,
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40),
                    );
                }
            });
    }

    /// Projects visible basemap labels into screen space, performs
    /// priority-ordered greedy decluttering, and expands every surviving
    /// string into per-glyph screen quads ready for the map callback's SDF
    /// text pipeline.
    fn collect_label_draws(
        &self,
        basemap_tiles: &[rgis_render::BasemapTileDraw],
        rect: egui::Rect,
    ) -> (
        Vec<LabelGlyphInstance>,
        GlyphBitmapRanges,
        bool,
        Vec<rgis_render::TileDraw>,
    ) {
        struct ProjectedLabel {
            priority: i32,
            pos: egui::Pos2,
            text: String,
            font_size: f32,
            color: [f32; 4],
            halo_color: [f32; 4],
            fontstack: String,
            angle: f32,
            /// Screen-space road polyline for line-placed labels (see
            /// `TileLabel::path`); `None` for point (place/poi) labels.
            path: Option<Vec<egui::Pos2>>,
            icon: Option<String>,
            icon_size: f32,
        }

        // Label positions feed the wgpu `MapCallback` (same as basemap
        // tiles' own `offset`/`scale`), which places everything in
        // rect-*local* pixel space (0,0 at the map panel's own top-left,
        // matching `width`/`height`) -- NOT absolute window space, unlike
        // the old egui-painter-based approach this replaced. So `rect.min`
        // must NOT be added here (that previously caused labels to drift
        // by the sidebar's width and desync from the tiles under
        // zoom/pan). The cull rect is likewise expressed in that same
        // rect-local space, with a little slack beyond the viewport so a
        // label whose anchor point is just off-screen (but whose text
        // would still be partially visible) isn't dropped before the
        // overlap pass even sees it.
        let cull_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, rect.size()).expand(64.0);
        let mut projected = Vec::new();
        for draw in basemap_tiles {
            for label in &draw.mesh.labels {
                let pos = egui::pos2(
                    label.position[0] * draw.scale + draw.offset[0],
                    label.position[1] * draw.scale + draw.offset[1],
                );
                if !cull_rect.contains(pos) {
                    continue;
                }
                projected.push(ProjectedLabel {
                    priority: label.priority,
                    pos,
                    text: label.text.clone(),
                    font_size: label.font_size,
                    color: label.color,
                    halo_color: label.halo_color,
                    fontstack: label.fontstack.clone(),
                    angle: label.angle,
                    path: label.path.as_ref().map(|path| {
                        path.iter()
                            .map(|p| {
                                egui::pos2(
                                    p[0] * draw.scale + draw.offset[0],
                                    p[1] * draw.scale + draw.offset[1],
                                )
                            })
                            .collect()
                    }),
                    icon: label.icon.clone(),
                    icon_size: label.icon_size,
                });
            }
        }
        projected.sort_by_key(|l| l.priority);

        let mut placed: Vec<egui::Rect> = Vec::with_capacity(projected.len());
        let mut glyphs = Vec::new();
        let mut glyph_bitmaps = GlyphBitmapRanges::default();
        let mut pending_glyphs = false;
        let mut icon_draws = Vec::new();

        for label in projected {
            if label.text.is_empty() {
                continue;
            }

            let fontstack = label.fontstack.as_str();
            let mut codepoints = Vec::with_capacity(label.text.chars().count());
            let mut ranges = std::collections::HashMap::new();
            let mut missing = false;
            for ch in label.text.chars() {
                let codepoint = ch as u32;
                codepoints.push(codepoint);
                let range_start = glyph_range_start(codepoint);
                let Some(range) = self.glyph_fetcher.get_cached(fontstack, codepoint) else {
                    self.glyph_fetcher.request(fontstack, codepoint);
                    pending_glyphs = true;
                    missing = true;
                    continue;
                };
                if !range.contains_key(&codepoint) {
                    self.glyph_fetcher.request(fontstack, codepoint);
                    pending_glyphs = true;
                    missing = true;
                    continue;
                }
                ranges.entry(range_start).or_insert(range);
            }
            if missing {
                continue;
            }

            let scale = label.font_size / GLYPH_PIXELS_PER_EM;
            let total_advance = codepoints
                .iter()
                .filter_map(|codepoint| {
                    let range = ranges.get(&glyph_range_start(*codepoint))?;
                    let glyph = range.get(codepoint)?;
                    Some(glyph.advance as f32 * scale)
                })
                .sum::<f32>();
            // Baseline-relative vertical center of this label's actual ink
            // (not a guessed constant): `top` is how far each glyph's ink
            // rises above the baseline and `height - top` is how far it
            // dips below, so the tallest ascender/descender across the
            // string's glyphs gives the true cap-height box to center on
            // the road line, matching how MapLibre centers line-placed
            // text vertically on the line itself rather than hanging it
            // below like point labels.
            let (mut max_ascent, mut max_descent) = (0i32, 0i32);
            for codepoint in &codepoints {
                if let Some(glyph) = ranges
                    .get(&glyph_range_start(*codepoint))
                    .and_then(|range| range.get(codepoint))
                {
                    max_ascent = max_ascent.max(glyph.top);
                    max_descent = max_descent.max(glyph.height as i32 - glyph.top);
                }
            }
            let baseline_offset = (max_ascent - max_descent) as f32 * 0.5 * scale;
            let mut label_glyphs = Vec::with_capacity(codepoints.len());
            let mut bounds: Option<egui::Rect> = None;

            if let Some(path) = &label.path {
                // Road label: thread glyphs along the actual line geometry
                // (like MapLibre's `symbol-placement: line`) instead of one
                // flat quad, so text curves with bends in the road.
                let Some(total_len) = path_length(path) else {
                    continue;
                };
                let mid_len = total_len * 0.5;
                // Sample the tangent near the anchor to decide whether
                // walking the path forward would lay the text out upside
                // down; if so, walk it in the other direction instead so
                // the label always reads left-to-right, upright.
                let (_, probe_angle) = point_and_angle_at(path, mid_len);
                let forward = probe_angle.cos() >= 0.0;
                let mut pen_len = mid_len - total_advance * 0.5;
                for codepoint in codepoints {
                    let range_start = glyph_range_start(codepoint);
                    let Some(range) = ranges.get(&range_start) else {
                        continue;
                    };
                    let Some(glyph) = range.get(&codepoint) else {
                        continue;
                    };
                    let sample_len = if forward {
                        pen_len
                    } else {
                        total_len - pen_len
                    };
                    let (anchor, mut angle) = point_and_angle_at(path, sample_len);
                    if !forward {
                        angle += std::f32::consts::PI;
                    }
                    let x = anchor.x + (glyph.left - GLYPH_BUFFER as i32) as f32 * scale;
                    let y = anchor.y + baseline_offset
                        - (glyph.top + GLYPH_BUFFER as i32) as f32 * scale;
                    let w = (glyph.width + 2 * GLYPH_BUFFER) as f32 * scale;
                    let h = (glyph.height + 2 * GLYPH_BUFFER) as f32 * scale;
                    let glyph_rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h));
                    bounds = Some(match bounds {
                        Some(existing) => existing.union(glyph_rect),
                        None => glyph_rect,
                    });
                    label_glyphs.push(LabelGlyphInstance {
                        rect: [x, y, w, h],
                        anchor: [anchor.x, anchor.y],
                        angle,
                        fontstack: fontstack.to_string(),
                        codepoint,
                        color: label.color,
                        halo_color: label.halo_color,
                    });
                    glyph_bitmaps
                        .entry((fontstack.to_string(), range_start))
                        .or_insert_with(|| Arc::clone(range));
                    pen_len += glyph.advance as f32 * scale;
                }
            } else {
                let baseline_y = label.pos.y + label.font_size * 0.35;
                let mut pen_x = label.pos.x - total_advance * 0.5;

                for codepoint in codepoints {
                    let range_start = glyph_range_start(codepoint);
                    let Some(range) = ranges.get(&range_start) else {
                        continue;
                    };
                    let Some(glyph) = range.get(&codepoint) else {
                        continue;
                    };

                    // `left`/`top` describe the ink box, while the packed
                    // atlas rect includes the standard SDF padding around
                    // it, so shift the screen quad by that buffer to keep
                    // the ink aligned with the layout metrics.
                    let x = pen_x + (glyph.left - GLYPH_BUFFER as i32) as f32 * scale;
                    let y = baseline_y - (glyph.top + GLYPH_BUFFER as i32) as f32 * scale;
                    let w = (glyph.width + 2 * GLYPH_BUFFER) as f32 * scale;
                    let h = (glyph.height + 2 * GLYPH_BUFFER) as f32 * scale;
                    let glyph_rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h));
                    bounds = Some(match bounds {
                        Some(existing) => existing.union(glyph_rect),
                        None => glyph_rect,
                    });
                    label_glyphs.push(LabelGlyphInstance {
                        rect: [x, y, w, h],
                        anchor: [label.pos.x, label.pos.y],
                        angle: label.angle,
                        fontstack: fontstack.to_string(),
                        codepoint,
                        color: label.color,
                        halo_color: label.halo_color,
                    });
                    glyph_bitmaps
                        .entry((fontstack.to_string(), range_start))
                        .or_insert_with(|| Arc::clone(range));
                    pen_x += glyph.advance as f32 * scale;
                }
            }

            // Real sprite icon quad (see `TileLabel::icon`), positioned
            // centered on the label's anchor point exactly like the text
            // label sitting alongside it. Included in `bounds` before the
            // overlap check below so an icon can't survive decluttering
            // while its paired text (or vice versa) doesn't.
            let icon_draw = label.icon.as_deref().and_then(|icon_name| {
                let atlas = self.sprite_atlas.as_ref()?;
                let sprite_rect = atlas.rects.get(icon_name)?;
                let (atlas_w, atlas_h) = atlas.image.dimensions();
                if atlas_w == 0 || atlas_h == 0 {
                    return None;
                }
                let w = sprite_rect.width as f32 * label.icon_size;
                let h = sprite_rect.height as f32 * label.icon_size;
                let x = label.pos.x - w * 0.5;
                let y = label.pos.y - h * 0.5;
                let icon_rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h));
                bounds = Some(match bounds {
                    Some(existing) => existing.union(icon_rect),
                    None => icon_rect,
                });
                Some(rgis_render::TileDraw {
                    key: SPRITE_ATLAS_TILE_KEY,
                    rect: [x, y, w, h],
                    rgba: Arc::clone(&atlas.image),
                    uv_rect: [
                        sprite_rect.x as f32 / atlas_w as f32,
                        sprite_rect.y as f32 / atlas_h as f32,
                        (sprite_rect.x + sprite_rect.width) as f32 / atlas_w as f32,
                        (sprite_rect.y + sprite_rect.height) as f32 / atlas_h as f32,
                    ],
                    opacity: 1.0,
                })
            });

            let Some(label_rect) = bounds.map(|rect| rect.expand2(egui::vec2(2.0, 2.0))) else {
                continue;
            };
            if placed
                .iter()
                .any(|placed_rect| placed_rect.intersects(label_rect))
            {
                continue;
            }
            placed.push(label_rect);
            glyphs.extend(label_glyphs);
            icon_draws.extend(icon_draw);
        }

        (glyphs, glyph_bitmaps, pending_glyphs, icon_draws)
    }
}

/// A stable GPU texture-cache key for a raster tile, distinguishing tiles
/// with the same `(z, x, y)` from different raster sources (unlikely to
/// collide in practice, but cheap to guard against).
fn tile_draw_key(source_id: &str, coord: TileCoord) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    source_id.hash(&mut hasher);
    coord.z.hash(&mut hasher);
    coord.x.hash(&mut hasher);
    coord.y.hash(&mut hasher);
    hasher.finish()
}

/// Builds a [`TileFetcher`] for every `"type": "raster"` source referenced
/// by a `raster` layer in `style` (e.g. `natural_earth` in the liberty
/// style), keyed by source id -- see `RgisApp::style`/`drain_ready_tiles`.
/// Sources with no `tiles` template (only a TileJSON `url`, unsupported
/// here) are silently skipped.
fn raster_fetchers_for_style(
    style: &rgis_render::StyleSheet,
) -> std::collections::HashMap<String, Arc<TileFetcher>> {
    let mut fetchers = std::collections::HashMap::new();
    for layer in style.layers_of_kind("raster") {
        let Some(source_id) = &layer.source else {
            continue;
        };
        if fetchers.contains_key(source_id) {
            continue;
        }
        let Some(source) = style.sources.get(source_id) else {
            continue;
        };
        let Some(template) = source.tiles.as_ref().and_then(|t| t.first()) else {
            continue;
        };
        let max_zoom = source.maxzoom.unwrap_or(22.0) as u8;
        let tile_size = source.tile_size.unwrap_or(256);
        let raster_source = StyleRasterSource::new(template.clone(), max_zoom, tile_size);
        fetchers.insert(source_id.clone(), Arc::new(TileFetcher::new(raster_source)));
    }
    fetchers
}

/// Total length of a polyline, or `None` for degenerate (single-point/
/// zero-length) paths that can't host line-following text.
fn path_length(path: &[egui::Pos2]) -> Option<f32> {
    let len: f32 = path.windows(2).map(|pair| pair[0].distance(pair[1])).sum();
    (len > f32::EPSILON).then_some(len)
}

/// Walks `path` by cumulative arc length and returns `(point, tangent
/// angle)` at `target_len` (clamped to the path's endpoints), used to
/// thread road-name glyphs along the actual line geometry.
fn point_and_angle_at(path: &[egui::Pos2], target_len: f32) -> (egui::Pos2, f32) {
    let mut walked = 0.0f32;
    for pair in path.windows(2) {
        let (p0, p1) = (pair[0], pair[1]);
        let seg_len = p0.distance(p1);
        if walked + seg_len >= target_len || seg_len <= f32::EPSILON {
            let t = if seg_len > f32::EPSILON {
                ((target_len - walked) / seg_len).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let point = p0 + (p1 - p0) * t;
            let angle = (p1.y - p0.y).atan2(p1.x - p0.x);
            return (point, angle);
        }
        walked += seg_len;
    }
    let n = path.len();
    if n >= 2 {
        let (p0, p1) = (path[n - 2], path[n - 1]);
        (*path.last().unwrap(), (p1.y - p0.y).atan2(p1.x - p0.x))
    } else {
        (*path.last().unwrap(), 0.0)
    }
}

impl eframe::App for RgisApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_pending_loads();
        self.drain_ready_tiles();
        let ready_glyphs = self.drain_ready_glyphs();

        #[cfg(target_arch = "wasm32")]
        if let Some((lon, lat, zoom)) = DEBUG_VIEWPORT_JUMP.take() {
            self.project.viewport.center = rgis_core::lonlat_to_mercator(lon, lat);
            self.project.viewport.zoom = zoom;
        }

        // Bottom panel must be added before the side panel so it spans the
        // full window width instead of just the area right of the sidebar.
        self.render_status_bar(ui);
        self.render_sidebar(ui);
        self.render_map(ui);

        if ready_glyphs || !self.pending_loads.is_empty() || !self.pending_tiles.is_empty() {
            ui.ctx().request_repaint();
        }
    }
}

/// Current byte size of the main thread's own wasm linear memory -- exposed
/// via `debug_wasm_memory_bytes`/`window.debugMemBytes()` for manual memory
/// profiling in the browser (see `rgis-render`'s `tile_mesh_byte_budget`
/// test for the offline, automatable equivalent).
#[cfg(target_arch = "wasm32")]
fn wasm_memory_bytes() -> u32 {
    use wasm_bindgen::JsCast;
    let memory: js_sys::WebAssembly::Memory = wasm_bindgen::memory().unchecked_into();
    let buffer: js_sys::ArrayBuffer = memory.buffer().unchecked_into();
    buffer.byte_length()
}

fn tree_row(ui: &mut egui::Ui, checked: &mut bool, label: &str, removable: bool) -> (bool, bool) {
    let row_height = ui.spacing().interact_size.y + ROW_VPAD * 2.0;
    let rect = egui::Rect::from_min_size(
        ui.cursor().min,
        egui::vec2(ui.available_width(), row_height),
    );
    // A passive geometric check (not a registered widget) so it doesn't
    // compete with the remove-button's own click hit-testing.
    let hovered = ui.rect_contains_pointer(rect);
    if hovered {
        ui.painter().rect_filled(rect, 2.0, ROW_HOVER_FILL);
    }

    let mut toggled = false;
    let mut remove_clicked = false;

    ui.horizontal(|ui| {
        // Force the row to its full padded height so content is vertically
        // centered within it (plain `ui.horizontal` only assumes
        // `interact_size.y` and top-aligns any extra space).
        ui.set_min_height(row_height);
        ui.add_space(ui.spacing().indent);
        if ui.checkbox(checked, label).changed() {
            toggled = true;
        }
        if removable {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(4.0);
                if hovered && icon_button(ui, "✕", "Remove layer").clicked() {
                    remove_clicked = true;
                }
            });
        }
    });

    (toggled, remove_clicked)
}

/// A small square icon button with the glyph painted centered in its rect
/// (avoids off-center glyphs from `Button`'s frame/padding), with a tooltip.
fn icon_button(ui: &mut egui::Ui, glyph: &str, tooltip: &str) -> egui::Response {
    let size = ui.spacing().icon_width.max(14.0);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&response);
        if response.hovered() {
            ui.painter().rect_filled(rect, 3.0, visuals.weak_bg_fill);
        }
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            glyph,
            egui::TextStyle::Small.resolve(ui.style()),
            visuals.text_color(),
        );
    }
    response.on_hover_text(tooltip)
}
