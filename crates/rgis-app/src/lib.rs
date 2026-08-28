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
use rgis_render::{SceneMesh, TileMesh};
use rgis_tiles::{OPENFREEMAP_MAX_ZOOM, TileCoord, VectorTileFetcher, visible_tiles_for_zoom};

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

/// Max newly-arrived tiles handed off to tessellation per `drain_ready_tiles`
/// call (once per frame) -- see the comment on its call site for why this
/// matters most on wasm, where tessellation isn't actually backgrounded.
const MAX_TESSELLATIONS_PER_FRAME: usize = 3;

/// Extra vertical padding added above/below the sidebar tree row content.
const ROW_VPAD: f32 = 5.0;
/// Background fill painted behind a hovered sidebar tree row.
const ROW_HOVER_FILL: egui::Color32 = egui::Color32::from_gray(40);

/// One or more layers finished (or failed) loading, each tagged with a
/// display name for error messages.
type LoadResults = Vec<(String, Result<LoadedLayer, IoError>)>;

pub struct RgisApp {
    project: Project,
    vector_tile_fetcher: Arc<VectorTileFetcher>,
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

        Self {
            project: Project::default(),
            vector_tile_fetcher: VectorTileFetcher::new_openfreemap(),
            tile_meshes: LruCache::new(NonZeroUsize::new(TILE_MESH_CACHE_SIZE).unwrap()),
            pending_tiles: std::collections::HashSet::new(),
            pending_tile_meshes: Vec::new(),
            #[cfg(target_arch = "wasm32")]
            tile_worker_pool: Rc::new(tile_worker_pool::TileWorkerPool::new()),
            pending_loads: Vec::new(),
            cursor_lonlat: None,
            last_error: None,
            layers_expanded: true,
            gpu_backend_label,
        }
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
                #[cfg(not(target_arch = "wasm32"))]
                let promise = Promise::spawn_thread("tessellate-tile", move || {
                    (coord, Some(rgis_render::build_tile_mesh(&tile, coord)))
                });
                #[cfg(target_arch = "wasm32")]
                let promise = Promise::spawn_local(async move {
                    (coord, Some(rgis_render::build_tile_mesh(&tile, coord)))
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
                    Promise::spawn_thread("decode-tessellate-tile", move || {
                        match fetcher.decode_and_cache(coord, &bytes) {
                            Ok(tile) => (coord, Some(rgis_render::build_tile_mesh(&tile, coord))),
                            Err(_) => (coord, None),
                        }
                    })
                };
                #[cfg(target_arch = "wasm32")]
                let promise = {
                    let pool = Rc::clone(&self.tile_worker_pool);
                    Promise::spawn_local(async move {
                        let mesh = pool.tessellate(coord, bytes).await;
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

    fn render_sidebar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("sidebar")
            .default_size(280.0)
            .show(ui, |ui| {
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
            });
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

                if response.dragged() {
                    let delta = response.drag_delta();
                    self.project.viewport.pan(delta.x, delta.y);
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

                self.cursor_lonlat = response.hover_pos().map(|pos| {
                    let local = pos - rect.min;
                    let world = self.project.viewport.screen_to_world([local.x, local.y]);
                    mercator_to_lonlat(world.x, world.y)
                });

                let mut mesh = SceneMesh::default();
                let mut background_index_count = 0;
                let mut basemap_tiles = Vec::new();
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
                    basemap_tiles.extend(current_draws);
                    mesh = rgis_render::build_background_mesh(&self.project.viewport);
                    background_index_count = mesh.indices.len() as u32;
                }
                mesh.extend(rgis_render::build_scene_mesh(
                    &self.project.layers,
                    &self.project.viewport,
                ));

                let callback = rgis_render::MapCallback {
                    mesh,
                    background_index_count,
                    basemap_tiles,
                    tiles: Vec::new(),
                    width: rect.width(),
                    height: rect.height(),
                };
                ui.painter()
                    .add(egui_wgpu::Callback::new_paint_callback(rect, callback));
            });
    }
}

impl eframe::App for RgisApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_pending_loads();
        self.drain_ready_tiles();

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

        if !self.pending_loads.is_empty() || !self.pending_tiles.is_empty() {
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
