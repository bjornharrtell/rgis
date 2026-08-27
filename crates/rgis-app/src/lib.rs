//! Shared `eframe::App` implementation for `rgis`, used by both the native
//! binary ([`crate`] via `main.rs`) and the browser build (`rgis-web`, via
//! `eframe::WebRunner`).

use std::collections::HashMap;
use std::sync::Arc;

use poll_promise::Promise;
use rgis_core::{Layer, LayerId, Project, mercator_to_lonlat};
use rgis_io::{IoError, LoadedLayer};
use rgis_tiles::{OsmTileSource, TileCoord, TileFetcher, tile_screen_rect, visible_tiles};

mod status_bar;

/// One or more layers finished (or failed) loading, each tagged with a
/// display name for error messages.
type LoadResults = Vec<(String, Result<LoadedLayer, IoError>)>;

pub struct RgisApp {
    project: Project,
    tile_fetcher: Arc<TileFetcher>,
    tile_images: HashMap<TileCoord, Arc<image::RgbaImage>>,
    pending_tiles: std::collections::HashSet<TileCoord>,
    pending_loads: Vec<Promise<LoadResults>>,
    cursor_lonlat: Option<(f64, f64)>,
    last_error: Option<String>,
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

        Self {
            project: Project::default(),
            tile_fetcher: Arc::new(TileFetcher::new(OsmTileSource)),
            tile_images: HashMap::new(),
            pending_tiles: std::collections::HashSet::new(),
            pending_loads: Vec::new(),
            cursor_lonlat: None,
            last_error: None,
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
        while let Ok(ready) = self.tile_fetcher.receiver.try_recv() {
            self.pending_tiles.remove(&ready.coord);
            self.tile_images.insert(ready.coord, ready.image);
        }
    }

    fn render_header(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("header").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("rgis");
                ui.label(format!("{} layers", self.project.layers.len()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Add Layer…").clicked() {
                        self.queue_pick_files();
                    }
                });
            });
        });
    }

    fn render_sidebar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("sidebar")
            .default_size(280.0)
            .show(ui, |ui| {
                ui.label(egui::RichText::new("LAYERS").weak().small());
                ui.separator();

                let mut to_toggle: Option<LayerId> = None;
                let mut to_remove: Option<LayerId> = None;

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for layer in self.project.layers.iter().rev() {
                        ui.horizontal(|ui| {
                            let mut visible = layer.visible;
                            if ui.checkbox(&mut visible, &layer.name).changed() {
                                to_toggle = Some(layer.id);
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("✕").clicked() {
                                        to_remove = Some(layer.id);
                                    }
                                },
                            );
                        });
                    }

                    ui.separator();
                    let mut show_tiles = self.project.show_tiles;
                    if ui.checkbox(&mut show_tiles, "OSM Background").changed() {
                        self.project.show_tiles = show_tiles;
                    }
                });

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
            ui.horizontal(|ui| {
                let coords = self
                    .cursor_lonlat
                    .map(|(lon, lat)| status_bar::format_coordinates(lon, lat))
                    .unwrap_or_else(|| "—".to_string());
                ui.label(coords);
                ui.separator();
                ui.label(status_bar::format_scale(self.project.viewport.resolution()));
                ui.separator();
                ui.label("EPSG:4326");
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
                        let zoom_delta = if scroll > 0.0 { 0.25 } else { -0.25 };
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

                let mesh =
                    rgis_render::build_scene_mesh(&self.project.layers, &self.project.viewport);

                let mut tiles = Vec::new();
                if self.project.show_tiles {
                    for coord in visible_tiles(&self.project.viewport, &OsmTileSource) {
                        if let Some(image) = self.tile_images.get(&coord) {
                            let rect = tile_screen_rect(coord, &self.project.viewport);
                            tiles.push(rgis_render::TileDraw {
                                key: tile_cache_key(coord),
                                rect,
                                rgba: Arc::clone(image),
                            });
                        } else if self.pending_tiles.insert(coord) {
                            self.tile_fetcher.request(coord);
                        }
                    }
                }

                let callback = rgis_render::MapCallback {
                    mesh,
                    tiles,
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

        self.render_header(ui);
        self.render_sidebar(ui);
        self.render_status_bar(ui);
        self.render_map(ui);

        if !self.pending_loads.is_empty() || !self.pending_tiles.is_empty() {
            ui.ctx().request_repaint();
        }
    }
}

fn tile_cache_key(coord: TileCoord) -> u64 {
    ((coord.z as u64) << 56) | ((coord.x as u64) << 28) | (coord.y as u64)
}
