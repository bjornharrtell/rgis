#![allow(dead_code)]

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::{fs, path::PathBuf};

use gpui::{
    App, Bounds, Context, CursorStyle, DevicePixels, MouseButton, Render, ResizeEdge, Window,
    WindowBounds, WindowDecorations, WindowOptions, div, prelude::*, px, rgb, rgba, size, svg,
};
use gpui_elements::editable_text::actions::{DEFAULT_INPUT_CONTEXT, default_bindings};
use gpui_platform::application;
use gpui_wgpu::{WgpuContextHandle, WgpuRenderTarget};
use lru::LruCache;
use poll_promise::Promise;
use rgis_app::labels;
use rgis_app::raster;
use rgis_app::ui::{self, LayerUi, LayerUiState, StyleColorTarget};
use rgis_core::{
    Bounds as GeoBounds, Color, Layer, LayerId, Project, ProjectState, mercator_to_lonlat,
};
use rgis_render::{MapCallback, MapRenderResources, SceneMesh};
use rgis_tiles::{OPENFREEMAP_MAX_ZOOM, TileCoord, VectorTileFetcher, visible_tiles_for_zoom};

const SIDEBAR_WIDTH: f32 = 280.0;
const TITLEBAR_HEIGHT: f32 = 32.0;
const STATUS_HEIGHT: f32 = 28.0;
const TILE_CACHE_SIZE: usize = 256;
const ZED_CANVAS: u32 = 0x121212;
const ZED_PANEL: u32 = 0x1b1b1b;
const ZED_TITLEBAR: u32 = 0x202020;
const ZED_SURFACE: u32 = 0x2a2a2a;
const ZED_BORDER: u32 = 0x343434;
const ZED_TEXT: u32 = 0xd4d4d4;
const ZED_MUTED: u32 = 0x929292;
const ZED_ACCENT: u32 = 0x8ab4f8;
const DEFAULT_STYLE_JSON: &str = include_str!("../../rgis-style/fixtures/liberty.json");

type LoadResults = Vec<(PathBuf, Result<rgis_io::LoadedLayer, rgis_io::IoError>)>;

struct LoadedProject {
    state: ProjectState,
    layers: Vec<(LayerId, PathBuf, Vec<rgis_core::Feature>)>,
    path: PathBuf,
}

struct PendingVectorRender {
    layer_key: u64,
    viewport: rgis_core::Viewport,
    generation: u64,
    promise: Promise<Option<image::RgbaImage>>,
}

fn prepare_loaded_layer(
    loaded: rgis_io::LoadedLayer,
    file_name: &str,
) -> Result<Option<rgis_io::LoadedLayer>, rgis_io::IoError> {
    if let Some(epsg) = loaded.epsg {
        let result = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Reproject layer?")
            .set_description(format!(
                "\"{file_name}\" declares EPSG:{epsg}. Rgis renders layers in \
                 Web Mercator (EPSG:3857). Transform this layer on the fly?"
            ))
            .set_buttons(rfd::MessageButtons::YesNo)
            .show();
        if result != rfd::MessageDialogResult::Yes {
            return Ok(None);
        }
    }
    loaded.into_web_mercator().map(Some)
}

fn load_project_file(path: PathBuf) -> Result<LoadedProject, String> {
    let yaml = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let state = ProjectState::from_yaml(&yaml).map_err(|error| {
        format!(
            "failed to parse project {}: {error}",
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("project")
        )
    })?;
    let base = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut layers = Vec::new();
    for layer in &state.layers {
        let Some(source) = &layer.source_path else {
            continue;
        };
        let source_path = if source.is_absolute() {
            source.clone()
        } else {
            base.join(source)
        };
        let loaded = rgis_io::load_path(&source_path).map_err(|error| {
            format!(
                "failed to load layer {} from {}: {error}",
                layer.name,
                source_path.display()
            )
        })?;
        layers.push((layer.id, source_path, loaded.features));
    }
    Ok(LoadedProject {
        state,
        layers,
        path,
    })
}

fn icon(path: &str, color: u32) -> impl IntoElement {
    let data = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
<path d="{path}" fill="none" stroke="#{color:06x}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"##
    );
    svg()
        .data(data.as_bytes())
        .size(px(14.0))
        .text_color(rgb(color))
        .flex_none()
}

fn color_to_rgba(color: Color) -> u32 {
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u32;
    (channel(color.r) << 24) | (channel(color.g) << 16) | (channel(color.b) << 8) | channel(color.a)
}

fn format_color(color: Color) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.b.clamp(0.0, 1.0) * 255.0).round() as u8
    )
}

fn resize_handle<T: 'static>(
    edge: ResizeEdge,
    cursor: CursorStyle,
    cx: &mut Context<T>,
) -> gpui::Div {
    div().cursor(cursor).on_mouse_down(
        MouseButton::Left,
        cx.listener(move |_, _, window, cx| {
            window.start_window_resize(edge);
            cx.stop_propagation();
        }),
    )
}

struct MapTarget {
    target: WgpuRenderTarget,
    msaa_view: Option<wgpu::TextureView>,
    size: (u32, u32),
}

impl MapTarget {
    fn new(context: &WgpuContextHandle, dimensions: (u32, u32)) -> Self {
        let target = WgpuRenderTarget::new(
            context,
            size(
                DevicePixels(dimensions.0 as i32),
                DevicePixels(dimensions.1 as i32),
            ),
        );
        let msaa_view = (rgis_render::MSAA_SAMPLES > 1).then(|| {
            context
                .device()
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("rgis-map-msaa"),
                    size: wgpu::Extent3d {
                        width: dimensions.0,
                        height: dimensions.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: rgis_render::MSAA_SAMPLES,
                    dimension: wgpu::TextureDimension::D2,
                    format: target.format(),
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        Self {
            target,
            msaa_view,
            size: dimensions,
        }
    }
}

pub struct RgisNativeApp {
    project: Project,
    style: Arc<rgis_render::StyleSheet>,
    vector_tile_fetcher: Arc<VectorTileFetcher>,
    glyph_fetcher: Arc<rgis_tiles::GlyphFetcher>,
    sprite_fetcher: Option<Arc<rgis_tiles::SpriteFetcher>>,
    sprite_atlas: Option<Arc<rgis_tiles::SpriteAtlas>>,
    raster_fetchers: raster::RasterFetchers,
    raster_tile_cache: raster::RasterTileCaches,
    tile_meshes: LruCache<TileCoord, Arc<rgis_render::TileMesh>>,
    pending_tiles: std::collections::HashSet<TileCoord>,
    pending_tile_meshes: Vec<Promise<(TileCoord, Option<rgis_render::TileMesh>)>>,
    vector_render_cache: rgis_render::VectorRenderCache,
    vector_render_generation: Arc<AtomicU64>,
    pending_vector_render: Option<PendingVectorRender>,
    resources: Option<MapRenderResources>,
    target: Option<MapTarget>,
    pending_loads: Vec<Promise<LoadResults>>,
    pending_project_loads: Vec<Promise<Result<Option<LoadedProject>, String>>>,
    pending_project_saves: Vec<Promise<Result<Option<PathBuf>, String>>>,
    status: String,
    last_cursor: Option<gpui::Point<gpui::Pixels>>,
    cursor_lonlat: Option<(f64, f64)>,
    bbox_zoom_start: Option<gpui::Point<gpui::Pixels>>,
    dragging: bool,
    sidebar_visible: bool,
    layer_ui_state: LayerUiState,
}

impl RgisNativeApp {
    fn new(startup_paths: &[std::path::PathBuf]) -> Self {
        let style = rgis_render::StyleSheet::parse(DEFAULT_STYLE_JSON)
            .expect("bundled default style JSON should parse");
        let mut project = Project::default();
        if startup_paths.is_empty() {
            let bytes = include_bytes!("../assets/sample.geojson");
            if let Ok(loaded) = rgis_io::load_bytes_with_crs("sample.geojson", bytes)
                .and_then(|loaded| loaded.into_web_mercator())
            {
                let id = project.next_layer_id();
                let layer = Layer::new(id, loaded.name, loaded.features);
                if let Some(bounds) = layer.bounds {
                    project.add_layer(layer);
                    project.viewport.fit_bounds(&bounds);
                } else {
                    project.add_layer(layer);
                }
            }
        } else {
            for path in startup_paths {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("layer")
                    .to_string();
                match rgis_io::load_path_with_crs(path) {
                    Ok(loaded) => match prepare_loaded_layer(loaded, &name) {
                        Ok(Some(loaded)) => {
                            let id = project.next_layer_id();
                            let mut layer = Layer::new(id, loaded.name, loaded.features);
                            layer.source_path = Some(path.clone());
                            if let Some(bounds) = layer.bounds {
                                project.add_layer(layer);
                                project.viewport.fit_bounds(&bounds);
                            } else {
                                project.add_layer(layer);
                            }
                        }
                        Ok(None) => eprintln!("Skipped {name}: reprojection declined"),
                        Err(error) => eprintln!("Failed to load {name}: {error}"),
                    },
                    Err(error) => eprintln!("Failed to load {name}: {error}"),
                }
            }
        }
        let style = Arc::new(style);
        let sprite_fetcher = style.sprite.as_deref().map(rgis_tiles::SpriteFetcher::new);
        let raster_fetchers = raster::fetchers_for_style(&style);
        Self {
            project,
            style,
            vector_tile_fetcher: VectorTileFetcher::new_openfreemap(),
            glyph_fetcher: rgis_tiles::GlyphFetcher::new(),
            sprite_fetcher,
            sprite_atlas: None,
            raster_fetchers,
            raster_tile_cache: raster::RasterTileCaches::new(),
            tile_meshes: LruCache::new(std::num::NonZeroUsize::new(TILE_CACHE_SIZE).unwrap()),
            pending_tiles: std::collections::HashSet::new(),
            pending_tile_meshes: Vec::new(),
            vector_render_cache: rgis_render::VectorRenderCache::default(),
            vector_render_generation: Arc::new(AtomicU64::new(0)),
            pending_vector_render: None,
            resources: None,
            target: None,
            pending_loads: Vec::new(),
            pending_project_loads: Vec::new(),
            pending_project_saves: Vec::new(),
            status: "GPUI native renderer".to_string(),
            last_cursor: None,
            cursor_lonlat: None,
            bbox_zoom_start: None,
            dragging: false,
            sidebar_visible: true,
            layer_ui_state: LayerUiState::default(),
        }
    }

    fn queue_pick_files(&mut self, window: &Window) {
        self.pending_loads
            .push(Promise::spawn_thread("pick-and-load-layers", move || {
                let Some(paths) = rfd::FileDialog::new()
                    .add_filter("GeoJSON", &["geojson", "json"])
                    .add_filter("Shapefile", &["shp"])
                    .add_filter("FlatGeobuf", &["fgb"])
                    .pick_files()
                else {
                    return Vec::new();
                };
                paths
                    .into_iter()
                    .map(|path| {
                        let result = rgis_io::load_path_with_crs(&path);
                        (path, result)
                    })
                    .collect()
            }));
        window.on_next_frame(|_, cx| cx.refresh_windows());
    }

    fn queue_open_project(&mut self, window: &Window) {
        if !self.pending_project_loads.is_empty() {
            self.status = "A project is already opening".to_string();
            return;
        }
        self.status = "Opening project...".to_string();
        self.pending_project_loads
            .push(Promise::spawn_thread("pick-and-load-project", || {
                let Some(path) = rfd::FileDialog::new()
                    .add_filter("Rgis project", &["rgis", "yaml", "yml"])
                    .pick_file()
                else {
                    return Ok(None);
                };
                load_project_file(path).map(Some)
            }));
        window.on_next_frame(|_, cx| cx.refresh_windows());
    }

    fn queue_save_project(&mut self, window: &Window) {
        if !self.pending_project_saves.is_empty() {
            self.status = "A project is already saving".to_string();
            return;
        }
        let yaml = match self.project.to_yaml() {
            Ok(yaml) => yaml,
            Err(error) => {
                self.status = format!("Failed to serialize project: {error}");
                return;
            }
        };
        self.status = "Saving project...".to_string();
        self.pending_project_saves.push(Promise::spawn_thread(
            "pick-and-save-project",
            move || {
                let Some(path) = rfd::FileDialog::new()
                    .add_filter("Rgis project", &["rgis", "yaml", "yml"])
                    .set_file_name("project.rgis")
                    .save_file()
                else {
                    return Ok(None);
                };
                fs::write(&path, yaml)
                    .map(|()| Some(path.clone()))
                    .map_err(|error| format!("failed to write {}: {error}", path.display()))
            },
        ));
        window.on_next_frame(|_, cx| cx.refresh_windows());
    }

    fn poll_project_io(&mut self, window: &Window) {
        let pending_loads = std::mem::take(&mut self.pending_project_loads);
        for promise in pending_loads {
            match promise.try_take() {
                Ok(Ok(Some(loaded))) => {
                    let missing_sources = loaded
                        .state
                        .layers
                        .iter()
                        .filter(|layer| layer.source_path.is_none())
                        .count();
                    let dimensions = (
                        self.project.viewport.width_px,
                        self.project.viewport.height_px,
                    );
                    match Project::from_state(loaded.state) {
                        Ok(mut project) => {
                            project.viewport.width_px = dimensions.0;
                            project.viewport.height_px = dimensions.1;
                            for (layer_id, source_path, features) in loaded.layers {
                                if let Some(layer) = project.get_layer_mut(layer_id) {
                                    layer.set_features(features);
                                    layer.source_path = Some(source_path);
                                }
                            }
                            self.project = project;
                            self.cancel_vector_render();
                            let name = loaded
                                .path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("project");
                            self.status = if missing_sources == 0 {
                                format!("Opened project {name}")
                            } else {
                                format!(
                                    "Opened project {name}; {missing_sources} layer(s) have no source file"
                                )
                            };
                        }
                        Err(error) => {
                            self.status = format!("Failed to open project: {error}");
                        }
                    }
                }
                Ok(Ok(None)) => {
                    self.status = "Open project cancelled".to_string();
                }
                Ok(Err(error)) => {
                    self.status = format!("Failed to open project: {error}");
                }
                Err(promise) => self.pending_project_loads.push(promise),
            }
        }

        let pending_saves = std::mem::take(&mut self.pending_project_saves);
        for promise in pending_saves {
            match promise.try_take() {
                Ok(Ok(Some(path))) => {
                    let name = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("project");
                    self.status = format!("Saved project {name}");
                }
                Ok(Ok(None)) => {
                    self.status = "Save project cancelled".to_string();
                }
                Ok(Err(error)) => {
                    self.status = format!("Failed to save project: {error}");
                }
                Err(promise) => self.pending_project_saves.push(promise),
            }
        }

        if !self.pending_project_loads.is_empty() || !self.pending_project_saves.is_empty() {
            window.on_next_frame(|_, cx| cx.refresh_windows());
        }
    }

    fn sidebar_offset(&self) -> f32 {
        if self.sidebar_visible {
            SIDEBAR_WIDTH
        } else {
            0.0
        }
    }

    fn poll_pending_loads(&mut self, window: &Window) {
        let pending = std::mem::take(&mut self.pending_loads);
        for promise in pending {
            match promise.try_take() {
                Ok(results) => {
                    for (path, result) in results {
                        let name = path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("layer")
                            .to_string();
                        match result {
                            Ok(loaded) => match prepare_loaded_layer(loaded, &name) {
                                Ok(Some(loaded)) => {
                                    let id = self.project.next_layer_id();
                                    let mut layer = Layer::new(id, loaded.name, loaded.features);
                                    layer.source_path = Some(path);
                                    if self.project.layers.is_empty()
                                        && let Some(bounds) = layer.bounds
                                    {
                                        self.project.viewport.fit_bounds(&bounds);
                                    }
                                    self.project.add_layer(layer);
                                    self.status = format!("Loaded {name}");
                                }
                                Ok(None) => {
                                    self.status = format!("Skipped {name}: reprojection declined");
                                }
                                Err(error) => {
                                    self.status = format!("Failed to load {name}: {error}");
                                }
                            },
                            Err(error) => {
                                self.status = format!("Failed to load {name}: {error}");
                            }
                        }
                    }
                }
                Err(promise) => self.pending_loads.push(promise),
            }
        }
        if !self.pending_loads.is_empty() {
            window.on_next_frame(|_, cx| cx.refresh_windows());
        }
    }

    fn drain_tiles(&mut self) {
        raster::drain_fetchers(&self.raster_fetchers, &mut self.raster_tile_cache);
        while let Ok(ready) = self.vector_tile_fetcher.receiver.try_recv() {
            let coord = ready.coord;
            let style = Arc::clone(&self.style);
            self.pending_tile_meshes.push(Promise::spawn_thread(
                "rgis-tessellate-tile",
                move || {
                    (
                        coord,
                        Some(rgis_render::build_tile_mesh(&ready.tile, coord, &style)),
                    )
                },
            ));
        }
        while let Ok(fetched) = self.vector_tile_fetcher.raw_receiver.try_recv() {
            let coord = fetched.coord;
            let bytes = fetched.bytes;
            let fetcher = Arc::clone(&self.vector_tile_fetcher);
            let style = Arc::clone(&self.style);
            self.pending_tile_meshes
                .push(Promise::spawn_thread("rgis-decode-tile", move || {
                    let mesh = fetcher
                        .decode_and_cache(coord, &bytes)
                        .ok()
                        .map(|tile| rgis_render::build_tile_mesh(&tile, coord, &style));
                    (coord, mesh)
                }));
        }
        let pending = std::mem::take(&mut self.pending_tile_meshes);
        for promise in pending {
            match promise.try_take() {
                Ok((coord, Some(mesh))) => {
                    self.pending_tiles.remove(&coord);
                    self.tile_meshes.put(coord, Arc::new(mesh));
                }
                Ok((coord, None)) => {
                    self.pending_tiles.remove(&coord);
                }
                Err(promise) => self.pending_tile_meshes.push(promise),
            }
        }
    }

    fn poll_vector_render(&mut self, window: &Window) {
        let Some(pending) = self.pending_vector_render.take() else {
            return;
        };
        match pending.promise.try_take() {
            Ok(Some(image)) => {
                if self.vector_render_generation.load(Ordering::Acquire) == pending.generation {
                    self.vector_render_cache
                        .install_rendered(&pending.viewport, image);
                }
            }
            Ok(None) => {}
            Err(promise) => {
                self.pending_vector_render = Some(PendingVectorRender { promise, ..pending });
                window.on_next_frame(|_, cx| cx.refresh_windows());
            }
        }
    }

    fn ensure_vector_render(&mut self) {
        let viewport = self.project.viewport.clone();
        let layers = self
            .vector_render_cache
            .snapshot_layers(&self.project.layers);
        let Some(layer_key) = self.vector_render_cache.layer_key() else {
            return;
        };

        if self.dragging {
            return;
        }
        if !self.vector_render_cache.needs_render(&viewport) {
            return;
        }
        if self
            .pending_vector_render
            .as_ref()
            .is_some_and(|pending| pending.layer_key == layer_key && pending.viewport == viewport)
        {
            return;
        }

        let generation = self.vector_render_generation.fetch_add(1, Ordering::AcqRel) + 1;
        let cancellation = Arc::clone(&self.vector_render_generation);
        let render_viewport = viewport.clone();
        let promise = Promise::spawn_thread("rgis-render-vectors", move || {
            rgis_render::render_vector_layers_cancellable(
                &layers,
                &render_viewport,
                &cancellation,
                generation,
            )
        });
        self.pending_vector_render = Some(PendingVectorRender {
            layer_key,
            viewport,
            generation,
            promise,
        });
    }

    fn cancel_vector_render(&mut self) {
        if self.pending_vector_render.take().is_some() {
            self.vector_render_generation.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn map_callback(&mut self, width: f32, height: f32) -> MapCallback {
        self.drain_tiles();
        if let Some(fetcher) = &self.sprite_fetcher
            && let Ok(ready) = fetcher.receiver.try_recv()
        {
            self.sprite_atlas = Some(ready.atlas);
        }
        self.project.viewport.width_px = width.max(1.0) as u32;
        self.project.viewport.height_px = height.max(1.0) as u32;
        self.ensure_vector_render();
        let mut basemap_tiles = Vec::new();
        if self.project.show_tiles {
            let mut exact_tiles = Vec::new();
            let mut fallback_coords = std::collections::HashSet::new();
            for coord in visible_tiles_for_zoom(&self.project.viewport, OPENFREEMAP_MAX_ZOOM) {
                if let Some(mesh) = self.tile_meshes.get(&coord) {
                    exact_tiles.push((coord, Arc::clone(mesh)));
                } else if self.pending_tiles.insert(coord) {
                    self.vector_tile_fetcher.request(coord);
                } else {
                    for ancestor in raster::ancestor_coords(coord) {
                        if self.tile_meshes.contains(&ancestor) {
                            fallback_coords.insert(ancestor);
                            break;
                        }
                        if self.pending_tiles.insert(ancestor) {
                            self.vector_tile_fetcher.request(ancestor);
                        }
                    }
                }
            }
            for coord in fallback_coords {
                if let Some(mesh) = self.tile_meshes.get(&coord) {
                    let transform =
                        rgis_render::tile_screen_transform(coord, &self.project.viewport);
                    basemap_tiles.push(rgis_render::BasemapTileDraw {
                        coord,
                        mesh: Arc::clone(mesh),
                        offset: transform.offset,
                        scale: transform.scale,
                        width_scale: transform.width_scale,
                        size: transform.size,
                    });
                }
            }
            for (coord, mesh) in exact_tiles {
                let transform = rgis_render::tile_screen_transform(coord, &self.project.viewport);
                basemap_tiles.push(rgis_render::BasemapTileDraw {
                    coord,
                    mesh,
                    offset: transform.offset,
                    scale: transform.scale,
                    width_scale: transform.width_scale,
                    size: transform.size,
                });
            }
        }
        let vector_tile = self.vector_render_cache.render_with_preview(
            &self.project.layers,
            &self.project.viewport,
            true,
        );
        let mut tiles = if self.project.show_tiles {
            raster::collect_draws(
                &self.style,
                &self.project.viewport,
                &self.raster_fetchers,
                &mut self.raster_tile_cache,
            )
        } else {
            Vec::new()
        };
        let raster_tile_count = tiles.len() as u32;
        if let Some(vector_tile) = vector_tile {
            tiles.push(rgis_render::TileDraw {
                key: vector_tile.key,
                rect: vector_tile.rect,
                rgba: vector_tile.image,
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                opacity: 1.0,
            });
        }
        let vector_tile_count = tiles.len() as u32 - raster_tile_count;
        let (labels, glyph_bitmaps, icons) = labels::collect_label_draws(
            &basemap_tiles,
            &self.glyph_fetcher,
            self.sprite_atlas.as_ref(),
        );
        tiles.extend(icons);
        let mesh = if self.project.show_tiles {
            rgis_render::build_background_mesh(&self.project.viewport, &self.style)
        } else {
            SceneMesh::default()
        };
        MapCallback {
            background_index_count: mesh.indices.len() as u32,
            mesh,
            basemap_tiles,
            tiles,
            raster_tile_count,
            vector_tile_count,
            labels,
            glyph_bitmaps,
            width,
            height,
        }
    }

    fn render_map(&mut self, window: &mut Window) {
        self.poll_pending_loads(window);
        self.poll_vector_render(window);
        let Some(context) = WgpuContextHandle::from_window(window) else {
            self.status = "GPUI GPU context unavailable".to_string();
            return;
        };
        if context.device_lost() {
            self.resources = None;
            self.target = None;
            self.status = "Recovering GPU device".to_string();
            return;
        }
        let device = context.device();
        let queue = context.queue();

        let scale_factor = window.scale_factor();
        let viewport_size = window.viewport_size();
        let width = (viewport_size.width - px(self.sidebar_offset())).max(px(1.0));
        let has_client_titlebar = matches!(
            window.window_decorations(),
            gpui::Decorations::Client { .. }
        );
        let titlebar_height = if has_client_titlebar {
            px(TITLEBAR_HEIGHT)
        } else {
            px(0.0)
        };
        let height = (viewport_size.height - titlebar_height - px(STATUS_HEIGHT)).max(px(1.0));
        let size = (
            (f32::from(width) * scale_factor).ceil() as u32,
            (f32::from(height) * scale_factor).ceil() as u32,
        );
        if self.target.as_ref().map(|target| target.size) != Some(size) {
            self.target = Some(MapTarget::new(&context, size));
            self.resources = Some(MapRenderResources::new(device, context.texture_format()));
        }
        let callback = self.map_callback(
            f32::from(width) * scale_factor,
            f32::from(height) * scale_factor,
        );
        let target = self.target.as_ref().expect("map target created");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rgis-map-encoder"),
        });
        let resources = self.resources.take().expect("map resources created");
        self.resources = Some(
            resources.render_into(
                device,
                queue,
                &mut encoder,
                target
                    .msaa_view
                    .as_ref()
                    .unwrap_or_else(|| target.target.view()),
                target.msaa_view.as_ref().map(|_| target.target.view()),
                &callback,
            ),
        );
        queue.submit([encoder.finish()]);
        if self.pending_vector_render.is_some() {
            window.on_next_frame(|_, cx| cx.refresh_windows());
        }
    }

    fn layer_row(
        &self,
        layer_id: LayerId,
        name: String,
        visible: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let visibility_path = if visible {
            "M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Zm10 2.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"
        } else {
            "m3 3 18 18M10.6 6.2A10.7 10.7 0 0 1 12 6c6.5 0 10 6 10 6a18 18 0 0 1-3.2 3.8M6.2 6.3C3.4 8.3 2 12 2 12s3.5 6 10 6c1.1 0 2.1-.2 3-.5"
        };
        div()
            .h(px(30.0))
            .w_full()
            .px_2()
            .gap_1()
            .flex()
            .items_center()
            .text_sm()
            .text_color(rgb(ZED_TEXT))
            .hover(|style| style.bg(rgb(ZED_SURFACE)))
            .child(
                div()
                    .w(px(22.0))
                    .h(px(24.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(icon(visibility_path, ZED_MUTED))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            if let Some(layer) = this.project.get_layer_mut(layer_id) {
                                layer.visible = !layer.visible;
                            }
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    ),
            )
            .child(
                div()
                    .w(px(22.0))
                    .h(px(24.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(icon(
                        "M12 2 3.5 6.5 12 11l8.5-4.5L12 2Zm-8.5 9.5L12 16l8.5-4.5M3.5 16.5 12 21l8.5-4.5",
                        ZED_ACCENT,
                    )),
            )
            .child(div().flex_1().child(name))
            .child(
                div()
                    .w(px(22.0))
                    .h(px(24.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(icon(
                        "m12 3 1.2 5.8L19 10l-5.8 1.2L12 17l-1.2-5.8L5 10l5.8-1.2L12 3Zm6.5 12 .6 2.4 2.4.6-2.4.6-.6 2.4-.6-2.4-2.4-.6 2.4-.6.6-2.4Z",
                        ZED_MUTED,
                    ))
                    .hover(|style| style.text_color(rgb(ZED_TEXT)))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            let layer = this.layer_ui_state.style_editor_layer();
                            this.layer_ui_state
                                .set_style_editor_layer((layer != Some(layer_id)).then_some(layer_id));
                            this.layer_ui_state
                                .set_style_color_target(StyleColorTarget::Fill);
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    ),
            )
            .child(
                div()
                    .w(px(22.0))
                    .h(px(24.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(icon("m6 6 12 12M18 6 6 18", ZED_MUTED))
                    .hover(|style| style.text_color(rgb(0xf38ba8)))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, _, cx| {
                            this.project.remove_layer(layer_id);
                            cx.stop_propagation();
                            cx.notify();
                        }),
                    ),
            )
    }

    fn sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut content = div()
            .w(px(SIDEBAR_WIDTH))
            .h_full()
            .p_2()
            .gap_1()
            .flex()
            .flex_col()
            .bg(rgb(ZED_PANEL))
            .border_r_1()
            .border_color(rgb(ZED_BORDER))
            .text_color(rgb(ZED_TEXT))
            .child(
                div()
                    .h(px(30.0))
                    .w_full()
                    .px_2()
                    .flex()
                    .items_center()
                    .text_xs()
                    .text_color(rgb(ZED_MUTED))
                    .child(
                        div()
                            .w(px(18.0))
                            .h(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(if self.layer_ui_state.layers_expanded() {
                                icon("m6 9 6 6 6-6", ZED_MUTED)
                            } else {
                                icon("m9 6 6 6-6 6", ZED_MUTED)
                            }),
                    )
                    .child(div().text_xs().child("LAYERS"))
                    .child(
                        div()
                            .ml_auto()
                            .w(px(24.0))
                            .h(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .hover(|style| style.bg(rgb(ZED_SURFACE)))
                            .child(icon("M12 5v14M5 12h14", ZED_MUTED))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(|this, _, window, cx| {
                                    this.queue_pick_files(window);
                                    cx.stop_propagation();
                                }),
                            ),
                    )
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            let expanded = this.layer_ui_state.layers_expanded();
                            this.layer_ui_state.set_layers_expanded(!expanded);
                            cx.notify();
                        }),
                    ),
            );

        if self.layer_ui_state.layers_expanded() {
            for (id, name, visible) in self
                .project
                .layers
                .iter()
                .rev()
                .map(|layer| (layer.id, layer.name.clone(), layer.visible))
            {
                content = content.child(self.layer_row(id, name, visible, cx));
                if self.layer_ui_state.style_editor_layer() == Some(id) {
                    content = content.child(self.style_panel(id, cx));
                }
            }
            content = content.child(
                div()
                    .h(px(30.0))
                    .w_full()
                    .px_2()
                    .gap_1()
                    .flex()
                    .items_center()
                    .text_sm()
                    .text_color(rgb(ZED_MUTED))
                    .hover(|style| style.bg(rgb(ZED_SURFACE)))
                    .child(
                        div()
                            .w(px(22.0))
                            .h(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(icon(
                                if self.project.show_tiles {
                                    "M3 6 9 3l6 3 6-3v15l-6 3-6-3-6 3V6Zm6-3v15m6-12v15"
                                } else {
                                    "M3 3 21 21M3 6l6-3 6 3 6-3v9M3 12v9l6-3 2.2 1.1"
                                },
                                ZED_MUTED,
                            )),
                    )
                    .child(div().flex_1().child("OpenFreeMap"))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.project.show_tiles = !this.project.show_tiles;
                            cx.notify();
                        }),
                    ),
            );
        }
        content
    }

    fn color_chip(
        &self,
        layer_id: LayerId,
        target: StyleColorTarget,
        color: Color,
        selected: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .w(px(24.0))
            .h(px(24.0))
            .rounded_sm()
            .bg(rgba(color_to_rgba(color)))
            .border_1()
            .border_color(rgb(if selected { ZED_ACCENT } else { ZED_BORDER }))
            .hover(|style| style.border_color(rgb(ZED_TEXT)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    this.layer_ui_state.set_style_color_target(target);
                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                        let alpha = match target {
                            StyleColorTarget::Fill => layer.style.fill.a,
                            StyleColorTarget::Stroke => layer.style.stroke.a,
                        };
                        let chosen = Color { a: alpha, ..color };
                        match target {
                            StyleColorTarget::Fill => layer.style.fill = chosen,
                            StyleColorTarget::Stroke => layer.style.stroke = chosen,
                        }
                    }
                    cx.stop_propagation();
                    cx.notify();
                }),
            )
    }

    fn style_panel(&self, layer_id: LayerId, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(layer) = self
            .project
            .layers
            .iter()
            .find(|layer| layer.id == layer_id)
        else {
            return div();
        };
        let fill = layer.style.fill;
        let stroke = layer.style.stroke;
        let stroke_width = layer.style.stroke_width;
        let point_radius = layer.style.point_radius;
        let target = self.layer_ui_state.style_color_target();
        let target_color = match target {
            StyleColorTarget::Fill => fill,
            StyleColorTarget::Stroke => stroke,
        };
        let palette_colors = [
            (243, 139, 168),
            (250, 179, 135),
            (249, 226, 175),
            (166, 227, 161),
            (148, 226, 213),
            (137, 220, 235),
            (137, 180, 250),
            (203, 166, 247),
            (245, 194, 231),
            (205, 214, 244),
            (147, 153, 178),
            (49, 50, 68),
        ];
        let mut palette = div().flex().flex_wrap().gap_1();
        for (r, g, b) in palette_colors {
            palette = palette.child(self.color_chip(
                layer_id,
                target,
                Color::from_u8(r, g, b, 255),
                false,
                cx,
            ));
        }
        div()
            .w_full()
            .pl(px(48.0))
            .pr_2()
            .py_2()
            .gap_2()
            .flex()
            .flex_col()
            .bg(rgb(0x222222))
            .border_l_1()
            .border_color(rgb(ZED_BORDER))
            .text_xs()
            .text_color(rgb(ZED_MUTED))
            .child(
                div().flex().items_center().child("APPEARANCE").child(
                    div()
                        .ml_auto()
                        .w(px(24.0))
                        .h(px(24.0))
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(icon("m6 6 12 12M18 6 6 18", ZED_MUTED))
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, _, _, cx| {
                                this.layer_ui_state.set_style_editor_layer(None);
                                cx.stop_propagation();
                                cx.notify();
                            }),
                        ),
                ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(44.0)).child("Fill"))
                    .child(self.color_chip(
                        layer_id,
                        StyleColorTarget::Fill,
                        fill,
                        target == StyleColorTarget::Fill,
                        cx,
                    ))
                    .child(format_color(fill)),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(44.0)).child("Stroke"))
                    .child(self.color_chip(
                        layer_id,
                        StyleColorTarget::Stroke,
                        stroke,
                        target == StyleColorTarget::Stroke,
                        cx,
                    ))
                    .child(format_color(stroke)),
            )
            .child(palette)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(84.0)).child("Opacity"))
                    .child(
                        div()
                            .w(px(24.0))
                            .h(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(rgb(ZED_SURFACE))
                            .child(icon("M5 12h14", ZED_MUTED))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                                        let color = match target {
                                            StyleColorTarget::Fill => &mut layer.style.fill,
                                            StyleColorTarget::Stroke => &mut layer.style.stroke,
                                        };
                                        color.a = (color.a - 0.05).max(0.0);
                                    }
                                    cx.stop_propagation();
                                    cx.notify();
                                }),
                            ),
                    )
                    .child(format!("{:.0}%", target_color.a * 100.0))
                    .child(
                        div()
                            .w(px(24.0))
                            .h(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(rgb(ZED_SURFACE))
                            .child(icon("M12 5v14M5 12h14", ZED_MUTED))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                                        let color = match target {
                                            StyleColorTarget::Fill => &mut layer.style.fill,
                                            StyleColorTarget::Stroke => &mut layer.style.stroke,
                                        };
                                        color.a = (color.a + 0.05).min(1.0);
                                    }
                                    cx.stop_propagation();
                                    cx.notify();
                                }),
                            ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(84.0)).child("Line width"))
                    .child(
                        div()
                            .w(px(24.0))
                            .h(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(rgb(ZED_SURFACE))
                            .child(icon("M5 12h14", ZED_MUTED))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                                        layer.style.stroke_width =
                                            (layer.style.stroke_width - 0.5).max(0.1);
                                    }
                                    cx.stop_propagation();
                                    cx.notify();
                                }),
                            ),
                    )
                    .child(format!("{stroke_width:.1}"))
                    .child(
                        div()
                            .w(px(24.0))
                            .h(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(rgb(ZED_SURFACE))
                            .child(icon("M12 5v14M5 12h14", ZED_MUTED))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                                        layer.style.stroke_width += 0.5;
                                    }
                                    cx.stop_propagation();
                                    cx.notify();
                                }),
                            ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(84.0)).child("Point radius"))
                    .child(
                        div()
                            .w(px(24.0))
                            .h(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(rgb(ZED_SURFACE))
                            .child(icon("M5 12h14", ZED_MUTED))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                                        layer.style.point_radius =
                                            (layer.style.point_radius - 1.0).max(1.0);
                                    }
                                    cx.stop_propagation();
                                    cx.notify();
                                }),
                            ),
                    )
                    .child(format!("{point_radius:.1}"))
                    .child(
                        div()
                            .w(px(24.0))
                            .h(px(24.0))
                            .flex()
                            .items_center()
                            .justify_center()
                            .bg(rgb(ZED_SURFACE))
                            .child(icon("M12 5v14M5 12h14", ZED_MUTED))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                                        layer.style.point_radius += 1.0;
                                    }
                                    cx.stop_propagation();
                                    cx.notify();
                                }),
                            ),
                    ),
            )
            .child(
                div()
                    .mt_1()
                    .flex()
                    .items_center()
                    .gap_2()
                    .text_color(rgb(ZED_MUTED))
                    .child("Color applies to selected target")
                    .child(
                        div()
                            .ml_auto()
                            .px_2()
                            .py_1()
                            .bg(rgb(ZED_SURFACE))
                            .child("Reset")
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _, _, cx| {
                                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                                        layer.style = rgis_core::Style::default();
                                    }
                                    cx.stop_propagation();
                                    cx.notify();
                                }),
                            ),
                    ),
            )
    }

    fn client_titlebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .h(px(TITLEBAR_HEIGHT))
            .w_full()
            .px_3()
            .flex()
            .items_center()
            .gap_2()
            .bg(rgb(ZED_TITLEBAR))
            .border_b_1()
            .border_color(rgb(ZED_BORDER))
            .text_sm()
            .text_color(rgb(ZED_TEXT))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|_, event: &gpui::MouseDownEvent, window, cx| {
                    if event.click_count == 2 {
                        window.zoom_window();
                    } else {
                        window.start_window_move();
                    }
                    cx.stop_propagation();
                }),
            )
            .child(icon(
                "M12 2 20 6.5v9L12 20l-8-4.5v-9L12 2Zm0 5v8m-4-6 4 2 4-2",
                ZED_ACCENT,
            ))
            .child(
                div()
                    .ml_auto()
                    .w(px(32.0))
                    .h(px(28.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .hover(|style| style.bg(rgb(0x5a2a34)))
                    .child(icon("m6 6 12 12M18 6 6 18", 0xf38ba8))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_, _, window, cx| {
                            window.remove_window();
                            cx.stop_propagation();
                        }),
                    ),
            )
    }
}

impl LayerUi for RgisNativeApp {
    fn project(&self) -> &Project {
        &self.project
    }

    fn project_mut(&mut self) -> &mut Project {
        &mut self.project
    }

    fn layer_ui_state(&self) -> &LayerUiState {
        &self.layer_ui_state
    }

    fn layer_ui_state_mut(&mut self) -> &mut LayerUiState {
        &mut self.layer_ui_state
    }

    fn sidebar_visible(&self) -> bool {
        self.sidebar_visible
    }

    fn set_sidebar_visible(&mut self, visible: bool) {
        self.sidebar_visible = visible;
    }

    fn status_text(&self) -> &str {
        &self.status
    }

    fn set_status(&mut self, status: String) {
        self.status = status;
    }

    fn cursor_lonlat(&self) -> Option<(f64, f64)> {
        self.cursor_lonlat
    }

    fn add_layer(&mut self, window: &mut Window) {
        self.queue_pick_files(window);
    }

    fn open_project(&mut self, window: &mut Window) {
        self.queue_open_project(window);
    }

    fn save_project(&mut self, window: &mut Window) {
        self.queue_save_project(window);
    }
}

impl Render for RgisNativeApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.poll_project_io(window);
        self.render_map(window);
        let has_client_titlebar = matches!(
            window.window_decorations(),
            gpui::Decorations::Client { .. }
        );
        let map_surface = self.target.as_ref().map(|target| target.target.surface());
        let map = div()
            .flex_1()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &gpui::MouseDownEvent, _, cx| {
                    this.cancel_vector_render();
                    this.dragging = true;
                    this.last_cursor = Some(event.position);
                    this.bbox_zoom_start = event.modifiers.shift.then_some(event.position);
                    cx.notify();
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &gpui::MouseUpEvent, window, cx| {
                    this.dragging = false;
                    this.last_cursor = None;
                    if let Some(start) = this.bbox_zoom_start.take() {
                        let end = event.position;
                        if (end.x - start.x).abs() > px(4.0) && (end.y - start.y).abs() > px(4.0) {
                            let scale = window.scale_factor();
                            let titlebar = if matches!(
                                window.window_decorations(),
                                gpui::Decorations::Client { .. }
                            ) {
                                TITLEBAR_HEIGHT
                            } else {
                                0.0
                            };
                            let to_map = |point: gpui::Point<gpui::Pixels>| {
                                [
                                    ((f32::from(point.x) - this.sidebar_offset()) * scale)
                                        .clamp(0.0, this.project.viewport.width_px as f32),
                                    ((f32::from(point.y) - titlebar) * scale)
                                        .clamp(0.0, this.project.viewport.height_px as f32),
                                ]
                            };
                            let world_a = this.project.viewport.screen_to_world(to_map(start));
                            let world_b = this.project.viewport.screen_to_world(to_map(end));
                            this.project.viewport.fit_bounds(&GeoBounds {
                                min_x: world_a.x.min(world_b.x),
                                min_y: world_a.y.min(world_b.y),
                                max_x: world_a.x.max(world_b.x),
                                max_y: world_a.y.max(world_b.y),
                            });
                        }
                    }
                    cx.notify();
                }),
            )
            .on_mouse_move(
                cx.listener(|this, event: &gpui::MouseMoveEvent, window, cx| {
                    let scale = window.scale_factor();
                    let titlebar = if matches!(
                        window.window_decorations(),
                        gpui::Decorations::Client { .. }
                    ) {
                        TITLEBAR_HEIGHT
                    } else {
                        0.0
                    };
                    let cursor = [
                        ((f32::from(event.position.x) - this.sidebar_offset()) * scale)
                            .clamp(0.0, this.project.viewport.width_px as f32),
                        ((f32::from(event.position.y) - titlebar) * scale)
                            .clamp(0.0, this.project.viewport.height_px as f32),
                    ];
                    let world = this.project.viewport.screen_to_world(cursor);
                    this.cursor_lonlat = Some(mercator_to_lonlat(world.x, world.y));
                    if this.dragging
                        && this.bbox_zoom_start.is_none()
                        && let Some(previous) = this.last_cursor.replace(event.position)
                    {
                        let delta = event.position - previous;
                        this.project
                            .viewport
                            .pan(f32::from(delta.x) * scale, f32::from(delta.y) * scale);
                        cx.notify();
                    }
                }),
            )
            .on_scroll_wheel(
                cx.listener(|this, event: &gpui::ScrollWheelEvent, window, cx| {
                    let delta = f32::from(event.delta.pixel_delta(px(16.0)).y) as f64 / 240.0;
                    let scale = window.scale_factor();
                    let cursor = [
                        ((f32::from(event.position.x) - this.sidebar_offset()) * scale)
                            .clamp(0.0, this.project.viewport.width_px as f32),
                        ((f32::from(event.position.y)
                            - if matches!(
                                window.window_decorations(),
                                gpui::Decorations::Client { .. }
                            ) {
                                TITLEBAR_HEIGHT
                            } else {
                                0.0
                            })
                            * scale)
                            .clamp(0.0, this.project.viewport.height_px as f32),
                    ];
                    this.project.viewport.zoom_toward(cursor, delta);
                    cx.notify();
                }),
            )
            .bg(rgb(ZED_CANVAS));
        let map = if let Some(surface) = map_surface {
            map.child(surface.size_full())
        } else {
            map.child("GPU surface unavailable")
        };
        let mut root = div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(ZED_CANVAS))
            .overflow_hidden();
        if !window.is_maximized() {
            root = root.border_1().rounded_lg().border_color(rgb(0x484848));
        }
        if has_client_titlebar {
            root = root.child(self.client_titlebar(cx));
        }
        let map_content = div()
            .flex_1()
            .flex()
            .when(self.sidebar_visible, |content| {
                content.child(ui::sidebar(self, window, cx))
            })
            .child(map);
        root = root.child(map_content).child(ui::status_bar(self, cx));
        if !window.is_maximized() {
            const RESIZE_ZONE: f32 = 8.0;
            root = root.child(
                div()
                    .absolute()
                    .top_0()
                    .left_0()
                    .size_full()
                    .child(
                        resize_handle(ResizeEdge::TopLeft, CursorStyle::ResizeUpLeftDownRight, cx)
                            .absolute()
                            .top_0()
                            .left_0()
                            .w(px(RESIZE_ZONE))
                            .h(px(RESIZE_ZONE)),
                    )
                    .child(
                        resize_handle(ResizeEdge::Top, CursorStyle::ResizeUpDown, cx)
                            .absolute()
                            .top_0()
                            .left(px(RESIZE_ZONE))
                            .right(px(RESIZE_ZONE))
                            .h(px(RESIZE_ZONE)),
                    )
                    .child(
                        resize_handle(ResizeEdge::TopRight, CursorStyle::ResizeUpRightDownLeft, cx)
                            .absolute()
                            .top_0()
                            .right_0()
                            .w(px(RESIZE_ZONE))
                            .h(px(RESIZE_ZONE)),
                    )
                    .child(
                        resize_handle(ResizeEdge::Left, CursorStyle::ResizeLeftRight, cx)
                            .absolute()
                            .top(px(RESIZE_ZONE))
                            .bottom(px(RESIZE_ZONE))
                            .left_0()
                            .w(px(RESIZE_ZONE)),
                    )
                    .child(
                        resize_handle(ResizeEdge::Right, CursorStyle::ResizeLeftRight, cx)
                            .absolute()
                            .top(px(RESIZE_ZONE))
                            .bottom(px(RESIZE_ZONE))
                            .right_0()
                            .w(px(RESIZE_ZONE)),
                    )
                    .child(
                        resize_handle(
                            ResizeEdge::BottomLeft,
                            CursorStyle::ResizeUpRightDownLeft,
                            cx,
                        )
                        .absolute()
                        .bottom_0()
                        .left_0()
                        .w(px(RESIZE_ZONE))
                        .h(px(RESIZE_ZONE)),
                    )
                    .child(
                        resize_handle(ResizeEdge::Bottom, CursorStyle::ResizeUpDown, cx)
                            .absolute()
                            .bottom_0()
                            .left(px(RESIZE_ZONE))
                            .right(px(RESIZE_ZONE))
                            .h(px(RESIZE_ZONE)),
                    )
                    .child(
                        resize_handle(
                            ResizeEdge::BottomRight,
                            CursorStyle::ResizeUpLeftDownRight,
                            cx,
                        )
                        .absolute()
                        .bottom_0()
                        .right_0()
                        .w(px(RESIZE_ZONE))
                        .h(px(RESIZE_ZONE)),
                    ),
            );
        }
        root
    }
}

pub fn run(startup_paths: Vec<std::path::PathBuf>) {
    application().run(move |cx: &mut App| {
        cx.bind_keys(default_bindings().as_keybindings(Some(DEFAULT_INPUT_CONTEXT)));
        let bounds = Bounds::centered(None, size(px(1280.0), px(800.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(gpui::TitlebarOptions::default()),
                is_resizable: true,
                // Draw the titlebar in the GPUI view. Wayland compositors are
                // not required to provide the server-decoration protocol.
                window_decorations: Some(WindowDecorations::Client),
                ..Default::default()
            },
            move |_, cx| cx.new(|_| RgisNativeApp::new(&startup_paths)),
        )
        .expect("failed to open rgis window");
        cx.activate(true);
    });
}
