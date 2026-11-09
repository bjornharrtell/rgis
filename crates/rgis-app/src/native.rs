#![allow(dead_code)]

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::{fs, path::PathBuf};

use gpui::component::{ActiveTheme as _, TitleBar};
use gpui::{
    App, Bounds, Context, CursorStyle, MouseButton, Render, ResizeEdge, Window, WindowBounds,
    WindowDecorations, WindowOptions, div, prelude::*, px, size,
};
use lru::LruCache;
use poll_promise::Promise;
use rgis_app::header;
use rgis_app::labels;
use rgis_app::raster;
use rgis_app::ui::{self, LayerUi, LayerUiState};
use rgis_core::{Bounds as GeoBounds, Color, Layer, LayerId, Project, ProjectState};
use rgis_render::{MapCallback, MapRenderResources};
use rgis_tiles::{OPENFREEMAP_MAX_ZOOM, TileCoord, VectorTileFetcher, visible_tiles_for_zoom};

const TITLEBAR_HEIGHT: f32 = ui::HEADER_HEIGHT;
const TILE_CACHE_SIZE: usize = 256;
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
    crs: u16,
) -> Result<Option<rgis_io::LoadedLayer>, rgis_io::IoError> {
    if let Some(epsg) = loaded.epsg.filter(|epsg| *epsg != crs) {
        let result = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Reproject layer?")
            .set_description(format!(
                "\"{file_name}\" declares EPSG:{epsg}. The project CRS is \
                 EPSG:{crs}. Transform this layer on the fly?"
            ))
            .set_buttons(rfd::MessageButtons::YesNo)
            .show();
        if result != rfd::MessageDialogResult::Yes {
            return Ok(None);
        }
    }
    loaded.into_crs(crs).map(Some)
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
        let loaded = rgis_io::load_path_with_crs(&source_path)
            .and_then(|loaded| loaded.into_crs(state.crs))
            .map_err(|error| {
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

/// Space between the window edge and our content: gpui-component draws the
/// client-side frame (shadow margin plus a 1px border) around the root view.
/// Returns (left, top, right, bottom) in logical pixels.
fn frame_insets(window: &Window) -> (f32, f32, f32, f32) {
    match window.window_decorations() {
        gpui::Decorations::Server => (0.0, 0.0, 0.0, 0.0),
        gpui::Decorations::Client { tiling } => {
            let paddings = gpui::component::window_paddings(window);
            let edge = |padding: gpui::Pixels, tiled: bool| {
                f32::from(padding) + if tiled { 0.0 } else { 1.0 }
            };
            (
                edge(paddings.left, tiling.left),
                edge(paddings.top, tiling.top),
                edge(paddings.right, tiling.right),
                edge(paddings.bottom, tiling.bottom),
            )
        }
    }
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

const MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// Our own wgpu device: gpui-pre exposes no way to render into the window's
/// device or to composite an external texture, so the map is drawn offscreen
/// here and shown as an image.
struct OffscreenGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl OffscreenGpu {
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        Some(Self { device, queue })
    }
}

struct MapTarget {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    msaa_view: Option<wgpu::TextureView>,
    readback: wgpu::Buffer,
    padded_row: u32,
    size: (u32, u32),
    image: Option<Arc<gpui::RenderImage>>,
}

impl MapTarget {
    fn new(device: &wgpu::Device, dimensions: (u32, u32)) -> Self {
        let (w, h) = dimensions;
        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgis-map-color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MAP_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let msaa_view = (rgis_render::MSAA_SAMPLES > 1).then(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("rgis-map-msaa"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: rgis_render::MSAA_SAMPLES,
                    dimension: wgpu::TextureDimension::D2,
                    format: MAP_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        let padded_row = (w * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rgis-map-readback"),
            size: padded_row as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            color,
            color_view,
            msaa_view,
            readback,
            padded_row,
            size: dimensions,
            image: None,
        }
    }

    /// Copies the rendered frame to the CPU and wraps it as a GPUI image.
    fn read_back(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut encoder: wgpu::CommandEncoder,
    ) {
        let (w, h) = self.size;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return;
        }
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        {
            let data = slice.get_mapped_range().expect("mapped range");
            if self.padded_row == w * 4 {
                pixels.extend_from_slice(&data);
            } else {
                for row in 0..h as usize {
                    let start = row * self.padded_row as usize;
                    pixels.extend_from_slice(&data[start..start + (w * 4) as usize]);
                }
            }
        }
        self.readback.unmap();
        // Frames are BGRA, matching GPUI's image path.
        if let Some(image) = image::RgbaImage::from_raw(w, h, pixels) {
            self.image = Some(Arc::new(gpui::RenderImage::new(smallvec::smallvec![
                image::Frame::new(image)
            ])));
        }
    }
}

/// Hash of everything a map frame is drawn from: the camera, the layers and
/// the identity of each tile and image drawn. Equal keys give identical pixels.
fn map_render_key(
    viewport: &rgis_core::Viewport,
    layer_key: Option<u64>,
    crs: u64,
    callback: &MapCallback,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    layer_key.hash(&mut hasher);
    crs.hash(&mut hasher);
    viewport.center.x.to_bits().hash(&mut hasher);
    viewport.center.y.to_bits().hash(&mut hasher);
    viewport.zoom.to_bits().hash(&mut hasher);
    viewport.width_px.hash(&mut hasher);
    viewport.height_px.hash(&mut hasher);
    for tile in &callback.basemap_tiles {
        (tile.coord.z, tile.coord.x, tile.coord.y).hash(&mut hasher);
        Arc::as_ptr(&tile.mesh).hash(&mut hasher);
    }
    for tile in &callback.tiles {
        tile.key.hash(&mut hasher);
        Arc::as_ptr(&tile.rgba).hash(&mut hasher);
        tile.opacity.to_bits().hash(&mut hasher);
    }
    callback.labels.len().hash(&mut hasher);
    callback.glyph_bitmaps.len().hash(&mut hasher);
    callback.mesh.vertices.len().hash(&mut hasher);
    callback.mesh.indices.len().hash(&mut hasher);
    hasher.finish()
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
    gpu: Option<OffscreenGpu>,
    resources: Option<MapRenderResources>,
    target: Option<MapTarget>,
    pending_loads: Vec<Promise<LoadResults>>,
    pending_project_loads: Vec<Promise<Result<Option<LoadedProject>, String>>>,
    pending_project_saves: Vec<Promise<Result<Option<PathBuf>, String>>>,
    status: String,
    last_cursor: Option<gpui::Point<gpui::Pixels>>,
    cursor_coords: Option<(f64, f64)>,
    bbox_zoom_start: Option<gpui::Point<gpui::Pixels>>,
    dragging: bool,
    zoom_animation: Option<rgis_core::ZoomAnimation>,
    /// Fingerprint of what the last rendered map frame was drawn from.
    last_render_key: Option<u64>,
    last_frame: Option<std::time::Instant>,
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
                    Ok(loaded) => match prepare_loaded_layer(loaded, &name, project.crs) {
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
            gpu: None,
            resources: None,
            target: None,
            pending_loads: Vec::new(),
            pending_project_loads: Vec::new(),
            pending_project_saves: Vec::new(),
            status: "GPUI native renderer".to_string(),
            last_cursor: None,
            cursor_coords: None,
            bbox_zoom_start: None,
            dragging: false,
            zoom_animation: None,
            last_render_key: None,
            last_frame: None,
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
        ui::map_left(self.sidebar_visible)
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
                            Ok(loaded) => {
                                match prepare_loaded_layer(loaded, &name, self.project.crs) {
                                    Ok(Some(loaded)) => {
                                        let id = self.project.next_layer_id();
                                        let mut layer =
                                            Layer::new(id, loaded.name, loaded.features);
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
                                        self.status =
                                            format!("Skipped {name}: reprojection declined");
                                    }
                                    Err(error) => {
                                        self.status = format!("Failed to load {name}: {error}");
                                    }
                                }
                            }
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

        // Wait for the viewport to settle instead of rendering every frame.
        if self.dragging || self.zoom_animation.is_some() {
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
        if self.project.basemap_visible() {
            let mut exact_tiles = Vec::new();
            let mut fallback_coords: Vec<TileCoord> = Vec::new();
            for coord in visible_tiles_for_zoom(&self.project.viewport, OPENFREEMAP_MAX_ZOOM) {
                if let Some(mesh) = self.tile_meshes.get(&coord) {
                    exact_tiles.push((coord, Arc::clone(mesh)));
                    continue;
                }
                if self.pending_tiles.insert(coord) {
                    self.vector_tile_fetcher.request(coord);
                }
                // Cover the gap with what is already loaded, in the same frame
                // the tile is first needed, so zooming never flashes background.
                let tile_meshes = &self.tile_meshes;
                let before = fallback_coords.len();
                for ancestor in raster::ancestor_coords(coord) {
                    if tile_meshes.contains(&ancestor) {
                        fallback_coords.push(ancestor);
                        break;
                    }
                    if self.pending_tiles.insert(ancestor) {
                        self.vector_tile_fetcher.request(ancestor);
                    }
                }
                if fallback_coords.len() == before {
                    fallback_coords.extend(
                        raster::descendant_coords(coord, 2)
                            .filter(|descendant| tile_meshes.contains(descendant)),
                    );
                }
            }
            // Coarse tiles first so finer ones draw over them.
            fallback_coords.sort_by_key(|coord| (coord.z, coord.x, coord.y));
            fallback_coords.dedup();
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
        let mut tiles = if self.project.basemap_visible() {
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
        // The style's background colour also paints the map without a basemap.
        let mesh = rgis_render::build_background_mesh(&self.project.viewport, &self.style);
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

    /// Advances a running zoom animation by the time since the last frame.
    fn step_zoom_animation(&mut self, window: &mut Window) {
        let now = std::time::Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map_or(1.0 / 60.0, |last| (now - last).as_secs_f64().min(0.1));
        let Some(animation) = &self.zoom_animation else {
            return;
        };
        if animation.step(&mut self.project.viewport, dt) {
            window.request_animation_frame();
        } else {
            self.zoom_animation = None;
        }
    }

    fn render_map(&mut self, window: &mut Window) {
        self.step_zoom_animation(window);
        self.poll_pending_loads(window);
        self.poll_vector_render(window);
        if self.gpu.is_none() {
            self.gpu = OffscreenGpu::new();
        }
        let Some(gpu) = self.gpu.as_ref() else {
            self.status = "No GPU adapter available".to_string();
            return;
        };
        let (device_owned, queue_owned) = (gpu.device.clone(), gpu.queue.clone());
        let (device, queue) = (&device_owned, &queue_owned);

        let scale_factor = window.scale_factor();
        let viewport_size = window.viewport_size();
        let (frame_l, frame_t, frame_r, frame_b) = frame_insets(window);
        let width =
            (viewport_size.width - px(self.sidebar_offset() + frame_l + frame_r)).max(px(1.0));
        let has_client_titlebar = matches!(
            window.window_decorations(),
            gpui::Decorations::Client { .. }
        );
        let titlebar_height = if has_client_titlebar {
            px(TITLEBAR_HEIGHT)
        } else {
            px(0.0)
        };
        let height =
            (viewport_size.height - titlebar_height - px(ui::STATUS_HEIGHT + frame_t + frame_b))
                .max(px(1.0));
        let size = (
            (f32::from(width) * scale_factor).ceil() as u32,
            (f32::from(height) * scale_factor).ceil() as u32,
        );
        if self.target.as_ref().map(|target| target.size) != Some(size) {
            self.target = Some(MapTarget::new(device, size));
            self.last_render_key = None;
        }
        if self.resources.is_none() {
            self.resources = Some(MapRenderResources::new(device, MAP_FORMAT));
        }
        let callback = self.map_callback(
            f32::from(width) * scale_factor,
            f32::from(height) * scale_factor,
        );
        // Hover, status and other UI-only updates re-render the window without
        // changing the map; skip the GPU pass and the CPU readback for those.
        let key = map_render_key(
            &self.project.viewport,
            self.vector_render_cache.layer_key(),
            u64::from(self.project.crs),
            &callback,
        );
        let target = self.target.as_mut().expect("map target created");
        if self.last_render_key == Some(key) && target.image.is_some() {
            return;
        }
        self.last_render_key = Some(key);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rgis-map-encoder"),
        });
        let resources = self.resources.take().expect("map resources created");
        self.resources = Some(resources.render_into(
            device,
            queue,
            &mut encoder,
            target.msaa_view.as_ref().unwrap_or(&target.color_view),
            target.msaa_view.as_ref().map(|_| &target.color_view),
            &callback,
        ));
        target.read_back(device, queue, encoder);
        if self.pending_vector_render.is_some() {
            window.on_next_frame(|_, cx| cx.refresh_windows());
        }
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

    fn cursor_coords(&self) -> Option<(f64, f64)> {
        self.cursor_coords
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

    fn crs_changed(&mut self) {
        self.cancel_vector_render();
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
        let map_image = self.target.as_ref().and_then(|target| target.image.clone());
        let map = div()
            .size_full()
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
                            let (frame_l, frame_t, _, _) = frame_insets(window);
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
                                    ((f32::from(point.x) - this.sidebar_offset() - frame_l)
                                        * scale)
                                        .clamp(0.0, this.project.viewport.width_px as f32),
                                    ((f32::from(point.y) - titlebar - frame_t) * scale)
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
                    let (frame_l, frame_t, _, _) = frame_insets(window);
                    let titlebar = if matches!(
                        window.window_decorations(),
                        gpui::Decorations::Client { .. }
                    ) {
                        TITLEBAR_HEIGHT
                    } else {
                        0.0
                    };
                    let cursor = [
                        ((f32::from(event.position.x) - this.sidebar_offset() - frame_l) * scale)
                            .clamp(0.0, this.project.viewport.width_px as f32),
                        ((f32::from(event.position.y) - titlebar - frame_t) * scale)
                            .clamp(0.0, this.project.viewport.height_px as f32),
                    ];
                    let world = this.project.viewport.screen_to_world(cursor);
                    this.cursor_coords = Some((world.x, world.y));
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
            .on_scroll(cx.listener(|this, event: &gpui::ScrollEvent, window, cx| {
                let delta = f32::from(event.delta.pixel_delta(px(16.0)).y) as f64 / 240.0;
                let scale = window.scale_factor();
                let (frame_l, frame_t, _, _) = frame_insets(window);
                let cursor = [
                    ((f32::from(event.position.x) - this.sidebar_offset() - frame_l) * scale)
                        .clamp(0.0, this.project.viewport.width_px as f32),
                    ((f32::from(event.position.y)
                        - frame_t
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
                this.cancel_vector_render();
                match &mut this.zoom_animation {
                    Some(animation) => animation.retarget(cursor, delta),
                    None => {
                        this.zoom_animation = Some(rgis_core::ZoomAnimation::new(
                            &this.project.viewport,
                            cursor,
                            delta,
                        ));
                        this.last_frame = None;
                    }
                }
                cx.notify();
            }))
            .bg(cx.theme().tokens.background);
        let map = if let Some(image) = map_image {
            map.child(
                gpui::canvas(
                    |_, _, _| (),
                    move |bounds, _, window, _| {
                        let _ = window.paint_image(
                            bounds,
                            bounds,
                            gpui::Corners::default(),
                            image.clone(),
                            0,
                            false,
                        );
                    },
                )
                .size_full(),
            )
        } else {
            map.child("GPU unavailable")
        };
        let radii = gpui::component::window_corner_radii(window, cx);
        let mut root = div()
            .size_full()
            .flex()
            .flex_col()
            .bg(cx.theme().title_bar)
            .rounded_tl(radii.top_left)
            .rounded_tr(radii.top_right)
            .rounded_bl(radii.bottom_left)
            .rounded_br(radii.bottom_right)
            .overflow_hidden();
        if has_client_titlebar {
            root = root.child(header::app_header(
                window,
                cx,
                self.project
                    .name
                    .clone()
                    .unwrap_or_else(|| "rgis".to_string()),
            ));
        }
        root = root
            .child(ui::workspace(self, map, window, cx))
            .child(ui::status_bar(self, window, cx));
        root
    }
}

pub fn run(startup_paths: Vec<std::path::PathBuf>) {
    gpui::application()
        .with_assets(gpui::assets::Assets)
        .run(move |cx: &mut App| {
            gpui::init(cx);
            ui::init_theme(cx);
            cx.text_system()
                .add_fonts(vec![std::borrow::Cow::Borrowed(include_bytes!(
                    "../assets/fonts/IBMPlexSans-Regular.ttf"
                ))])
                .expect("failed to load embedded font");
            let bounds = Bounds::centered(None, size(px(1280.0), px(800.0)), cx);
            gpui::open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    titlebar: Some(TitleBar::title_bar_options()),
                    is_resizable: true,
                    // Draw the titlebar in the GPUI view. Wayland compositors are
                    // not required to provide the server-decoration protocol.
                    window_decorations: Some(WindowDecorations::Client),
                    // gpui-pre defaults to an opaque window; the rounded client-side
                    // frame needs transparent corners.
                    window_background: gpui::WindowBackgroundAppearance::Transparent,
                    ..Default::default()
                },
                cx,
                move |_, cx| cx.new(|_| RgisNativeApp::new(&startup_paths)),
            )
            .expect("failed to open rgis window");
            cx.activate(true);
        });
}
