#![allow(dead_code)]

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    rc::Rc,
    sync::Arc,
};

use crate::labels;
use crate::raster;
use crate::ui::{self, LayerUi, LayerUiState};
use gpui::component::ActiveTheme as _;
use gpui::web::WebPlatform;
use gpui::{
    App, Bounds, Context, MouseButton, Render, Window, WindowBounds, WindowOptions, div,
    prelude::*, px, rgb, size,
};
use rgis_core::{
    Bounds as GeoBounds, Layer, Project, Viewport, lonlat_to_mercator, mercator_to_lonlat,
};
use rgis_render::{
    MapCallback, MapRenderResources, StyleSheet, TileMesh, build_background_mesh, build_tile_mesh,
};
use rgis_tiles::{OPENFREEMAP_MAX_ZOOM, TileCoord, VectorTileFetcher, visible_tiles_for_zoom};
use wasm_bindgen::JsCast;
use wgpu;

const SIDEBAR_WIDTH: f32 = 280.0;
const STATUS_HEIGHT: f32 = 28.0;

// WebGL2 has no BGRA render targets, so frames are rendered as RGBA and
// swizzled on the CPU while being copied into GPUI's BGRA image.
const MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Our own wgpu device. gpui-pre has no external-texture surface, so the map
/// is rendered offscreen here and shown as an image.
struct OffscreenGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    _surface: Option<wgpu::Surface<'static>>,
}

#[derive(Debug)]
struct WebDisplaySource;

impl raw_window_handle::HasDisplayHandle for WebDisplaySource {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(raw_window_handle::DisplayHandle::web())
    }
}

enum GpuState {
    Idle,
    Starting,
    Ready(OffscreenGpu),
    Failed(String),
}

fn start_gpu(slot: Rc<std::cell::RefCell<GpuState>>) {
    *slot.borrow_mut() = GpuState::Starting;
    wasm_bindgen_futures::spawn_local(async move {
        let instance = wgpu::util::new_instance_with_webgpu_detection(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_with_display_handle(Box::new(WebDisplaySource))
        })
        .await;
        // WebGL needs a surface to find an adapter. A detached canvas that is
        // never presented to is enough; frames are read back instead.
        let canvas = match web_sys::window() {
            None => Err("no window".to_string()),
            Some(window) => match window.document() {
                None => Err("no document".to_string()),
                Some(document) => match document.create_element("canvas") {
                    Err(error) => Err(format!("create_element: {error:?}")),
                    Ok(element) => element
                        .dyn_into::<web_sys::HtmlCanvasElement>()
                        .map_err(|_| "element is not a canvas".to_string()),
                },
            },
        };
        let mut surface_error = String::new();
        let surface = match canvas {
            Ok(canvas) => match instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)) {
                Ok(surface) => Some(surface),
                Err(error) => {
                    surface_error = format!("surface: {error}");
                    None
                }
            },
            Err(error) => {
                surface_error = error;
                None
            }
        };
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: surface.as_ref(),
                ..Default::default()
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(error) => {
                *slot.borrow_mut() = GpuState::Failed({
                    let text = error.to_string();
                    let useful: Vec<&str> = text
                        .split(';')
                        .filter(|part| !part.contains("not compiled in"))
                        .collect();
                    format!("GPU adapter failed ({surface_error}): {}", useful.join(";"))
                });
                return;
            }
        };
        let limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
        match adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                ..Default::default()
            })
            .await
        {
            Ok((device, queue)) => {
                *slot.borrow_mut() = GpuState::Ready(OffscreenGpu {
                    device,
                    queue,
                    _surface: surface,
                });
            }
            Err(error) => {
                *slot.borrow_mut() = GpuState::Failed(format!("no GPU device: {error}"));
            }
        }
    });
}

struct MapTarget {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    msaa_view: Option<wgpu::TextureView>,
    readback: wgpu::Buffer,
    padded_row: u32,
    size: (u32, u32),
    image: Option<Arc<gpui::RenderImage>>,
    /// Set by the `map_async` callback once the readback buffer is mapped.
    in_flight: Option<Rc<std::cell::Cell<bool>>>,
}

impl LayerUi for RgisWebApp {
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

    fn add_layer(&mut self, _window: &mut Window) {}

    fn open_project(&mut self, _window: &mut Window) {
        self.status = "Opening project files is only available in the native app".to_string();
    }

    fn save_project(&mut self, _window: &mut Window) {
        self.status = "Saving project files is only available in the native app".to_string();
    }
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
            in_flight: None,
        }
    }

    /// Collects a finished readback, if any. Returns true while one is still
    /// pending (the buffer can not be reused until then).
    fn poll_readback(&mut self, device: &wgpu::Device) -> bool {
        let Some(done) = self.in_flight.clone() else {
            return false;
        };
        let _ = device.poll(wgpu::PollType::Poll);
        if !done.get() {
            return true;
        }
        let (w, h) = self.size;
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        {
            let data = self
                .readback
                .slice(..)
                .get_mapped_range()
                .expect("mapped range");
            for row in 0..h as usize {
                let start = row * self.padded_row as usize;
                pixels.extend_from_slice(&data[start..start + (w * 4) as usize]);
            }
        }
        self.readback.unmap();
        self.in_flight = None;
        pixels
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .for_each(|px| px.swap(0, 2));
        if let Some(image) = image::RgbaImage::from_raw(w, h, pixels) {
            self.image = Some(Arc::new(gpui::RenderImage::new(smallvec::smallvec![
                image::Frame::new(image)
            ])));
        }
        false
    }

    /// Copies the rendered frame into the readback buffer and starts mapping it.
    fn start_readback(&mut self, queue: &wgpu::Queue, mut encoder: wgpu::CommandEncoder) {
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
        let done = Rc::new(std::cell::Cell::new(false));
        let flag = done.clone();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| flag.set(result.is_ok()));
        self.in_flight = Some(done);
    }
}

thread_local! {
    static DEBUG_VIEWPORT_JUMP: std::cell::Cell<Option<(f64, f64, f64)>> =
        const { std::cell::Cell::new(None) };
    static DISTINCT_TILES_SEEN: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static APPLICATION: std::cell::RefCell<Option<gpui::ApplicationHandle>> =
        const { std::cell::RefCell::new(None) };
}

pub fn debug_jump_viewport(lon: f64, lat: f64, zoom: f64) {
    DEBUG_VIEWPORT_JUMP.set(Some((lon, lat, zoom)));
}

pub fn debug_wasm_memory_bytes() -> u32 {
    wasm_memory_bytes()
}

pub fn debug_distinct_tile_count() -> u32 {
    DISTINCT_TILES_SEEN.with(std::cell::Cell::get)
}

fn wasm_memory_bytes() -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        let memory = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
        let buffer = memory.buffer().unchecked_into::<js_sys::ArrayBuffer>();
        return buffer.byte_length();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

pub struct RgisWebApp {
    project: Project,
    style: Arc<StyleSheet>,
    vector_tile_fetcher: Arc<VectorTileFetcher>,
    glyph_fetcher: Arc<rgis_tiles::GlyphFetcher>,
    sprite_fetcher: Option<Arc<rgis_tiles::SpriteFetcher>>,
    sprite_atlas: Option<Arc<rgis_tiles::SpriteAtlas>>,
    raster_fetchers: raster::RasterFetchers,
    raster_tile_cache: raster::RasterTileCaches,
    gpu_basemap_meshes: HashMap<TileCoord, Arc<TileMesh>>,
    pending_tiles: HashSet<TileCoord>,
    vector_render_cache: rgis_render::VectorRenderCache,
    gpu: Rc<std::cell::RefCell<GpuState>>,
    resources: Option<MapRenderResources>,
    target: Option<MapTarget>,
    last_render_key: Option<[u64; 9]>,
    dragging: bool,
    last_cursor: Option<gpui::Point<gpui::Pixels>>,
    last_map_size: Option<(u32, u32)>,
    status: String,
    sidebar_visible: bool,
    layer_ui_state: LayerUiState,
    cursor_lonlat: Option<(f64, f64)>,
    bbox_zoom_start: Option<gpui::Point<gpui::Pixels>>,
}

impl RgisWebApp {
    fn new(_cx: &mut Context<Self>) -> Self {
        let mut project = Project::default();
        if let Ok(loaded) = rgis_io::load_bytes_with_crs(
            "sample.geojson",
            include_bytes!("../assets/sample.geojson"),
        )
        .and_then(|loaded| loaded.into_web_mercator())
        {
            let id = project.next_layer_id();
            let layer = Layer::new(id, loaded.name, loaded.features);
            project.add_layer(layer);
        }
        let style = Arc::new(
            StyleSheet::parse(include_str!("../../rgis-style/fixtures/liberty.json"))
                .expect("failed to parse embedded OpenFreeMap style"),
        );
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
            gpu_basemap_meshes: HashMap::new(),
            pending_tiles: HashSet::new(),
            vector_render_cache: rgis_render::VectorRenderCache::default(),
            gpu: Rc::new(std::cell::RefCell::new(GpuState::Idle)),
            resources: None,
            target: None,
            last_render_key: None,
            dragging: false,
            last_cursor: None,
            last_map_size: None,
            status: "GPUI browser renderer".to_string(),
            sidebar_visible: true,
            layer_ui_state: LayerUiState::default(),
            cursor_lonlat: None,
            bbox_zoom_start: None,
        }
    }

    fn apply_debug_viewport(&mut self) {
        if let Some((lon, lat, zoom)) = DEBUG_VIEWPORT_JUMP.take() {
            self.project.viewport.center = lonlat_to_mercator(lon, lat);
            self.project.viewport.zoom = zoom;
            self.status = format!("Viewport {lon:.4}, {lat:.4} · zoom {zoom:.2}");
        }
    }

    fn sidebar_offset(&self) -> f32 {
        if self.sidebar_visible {
            SIDEBAR_WIDTH
        } else {
            0.0
        }
    }

    fn process_basemap_tiles(&mut self) {
        raster::drain_fetchers(&self.raster_fetchers, &mut self.raster_tile_cache);
        for _ in 0..2 {
            let Ok(fetched) = self.vector_tile_fetcher.raw_receiver.try_recv() else {
                break;
            };
            self.pending_tiles.remove(&fetched.coord);
            if let Ok(tile) = self
                .vector_tile_fetcher
                .decode_and_cache(fetched.coord, &fetched.bytes)
            {
                let mesh = build_tile_mesh(&tile, fetched.coord, &self.style);
                self.gpu_basemap_meshes
                    .insert(fetched.coord, Arc::new(mesh.clone()));
                DISTINCT_TILES_SEEN.with(|count| count.set(count.get().saturating_add(1)));
            }
        }
    }

    fn request_basemap_tiles(&mut self, viewport: &Viewport) {
        for coord in visible_tiles_for_zoom(viewport, OPENFREEMAP_MAX_ZOOM) {
            if self.gpu_basemap_meshes.contains_key(&coord) || !self.pending_tiles.insert(coord) {
                continue;
            }
            self.vector_tile_fetcher.request(coord);
        }
    }

    fn map_callback(&mut self, width: f32, height: f32) -> MapCallback {
        self.process_basemap_tiles();
        if let Some(fetcher) = &self.sprite_fetcher
            && let Ok(ready) = fetcher.receiver.try_recv()
        {
            self.sprite_atlas = Some(ready.atlas);
        }
        self.project.viewport.width_px = width.max(1.0) as u32;
        self.project.viewport.height_px = height.max(1.0) as u32;

        let mut basemap_tiles = Vec::new();
        if self.project.show_tiles {
            let mut exact_tiles = Vec::new();
            let mut fallback_coords = HashSet::new();
            for coord in visible_tiles_for_zoom(&self.project.viewport, OPENFREEMAP_MAX_ZOOM) {
                if let Some(mesh) = self.gpu_basemap_meshes.get(&coord) {
                    exact_tiles.push((coord, Arc::clone(mesh)));
                } else if self.pending_tiles.insert(coord) {
                    self.vector_tile_fetcher.request(coord);
                } else {
                    for ancestor in raster::ancestor_coords(coord) {
                        if self.gpu_basemap_meshes.contains_key(&ancestor) {
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
                if let Some(mesh) = self.gpu_basemap_meshes.get(&coord) {
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
        if let Some(vector_tile) = self.vector_render_cache.render_with_preview(
            &self.project.layers,
            &self.project.viewport,
            self.dragging,
        ) {
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
        let background = if self.project.show_tiles {
            build_background_mesh(&self.project.viewport, &self.style)
        } else {
            rgis_render::SceneMesh::default()
        };
        MapCallback {
            background_index_count: background.indices.len() as u32,
            mesh: background,
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

    fn render_map(&mut self, window: &mut Window, width: f32, height: f32) {
        if matches!(*self.gpu.borrow(), GpuState::Idle) {
            start_gpu(self.gpu.clone());
        }
        let gpu_slot = self.gpu.clone();
        let gpu_state = gpu_slot.borrow();
        let gpu = match &*gpu_state {
            GpuState::Ready(gpu) => gpu,
            GpuState::Failed(message) => {
                self.status = message.clone();
                return;
            }
            _ => {
                // Device request is still in flight; try again next frame.
                window.on_next_frame(|_, cx| cx.refresh_windows());
                return;
            }
        };
        let (device, queue) = (&gpu.device, &gpu.queue);
        let size = (width.ceil().max(1.0) as u32, height.ceil().max(1.0) as u32);
        if self.target.as_ref().map(|target| target.size) != Some(size) {
            self.target = Some(MapTarget::new(device, size));
            self.last_render_key = None;
        }
        if self.resources.is_none() {
            self.resources = Some(MapRenderResources::new(device, MAP_FORMAT));
        }
        let target = self.target.as_mut().expect("map target created");
        if target.poll_readback(device) {
            window.on_next_frame(|_, cx| cx.refresh_windows());
            return;
        }
        let callback = self.map_callback(width, height);
        // Only redraw when something that affects the picture changed; the
        // readback is async, so an unconditional redraw would spin forever.
        let viewport = &self.project.viewport;
        let layers_key = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for layer in &self.project.layers {
                layer.id.hash(&mut hasher);
                layer.visible.hash(&mut hasher);
                let style = &layer.style;
                for value in [
                    style.fill.r,
                    style.fill.g,
                    style.fill.b,
                    style.fill.a,
                    style.stroke.r,
                    style.stroke.g,
                    style.stroke.b,
                    style.stroke.a,
                    style.stroke_width,
                    style.point_radius,
                ] {
                    value.to_bits().hash(&mut hasher);
                }
            }
            hasher.finish()
        };
        let key = [
            layers_key,
            viewport.center.x.to_bits(),
            viewport.center.y.to_bits(),
            viewport.zoom.to_bits(),
            callback.basemap_tiles.len() as u64,
            callback.tiles.len() as u64,
            callback.labels.len() as u64,
            callback.mesh.indices.len() as u64,
            callback.mesh.vertices.len() as u64,
        ];
        if self.last_render_key == Some(key) {
            return;
        }
        self.last_render_key = Some(key);
        let target = self.target.as_mut().expect("map target created");
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
        target.start_readback(queue, encoder);
        window.on_next_frame(|_, cx| cx.refresh_windows());
    }
}

impl Render for RgisWebApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        window.on_next_frame(|_, cx| cx.refresh_windows());
        self.apply_debug_viewport();
        let window_size = window.viewport_size();
        let scale = window.scale_factor();
        let window_width = f32::from(window_size.width);
        let sidebar_width = self.sidebar_offset();
        self.project.viewport.width_px =
            ((window_width - sidebar_width).max(1.0) * scale).round() as u32;
        self.project.viewport.height_px =
            ((f32::from(window_size.height) - STATUS_HEIGHT).max(1.0) * scale).round() as u32;
        let map_size = (
            self.project.viewport.width_px,
            self.project.viewport.height_px,
        );
        if self.last_map_size != Some(map_size) {
            let bounds = self
                .project
                .layers
                .iter()
                .filter_map(|layer| layer.bounds)
                .reduce(|left, right| left.union(&right));
            if let Some(bounds) = bounds {
                self.project.viewport.fit_bounds(&bounds);
            }
            self.last_map_size = Some(map_size);
        }
        let viewport = self.project.viewport.clone();
        self.request_basemap_tiles(&viewport);
        self.render_map(
            window,
            self.project.viewport.width_px as f32,
            self.project.viewport.height_px as f32,
        );
        let map_image = self.target.as_ref().and_then(|target| target.image.clone());
        let mut map_content = div().size_full().relative();
        if let Some(image) = map_image {
            map_content = map_content.child(
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
            );
        }
        let map = div()
            .flex_1()
            .bg(rgb(0xf8f4f0))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &gpui::MouseDownEvent, _, cx| {
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
                            let to_map = |point: gpui::Point<gpui::Pixels>| {
                                [
                                    ((f32::from(point.x) - this.sidebar_offset()) * scale)
                                        .clamp(0.0, this.project.viewport.width_px as f32),
                                    (f32::from(point.y) * scale)
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
                    let cursor = [
                        ((f32::from(event.position.x) - this.sidebar_offset()) * scale)
                            .clamp(0.0, this.project.viewport.width_px as f32),
                        (f32::from(event.position.y) * scale)
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
            .on_scroll(cx.listener(|this, event: &gpui::ScrollEvent, window, cx| {
                let delta = f32::from(event.delta.pixel_delta(px(16.0)).y) as f64 / 240.0;
                let scale = window.scale_factor();
                let cursor = [
                    ((f32::from(event.position.x) - this.sidebar_offset()) * scale)
                        .clamp(0.0, this.project.viewport.width_px as f32),
                    (f32::from(event.position.y) * scale)
                        .clamp(0.0, this.project.viewport.height_px as f32),
                ];
                this.project.viewport.zoom_toward(cursor, delta);
                cx.notify();
            }))
            .child(map_content);
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(cx.theme().tokens.background)
            .overflow_hidden()
            .child(
                div()
                    .flex_1()
                    .flex()
                    .when(self.sidebar_visible, |content| {
                        content.child(ui::sidebar(self, window, cx))
                    })
                    .child(map),
            )
            .child(ui::status_bar(self, cx))
    }
}

/// Directory the page is served from (for example `/rgis`), where trunk copies
/// the `assets/icons` that gpui-kit's wasm asset source downloads on demand.
fn asset_endpoint() -> String {
    // The loader hands this to `reqwest`, which needs an absolute URL.
    let Some(location) = web_sys::window().map(|window| window.location()) else {
        return String::new();
    };
    let origin = location.origin().unwrap_or_default();
    let path = location.pathname().unwrap_or_default();
    let dir = path.rsplit_once('/').map_or("", |(dir, _)| dir);
    format!("{origin}{dir}")
}

pub fn run() {
    gpui::web::init_logging();
    let platform = Rc::new(WebPlatform::new(false));
    let http_client = std::sync::Arc::new(platform.fetch_http_client());
    let application = gpui::Application::with_platform(platform)
        .with_http_client(http_client)
        .with_assets(gpui::assets::Assets::new(asset_endpoint()))
        .run_embedded(|cx: &mut App| {
            gpui::init(cx);
            gpui::component::Theme::change(gpui::component::ThemeMode::Dark, None, cx);
            cx.text_system()
                .add_fonts(vec![Cow::Borrowed(include_bytes!(
                    "../assets/fonts/IBMPlexSans-Regular.ttf"
                ))])
                .expect("failed to load embedded browser font");
            let bounds = Bounds::centered(None, size(px(1280.0), px(800.0)), cx);
            gpui::open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                cx,
                |_, cx| cx.new(RgisWebApp::new),
            )
            .expect("failed to open rgis browser window");
            cx.activate(true);
        });
    APPLICATION.with(|slot| *slot.borrow_mut() = Some(application));
}
