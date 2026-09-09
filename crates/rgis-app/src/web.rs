#![allow(dead_code)]

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    rc::Rc,
    sync::Arc,
};

use crate::labels;
use crate::raster;
use crate::ui::{self, LayerUi, LayerUiState, StyleColorTarget};
use gpui::{
    App, Bounds, Context, DevicePixels, MouseButton, Render, Window, WindowBounds, WindowOptions,
    div, prelude::*, px, rgb, size,
};
use gpui_web::WebPlatform;
use gpui_wgpu::{WgpuContextHandle, WgpuRenderTarget};
use rgis_core::{
    Bounds as GeoBounds, Color, Layer, LayerId, Project, Viewport, lonlat_to_mercator,
    mercator_to_lonlat,
};
use rgis_render::{
    MapCallback, MapRenderResources, StyleSheet, TileMesh, build_background_mesh, build_tile_mesh,
    render_vector_layers,
};
use rgis_tiles::{OPENFREEMAP_MAX_ZOOM, TileCoord, VectorTileFetcher, visible_tiles_for_zoom};
use wasm_bindgen::JsCast;
use wgpu;

const SIDEBAR_WIDTH: f32 = 280.0;
const STATUS_HEIGHT: f32 = 28.0;
const ZED_CANVAS: u32 = 0x121212;
const ZED_PANEL: u32 = 0x1b1b1b;
const ZED_TITLEBAR: u32 = 0x202020;
const ZED_SURFACE: u32 = 0x2a2a2a;
const ZED_BORDER: u32 = 0x343434;
const ZED_TEXT: u32 = 0xd4d4d4;
const ZED_MUTED: u32 = 0x929292;
const ZED_ACCENT: u32 = 0x8ab4f8;

struct MapTarget {
    target: WgpuRenderTarget,
    msaa_view: Option<wgpu::TextureView>,
    size: (u32, u32),
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

    fn cursor_lonlat(&self) -> Option<(f64, f64)> {
        self.cursor_lonlat
    }

    fn add_layer(&mut self, _window: &mut Window) {}
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

fn vector_draw_key(image: &image::RgbaImage) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    image.width().hash(&mut hasher);
    image.height().hash(&mut hasher);
    image.as_raw().hash(&mut hasher);
    hasher.finish() | (1 << 63)
}

fn icon(path: &str, color: u32) -> impl IntoElement {
    let data = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
<path d="{path}" fill="none" stroke="#{color:06x}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"##
    );
    gpui::svg()
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
    resources: Option<MapRenderResources>,
    target: Option<MapTarget>,
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
        if let Ok(loaded) =
            rgis_io::load_bytes("sample.geojson", include_bytes!("../assets/sample.geojson"))
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
            resources: None,
            target: None,
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
            .bg(gpui::rgba(color_to_rgba(color)))
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
        let target = self.layer_ui_state.style_color_target();
        let target_color = if target == StyleColorTarget::Fill {
            fill
        } else {
            stroke
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
        let stroke_width = layer.style.stroke_width;
        let point_radius = layer.style.point_radius;
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
                    .child(self.adjust_style_button(layer_id, target, false, cx))
                    .child(format!("{:.0}%", target_color.a * 100.0))
                    .child(self.adjust_style_button(layer_id, target, true, cx)),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(84.0)).child("Line width"))
                    .child(self.adjust_numeric_button(layer_id, false, cx))
                    .child(format!("{stroke_width:.1}"))
                    .child(self.adjust_numeric_button(layer_id, true, cx)),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(84.0)).child("Point radius"))
                    .child(self.adjust_radius_button(layer_id, false, cx))
                    .child(format!("{point_radius:.1}"))
                    .child(self.adjust_radius_button(layer_id, true, cx)),
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

    fn adjust_style_button(
        &self,
        layer_id: LayerId,
        target: StyleColorTarget,
        increase: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .w(px(24.0))
            .h(px(24.0))
            .flex()
            .items_center()
            .justify_center()
            .bg(rgb(ZED_SURFACE))
            .child(icon(
                if increase {
                    "M12 5v14M5 12h14"
                } else {
                    "M5 12h14"
                },
                ZED_MUTED,
            ))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                        let color = match target {
                            StyleColorTarget::Fill => &mut layer.style.fill,
                            StyleColorTarget::Stroke => &mut layer.style.stroke,
                        };
                        color.a = if increase {
                            (color.a + 0.05).min(1.0)
                        } else {
                            (color.a - 0.05).max(0.0)
                        };
                    }
                    cx.stop_propagation();
                    cx.notify();
                }),
            )
    }

    fn adjust_numeric_button(
        &self,
        layer_id: LayerId,
        increase: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .w(px(24.0))
            .h(px(24.0))
            .flex()
            .items_center()
            .justify_center()
            .bg(rgb(ZED_SURFACE))
            .child(icon(
                if increase {
                    "M12 5v14M5 12h14"
                } else {
                    "M5 12h14"
                },
                ZED_MUTED,
            ))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                        layer.style.stroke_width = if increase {
                            layer.style.stroke_width + 0.5
                        } else {
                            (layer.style.stroke_width - 0.5).max(0.1)
                        };
                    }
                    cx.stop_propagation();
                    cx.notify();
                }),
            )
    }

    fn adjust_radius_button(
        &self,
        layer_id: LayerId,
        increase: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .w(px(24.0))
            .h(px(24.0))
            .flex()
            .items_center()
            .justify_center()
            .bg(rgb(ZED_SURFACE))
            .child(icon(
                if increase {
                    "M12 5v14M5 12h14"
                } else {
                    "M5 12h14"
                },
                ZED_MUTED,
            ))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    if let Some(layer) = this.project.get_layer_mut(layer_id) {
                        layer.style.point_radius = if increase {
                            layer.style.point_radius + 1.0
                        } else {
                            (layer.style.point_radius - 1.0).max(1.0)
                        };
                    }
                    cx.stop_propagation();
                    cx.notify();
                }),
            )
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
        if let Some(rgba) = render_vector_layers(&self.project.layers, &self.project.viewport) {
            tiles.push(rgis_render::TileDraw {
                key: vector_draw_key(&rgba),
                rect: [0.0, 0.0, width, height],
                rgba: Arc::new(rgba),
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
        let size = (width.ceil().max(1.0) as u32, height.ceil().max(1.0) as u32);
        if self.target.as_ref().map(|target| target.size) != Some(size) {
            self.target = Some(MapTarget::new(&context, size));
            self.resources = Some(MapRenderResources::new(device, context.texture_format()));
        }
        let callback = self.map_callback(width, height);
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
    }

    fn sidebar(&self, width: f32, cx: &mut Context<Self>) -> impl IntoElement {
        let mut content = div()
            .w(px(width))
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
                    .child("LAYERS")
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
            for layer in self.project.layers.iter().rev() {
                content =
                    content.child(self.layer_row(layer.id, layer.name.clone(), layer.visible, cx));
                if self.layer_ui_state.style_editor_layer() == Some(layer.id) {
                    content = content.child(self.style_panel(layer.id, cx));
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
}

impl Render for RgisWebApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        window.request_animation_frame();
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
        let map_surface = self.target.as_ref().map(|target| target.target.surface());
        let mut map_content = div().size_full().relative();
        if let Some(surface) = map_surface {
            map_content = map_content.child(surface.size_full());
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
            .on_scroll_wheel(
                cx.listener(|this, event: &gpui::ScrollWheelEvent, window, cx| {
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
                }),
            )
            .child(map_content);
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(ZED_CANVAS))
            .overflow_hidden()
            .child(
                div()
                    .flex_1()
                    .flex()
                    .when(self.sidebar_visible, |content| {
                        content.child(ui::sidebar(self, cx))
                    })
                    .child(map),
            )
            .child(ui::status_bar(self, cx))
    }
}

pub fn run() {
    gpui_web::init_logging();
    let platform = Rc::new(WebPlatform::new(false));
    let http_client = std::sync::Arc::new(platform.fetch_http_client());
    let application = gpui::Application::with_platform(platform)
        .with_http_client(http_client)
        .run_embedded(|cx: &mut App| {
            cx.text_system()
                .add_fonts(vec![Cow::Borrowed(include_bytes!(
                    "../assets/fonts/IBMPlexSans-Regular.ttf"
                ))])
                .expect("failed to load embedded browser font");
            let bounds = Bounds::centered(None, size(px(1280.0), px(800.0)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_, cx| cx.new(RgisWebApp::new),
            )
            .expect("failed to open rgis browser window");
            cx.activate(true);
        });
    APPLICATION.with(|slot| *slot.borrow_mut() = Some(application));
}
