mod status_bar;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::PathBuf,
    rc::Rc,
    sync::Arc,
};

use gpui::{
    AnyElement, App, Application, AsyncApp, Bounds, ClickEvent, Context, Hsla, InteractiveElement,
    IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathPromptOptions,
    Pixels, Point, Render, RenderImage, ScrollDelta, ScrollWheelEvent, SharedString, Styled,
    TitlebarOptions, WeakEntity, Window, WindowBounds, WindowOptions, black, canvas, div,
    prelude::*, px, rgb, size,
};
use image::Frame;
use rgis_core::{Layer, LayerId, Project, Viewport, mercator_to_lonlat};
use rgis_render::{LayerPaths, build_project_paths_with_offset};
use rgis_tiles::{
    OsmTileSource, TileCoord, TileFetcher, TileReady, tile_screen_rect, visible_tiles,
};

const APP_ID: &str = "rs.rgis.app";
const SIDEBAR_WIDTH: f32 = 280.0;

#[derive(Default)]
struct DragState {
    active: bool,
    last_position: Option<Point<Pixels>>,
}

struct RgisApp {
    project: Project,
    tile_fetcher: Arc<TileFetcher>,
    tile_images: HashMap<TileCoord, Arc<RenderImage>>,
    pending_tiles: Rc<RefCell<HashSet<TileCoord>>>,
    cursor_lonlat: Option<(f64, f64)>,
    last_error: Option<String>,
    map_bounds: Rc<RefCell<Bounds<Pixels>>>,
    drag: DragState,
}

impl RgisApp {
    fn new(startup_paths: Vec<PathBuf>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        window.set_window_title("rgis");

        let tile_fetcher = Arc::new(TileFetcher::new(OsmTileSource));
        let map_bounds = Rc::new(RefCell::new(Bounds::new(
            gpui::point(px(SIDEBAR_WIDTH), px(0.0)),
            size(px(1000.0), px(764.0)),
        )));

        let this = Self {
            project: Project::default(),
            tile_fetcher: Arc::clone(&tile_fetcher),
            tile_images: HashMap::new(),
            pending_tiles: Rc::new(RefCell::new(HashSet::new())),
            cursor_lonlat: None,
            last_error: None,
            map_bounds,
            drag: DragState::default(),
        };

        Self::spawn_tile_listener(Arc::clone(&tile_fetcher), cx);
        cx.observe_window_bounds(window, |this, window, _| {
            window.set_window_title("rgis");
            this.sync_viewport_dimensions();
        })
        .detach();

        for path in startup_paths {
            this.queue_load_path(path, cx);
        }

        this
    }

    fn spawn_tile_listener(tile_fetcher: Arc<TileFetcher>, cx: &mut Context<Self>) {
        let receiver = tile_fetcher.receiver.clone();
        cx.spawn(move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                while let Ok(ready) = receiver.recv().await {
                    if this
                        .update(&mut cx, move |this, cx| {
                            this.on_tile_ready(ready);
                            cx.notify();
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            }
        })
        .detach();
    }

    fn on_tile_ready(&mut self, ready: TileReady) {
        self.pending_tiles.borrow_mut().remove(&ready.coord);
        self.tile_images
            .insert(ready.coord, rgba_to_render_image(&ready.image));
    }

    fn queue_load_path(&self, path: PathBuf, cx: &mut Context<Self>) {
        cx.spawn(move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                let load_path = path.clone();
                let result = tokio::spawn(async move { rgis_io::load(&load_path).await }).await;
                let _ = this.update(&mut cx, move |this, cx| {
                    this.finish_load(path, result);
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn finish_load(
        &mut self,
        path: PathBuf,
        result: Result<Result<rgis_io::LoadedLayer, rgis_io::IoError>, tokio::task::JoinError>,
    ) {
        match result {
            Ok(Ok(loaded)) => {
                self.sync_viewport_dimensions();
                let id = self.project.next_layer_id();
                let mut layer = Layer::new(id, loaded.name, loaded.features);
                layer.source_path = Some(path);
                self.project.add_layer(layer);
                if let Some(bounds) = self
                    .project
                    .layers
                    .iter()
                    .find(|layer| layer.id == id)
                    .and_then(|layer| layer.bounds)
                {
                    self.project.viewport.fit_bounds(&bounds);
                }
                self.last_error = None;
            }
            Ok(Err(error)) => {
                self.last_error = Some(format!("Failed to load {}: {error}", display_name(&path)));
            }
            Err(error) => {
                self.last_error = Some(format!(
                    "Layer load task failed for {}: {error}",
                    display_name(&path)
                ));
            }
        }
    }

    fn prompt_for_layer_paths(
        &mut self,
        _event: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let prompt = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some(SharedString::from("Add Layer…")),
        });

        cx.spawn(move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let mut cx = cx.clone();
            async move {
                let selected = match prompt.await {
                    Ok(Ok(paths)) => paths,
                    Ok(Err(error)) => {
                        let _ = this.update(&mut cx, move |this, cx| {
                            this.last_error = Some(format!("File picker failed: {error}"));
                            cx.notify();
                        });
                        return;
                    }
                    Err(_) => return,
                };

                let Some(paths) = selected else {
                    return;
                };

                let _ = this.update(&mut cx, move |this, cx| {
                    for path in paths {
                        this.queue_load_path(path, cx);
                    }
                });
            }
        })
        .detach();
    }

    fn toggle_tiles(&mut self, _event: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        self.project.show_tiles = !self.project.show_tiles;
        cx.notify();
    }

    fn toggle_layer(
        &mut self,
        id: LayerId,
        _event: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(layer) = self.project.get_layer_mut(id) {
            layer.visible = !layer.visible;
            cx.notify();
        }
    }

    fn remove_layer(
        &mut self,
        id: LayerId,
        _event: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.project.remove_layer(id);
        cx.notify();
    }

    fn on_map_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if event.button != MouseButton::Left {
            return;
        }
        if self.map_contains(event.position) {
            self.drag.active = true;
            self.drag.last_position = Some(self.window_to_map(event.position));
            self.update_cursor(event.position);
            cx.notify();
        }
    }

    fn on_map_mouse_up(
        &mut self,
        _event: &MouseUpEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.drag.active = false;
        self.drag.last_position = None;
        cx.notify();
    }

    fn on_map_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.map_contains(event.position) {
            self.cursor_lonlat = None;
            self.drag.last_position = None;
            cx.notify();
            return;
        }

        let local = self.window_to_map(event.position);
        self.update_cursor(event.position);

        if self.drag.active && event.dragging() {
            self.sync_viewport_dimensions();
            if let Some(previous) = self.drag.last_position.replace(local) {
                self.project.viewport.pan(
                    f32::from(local.x) - f32::from(previous.x),
                    f32::from(local.y) - f32::from(previous.y),
                );
            }
        } else {
            self.drag.last_position = Some(local);
        }

        cx.notify();
    }

    fn on_map_scroll(
        &mut self,
        event: &ScrollWheelEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.map_contains(event.position) {
            return;
        }

        let delta_y = match event.delta {
            ScrollDelta::Pixels(delta) => f32::from(delta.y),
            ScrollDelta::Lines(delta) => delta.y * 32.0,
        };
        if delta_y.abs() < f32::EPSILON {
            return;
        }

        self.sync_viewport_dimensions();
        let local = self.window_to_map(event.position);
        let zoom_delta = if delta_y < 0.0 { 0.25 } else { -0.25 };
        self.project
            .viewport
            .zoom_toward([f32::from(local.x), f32::from(local.y)], zoom_delta);
        self.update_cursor(event.position);
        cx.notify();
    }

    fn sync_viewport_dimensions(&mut self) {
        let bounds = *self.map_bounds.borrow();
        let width = f32::from(bounds.size.width).max(1.0).round() as u32;
        let height = f32::from(bounds.size.height).max(1.0).round() as u32;
        self.project.viewport.width_px = width;
        self.project.viewport.height_px = height;
    }

    fn prepared_viewport(&self) -> Viewport {
        let mut viewport = self.project.viewport.clone();
        let bounds = *self.map_bounds.borrow();
        viewport.width_px = f32::from(bounds.size.width).max(1.0).round() as u32;
        viewport.height_px = f32::from(bounds.size.height).max(1.0).round() as u32;
        viewport
    }

    fn map_contains(&self, position: Point<Pixels>) -> bool {
        self.map_bounds.borrow().contains(&position)
    }

    fn window_to_map(&self, position: Point<Pixels>) -> Point<Pixels> {
        let bounds = self.map_bounds.borrow();
        gpui::point(position.x - bounds.origin.x, position.y - bounds.origin.y)
    }

    fn update_cursor(&mut self, window_position: Point<Pixels>) {
        if !self.map_contains(window_position) {
            self.cursor_lonlat = None;
            return;
        }

        self.sync_viewport_dimensions();
        let local = self.window_to_map(window_position);
        let world = self
            .project
            .viewport
            .screen_to_world([f32::from(local.x), f32::from(local.y)]);
        self.cursor_lonlat = Some(mercator_to_lonlat(world.x, world.y));
    }

    fn layer_row(&self, layer: &Layer, cx: &mut Context<Self>) -> AnyElement {
        let id = layer.id;
        let label = format!("{} {}", checkbox_label(layer.visible), layer.name);
        div()
            .flex()
            .items_center()
            .justify_between()
            .gap_2()
            .px_2()
            .py_1()
            .border_b_1()
            .border_color(rgb(0x26303c))
            .child(
                div()
                    .flex_1()
                    .id(SharedString::from(format!("layer-toggle-{}", id.0)))
                    .cursor_pointer()
                    .child(label)
                    .on_click(cx.listener(move |this, event, window, cx| {
                        this.toggle_layer(id, event, window, cx);
                    })),
            )
            .child(
                div()
                    .id(SharedString::from(format!("layer-remove-{}", id.0)))
                    .cursor_pointer()
                    .text_color(Hsla::red())
                    .child("✕")
                    .on_click(cx.listener(move |this, event, window, cx| {
                        this.remove_layer(id, event, window, cx);
                    })),
            )
            .into_any_element()
    }

    fn render_sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut rows: Vec<AnyElement> = self
            .project
            .layers
            .iter()
            .rev()
            .map(|layer| self.layer_row(layer, cx))
            .collect();

        rows.push(
            div()
                .flex()
                .id("toggle-osm")
                .items_center()
                .px_2()
                .py_1()
                .border_b_1()
                .border_color(rgb(0x26303c))
                .cursor_pointer()
                .child(format!(
                    "{} OSM Background",
                    checkbox_label(self.project.show_tiles)
                ))
                .on_click(cx.listener(|this, event, window, cx| {
                    this.toggle_tiles(event, window, cx);
                }))
                .into_any_element(),
        );

        let mut sidebar = div()
            .w(px(SIDEBAR_WIDTH))
            .h_full()
            .flex()
            .flex_col()
            .bg(rgb(0x161d26))
            .overflow_hidden()
            .text_color(rgb(0xf4f7fb))
            .border_r_1()
            .border_color(rgb(0x26303c))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_3()
                    .py_2()
                    .bg(rgb(0x1d2631))
                    .child("Layers")
                    .child(
                        div()
                            .id("add-layer")
                            .cursor_pointer()
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0x2f80ed))
                            .child("Add Layer…")
                            .on_click(cx.listener(|this, event, window, cx| {
                                this.prompt_for_layer_paths(event, window, cx);
                            })),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .id("layer-list")
                    .overflow_y_scroll()
                    .children(rows),
            );

        if let Some(error) = &self.last_error {
            sidebar = sidebar.child(
                div()
                    .m_2()
                    .p_2()
                    .rounded_sm()
                    .bg(rgb(0x4a1f24))
                    .text_sm()
                    .child(error.clone()),
            );
        }

        sidebar
    }

    fn render_status_bar(&self) -> impl IntoElement {
        let coords = self
            .cursor_lonlat
            .map(|(lon, lat)| status_bar::format_coordinates(lon, lat))
            .unwrap_or_else(|| "—".to_string());
        let scale = status_bar::format_scale(self.prepared_viewport().resolution());
        div()
            .h(px(36.0))
            .w_full()
            .flex()
            .items_center()
            .justify_between()
            .px_3()
            .bg(rgb(0x14181d))
            .text_color(rgb(0xd8dee9))
            .text_sm()
            .child(div().child(coords))
            .child(div().child(scale))
            .child(div().child("EPSG:4326"))
    }

    fn render_map(&self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let map_bounds = Rc::clone(&self.map_bounds);
        let viewport = self.prepared_viewport();
        let map_origin = {
            let bounds = self.map_bounds.borrow();
            [f32::from(bounds.origin.x), f32::from(bounds.origin.y)]
        };
        let layer_paths =
            build_project_paths_with_offset(&self.project.layers, &viewport, map_origin)
                .unwrap_or_default();
        let tile_fetcher = Arc::clone(&self.tile_fetcher);
        let tile_images = self.tile_images.clone();
        let pending_tiles = Rc::clone(&self.pending_tiles);
        let show_tiles = self.project.show_tiles;

        div()
            .flex_1()
            .h_full()
            .bg(rgb(0x222b34))
            .child(
                canvas(
                    move |bounds, _window, _cx| {
                        *map_bounds.borrow_mut() = bounds;
                        let mut tiles = Vec::new();
                        let mut requested = Vec::new();
                        if show_tiles {
                            let mut pending = pending_tiles.borrow_mut();
                            for coord in visible_tiles(&viewport, &OsmTileSource) {
                                if let Some(image) = tile_images.get(&coord) {
                                    let [x, y, w, h] = tile_screen_rect(coord, &viewport);
                                    tiles.push((
                                        Arc::clone(image),
                                        Bounds::new(
                                            gpui::point(
                                                bounds.origin.x + px(x),
                                                bounds.origin.y + px(y),
                                            ),
                                            size(px(w), px(h)),
                                        ),
                                    ));
                                } else if pending.insert(coord) {
                                    requested.push(coord);
                                }
                            }
                        }
                        (tiles, requested)
                    },
                    move |bounds, (tiles, requested), window, _cx| {
                        for coord in requested {
                            tile_fetcher.request(coord);
                        }

                        window.paint_quad(gpui::fill(bounds, rgb(0x222b34)));

                        for (image, bounds) in &tiles {
                            let _ = window.paint_image(
                                *bounds,
                                Default::default(),
                                Arc::clone(image),
                                0,
                                false,
                            );
                        }

                        for layer in &layer_paths {
                            paint_layer_paths(window, layer);
                        }
                    },
                )
                .size_full(),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event, window, cx| this.on_map_mouse_down(event, window, cx)),
            )
            .on_mouse_move(
                cx.listener(|this, event, window, cx| this.on_map_mouse_move(event, window, cx)),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event, window, cx| this.on_map_mouse_up(event, window, cx)),
            )
            .on_scroll_wheel(
                cx.listener(|this, event, window, cx| this.on_map_scroll(event, window, cx)),
            )
    }
}

impl Render for RgisApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(black())
            .child(
                div()
                    .flex_1()
                    .flex()
                    .child(self.render_sidebar(cx))
                    .child(self.render_map(window, cx)),
            )
            .child(self.render_status_bar())
    }
}

fn checkbox_label(visible: bool) -> &'static str {
    if visible { "[x]" } else { "[ ]" }
}

fn display_name(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("layer")
        .to_string()
}

fn paint_layer_paths(window: &mut Window, layer: &LayerPaths) {
    for (path, color) in &layer.fills {
        window.paint_path(path.clone(), *color);
    }
    for (path, color) in &layer.strokes {
        window.paint_path(path.clone(), *color);
    }
    for (path, color) in &layer.points {
        window.paint_path(path.clone(), *color);
    }
}

fn rgba_to_render_image(image: &Arc<image::RgbaImage>) -> Arc<RenderImage> {
    let mut buffer = image.as_ref().clone();
    for pixel in buffer.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    Arc::new(RenderImage::new(vec![Frame::new(buffer)]))
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    let handle = runtime.handle().clone();
    let _ = std::thread::Builder::new()
        .name("rgis-tokio-runtime".into())
        .spawn(move || {
            runtime.block_on(std::future::pending::<()>());
        })
        .expect("failed to spawn tokio runtime thread");
    let guard = handle.enter();
    std::mem::forget(guard);

    let startup_paths: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    Application::new().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1280.0), px(800.0)), cx);
        cx.open_window(
            WindowOptions {
                app_id: Some(APP_ID.to_string()),
                focus: true,
                titlebar: Some(TitlebarOptions {
                    title: Some("rgis".into()),
                    appears_transparent: false,
                    traffic_light_position: None,
                }),
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            move |window, cx| {
                let startup_paths = startup_paths.clone();
                cx.new(|cx| RgisApp::new(startup_paths, window, cx))
            },
        )
        .unwrap();
        cx.on_window_closed(|cx| cx.quit()).detach();
        cx.activate(true);
    });
}
