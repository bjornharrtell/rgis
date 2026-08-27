//! Browser (wasm32) build of rgis.
//!
//! Renders GeoJSON layers on an HTML5 canvas using the 2D rendering context,
//! reusing `rgis-core` for viewport math (pan/zoom) and Web Mercator
//! projection so that behaviour matches the native desktop build.

use std::cell::RefCell;
use std::rc::Rc;

use geo::MapCoords;
use geo_types::Geometry;
use rgis_core::{Bounds, Feature, Layer, LayerId, Viewport, lonlat_to_mercator};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

/// A demo dataset (world borders, simplified) bundled with the wasm binary so
/// the map has something to show without requiring a file picker or a
/// network fetch.
const SAMPLE_GEOJSON: &str = include_str!("../assets/sample.geojson");

struct AppState {
    viewport: Viewport,
    layers: Vec<Layer>,
    dragging: bool,
    last_pointer: [f32; 2],
}

fn load_sample_layer() -> Layer {
    let geojson: geojson::GeoJson = SAMPLE_GEOJSON
        .parse()
        .expect("bundled sample.geojson must be valid GeoJSON");
    let collection = match geojson {
        geojson::GeoJson::FeatureCollection(fc) => fc,
        _ => panic!("bundled sample.geojson must be a FeatureCollection"),
    };

    let mut features = Vec::with_capacity(collection.features.len());
    for f in collection.features {
        let Some(geom_raw) = f.geometry else { continue };
        let geo_geom: Geometry = (&geom_raw)
            .try_into()
            .expect("bundled sample.geojson must contain valid geometries");
        let mercator = geo_geom.map_coords(|c| lonlat_to_mercator(c.x, c.y));
        let properties = f
            .properties
            .map(serde_json::Value::Object)
            .unwrap_or(serde_json::Value::Null);
        features.push(Feature {
            geometry: mercator,
            properties,
        });
    }

    Layer::new(LayerId(0), "sample", features)
}

fn canvas_and_ctx() -> (HtmlCanvasElement, CanvasRenderingContext2d) {
    let window = web_sys::window().expect("no global `window`");
    let document = window.document().expect("no document on window");
    let canvas = document
        .get_element_by_id("rgis-canvas")
        .expect("missing #rgis-canvas element")
        .dyn_into::<HtmlCanvasElement>()
        .expect("#rgis-canvas must be a <canvas>");
    let ctx = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();
    (canvas, ctx)
}

fn resize_canvas_to_container(canvas: &HtmlCanvasElement) -> (u32, u32) {
    let window = web_sys::window().unwrap();
    let width = window.inner_width().unwrap().as_f64().unwrap_or(800.0) as u32;
    let height = window.inner_height().unwrap().as_f64().unwrap_or(600.0) as u32;
    canvas.set_width(width);
    canvas.set_height(height);
    (width, height)
}

fn render(ctx: &CanvasRenderingContext2d, state: &AppState) {
    let vp = &state.viewport;
    let w = vp.width_px as f64;
    let h = vp.height_px as f64;

    ctx.set_fill_style_str("#eef2f5");
    ctx.fill_rect(0.0, 0.0, w, h);

    let mut layers: Vec<&Layer> = state.layers.iter().filter(|l| l.visible).collect();
    layers.sort_by_key(|l| l.z_order);

    for layer in layers {
        ctx.set_fill_style_str(&color_css(layer.style.fill));
        ctx.set_stroke_style_str(&color_css(layer.style.stroke));
        ctx.set_line_width(layer.style.stroke_width as f64);

        for feature in &layer.features {
            draw_geometry(ctx, vp, &feature.geometry, layer.style.point_radius as f64);
        }
    }
}

fn color_css(c: rgis_core::Color) -> String {
    format!(
        "rgba({}, {}, {}, {})",
        (c.r * 255.0) as u8,
        (c.g * 255.0) as u8,
        (c.b * 255.0) as u8,
        c.a
    )
}

fn draw_geometry(
    ctx: &CanvasRenderingContext2d,
    vp: &Viewport,
    geometry: &Geometry,
    point_radius: f64,
) {
    use geo_types::Geometry::*;
    match geometry {
        Point(p) => draw_point(ctx, vp, (p.x(), p.y()), point_radius),
        MultiPoint(mp) => {
            for p in mp {
                draw_point(ctx, vp, (p.x(), p.y()), point_radius);
            }
        }
        LineString(ls) => draw_line_string(ctx, vp, ls, false),
        MultiLineString(mls) => {
            for ls in mls {
                draw_line_string(ctx, vp, ls, false);
            }
        }
        Polygon(poly) => draw_polygon(ctx, vp, poly),
        MultiPolygon(mp) => {
            for poly in mp {
                draw_polygon(ctx, vp, poly);
            }
        }
        GeometryCollection(gc) => {
            for g in gc {
                draw_geometry(ctx, vp, g, point_radius);
            }
        }
        Line(l) => {
            ctx.begin_path();
            let [x0, y0] = vp.world_to_screen(l.start);
            let [x1, y1] = vp.world_to_screen(l.end);
            ctx.move_to(x0 as f64, y0 as f64);
            ctx.line_to(x1 as f64, y1 as f64);
            ctx.stroke();
        }
        Triangle(t) => {
            let ring: Vec<_> = [t.v1(), t.v2(), t.v3(), t.v1()]
                .into_iter()
                .map(|c| vp.world_to_screen(c))
                .collect();
            path_ring(ctx, &ring);
            ctx.fill();
            ctx.stroke();
        }
        Rect(r) => {
            let (min, max) = (r.min(), r.max());
            let corners = [
                min,
                geo_types::Coord { x: max.x, y: min.y },
                max,
                geo_types::Coord { x: min.x, y: max.y },
                min,
            ];
            let ring: Vec<_> = corners.into_iter().map(|c| vp.world_to_screen(c)).collect();
            path_ring(ctx, &ring);
            ctx.fill();
            ctx.stroke();
        }
    }
}

fn draw_point(ctx: &CanvasRenderingContext2d, vp: &Viewport, (x, y): (f64, f64), radius: f64) {
    let [sx, sy] = vp.world_to_screen(geo_types::Coord { x, y });
    ctx.begin_path();
    let _ = ctx.arc(sx as f64, sy as f64, radius, 0.0, std::f64::consts::TAU);
    ctx.fill();
    ctx.stroke();
}

fn draw_line_string(
    ctx: &CanvasRenderingContext2d,
    vp: &Viewport,
    ls: &geo_types::LineString,
    _closed: bool,
) {
    let points: Vec<_> = ls.coords().map(|c| vp.world_to_screen(*c)).collect();
    path_ring(ctx, &points);
    ctx.stroke();
}

fn draw_polygon(ctx: &CanvasRenderingContext2d, vp: &Viewport, poly: &geo_types::Polygon) {
    ctx.begin_path();
    add_ring(ctx, vp, poly.exterior());
    for interior in poly.interiors() {
        add_ring(ctx, vp, interior);
    }
    ctx.fill();
    ctx.stroke();
}

fn add_ring(ctx: &CanvasRenderingContext2d, vp: &Viewport, ring: &geo_types::LineString) {
    let mut coords = ring.coords();
    if let Some(first) = coords.next() {
        let [x, y] = vp.world_to_screen(*first);
        ctx.move_to(x as f64, y as f64);
        for c in coords {
            let [x, y] = vp.world_to_screen(*c);
            ctx.line_to(x as f64, y as f64);
        }
        ctx.close_path();
    }
}

fn path_ring(ctx: &CanvasRenderingContext2d, points: &[[f32; 2]]) {
    ctx.begin_path();
    let mut iter = points.iter();
    if let Some([x, y]) = iter.next() {
        ctx.move_to(*x as f64, *y as f64);
        for [x, y] in iter {
            ctx.line_to(*x as f64, *y as f64);
        }
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();

    let (canvas, ctx) = canvas_and_ctx();
    let (width, height) = resize_canvas_to_container(&canvas);

    let mut viewport = Viewport {
        width_px: width,
        height_px: height,
        ..Viewport::default()
    };
    let sample_layer = load_sample_layer();
    if let Some(bounds) = sample_layer.bounds {
        fit(&mut viewport, &bounds);
    }

    let state = Rc::new(RefCell::new(AppState {
        viewport,
        layers: vec![sample_layer],
        dragging: false,
        last_pointer: [0.0, 0.0],
    }));

    render(&ctx, &state.borrow());
    install_listeners(canvas, ctx, state);
}

fn fit(vp: &mut Viewport, bounds: &Bounds) {
    vp.fit_bounds(bounds);
}

fn install_listeners(
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    state: Rc<RefCell<AppState>>,
) {
    use wasm_bindgen::closure::Closure;

    {
        let state = state.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |ev: web_sys::MouseEvent| {
            let mut s = state.borrow_mut();
            s.dragging = true;
            s.last_pointer = [ev.offset_x() as f32, ev.offset_y() as f32];
        });
        canvas
            .add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    {
        let state = state.clone();
        let ctx = ctx.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |ev: web_sys::MouseEvent| {
            let mut s = state.borrow_mut();
            if !s.dragging {
                return;
            }
            let pos = [ev.offset_x() as f32, ev.offset_y() as f32];
            let dx = pos[0] - s.last_pointer[0];
            let dy = pos[1] - s.last_pointer[1];
            s.viewport.pan(dx, dy);
            s.last_pointer = pos;
            render(&ctx, &s);
        });
        canvas
            .add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    for event in ["mouseup", "mouseleave"] {
        let state = state.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |_ev: web_sys::MouseEvent| {
            state.borrow_mut().dragging = false;
        });
        canvas
            .add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    {
        let state = state.clone();
        let ctx = ctx.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |ev: web_sys::WheelEvent| {
            ev.prevent_default();
            let mut s = state.borrow_mut();
            let pos = [ev.offset_x() as f32, ev.offset_y() as f32];
            let delta = -ev.delta_y().signum() * 0.5;
            s.viewport.zoom_toward(pos, delta);
            render(&ctx, &s);
        });
        canvas
            .add_event_listener_with_callback("wheel", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    {
        let state = state.clone();
        let window = web_sys::window().unwrap();
        let closure = Closure::<dyn FnMut()>::new(move || {
            let (width, height) = resize_canvas_to_container(&canvas);
            let mut s = state.borrow_mut();
            s.viewport.width_px = width;
            s.viewport.height_px = height;
            render(&ctx, &s);
        });
        window
            .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }
}
