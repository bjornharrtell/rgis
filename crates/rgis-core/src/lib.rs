use geo_types::{Coord, Geometry, Rect};
use serde::{Deserialize, Serialize};

// ── IDs ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerId(pub u64);

// ── Style ─────────────────────────────────────────────────────────────────────

/// RGBA colour, components in [0.0, 1.0].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    pub const fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Style {
    pub fill: Color,
    pub stroke: Color,
    pub stroke_width: f32,
    pub point_radius: f32,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            fill: Color::rgba(0.3, 0.6, 0.9, 0.5),
            stroke: Color::rgba(0.1, 0.3, 0.7, 1.0),
            stroke_width: 1.5,
            point_radius: 5.0,
        }
    }
}

// ── Bounds ────────────────────────────────────────────────────────────────────

/// Geographic bounding box in the layer's native CRS (lon/lat for WGS-84).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Bounds {
    pub fn from_rect(r: Rect) -> Self {
        Self {
            min_x: r.min().x,
            min_y: r.min().y,
            max_x: r.max().x,
            max_y: r.max().y,
        }
    }

    pub fn center(&self) -> Coord {
        Coord {
            x: (self.min_x + self.max_x) * 0.5,
            y: (self.min_y + self.max_y) * 0.5,
        }
    }

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn union(&self, other: &Bounds) -> Bounds {
        Bounds {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }
}

// ── Viewport ──────────────────────────────────────────────────────────────────

/// Viewport in Web Mercator (EPSG:3857) metres.
///
/// `center` is in EPSG:3857 metres.
/// `zoom`   is a slippy-map zoom level (0 = whole world fits in 256 px).
/// `width_px` / `height_px` are the canvas size in device pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Viewport {
    pub center: Coord,
    pub zoom: f64,
    pub width_px: u32,
    pub height_px: u32,
}

/// Earth half-circumference in Web Mercator metres (EPSG:3857 extent).
pub const EARTH_HALF_CIRC: f64 = 20_037_508.342_789_244;

impl Viewport {
    /// Metres per pixel at the current zoom level (tile size = 256 px).
    pub fn resolution(&self) -> f64 {
        (2.0 * EARTH_HALF_CIRC) / (256.0 * 2_f64.powf(self.zoom))
    }

    /// Convert a Web Mercator coordinate to screen pixel position (top-left origin).
    pub fn world_to_screen(&self, world: Coord) -> [f32; 2] {
        let res = self.resolution();
        let cx = self.center.x;
        let cy = self.center.y;
        let x = (world.x - cx) / res + self.width_px as f64 * 0.5;
        // Y axis is flipped: north is up in mercator, down on screen.
        let y = (cy - world.y) / res + self.height_px as f64 * 0.5;
        [x as f32, y as f32]
    }

    /// Convert a screen pixel position to Web Mercator coordinate.
    pub fn screen_to_world(&self, px: [f32; 2]) -> Coord {
        let res = self.resolution();
        let cx = self.center.x;
        let cy = self.center.y;
        let x = cx + (px[0] as f64 - self.width_px as f64 * 0.5) * res;
        let y = cy - (px[1] as f64 - self.height_px as f64 * 0.5) * res;
        Coord { x, y }
    }

    /// Pan by `dx, dy` screen pixels.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let res = self.resolution();
        self.center.x -= dx as f64 * res;
        self.center.y += dy as f64 * res;
    }

    /// Zoom toward screen position `px` by `delta` zoom levels.
    pub fn zoom_toward(&mut self, px: [f32; 2], delta: f64) {
        let before = self.screen_to_world(px);
        self.zoom = (self.zoom + delta).clamp(0.0, 22.0);
        let after = self.screen_to_world(px);
        self.center.x += before.x - after.x;
        self.center.y += before.y - after.y;
    }

    /// Set center and zoom so that `bounds` (in EPSG:3857 metres) fills 80% of
    /// the viewport. Safe to call with zero-size bounds (single point/line).
    pub fn fit_bounds(&mut self, bounds: &Bounds) {
        self.center = bounds.center();
        let w = bounds.width().max(1.0);
        let h = bounds.height().max(1.0);
        let res_x = w / (self.width_px as f64 * 0.8);
        let res_y = h / (self.height_px as f64 * 0.8);
        let res = res_x.max(res_y);
        // zoom = log2( world_width / (tile_size_px * res) )
        self.zoom = ((2.0 * EARTH_HALF_CIRC) / (256.0 * res))
            .log2()
            .clamp(0.0, 22.0);
    }
}

impl Default for Viewport {
    /// Starts at zoom 2, centered on (0, 0) in EPSG:3857.
    fn default() -> Self {
        Self {
            center: Coord { x: 0.0, y: 0.0 },
            zoom: 2.0,
            width_px: 800,
            height_px: 600,
        }
    }
}

// ── Projection helpers ────────────────────────────────────────────────────────

/// Convert WGS-84 (lon, lat) degrees to EPSG:3857 metres.
pub fn lonlat_to_mercator(lon: f64, lat: f64) -> Coord {
    use std::f64::consts::PI;
    let x = lon.to_radians() * EARTH_HALF_CIRC / PI;
    let y = (lat.to_radians().tan() + 1.0 / lat.to_radians().cos())
        .ln()
        .clamp(-PI, PI)
        * EARTH_HALF_CIRC
        / PI;
    Coord { x, y }
}

/// Convert EPSG:3857 metres to WGS-84 (lon, lat) degrees.
pub fn mercator_to_lonlat(x: f64, y: f64) -> (f64, f64) {
    use std::f64::consts::PI;
    let lon = x * PI / EARTH_HALF_CIRC * 180.0 / PI;
    let lat = (2.0 * (y * PI / EARTH_HALF_CIRC).exp().atan() - PI / 2.0).to_degrees();
    (lon, lat)
}

// ── Feature ───────────────────────────────────────────────────────────────────

/// A single geographic feature with optional attribute properties.
#[derive(Debug, Clone)]
pub struct Feature {
    pub geometry: Geometry,
    pub properties: serde_json::Value,
}

// ── Layer ─────────────────────────────────────────────────────────────────────

/// A loaded data layer.
#[derive(Debug, Clone)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub source_path: Option<std::path::PathBuf>,
    pub features: Vec<Feature>,
    pub bounds: Option<Bounds>,
    pub style: Style,
    pub visible: bool,
    pub z_order: u32,
}

impl Layer {
    pub fn new(id: LayerId, name: impl Into<String>, features: Vec<Feature>) -> Self {
        let bounds = compute_bounds(&features);
        Self {
            id,
            name: name.into(),
            source_path: None,
            features,
            bounds,
            style: Style::default(),
            visible: true,
            z_order: 0,
        }
    }
}

fn compute_bounds(features: &[Feature]) -> Option<Bounds> {
    use geo::BoundingRect;
    let mut result: Option<Bounds> = None;
    for f in features {
        if let Some(rect) = f.geometry.bounding_rect() {
            let b = Bounds::from_rect(rect);
            result = Some(match result {
                None => b,
                Some(existing) => existing.union(&b),
            });
        }
    }
    result
}

// ── Project ───────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Project {
    pub layers: Vec<Layer>,
    pub viewport: Viewport,
    /// Whether the OSM tile background is visible.
    pub show_tiles: bool,
    next_id: u64,
}

impl Default for Project {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            viewport: Viewport::default(),
            show_tiles: true,
            next_id: 0,
        }
    }
}

impl Project {
    pub fn next_layer_id(&mut self) -> LayerId {
        let id = LayerId(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn add_layer(&mut self, mut layer: Layer) {
        layer.z_order = self.layers.len() as u32;
        self.layers.push(layer);
    }

    pub fn remove_layer(&mut self, id: LayerId) {
        self.layers.retain(|l| l.id != id);
    }

    pub fn get_layer_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_projection() {
        let (lon, lat) = (13.405_f64, 52.52_f64); // Berlin
        let merc = lonlat_to_mercator(lon, lat);
        let (lon2, lat2) = mercator_to_lonlat(merc.x, merc.y);
        assert!((lon - lon2).abs() < 1e-8, "lon round-trip: {lon} vs {lon2}");
        assert!((lat - lat2).abs() < 1e-8, "lat round-trip: {lat} vs {lat2}");
    }

    #[test]
    fn viewport_resolution_zoom0() {
        let vp = Viewport {
            zoom: 0.0,
            ..Default::default()
        };
        let expected = 2.0 * EARTH_HALF_CIRC / 256.0;
        let diff = (vp.resolution() - expected).abs();
        assert!(diff < 1e-6, "resolution at zoom 0: {diff}");
    }

    #[test]
    fn world_to_screen_round_trip() {
        let vp = Viewport {
            center: Coord { x: 0.0, y: 0.0 },
            zoom: 5.0,
            width_px: 800,
            height_px: 600,
        };
        let world = Coord { x: 100_000.0, y: -50_000.0 };
        let screen = vp.world_to_screen(world);
        let back = vp.screen_to_world(screen);
        assert!((world.x - back.x).abs() < 1.0, "x round-trip: delta={}", (world.x - back.x).abs());
        assert!((world.y - back.y).abs() < 1.0, "y round-trip: delta={}", (world.y - back.y).abs());
    }

    #[test]
    fn zoom_toward_preserves_cursor_position() {
        let mut vp = Viewport {
            center: Coord { x: 0.0, y: 0.0 },
            zoom: 5.0,
            width_px: 800,
            height_px: 600,
        };
        let cursor = [300.0_f32, 200.0_f32];
        let world_before = vp.screen_to_world(cursor);
        vp.zoom_toward(cursor, 1.0);
        let world_after = vp.screen_to_world(cursor);
        assert!(
            (world_before.x - world_after.x).abs() < 1.0,
            "cursor world x should be stable after zoom"
        );
        assert!(
            (world_before.y - world_after.y).abs() < 1.0,
            "cursor world y should be stable after zoom"
        );
    }
}
