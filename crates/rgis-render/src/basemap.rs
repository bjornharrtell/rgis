//! Tessellates decoded OpenFreeMap vector tiles (OpenMapTiles schema) into
//! the same `Vertex`/`SceneMesh` used for regular layers, via a small static
//! per-layer style table rather than a full MapLibre style-spec interpreter.
//!
//! Colors and widths are ported from OpenFreeMap's own "liberty" MapLibre
//! style (<https://tiles.openfreemap.org/styles/liberty>) so the basemap
//! looks close to the reference MapLibre rendering.

use bytemuck::{Pod, Zeroable};
use geo_types::{Geometry, LineString, Polygon};
use lyon::math::point;
use lyon::path::{Builder, Path};
use lyon::tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use rgis_core::{EARTH_HALF_CIRC, Viewport};
use rgis_tiles::{TileCoord, VectorFeature, VectorTile};

use crate::mesh::{SceneMesh, Vertex};

/// `liberty` style's `background` layer color.
const BACKGROUND: [f32; 4] = [0.973, 0.957, 0.941, 1.0];

/// Bottom-to-top paint order, mirroring OpenMapTiles-based basemaps
/// (positron/liberty/bright). Layers not listed here (e.g. road-label line
/// placement, house numbers) are skipped because this renderer currently
/// only extracts point labels (`place`/`poi`) plus vector geometry.
const LAYER_ORDER: &[&str] = &[
    "landcover",
    "landuse",
    "park",
    "water",
    "waterway",
    "aeroway",
    "building",
    "transportation",
    "boundary",
];

/// A basemap tile's tessellated geometry, split into fills (polygons —
/// scale naturally with zoom, like MapLibre fill layers) and lines (roads,
/// waterways, boundaries, polygon outlines — rendered via GPU-side
/// extrusion so their width stays in constant screen pixels rather than
/// stretching with the tile's own position scale; see [`LineVertex`]).
#[derive(Debug, Default, Clone)]
pub struct TileMesh {
    pub(crate) fill: SceneMesh,
    pub(crate) lines: LineMesh,
    /// Point labels (place names, POIs) extracted from the tile, in the
    /// same tile-local-metres space as `fill`/`lines`. Unlike fills/lines
    /// these aren't tessellated into triangles here -- the caller projects
    /// and shapes them into screen-space SDF glyph quads each frame, so
    /// this stays plain per-feature data.
    pub labels: Vec<TileLabel>,
}

impl TileMesh {
    /// (fill vertices, fill indices, line vertices, line indices) -- used by
    /// the `tile_mesh_byte_budget` regression test to measure per-tile
    /// tessellation output.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        (
            self.fill.vertices.len(),
            self.fill.indices.len(),
            self.lines.vertices.len(),
            self.lines.indices.len(),
        )
    }
}

/// A single point label (a `place` or `poi` layer feature with a `name`),
/// positioned in tile-local metres exactly like fill/line mesh vertices --
/// see [`tile_screen_transform`] for projecting it to screen space.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TileLabel {
    pub position: [f32; 2],
    pub text: String,
    /// Font size in constant screen pixels (labels don't scale with the
    /// tile, matching how map labels stay legible at any zoom).
    pub font_size: f32,
    pub color: [f32; 4],
    /// A lighter halo/outline color drawn behind the text for legibility
    /// over busy basemap tiles.
    pub halo_color: [f32; 4],
    /// Lower draws/keeps priority over higher when decluttering
    /// overlapping labels (place names before POIs; within a class, by the
    /// source layer's own `rank` property).
    pub priority: i32,
    /// Clockwise rotation in radians applied around `position` when laying
    /// out the label's glyphs (see `LabelGlyphInstance::angle`); `0.0` for
    /// point labels (place/poi), non-zero for road names, which follow
    /// their line's on-screen direction like MapLibre's `symbol-placement:
    /// line` road labels.
    pub angle: f32,
}

/// A line/stroke vertex: `center` is the tile-local-metres position
/// (transformed exactly like fill vertices), while `extrude` is a
/// direction+magnitude offset applied by the vertex shader in SCREEN
/// PIXELS after scaling `center` (see `shaders/basemap_line.wgsl`), so
/// line width stays constant in device pixels instead of stretching with
/// the tile's own position scale. `half_width` is the same (unmargined)
/// half-width the vertex's `extrude` was derived from -- signed on the two
/// sides of a straight segment (+ on the `extrude` side, - on the
/// `neg_extrude` side), unsigned (0 at the hub, +radius at the rim) for a
/// join/cap disc -- so the shader can push geometry out by a constant
/// device-pixel margin and compute a matching normalized distance for
/// analytic (MSAA-independent) edge antialiasing; see `basemap_line.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct LineVertex {
    pub center: [f32; 2],
    pub extrude: [f32; 2],
    pub half_width: f32,
    pub color: [f32; 4],
}

#[derive(Debug, Default, Clone)]
pub(crate) struct LineMesh {
    pub(crate) vertices: Vec<LineVertex>,
    pub(crate) indices: Vec<u32>,
}

/// A flat, structured-clone-friendly wire representation of a [`TileMesh`],
/// for shipping tessellation results across a Web Worker boundary (see
/// `rgis-app`'s wasm-only worker pool) as plain `Float32Array`/`Uint32Array`
/// typed arrays rather than `TileMesh` itself, whose fields are private to
/// this crate and which embeds no serialization support of its own.
#[derive(Debug, Default, Clone)]
pub struct TileMeshWire {
    /// Flattened `Vertex { position: [f32; 2], color: [f32; 4] }`, 6 floats
    /// per vertex.
    pub fill_vertices: Vec<f32>,
    pub fill_indices: Vec<u32>,
    /// Flattened `LineVertex { center: [f32; 2], extrude: [f32; 2],
    /// half_width: f32, color: [f32; 4] }`, 9 floats per vertex.
    pub line_vertices: Vec<f32>,
    pub line_indices: Vec<u32>,
    /// `TileMesh::labels`, JSON-encoded (`Vec<TileLabel>` isn't a flat
    /// numeric buffer like the vertex arrays above, but it's small -- at
    /// most a few hundred short strings per tile -- so plain JSON avoids
    /// hand-rolling a second binary wire format just for this).
    pub labels_json: String,
}

impl From<&TileMesh> for TileMeshWire {
    fn from(mesh: &TileMesh) -> Self {
        let mut fill_vertices = Vec::with_capacity(mesh.fill.vertices.len() * 6);
        for v in &mesh.fill.vertices {
            fill_vertices.extend_from_slice(&v.position);
            fill_vertices.extend_from_slice(&v.color);
        }
        let mut line_vertices = Vec::with_capacity(mesh.lines.vertices.len() * 9);
        for v in &mesh.lines.vertices {
            line_vertices.extend_from_slice(&v.center);
            line_vertices.extend_from_slice(&v.extrude);
            line_vertices.push(v.half_width);
            line_vertices.extend_from_slice(&v.color);
        }
        Self {
            fill_vertices,
            fill_indices: mesh.fill.indices.clone(),
            line_vertices,
            line_indices: mesh.lines.indices.clone(),
            labels_json: serde_json::to_string(&mesh.labels).unwrap_or_default(),
        }
    }
}

impl TileMeshWire {
    /// Reconstructs the [`TileMesh`] this wire form was built from (see
    /// `From<&TileMesh>`). Panics if the flat arrays' lengths aren't
    /// multiples of the expected per-vertex float count -- only expected to
    /// be called on data produced by `TileMeshWire::from`.
    pub fn into_tile_mesh(self) -> TileMesh {
        let fill_vertices = self
            .fill_vertices
            .as_chunks::<6>()
            .0
            .iter()
            .map(|c| Vertex {
                position: [c[0], c[1]],
                color: [c[2], c[3], c[4], c[5]],
            })
            .collect();
        let line_vertices = self
            .line_vertices
            .as_chunks::<9>()
            .0
            .iter()
            .map(|c| LineVertex {
                center: [c[0], c[1]],
                extrude: [c[2], c[3]],
                half_width: c[4],
                color: [c[5], c[6], c[7], c[8]],
            })
            .collect();
        let labels = serde_json::from_str(&self.labels_json).unwrap_or_default();
        TileMesh {
            fill: SceneMesh {
                vertices: fill_vertices,
                indices: self.fill_indices,
            },
            lines: LineMesh {
                vertices: line_vertices,
                indices: self.line_indices,
            },
            labels,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Paint {
    fill: Option<[f32; 4]>,
    /// Thin stroke around a filled polygon's own outline (buildings, parks).
    fill_outline: Option<[f32; 4]>,
    /// Wider stroke drawn under `stroke`, giving roads a two-tone casing.
    casing: Option<([f32; 4], f32)>,
    stroke: Option<([f32; 4], f32)>,
    /// Paint order within a layer: higher draws on top (e.g. motorways over
    /// residential streets at intersections).
    rank: u8,
}

/// Line-width scale factor relative to zoom 10 (where `base_width` values in
/// `style_for` are calibrated). Previously clamped to 1.0 below zoom 10, so
/// e.g. a motorway that just became visible around z5-7 was already drawn
/// at its full "high zoom" width — MapLibre's own line-width stops instead
/// keep shrinking below zoom 10 too (fading toward 0 as a road approaches
/// its own appear-zoom), so this now keeps scaling down continuously,
/// floored at 0 (fully invisible) rather than pinned at 1.0.
fn zoom_scale(zoom: f64) -> f32 {
    (1.0 + (zoom - 10.0) * 0.15).max(0.0) as f32
}

fn style_for(layer_name: &str, feature: &VectorFeature, zoom: f64) -> Option<Paint> {
    match layer_name {
        "landcover" => {
            let fill = match feature.get_str("class").unwrap_or("") {
                "wood" => [0.675, 0.891, 0.549, 0.28],
                "grass" => [0.690, 0.835, 0.604, 0.30],
                "ice" | "glacier" => [0.878, 0.925, 0.925, 0.80],
                "sand" => [0.969, 0.937, 0.765, 1.00],
                _ => return None,
            };
            Some(Paint {
                fill: Some(fill),
                ..Default::default()
            })
        }
        "landuse" => {
            let fill = match feature.get_str("class").unwrap_or("") {
                "residential" => [0.950, 0.890, 0.810, 0.50],
                "commercial" => [0.94, 0.87, 0.87, 0.60],
                "industrial" => [0.90, 0.87, 0.90, 0.50],
                "cemetery" => [0.845, 0.880, 0.740, 1.00],
                "hospital" => [1.00, 0.867, 0.933, 1.00],
                "school" => [0.925, 0.933, 0.800, 1.00],
                "pitch" | "track" => [0.871, 0.890, 0.804, 1.00],
                _ => return None,
            };
            Some(Paint {
                fill: Some(fill),
                ..Default::default()
            })
        }
        "park" => Some(Paint {
            fill: Some([0.847, 0.910, 0.784, 0.70]),
            ..Default::default()
        }),
        "water" => Some(Paint {
            fill: Some([0.620, 0.741, 1.000, 1.00]),
            ..Default::default()
        }),
        "waterway" => Some(Paint {
            stroke: Some(([0.627, 0.784, 0.941, 1.00], 1.0)),
            ..Default::default()
        }),
        "building" => {
            if zoom < 13.0 {
                return None;
            }
            Some(Paint {
                fill: Some([0.862, 0.852, 0.838, 1.00]),
                fill_outline: Some([0.803, 0.792, 0.777, 0.60]),
                ..Default::default()
            })
        }
        "aeroway" => {
            if zoom < 11.0 {
                return None;
            }
            match &feature.geometry {
                Geometry::Polygon(_) | Geometry::MultiPolygon(_) => Some(Paint {
                    fill: Some([0.898, 0.894, 0.878, 0.70]),
                    ..Default::default()
                }),
                Geometry::LineString(_) | Geometry::MultiLineString(_) => {
                    let width = match feature.get_str("class").unwrap_or("") {
                        "runway" => 3.0,
                        "taxiway" => 1.2,
                        _ => return None,
                    };
                    Some(Paint {
                        stroke: Some(([0.941, 0.929, 0.914, 1.00], width)),
                        ..Default::default()
                    })
                }
                _ => None,
            }
        }
        "boundary" => {
            let admin_level = feature.get_number("admin_level").unwrap_or(10.0) as i64;
            if admin_level <= 2 {
                Some(Paint {
                    stroke: Some(([0.40, 0.40, 0.42, 0.90], 1.4)),
                    rank: 1,
                    ..Default::default()
                })
            } else if admin_level <= 6 {
                if zoom < 5.0 {
                    return None;
                }
                Some(Paint {
                    stroke: Some(([0.70, 0.70, 0.70, 0.80], 0.8)),
                    ..Default::default()
                })
            } else {
                None
            }
        }
        "transportation" => {
            if !matches!(
                &feature.geometry,
                Geometry::LineString(_) | Geometry::MultiLineString(_)
            ) {
                // `road_area_pattern` (pedestrian plazas, etc): needs a
                // fill-pattern texture we don't support; skip.
                return None;
            }
            let casing_major = [0.914, 0.675, 0.467, 1.00];
            let casing_minor = [0.812, 0.804, 0.792, 1.00];
            // In MapLibre's own "liberty" style, a road class's casing/fill
            // pair don't fade in together: the casing line-width ramps up
            // from ~1-2 zoom levels *before* the fill line-width does (e.g.
            // `road_minor_casing` fades in from z12, but `road_minor` stays
            // width-0 until z13.5). So a road first appears as a single
            // plain-colored line, only gaining the two-tone "outline" look
            // once you zoom in further. `casing_min_zoom` approximates that
            // per-class delay.
            let (casing, fill, base_width, rank, casing_min_zoom) =
                match feature.get_str("class").unwrap_or("") {
                    "motorway" => (Some(casing_major), [1.000, 0.800, 0.533, 1.00], 2.2, 6, 7.0),
                    "trunk" => (Some(casing_major), [1.000, 0.933, 0.667, 1.00], 1.9, 5, 7.0),
                    "primary" => (Some(casing_major), [1.000, 0.933, 0.667, 1.00], 1.8, 4, 7.0),
                    "secondary" | "tertiary" => {
                        (Some(casing_major), [1.000, 0.933, 0.667, 1.00], 1.4, 3, 9.0)
                    }
                    "minor" => (Some(casing_minor), [1.0, 1.0, 1.0, 1.0], 1.0, 2, 14.0),
                    "service" | "track" => (Some(casing_minor), [1.0, 1.0, 1.0, 1.0], 0.6, 1, 16.0),
                    "path" | "pedestrian" => (None, [0.85, 0.85, 0.85, 1.0], 0.5, 0, 0.0),
                    "rail" | "transit" => (None, [0.733, 0.733, 0.733, 1.0], 0.8, 0, 0.0),
                    _ => return None,
                };
            let casing = if zoom >= casing_min_zoom {
                casing
            } else {
                None
            };
            Some(Paint {
                casing: casing.map(|c| (c, base_width + 0.8)),
                stroke: Some((fill, base_width)),
                rank,
                ..Default::default()
            })
        }
        _ => None,
    }
}

/// Tessellates a single decoded vector tile into fill + line meshes whose
/// vertex positions are in mercator METRES relative to the tile's own
/// top-left corner, NOT screen space — this makes the result independent
/// of the viewport, so callers can tessellate a tile once and cache it
/// indefinitely (see [`tile_screen_transform`] for the cheap per-frame
/// screen transform applied on the GPU).
pub fn build_tile_mesh(tile: &VectorTile, coord: TileCoord) -> TileMesh {
    let mut fill_buffers: VertexBuffers<Vertex, u32> = VertexBuffers::new();
    let mut fill_tess = FillTessellator::new();
    let mut lines = LineMesh::default();
    let zoom = coord.z as f64;
    let tile_size_m = TileMercatorBounds::for_coord(coord).size;

    for layer_name in LAYER_ORDER {
        let Some(layer) = tile.layers.iter().find(|l| &l.name == layer_name) else {
            continue;
        };
        let ctx = TileContext {
            extent: layer.extent,
            tile_size_m,
        };
        let mut styled: Vec<(&VectorFeature, Paint)> = Vec::new();
        for feature in &layer.features {
            if let Some(mut paint) = style_for(layer_name, feature, zoom) {
                // Every fill gets a thin same-color edge by default (unless
                // a layer already styled its own, e.g. buildings) so its
                // tessellated boundary gets the same analytic antialiasing
                // as stroked lines (see `LineVertex`), instead of a raw
                // jaggy triangle edge -- mirrors MapLibre's default
                // `fill-antialias` behavior.
                if paint.fill.is_some() && paint.fill_outline.is_none() {
                    paint.fill_outline = paint.fill;
                }
                styled.push((feature, paint));
            }
        }
        // Draw more important features (e.g. motorways) last within each
        // pass, so they render on top of less important ones at junctions.
        styled.sort_by_key(|(_, paint)| paint.rank);

        for (feature, paint) in &styled {
            if let Some(color) = paint.fill {
                append_fill(&mut fill_buffers, &mut fill_tess, feature, &ctx, color);
            }
        }
        for (feature, paint) in &styled {
            if let Some(color) = paint.fill_outline {
                append_outline(&mut lines, feature, &ctx, color, 1.0);
            }
        }
        for (feature, paint) in &styled {
            if let Some((color, width)) = paint.casing {
                append_line(&mut lines, feature, &ctx, color, width);
            }
        }
        for (feature, paint) in &styled {
            if let Some((color, width)) = paint.stroke {
                append_line(&mut lines, feature, &ctx, color, width);
            }
        }
    }

    let labels = extract_labels(tile, coord, tile_size_m, &mut lines);

    TileMesh {
        fill: SceneMesh {
            vertices: fill_buffers.vertices,
            indices: fill_buffers.indices,
        },
        lines,
        labels,
    }
}

/// OpenMapTiles `place` layer classes we label, from most to least
/// prominent, mirroring roughly how MapLibre's "liberty" style ranks them
/// (used both for the minimum zoom a class appears at and as a tie-break
/// alongside the feature's own `rank` property for decluttering priority).
fn place_label_style(feature: &VectorFeature, zoom: f64) -> Option<(f32, i32)> {
    let class = feature.get_str("class").unwrap_or("");
    let rank = feature.get_number("rank").unwrap_or(20.0) as i32;
    let (min_zoom, font_size, class_priority) = match class {
        "country" => (0.0, 15.0, 0),
        "state" => (4.0, 12.0, 1),
        "city" => (3.0, 14.0, 1),
        "town" => (6.0, 12.0, 2),
        "village" => (9.0, 11.0, 3),
        "hamlet" | "suburb" | "neighbourhood" => (11.0, 10.0, 4),
        _ => (12.0, 10.0, 5),
    };
    if zoom < min_zoom {
        return None;
    }
    Some((font_size, class_priority * 1000 + rank))
}

/// OpenMapTiles `poi` layer: gated by both zoom and the feature's own
/// `rank` (lower = more important), mirroring the "liberty" style's
/// `poi_r1`/`poi_r7`/`poi_r20` layers (rank 1-6 from z15, 7-19 from z16,
/// 20+ only from z17) rather than showing every POI regardless of rank
/// once some single zoom threshold is reached -- without this tiering,
/// far more POIs show up per zoom level than the reference style/MapLibre
/// client renders, which is why labels looked over-dense.
fn poi_label_style(feature: &VectorFeature, zoom: f64) -> Option<(f32, i32)> {
    let rank = feature.get_number("rank").unwrap_or(30.0) as i32;
    let min_zoom = if rank >= 20 {
        17.0
    } else if rank >= 7 {
        16.0
    } else {
        15.0
    };
    if zoom < min_zoom {
        return None;
    }
    Some((10.0, 10_000 + rank))
}

/// A per-feature label styling function: returns `(font size, decluttering
/// priority)`, or `None` to skip the feature.
type LabelStyleFn = fn(&VectorFeature, f64) -> Option<(f32, i32)>;

/// Constant on-screen radius (device pixels) of the small dot marker drawn
/// under `place`/`poi` point labels, mirroring the "liberty" style's
/// `circle_11_black` sprite / POI icon dots (see `marker_style` below).
const MARKER_RADIUS_PX: f32 = 2.5;
const MARKER_FILL: [f32; 4] = [0.13, 0.13, 0.13, 1.0];
const MARKER_HALO: [f32; 4] = [1.0, 1.0, 1.0, 0.9];

/// Whether a `place`/`poi` feature should get a small dot marker alongside
/// its label, mirroring the reference style's `icon-image` rules: `place`
/// features only show a generic dot for `village`/`town`/`city` classes and
/// only below zoom 10 (the style swaps to `''`, i.e. no icon, at z>=10);
/// `poi` features always show an icon (we approximate every POI class/
/// subclass sprite with a plain dot, since this renderer has no icon-sprite
/// atlas yet).
fn marker_style(layer_name: &str, feature: &VectorFeature, zoom: f64) -> bool {
    match layer_name {
        "place" => {
            let class = feature.get_str("class").unwrap_or("");
            matches!(class, "village" | "town" | "city") && zoom < 10.0
        }
        "poi" => true,
        _ => false,
    }
}

/// Extracts point labels (named `place`/`poi` features) from a decoded
/// tile, in the same tile-local-metres space `build_tile_mesh` uses for its
/// fill/line vertices, appending a small dot marker per labeled point into
/// `lines` (see `marker_style`).
fn extract_labels(
    tile: &VectorTile,
    coord: TileCoord,
    tile_size_m: f64,
    lines: &mut LineMesh,
) -> Vec<TileLabel> {
    let zoom = coord.z as f64;
    // (layer name, halo color, per-feature style fn).
    let sources: [(&str, [f32; 4], LabelStyleFn); 2] = [
        ("place", [1.0, 1.0, 1.0, 0.9], place_label_style),
        ("poi", [1.0, 1.0, 1.0, 0.85], poi_label_style),
    ];

    let mut labels = Vec::new();
    for (layer_name, halo_color, style_fn) in sources {
        let Some(layer) = tile.layers.iter().find(|l| l.name == layer_name) else {
            continue;
        };
        let ctx = TileContext {
            extent: layer.extent,
            tile_size_m,
        };
        for feature in &layer.features {
            let Geometry::Point(p) = &feature.geometry else {
                continue;
            };
            let Some(text) = feature.get_str("name").filter(|s| !s.is_empty()) else {
                continue;
            };
            let Some((font_size, priority)) = style_fn(feature, zoom) else {
                continue;
            };
            let pos = ctx.project_point(p.x(), p.y());
            if marker_style(layer_name, feature, zoom) {
                append_disc(lines, [pos.x, pos.y], MARKER_RADIUS_PX + 1.0, MARKER_HALO);
                append_disc(lines, [pos.x, pos.y], MARKER_RADIUS_PX, MARKER_FILL);
            }
            labels.push(TileLabel {
                position: [pos.x, pos.y],
                text: text.to_string(),
                font_size,
                color: [0.15, 0.15, 0.17, 1.0],
                halo_color,
                priority,
                angle: 0.0,
            });
        }
    }
    labels.extend(extract_road_labels(tile, coord, tile_size_m));
    labels.sort_by_key(|l| l.priority);
    labels
}

/// OpenMapTiles `transportation_name` layer: mirrors the "liberty" style's
/// `highway-name-major`/`highway-name-minor`/`highway-name-path` layers
/// (major roads visible from z12.2, minor/service/track from z15, paths
/// from z15.5) rather than showing every road name regardless of class.
fn road_label_style(feature: &VectorFeature, zoom: f64) -> Option<(f32, i32)> {
    let class = feature.get_str("class").unwrap_or("");
    let (min_zoom, font_size, class_priority) = match class {
        "motorway" | "trunk" | "primary" | "secondary" | "tertiary" => (12.2, 11.0, 0),
        "minor" | "service" | "track" => (15.0, 10.0, 1),
        "path" => (15.5, 9.0, 2),
        _ => (14.0, 10.0, 3),
    };
    if zoom < min_zoom {
        return None;
    }
    Some((font_size, class_priority))
}

/// Extracts road-name labels (`transportation_name` layer) from a decoded
/// tile, placing each label at the midpoint of its line's longest segment
/// and rotating it to follow that segment's on-screen direction, like
/// MapLibre's `symbol-placement: line` road labels -- unlike `place`/`poi`
/// labels (see `extract_labels`), these aren't drawn axis-aligned.
fn extract_road_labels(tile: &VectorTile, coord: TileCoord, tile_size_m: f64) -> Vec<TileLabel> {
    let zoom = coord.z as f64;
    let Some(layer) = tile.layers.iter().find(|l| l.name == "transportation_name") else {
        return Vec::new();
    };
    let ctx = TileContext {
        extent: layer.extent,
        tile_size_m,
    };
    let mut labels = Vec::new();
    for feature in &layer.features {
        let Some(text) = feature.get_str("name").filter(|s| !s.is_empty()) else {
            continue;
        };
        let Some((font_size, class_priority)) = road_label_style(feature, zoom) else {
            continue;
        };
        let points = match &feature.geometry {
            Geometry::LineString(line) => line_points(line, &ctx),
            Geometry::MultiLineString(lines) => lines
                .0
                .iter()
                .max_by(|a, b| line_length(a, &ctx).total_cmp(&line_length(b, &ctx)))
                .map(|line| line_points(line, &ctx))
                .unwrap_or_default(),
            _ => continue,
        };
        if points.len() < 2 {
            continue;
        }
        // Longest single segment, so the label sits on the straightest run
        // of road rather than spanning a sharp bend.
        let (mut a, mut b, mut best_len) = (points[0], points[1], 0.0f32);
        for pair in points.windows(2) {
            let [p0, p1] = [pair[0], pair[1]];
            let len = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
            if len > best_len {
                best_len = len;
                a = p0;
                b = p1;
            }
        }
        let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let mut angle = (b[1] - a[1]).atan2(b[0] - a[0]);
        // Keep text upright (never upside-down) by flipping direction when
        // it would otherwise point into the left half-plane.
        if angle > std::f32::consts::FRAC_PI_2 || angle < -std::f32::consts::FRAC_PI_2 {
            angle += std::f32::consts::PI;
        }
        labels.push(TileLabel {
            position: mid,
            text: text.to_string(),
            font_size,
            color: [0.15, 0.15, 0.17, 1.0],
            halo_color: [1.0, 1.0, 1.0, 0.9],
            priority: 20_000 + class_priority,
            angle,
        });
    }
    labels
}

fn line_length(line: &LineString<i32>, ctx: &TileContext) -> f32 {
    let points = line_points(line, ctx);
    points
        .windows(2)
        .map(|pair| ((pair[1][0] - pair[0][0]).powi(2) + (pair[1][1] - pair[0][1]).powi(2)).sqrt())
        .sum()
}

/// The small per-tile screen transform needed to position an already
/// tessellated (tile-local metres) [`build_tile_mesh`] result on screen:
/// `screen = local * scale + offset`. Computing this is pure arithmetic (no
/// tessellation, no per-vertex work), so unlike building the tile mesh
/// itself, it's cheap enough to recompute every frame, including during
/// pan/zoom — the actual per-vertex transform happens on the GPU (see
/// `rgis-render/src/shaders/basemap.wgsl`), not on the CPU.
pub struct TileTransform {
    pub offset: [f32; 2],
    pub scale: f32,
    /// Multiplier applied to line/stroke `extrude` offsets (see
    /// [`LineVertex`]), derived from the *current* viewport zoom rather
    /// than the tile's own zoom — this is what makes line width respond
    /// smoothly and continuously to zoom instead of stretching with the
    /// tile's position scale or snapping when a new tile loads.
    pub width_scale: f32,
    /// On-screen width/height (the tile is square) in the same units as
    /// `offset`, used to scissor-clip each tile's draws to its own bounds
    /// so MVT buffer-zone geometry (features duplicated a little past the
    /// tile edge, so strokes don't get cut off mid-width at the seam)
    /// doesn't get drawn twice where adjacent tiles overlap.
    pub size: f32,
}

pub fn tile_screen_transform(coord: TileCoord, viewport: &Viewport) -> TileTransform {
    let bounds = TileMercatorBounds::for_coord(coord);
    let offset = viewport.world_to_screen(geo_types::Coord {
        x: bounds.left,
        y: bounds.top,
    });
    let scale = (1.0 / viewport.resolution()) as f32;
    TileTransform {
        offset,
        scale,
        width_scale: zoom_scale(viewport.zoom),
        size: bounds.size as f32 * scale,
    }
}

/// A full-viewport quad in the `liberty` style's `background` color, drawn
/// beneath the basemap tiles.
pub fn build_background_mesh(viewport: &Viewport) -> SceneMesh {
    let w = viewport.width_px as f32;
    let h = viewport.height_px as f32;
    SceneMesh {
        vertices: vec![
            Vertex {
                position: [0.0, 0.0],
                color: BACKGROUND,
            },
            Vertex {
                position: [w, 0.0],
                color: BACKGROUND,
            },
            Vertex {
                position: [w, h],
                color: BACKGROUND,
            },
            Vertex {
                position: [0.0, h],
                color: BACKGROUND,
            },
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

#[derive(Clone, Copy)]
struct TileMercatorBounds {
    left: f64,
    top: f64,
    size: f64,
}

impl TileMercatorBounds {
    fn for_coord(coord: TileCoord) -> Self {
        let n = 2_f64.powi(coord.z as i32);
        let size = 2.0 * EARTH_HALF_CIRC / n;
        Self {
            left: coord.x as f64 * size - EARTH_HALF_CIRC,
            top: EARTH_HALF_CIRC - coord.y as f64 * size,
            size,
        }
    }
}

/// Bundles the per-tile state needed to project tile-local coordinates into
/// metres relative to the tile's own origin, so helper functions don't need
/// a growing list of separate arguments.
#[derive(Clone, Copy)]
struct TileContext {
    extent: u32,
    tile_size_m: f64,
}

impl TileContext {
    fn project_point(&self, local_x: i32, local_y: i32) -> lyon::math::Point {
        let fx = local_x as f64 / self.extent as f64;
        let fy = local_y as f64 / self.extent as f64;
        point(
            (fx * self.tile_size_m) as f32,
            (fy * self.tile_size_m) as f32,
        )
    }
}

fn append_fill(
    buffers: &mut VertexBuffers<Vertex, u32>,
    fill_tess: &mut FillTessellator,
    feature: &VectorFeature,
    ctx: &TileContext,
    color: [f32; 4],
) {
    match &feature.geometry {
        Geometry::Polygon(polygon) => fill_polygon(buffers, fill_tess, polygon, ctx, color),
        Geometry::MultiPolygon(polygons) => {
            for polygon in &polygons.0 {
                fill_polygon(buffers, fill_tess, polygon, ctx, color);
            }
        }
        _ => {}
    }
}

fn fill_polygon(
    buffers: &mut VertexBuffers<Vertex, u32>,
    fill_tess: &mut FillTessellator,
    polygon: &Polygon<i32>,
    ctx: &TileContext,
    color: [f32; 4],
) {
    if let Some(path) = ring_path(polygon, ctx) {
        fill_path(buffers, fill_tess, path, color);
    }
}

fn append_outline(
    buffers: &mut LineMesh,
    feature: &VectorFeature,
    ctx: &TileContext,
    color: [f32; 4],
    width_px: f32,
) {
    // No join/cap discs: this is the automatic 1px fill-antialiasing edge
    // added to every polygon feature (see `build_tile_mesh`), so with
    // potentially thousands of many-cornered polygons per tile (buildings,
    // landuse, water...) the disc-per-corner cost used for genuinely
    // visible stroked lines below would dominate tile memory for an
    // effect that's imperceptible at 1px -- see repo memory notes on the
    // OOM investigation.
    let mut outline = |polygon: &Polygon<i32>| {
        append_polyline(
            buffers,
            &ring_points(polygon.exterior(), ctx),
            color,
            width_px,
            false,
        );
        for ring in polygon.interiors() {
            append_polyline(buffers, &ring_points(ring, ctx), color, width_px, false);
        }
    };
    match &feature.geometry {
        Geometry::Polygon(polygon) => outline(polygon),
        Geometry::MultiPolygon(polygons) => polygons.0.iter().for_each(outline),
        _ => {}
    }
}

/// Projected points of a polygon ring, closed (first point repeated at the
/// end) so [`append_polyline`]'s segment loop naturally covers the closing
/// edge too.
fn ring_points(ring: &LineString<i32>, ctx: &TileContext) -> Vec<[f32; 2]> {
    let mut points = line_points(ring, ctx);
    if let Some(&first) = points.first() {
        points.push(first);
    }
    points
}

/// Builds a `Path` covering a polygon's exterior + interior rings, for
/// fill tessellation.
fn ring_path(polygon: &Polygon<i32>, ctx: &TileContext) -> Option<Path> {
    let mut builder = Path::builder();
    let mut any_ring = build_ring(&mut builder, polygon.exterior(), ctx);
    for ring in polygon.interiors() {
        any_ring |= build_ring(&mut builder, ring, ctx);
    }
    any_ring.then(|| builder.build())
}

fn build_ring(builder: &mut Builder, ring: &LineString<i32>, ctx: &TileContext) -> bool {
    let mut coords = ring.coords();
    let Some(first) = coords.next() else {
        return false;
    };
    builder.begin(ctx.project_point(first.x, first.y));
    for coord in coords {
        builder.line_to(ctx.project_point(coord.x, coord.y));
    }
    builder.end(true);
    true
}

fn append_line(
    buffers: &mut LineMesh,
    feature: &VectorFeature,
    ctx: &TileContext,
    color: [f32; 4],
    width_px: f32,
) {
    match &feature.geometry {
        Geometry::LineString(line) => {
            append_polyline(buffers, &line_points(line, ctx), color, width_px, true)
        }
        Geometry::MultiLineString(lines) => {
            for line in &lines.0 {
                append_polyline(buffers, &line_points(line, ctx), color, width_px, true);
            }
        }
        _ => {}
    }
}

fn line_points(line: &LineString<i32>, ctx: &TileContext) -> Vec<[f32; 2]> {
    line.coords()
        .map(|c| {
            let p = ctx.project_point(c.x, c.y);
            [p.x, p.y]
        })
        .collect()
}

/// Tessellates a polyline into a screen-pixel-width "ribbon": each segment
/// becomes a quad extruded perpendicular to its direction, and a small
/// round disc is added at the two endpoints (round caps) plus any interior
/// point where the path actually changes direction (a real join) — see
/// the loop below for why collinear interior points can safely skip the
/// disc entirely, not just approximately.
/// `center` stays in tile-local metres (scaled like fill vertices);
/// `extrude` is a direction+magnitude offset in that same local space,
/// applied in SCREEN PIXELS by the shader (see [`LineVertex`]) — since the
/// tile's own position transform is a uniform (isotropic) scale, a
/// direction computed here is identical to its screen-space direction, so
/// this is valid even though `extrude`'s on-screen magnitude is meant to
/// stay constant regardless of that scale.
fn append_polyline(
    buffers: &mut LineMesh,
    points: &[[f32; 2]],
    color: [f32; 4],
    width_px: f32,
    with_joins: bool,
) {
    if points.len() < 2 || width_px <= 0.0 {
        return;
    }
    let half_width = width_px * 0.5;

    // One entry per segment; `None` marks a degenerate (near-zero-length)
    // segment, in which case the neighboring joints are conservatively
    // treated as real joins below (rather than trying to divide by ~0).
    let mut directions: Vec<Option<[f32; 2]>> = Vec::with_capacity(points.len() - 1);
    for pair in points.windows(2) {
        let (p0, p1) = (pair[0], pair[1]);
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len <= f32::EPSILON {
            directions.push(None);
            continue;
        }
        let dir = [dx / len, dy / len];
        directions.push(Some(dir));
        let extrude = [-dir[1] * half_width, dir[0] * half_width];
        let neg_extrude = [-extrude[0], -extrude[1]];
        let base = buffers.vertices.len() as u32;
        buffers.vertices.push(LineVertex {
            center: p0,
            extrude,
            half_width,
            color,
        });
        buffers.vertices.push(LineVertex {
            center: p0,
            extrude: neg_extrude,
            half_width: -half_width,
            color,
        });
        buffers.vertices.push(LineVertex {
            center: p1,
            extrude,
            half_width,
            color,
        });
        buffers.vertices.push(LineVertex {
            center: p1,
            extrude: neg_extrude,
            half_width: -half_width,
            color,
        });
        buffers.indices.extend_from_slice(&[
            base,
            base + 1,
            base + 2,
            base + 1,
            base + 3,
            base + 2,
        ]);
    }

    if !with_joins {
        return;
    }

    // No cap discs: this matches MapLibre's own default `line-cap: butt`
    // -- a flat end needs no extra geometry at all, since the segment
    // quad's own perpendicular edge already forms it. Real-world OSM ways
    // are frequently split into many short adjoining LineString features
    // (way segments), so a per-endpoint round cap disc would multiply
    // cost by roughly 2x the total line count for a visual effect
    // MapLibre itself doesn't apply here.
    //
    // Interior points get a join: skipped entirely if collinear-enough
    // that the gap would be sub-pixel (see `join_cos_threshold`),
    // otherwise filled with a single cheap 3-vertex bevel triangle on the
    // turn's outer side (matching MapLibre's default `line-join: miter`
    // behavior, approximated with a bevel) rather than a full round join
    // disc -- avoids ~9 vertices/24 indices per turn for ~3 vertices/3
    // indices.
    let join_cos_threshold = join_cos_threshold(half_width);
    let last = points.len() - 1;
    for i in 1..last {
        if let (Some(a), Some(b)) = (directions[i - 1], directions[i]) {
            let cos = a[0] * b[0] + a[1] * b[1];
            if cos < join_cos_threshold {
                append_bevel_join(buffers, points[i], a, b, half_width, color);
            }
        } else {
            // Degenerate (near-zero-length) neighboring segment: fall back
            // to a disc since there's no reliable direction to bevel from.
            append_disc(buffers, points[i], half_width, color);
        }
    }
}

/// A join disc's outer corner leaves a gap of about `half_width * angle`
/// (small-angle approx) if skipped; solving `half_width * angle =
/// MAX_JOIN_GAP_PX` for the angle gives the largest deviation that's still
/// sub-pixel for a given width. Clamped to a sane range so hairline widths
/// don't skip joins outright and very wide ones don't regress past the old
/// fixed tolerance.
const MAX_JOIN_GAP_PX: f32 = 0.5;
const LINE_DISC_SEGMENTS: u32 = 8;

fn join_cos_threshold(half_width: f32) -> f32 {
    let angle = (MAX_JOIN_GAP_PX / half_width.max(0.05)).clamp(0.0, 1.05);
    angle.cos()
}

/// Fills the wedge-shaped gap left on a turn's outer side by two segments'
/// straight-extruded quads (see `append_polyline`) with a single triangle:
/// the shared point plus each segment's extrude endpoint on the outer
/// side. Cheap bevel-style join (3 vertices/3 indices) vs. a full round
/// disc (`1 + LINE_DISC_SEGMENTS` vertices) -- visually indistinguishable
/// from a round join at typical road widths (a few px), since the
/// difference is confined to the tiny wedge itself.
fn append_bevel_join(
    buffers: &mut LineMesh,
    center: [f32; 2],
    dir_in: [f32; 2],
    dir_out: [f32; 2],
    half_width: f32,
    color: [f32; 4],
) {
    // Cross product's sign tells us which side is the "outer" (convex)
    // side of the turn -- extrude is a +90-degree rotation of direction.
    let cross = dir_in[0] * dir_out[1] - dir_in[1] * dir_out[0];
    let side = if cross < 0.0 { 1.0 } else { -1.0 };
    let extrude_in = [
        -dir_in[1] * half_width * side,
        dir_in[0] * half_width * side,
    ];
    let extrude_out = [
        -dir_out[1] * half_width * side,
        dir_out[0] * half_width * side,
    ];
    let base = buffers.vertices.len() as u32;
    buffers.vertices.push(LineVertex {
        center,
        extrude: [0.0, 0.0],
        half_width: 0.0,
        color,
    });
    buffers.vertices.push(LineVertex {
        center,
        extrude: extrude_in,
        half_width: half_width * side,
        color,
    });
    buffers.vertices.push(LineVertex {
        center,
        extrude: extrude_out,
        half_width: half_width * side,
        color,
    });
    buffers
        .indices
        .extend_from_slice(&[base, base + 1, base + 2]);
}

fn append_disc(buffers: &mut LineMesh, center: [f32; 2], radius: f32, color: [f32; 4]) {
    let base = buffers.vertices.len() as u32;
    buffers.vertices.push(LineVertex {
        center,
        extrude: [0.0, 0.0],
        half_width: 0.0,
        color,
    });
    for i in 0..LINE_DISC_SEGMENTS {
        let angle = i as f32 / LINE_DISC_SEGMENTS as f32 * std::f32::consts::TAU;
        buffers.vertices.push(LineVertex {
            center,
            extrude: [radius * angle.cos(), radius * angle.sin()],
            half_width: radius,
            color,
        });
    }
    for i in 0..LINE_DISC_SEGMENTS {
        let next = if i + 1 == LINE_DISC_SEGMENTS {
            1
        } else {
            i + 2
        };
        buffers
            .indices
            .extend_from_slice(&[base, base + i + 1, base + next]);
    }
}

fn fill_path(
    buffers: &mut VertexBuffers<Vertex, u32>,
    fill_tess: &mut FillTessellator,
    path: Path,
    color: [f32; 4],
) {
    let _ = fill_tess.tessellate_path(
        &path,
        &FillOptions::default(),
        &mut BuffersBuilder::new(buffers, move |vertex: FillVertex| {
            let p = vertex.position();
            Vertex {
                position: [p.x, p.y],
                color,
            }
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_tile_mesh` should extract point labels from a dense real-world
    /// tile's `place`/`poi` layers, with the wire round-trip (used to ship
    /// results across the wasm worker boundary; see `TileMeshWire`)
    /// preserving them.
    #[test]
    fn labels_are_extracted_and_survive_the_wire_round_trip() {
        let full_path = format!("{}/fixtures/paris_14.pbf", env!("CARGO_MANIFEST_DIR"));
        let bytes = std::fs::read(&full_path).expect("failed to read fixture");
        let tile = rgis_tiles::decode_vector_tile(&bytes).expect("decode fixture MVT");
        let coord = TileCoord { z: 14, x: 0, y: 0 };
        let mesh = build_tile_mesh(&tile, coord);

        assert!(
            !mesh.labels.is_empty(),
            "expected at least one place/poi label in a dense real-world tile"
        );
        assert!(mesh.labels.iter().all(|l| !l.text.is_empty()));
        // Priorities are non-decreasing (place labels sort before poi
        // labels, each internally ranked).
        assert!(
            mesh.labels
                .windows(2)
                .all(|w| w[0].priority <= w[1].priority)
        );

        let wire = TileMeshWire::from(&mesh);
        let round_tripped = wire.into_tile_mesh();
        assert_eq!(round_tripped.labels.len(), mesh.labels.len());
        assert_eq!(round_tripped.labels[0].text, mesh.labels[0].text);
    }

    /// Vertex/index count contributed by exactly one bevel join triangle.
    const BEVEL_VERTS: usize = 3;
    const BEVEL_INDICES: usize = 3;
    /// Vertex/index count contributed by exactly one segment quad.
    const SEGMENT_VERTS: usize = 4;
    const SEGMENT_INDICES: usize = 6;

    #[test]
    fn collinear_interior_points_get_no_join_disc() {
        let mut mesh = LineMesh::default();
        let points = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]];
        append_polyline(&mut mesh, &points, [1.0, 1.0, 1.0, 1.0], 2.0, true);

        // 3 segments, no caps (butt caps need no extra geometry) and no
        // interior joins, since all 3 segments share the same direction.
        assert_eq!(mesh.vertices.len(), 3 * SEGMENT_VERTS);
        assert_eq!(mesh.indices.len(), 3 * SEGMENT_INDICES);
    }

    #[test]
    fn real_turn_gets_a_bevel_join() {
        let mut mesh = LineMesh::default();
        let points = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]];
        append_polyline(&mut mesh, &points, [1.0, 1.0, 1.0, 1.0], 2.0, true);

        // 2 segments, no caps, + 1 interior bevel join for the 90-degree
        // turn.
        assert_eq!(mesh.vertices.len(), 2 * SEGMENT_VERTS + BEVEL_VERTS);
        assert_eq!(mesh.indices.len(), 2 * SEGMENT_INDICES + BEVEL_INDICES);
    }

    #[test]
    fn with_joins_false_skips_all_joins() {
        let mut mesh = LineMesh::default();
        let points = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]];
        append_polyline(&mut mesh, &points, [1.0, 1.0, 1.0, 1.0], 2.0, false);

        assert_eq!(mesh.vertices.len(), 2 * SEGMENT_VERTS);
        assert_eq!(mesh.indices.len(), 2 * SEGMENT_INDICES);
    }

    /// Offline, deterministic reproduction of the "multi-GB memory on dense
    /// urban tiles" investigation, using real MVT fixtures checked into
    /// `fixtures/` (fetched once from OpenFreeMap; see `fixtures/README.md`)
    /// rather than a browser stress test. Run with
    /// `cargo test -p rgis-render tile_mesh_byte_budget -- --nocapture` to
    /// see a per-tile breakdown; the assertions guard against regressions
    /// that would blow the per-tile memory budget back up.
    #[test]
    fn tile_mesh_byte_budget() {
        // (name, raw .pbf byte size, max acceptable total mesh bytes).
        // Budgets are generous headroom above current measured output,
        // just tight enough to catch a regression back toward the
        // multi-ten-MB-per-tile behaviour seen before the `with_joins`
        // fix (auto-antialiasing outlines emitting full round-join discs
        // on every polygon corner, e.g. every building).
        const FIXTURES: &[(&str, &str, usize)] = &[
            ("paris_12", "fixtures/paris_12.pbf", 5_000_000),
            ("paris_14", "fixtures/paris_14.pbf", 15_000_000),
            ("london_14", "fixtures/london_14.pbf", 9_000_000),
            ("nyc_14", "fixtures/nyc_14.pbf", 9_000_000),
            ("tokyo_14", "fixtures/tokyo_14.pbf", 16_000_000),
        ];

        let mut total_bytes = 0usize;
        for (name, path, budget) in FIXTURES {
            let full_path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), path);
            let bytes = std::fs::read(&full_path)
                .unwrap_or_else(|e| panic!("failed to read fixture {full_path}: {e}"));
            let tile = rgis_tiles::decode_vector_tile(&bytes).expect("decode fixture MVT");
            let coord = TileCoord { z: 14, x: 0, y: 0 };
            let mesh = build_tile_mesh(&tile, coord);

            let (fv, fi, lv, li) = mesh.counts();
            let mesh_bytes = fv * std::mem::size_of::<Vertex>()
                + fi * std::mem::size_of::<u32>()
                + lv * std::mem::size_of::<LineVertex>()
                + li * std::mem::size_of::<u32>();
            total_bytes += mesh_bytes;

            println!(
                "{name}: raw={} bytes, fill_verts={fv} fill_idx={fi} line_verts={lv} line_idx={li} mesh_bytes={mesh_bytes} ({:.1} MB)",
                bytes.len(),
                mesh_bytes as f64 / 1_048_576.0
            );

            assert!(
                mesh_bytes <= *budget,
                "{name}: mesh_bytes={mesh_bytes} exceeds budget={budget} \
                 ({:.1} MB > {:.1} MB) -- tessellation output grew, check for \
                 an unnecessary source of extra vertices/discs",
                mesh_bytes as f64 / 1_048_576.0,
                *budget as f64 / 1_048_576.0
            );
        }
        println!(
            "total mesh bytes across {} fixtures: {} ({:.1} MB)",
            FIXTURES.len(),
            total_bytes,
            total_bytes as f64 / 1_048_576.0
        );
    }
}
