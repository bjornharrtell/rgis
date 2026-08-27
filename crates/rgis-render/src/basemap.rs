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
/// (positron/liberty/bright). Layers not listed here (POIs, place/road
/// labels, house numbers, …) are skipped since there's no text rendering.
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
}

/// A line/stroke vertex: `center` is the tile-local-metres position
/// (transformed exactly like fill vertices), while `extrude` is a
/// direction+magnitude offset applied by the vertex shader in SCREEN
/// PIXELS after scaling `center` (see `shaders/basemap_line.wgsl`), so
/// line width stays constant in device pixels instead of stretching with
/// the tile's own position scale.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct LineVertex {
    pub center: [f32; 2],
    pub extrude: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Debug, Default, Clone)]
pub(crate) struct LineMesh {
    pub(crate) vertices: Vec<LineVertex>,
    pub(crate) indices: Vec<u32>,
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

fn zoom_scale(zoom: f64) -> f32 {
    (1.0 + (zoom - 10.0).max(0.0) * 0.15) as f32
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
            let (casing, fill, base_width, rank) = match feature.get_str("class").unwrap_or("") {
                "motorway" => (Some(casing_major), [1.000, 0.800, 0.533, 1.00], 2.2, 6),
                "trunk" => (Some(casing_major), [1.000, 0.933, 0.667, 1.00], 1.9, 5),
                "primary" => (Some(casing_major), [1.000, 0.933, 0.667, 1.00], 1.8, 4),
                "secondary" | "tertiary" => {
                    (Some(casing_major), [1.000, 0.933, 0.667, 1.00], 1.4, 3)
                }
                "minor" => (Some(casing_minor), [1.0, 1.0, 1.0, 1.0], 1.0, 2),
                "service" | "track" => (Some(casing_minor), [1.0, 1.0, 1.0, 1.0], 0.6, 1),
                "path" | "pedestrian" => (None, [0.85, 0.85, 0.85, 1.0], 0.5, 0),
                "rail" | "transit" => (None, [0.733, 0.733, 0.733, 1.0], 0.8, 0),
                _ => return None,
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
            if let Some(paint) = style_for(layer_name, feature, zoom) {
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

    TileMesh {
        fill: SceneMesh {
            vertices: fill_buffers.vertices,
            indices: fill_buffers.indices,
        },
        lines,
    }
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
    let mut outline = |polygon: &Polygon<i32>| {
        append_polyline(
            buffers,
            &ring_points(polygon.exterior(), ctx),
            color,
            width_px,
        );
        for ring in polygon.interiors() {
            append_polyline(buffers, &ring_points(ring, ctx), color, width_px);
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
            append_polyline(buffers, &line_points(line, ctx), color, width_px)
        }
        Geometry::MultiLineString(lines) => {
            for line in &lines.0 {
                append_polyline(buffers, &line_points(line, ctx), color, width_px);
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
/// round disc is added at every point (endpoints and interior joints
/// alike) to approximate round caps/joins without miter-angle math.
/// `center` stays in tile-local metres (scaled like fill vertices);
/// `extrude` is a direction+magnitude offset in that same local space,
/// applied in SCREEN PIXELS by the shader (see [`LineVertex`]) — since the
/// tile's own position transform is a uniform (isotropic) scale, a
/// direction computed here is identical to its screen-space direction, so
/// this is valid even though `extrude`'s on-screen magnitude is meant to
/// stay constant regardless of that scale.
fn append_polyline(buffers: &mut LineMesh, points: &[[f32; 2]], color: [f32; 4], width_px: f32) {
    if points.len() < 2 || width_px <= 0.0 {
        return;
    }
    let half_width = width_px * 0.5;

    for pair in points.windows(2) {
        let (p0, p1) = (pair[0], pair[1]);
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len <= f32::EPSILON {
            continue;
        }
        let extrude = [-dy / len * half_width, dx / len * half_width];
        let neg_extrude = [-extrude[0], -extrude[1]];
        let base = buffers.vertices.len() as u32;
        buffers.vertices.push(LineVertex {
            center: p0,
            extrude,
            color,
        });
        buffers.vertices.push(LineVertex {
            center: p0,
            extrude: neg_extrude,
            color,
        });
        buffers.vertices.push(LineVertex {
            center: p1,
            extrude,
            color,
        });
        buffers.vertices.push(LineVertex {
            center: p1,
            extrude: neg_extrude,
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

    for &center in points {
        append_disc(buffers, center, half_width, color);
    }
}

const LINE_DISC_SEGMENTS: u32 = 8;

fn append_disc(buffers: &mut LineMesh, center: [f32; 2], radius: f32, color: [f32; 4]) {
    let base = buffers.vertices.len() as u32;
    buffers.vertices.push(LineVertex {
        center,
        extrude: [0.0, 0.0],
        color,
    });
    for i in 0..LINE_DISC_SEGMENTS {
        let angle = i as f32 / LINE_DISC_SEGMENTS as f32 * std::f32::consts::TAU;
        buffers.vertices.push(LineVertex {
            center,
            extrude: [radius * angle.cos(), radius * angle.sin()],
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
