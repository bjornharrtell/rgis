//! Tessellates decoded vector tiles into the same `Vertex`/`SceneMesh` used
//! for regular layers, driven by a runtime-parsed MapLibre/Mapbox style
//! document (see `rgis_style`) rather than a hardcoded per-layer style
//! table -- so any style-spec-compliant style JSON can be rendered, and
//! switched live by swapping the `StyleSheet` passed to [`build_tile_mesh`].
//!
//! Defaults to OpenFreeMap's own "liberty" MapLibre style
//! (<https://tiles.openfreemap.org/styles/liberty>), matching the
//! OpenMapTiles-schema vector tiles this app fetches, but nothing here is
//! specific to that style beyond that default.

use bytemuck::{Pod, Zeroable};
use geo_types::{Geometry, LineString, Polygon};
use rearcut::Earcut;
use rgis_core::{EARTH_HALF_CIRC, Viewport};
use rgis_style::{Color, EvalContext, Layer, StyleSheet};
use rgis_tiles::{TileCoord, VectorFeature, VectorTile};

use crate::mesh::{SceneMesh, Vertex};

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
    /// point labels (place/poi). Ignored when `path` is `Some` (road
    /// labels), which instead follow the path's own local tangent per
    /// glyph.
    pub angle: f32,
    /// For road-name labels (`symbol-placement: line`, mirroring MapLibre):
    /// the full road polyline in the same tile-local-metres space as
    /// `position`, so the label's glyphs can be threaded along the actual
    /// road geometry (curving with it) instead of sitting on one flat,
    /// straight quad. `None` for point labels (place/poi), which are laid
    /// out as a single horizontal run instead.
    pub path: Option<Vec<[f32; 2]>>,
    /// Evaluated `icon-image` sprite name (e.g. `"circle_11_black"`),
    /// looked up in the fetched sprite atlas by the caller; `None`/empty
    /// means no icon for this feature. Point labels only -- MapLibre only
    /// places icons for point symbols, not line-placed ones.
    pub icon: Option<String>,
    /// Evaluated `icon-size` multiplier applied to the sprite's native
    /// pixel dimensions (`1.0` if the style doesn't specify one).
    pub icon_size: f32,
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

/// Line-width scale factor for continuous (device-pixel-stable) zoom
/// behaviour: baked line half-widths are evaluated once at the *tile's
/// own* zoom (`coord.z`, the zoom a style's `line-width` expression is
/// designed to produce a "correct" screen-pixel width for), then this
/// factor rescales them for the *current* viewport zoom every frame
/// (see [`TileTransform::width_scale`]) so overzoomed tiles keep growing
/// continuously instead of snapping when a new, more-detailed tile loads.
/// `delta` is `viewport.zoom - tile.z`.
fn zoom_scale(delta: f64) -> f32 {
    (1.0 + delta * 0.15).max(0.0) as f32
}

/// Evaluates `layer`'s fill paint properties for `feature` at `zoom`:
/// `fill-color`/`fill-opacity` for `fill` layers, or
/// `fill-extrusion-color`/`fill-extrusion-opacity` for `fill-extrusion`
/// layers -- rendered identically (flat, top-down) since this app has no
/// camera pitch/tilt, matching how MapLibre itself renders an extrusion at
/// pitch 0 (no visible side walls, just the flat top polygon).
fn eval_fill_paint(layer: &Layer, ctx: &EvalContext) -> [f32; 4] {
    let (color_key, opacity_key) = if layer.kind == "fill-extrusion" {
        ("fill-extrusion-color", "fill-extrusion-opacity")
    } else {
        ("fill-color", "fill-opacity")
    };
    let mut color = layer
        .paint(color_key)
        .eval_color(ctx, Color([0.0, 0.0, 0.0, 1.0]));
    let opacity = layer.paint(opacity_key).eval_f64(ctx, 1.0) as f32;
    color.0[3] *= opacity;
    color.to_array()
}

/// Evaluates `layer`'s `fill-outline-color`, defaulting (per the style
/// spec) to the layer's own `fill-color` when unset -- this default is also
/// what gives every fill its analytic-antialiased boundary edge (see
/// `append_outline`'s doc comment). Unlike real MapLibre this ignores an
/// explicit `fill-antialias: false`: this renderer's line-based edge
/// antialiasing is a different technique from MapLibre's coverage-based
/// one, and turning it off here would only make edges jaggier without
/// reproducing MapLibre's actual look.
fn eval_fill_outline_paint(layer: &Layer, ctx: &EvalContext, fill_color: [f32; 4]) -> [f32; 4] {
    layer
        .paint("fill-outline-color")
        .eval(ctx)
        .and_then(|v| v.as_color())
        .map(Color::to_array)
        .unwrap_or(fill_color)
}

/// Evaluates `layer`'s `line-color`/`line-opacity`/`line-width` paint
/// properties for `feature` at `zoom`, returning `(color, width_px)`.
fn eval_line_paint(layer: &Layer, ctx: &EvalContext) -> ([f32; 4], f32) {
    let mut color = layer
        .paint("line-color")
        .eval_color(ctx, Color([0.0, 0.0, 0.0, 1.0]));
    let opacity = layer.paint("line-opacity").eval_f64(ctx, 1.0) as f32;
    color.0[3] *= opacity;
    let width = layer.paint("line-width").eval_f64(ctx, 1.0) as f32;
    (color.to_array(), width)
}
/// Tessellates a single decoded vector tile into fill + line meshes whose
/// vertex positions are in mercator METRES relative to the tile's own
/// top-left corner, NOT screen space — this makes the result independent
/// of the viewport, so callers can tessellate a tile once and cache it
/// indefinitely (see [`tile_screen_transform`] for the cheap per-frame
/// screen transform applied on the GPU).
///
/// Iterates `style`'s layers in their own bottom-to-top order (skipping
/// `background`/`raster`, handled separately -- see [`build_background_mesh`]
/// -- and `symbol`, handled by [`extract_labels`]), so draw/blend order
/// across the whole tile always matches the style document's own layer
/// order, exactly like MapLibre, rather than a fixed hardcoded pass order.
pub fn build_tile_mesh(tile: &VectorTile, coord: TileCoord, style: &StyleSheet) -> TileMesh {
    let mut fill_mesh = SceneMesh::default();
    let mut earcut: Earcut = Earcut::new();
    let mut earcut_buf: Vec<u32> = Vec::new();
    let mut earcut_flat: Vec<f64> = Vec::new();
    let mut lines = LineMesh::default();
    let zoom = coord.z as f64;
    let tile_size_m = TileMercatorBounds::for_coord(coord).size;

    for layer in &style.layers {
        if !matches!(layer.kind.as_str(), "fill" | "fill-extrusion" | "line") {
            continue;
        }
        if !layer.matches_zoom(zoom) {
            continue;
        }
        // `fill-pattern` (a tiled sprite texture instead of a solid color)
        // isn't implemented -- these layers have no `fill-color` to fall
        // back to, so without this check `eval_fill_paint`'s color-eval
        // failure would silently default to opaque black, painting solid
        // black blobs over e.g. pedestrian plazas/bridge decks. Skipping
        // the layer entirely (closest to "just don't draw the pattern")
        // is a much closer visual approximation than that.
        if layer.paint("fill-pattern").0.is_some() {
            continue;
        }
        let Some(source_layer_name) = layer.source_layer.as_deref() else {
            continue;
        };
        let Some(tile_layer) = tile.layers.iter().find(|l| l.name == source_layer_name) else {
            continue;
        };
        let ctx = TileContext {
            extent: tile_layer.extent,
            tile_size_m,
        };

        for feature in &tile_layer.features {
            if !layer.matches_feature(feature, zoom) {
                continue;
            }
            let eval_ctx = EvalContext::with_feature(zoom, feature);

            if layer.kind == "line" {
                let (color, width) = eval_line_paint(layer, &eval_ctx);
                append_line(&mut lines, feature, &ctx, color, width);
                continue;
            }

            // "fill" or "fill-extrusion".
            let fill_color = eval_fill_paint(layer, &eval_ctx);
            append_fill(
                &mut fill_mesh,
                &mut earcut,
                &mut earcut_buf,
                &mut earcut_flat,
                feature,
                &ctx,
                fill_color,
            );
            // Every fill gets a same-color (by default) edge so its
            // tessellated boundary gets the same analytic antialiasing as
            // stroked lines (see `LineVertex`), instead of a raw jaggy
            // triangle edge -- see `eval_fill_outline_paint`.
            let outline_color = eval_fill_outline_paint(layer, &eval_ctx, fill_color);
            append_outline(&mut lines, feature, &ctx, outline_color, 1.0);
        }
    }

    let labels = extract_labels(tile, coord, tile_size_m, style);

    TileMesh {
        fill: fill_mesh,
        lines,
        labels,
    }
}

/// Extracts every labeled feature (`symbol` layers with a `text-field`)
/// from a decoded tile, in the same tile-local-metres space
/// `build_tile_mesh` uses for its fill/line vertices, evaluating
/// `text-field`/`text-size`/`text-color`/`text-halo-color`/
/// `symbol-placement`/`icon-image`/`icon-size` from `style` -- driven
/// entirely by the style document instead of a hardcoded per-source-layer
/// table. `icon-image` is resolved to a sprite name here; the caller looks
/// it up in the fetched sprite atlas and shapes the actual textured quad
/// (see `RgisApp::collect_label_draws`), since sprite fetching/atlas
/// storage lives above this crate. Symbol layers with an `icon-image` but
/// no `text-field` (road shields, one-way arrows) are still skipped, since
/// this function only walks features that have label text.
fn extract_labels(
    tile: &VectorTile,
    coord: TileCoord,
    tile_size_m: f64,
    style: &StyleSheet,
) -> Vec<TileLabel> {
    let zoom = coord.z as f64;
    let mut labels = Vec::new();

    for layer in &style.layers {
        if layer.kind != "symbol" || !layer.matches_zoom(zoom) {
            continue;
        }
        let Some(source_layer_name) = layer.source_layer.as_deref() else {
            continue;
        };
        let Some(tile_layer) = tile.layers.iter().find(|l| l.name == source_layer_name) else {
            continue;
        };
        let ctx = TileContext {
            extent: tile_layer.extent,
            tile_size_m,
        };

        for feature in &tile_layer.features {
            if !layer.matches_feature(feature, zoom) {
                continue;
            }
            let eval_ctx = EvalContext::with_feature(zoom, feature);
            let Some(text) = layer.layout("text-field").eval_string(&eval_ctx) else {
                continue;
            };
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            let font_size = layer.layout("text-size").eval_f64(&eval_ctx, 16.0) as f32;
            let color = layer
                .paint("text-color")
                .eval_color(&eval_ctx, Color([0.15, 0.15, 0.17, 1.0]))
                .to_array();
            let halo_color = layer
                .paint("text-halo-color")
                .eval_color(&eval_ctx, Color([1.0, 1.0, 1.0, 0.9]))
                .to_array();

            // No full mapbox symbol-sort-key/collision port here -- this
            // approximates it generically (without hardcoding per-source-
            // layer knowledge) from the label's own evaluated size, which
            // correlates well with a style's intended visual hierarchy
            // (country names render larger than POI labels, etc), tie-
            // broken by the feature's own `rank` property when present
            // (place/poi layers set this).
            let rank = feature
                .get_number("rank")
                .unwrap_or(50.0)
                .clamp(0.0, 9_999.0) as i32;
            let size_rank = ((200.0 - font_size).max(0.0) * 10.0) as i32;
            let priority = size_rank * 10_000 + rank;

            let placement = layer.layout("symbol-placement").eval_string(&eval_ctx);
            if placement.as_deref() == Some("line") {
                let points = match &feature.geometry {
                    Geometry::LineString(line) => line_points(line, &ctx),
                    Geometry::MultiLineString(mlines) => mlines
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
                // Anchor at the arc-length midpoint of the whole road,
                // matching where MapLibre centers a line-placed label
                // along its geometry.
                let mid = point_at_arc_length(&points, line_length_points(&points) * 0.5);
                labels.push(TileLabel {
                    position: mid,
                    text: text.to_string(),
                    font_size,
                    color,
                    halo_color,
                    priority,
                    angle: 0.0,
                    path: Some(points),
                    icon: None,
                    icon_size: 1.0,
                });
            } else {
                let Geometry::Point(p) = &feature.geometry else {
                    continue;
                };
                let pos = ctx.project_point(p.x(), p.y());
                let icon_image = layer
                    .layout("icon-image")
                    .eval_string(&eval_ctx)
                    .filter(|s| !s.is_empty());
                let icon_size = layer.layout("icon-size").eval_f64(&eval_ctx, 1.0) as f32;
                labels.push(TileLabel {
                    position: pos,
                    text: text.to_string(),
                    font_size,
                    color,
                    halo_color,
                    priority,
                    angle: 0.0,
                    path: None,
                    icon: icon_image,
                    icon_size,
                });
            }
        }
    }
    labels.sort_by_key(|l| l.priority);
    labels
}

fn line_length(line: &LineString<i32>, ctx: &TileContext) -> f32 {
    line_length_points(&line_points(line, ctx))
}

fn line_length_points(points: &[[f32; 2]]) -> f32 {
    points
        .windows(2)
        .map(|pair| ((pair[1][0] - pair[0][0]).powi(2) + (pair[1][1] - pair[0][1]).powi(2)).sqrt())
        .sum()
}

/// Walks `points` by cumulative arc length and returns the point at
/// `target_len` along the polyline (clamped to its endpoints).
fn point_at_arc_length(points: &[[f32; 2]], target_len: f32) -> [f32; 2] {
    let mut walked = 0.0f32;
    for pair in points.windows(2) {
        let [p0, p1] = [pair[0], pair[1]];
        let seg_len = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
        if walked + seg_len >= target_len || seg_len <= f32::EPSILON {
            let t = if seg_len > f32::EPSILON {
                (target_len - walked) / seg_len
            } else {
                0.0
            };
            return [p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t];
        }
        walked += seg_len;
    }
    points.last().copied().unwrap_or([0.0, 0.0])
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
        width_scale: zoom_scale(viewport.zoom - coord.z as f64),
        size: bounds.size as f32 * scale,
    }
}

/// A full-viewport quad in `style`'s `background` layer color (falling
/// back to opaque white if the style defines none), drawn beneath the
/// basemap tiles. `background-color`/`background-opacity` may themselves
/// be zoom-interpolated expressions (e.g. a raster-only style fading in a
/// solid color at low zoom), so this re-evaluates them against the
/// viewport's current zoom every call rather than caching a fixed color.
pub fn build_background_mesh(viewport: &Viewport, style: &StyleSheet) -> SceneMesh {
    let ctx = EvalContext::new(viewport.zoom);
    let mut color = Color([1.0, 1.0, 1.0, 1.0]);
    for layer in style.layers_of_kind("background") {
        color = layer.paint("background-color").eval_color(&ctx, color);
        let opacity = layer.paint("background-opacity").eval_f64(&ctx, 1.0) as f32;
        color.0[3] *= opacity;
    }
    let color = color.to_array();
    let w = viewport.width_px as f32;
    let h = viewport.height_px as f32;
    SceneMesh {
        vertices: vec![
            Vertex {
                position: [0.0, 0.0],
                color,
            },
            Vertex {
                position: [w, 0.0],
                color,
            },
            Vertex {
                position: [w, h],
                color,
            },
            Vertex {
                position: [0.0, h],
                color,
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
    fn project_point(&self, local_x: i32, local_y: i32) -> [f32; 2] {
        let fx = local_x as f64 / self.extent as f64;
        let fy = local_y as f64 / self.extent as f64;
        [
            (fx * self.tile_size_m) as f32,
            (fy * self.tile_size_m) as f32,
        ]
    }
}

fn append_fill(
    fill_mesh: &mut SceneMesh,
    earcut: &mut Earcut,
    earcut_buf: &mut Vec<u32>,
    earcut_flat: &mut Vec<f64>,
    feature: &VectorFeature,
    ctx: &TileContext,
    color: [f32; 4],
) {
    match &feature.geometry {
        Geometry::Polygon(polygon) => fill_polygon(
            fill_mesh,
            earcut,
            earcut_buf,
            earcut_flat,
            polygon,
            ctx,
            color,
        ),
        Geometry::MultiPolygon(polygons) => {
            for polygon in &polygons.0 {
                fill_polygon(
                    fill_mesh,
                    earcut,
                    earcut_buf,
                    earcut_flat,
                    polygon,
                    ctx,
                    color,
                );
            }
        }
        _ => {}
    }
}

fn fill_polygon(
    fill_mesh: &mut SceneMesh,
    earcut: &mut Earcut,
    earcut_buf: &mut Vec<u32>,
    earcut_flat: &mut Vec<f64>,
    polygon: &Polygon<i32>,
    ctx: &TileContext,
    color: [f32; 4],
) {
    let mut data = ring_points_no_close(polygon.exterior(), ctx);
    if data.len() < 3 {
        return;
    }
    let mut hole_indices: Vec<usize> = Vec::new();
    for ring in polygon.interiors() {
        let ring = ring_points_no_close(ring, ctx);
        if ring.len() < 3 {
            continue;
        }
        hole_indices.push(data.len());
        data.extend(ring);
    }

    earcut_flat.clear();
    earcut_flat.extend(data.iter().flat_map(|&[x, y]| [x as f64, y as f64]));
    earcut.earcut_into(earcut_flat, &hole_indices, 2, earcut_buf);
    if earcut_buf.is_empty() {
        return;
    }

    let base = fill_mesh.vertices.len() as u32;
    fill_mesh
        .vertices
        .extend(data.iter().map(|&position| Vertex { position, color }));
    fill_mesh
        .indices
        .extend(earcut_buf.iter().map(|&i| base + i));
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

/// Builds a polygon ring's projected points for `earcut`, without a
/// duplicated closing coordinate (unlike [`ring_points`], which is used
/// for line-stroke rendering and keeps it).
fn ring_points_no_close(ring: &LineString<i32>, ctx: &TileContext) -> Vec<[f32; 2]> {
    line_points(ring, ctx)
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
    line.coords().map(|c| ctx.project_point(c.x, c.y)).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The "liberty" style JSON used both here and by `rgis-style`'s own
    /// tests (see `crates/rgis-style/fixtures/liberty.json`) -- a single
    /// canonical copy, loaded across the crate boundary via a relative
    /// path rather than duplicated.
    fn liberty_style() -> StyleSheet {
        let path = format!(
            "{}/../rgis-style/fixtures/liberty.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let json = std::fs::read_to_string(&path).expect("failed to read liberty.json fixture");
        StyleSheet::parse(&json).expect("liberty style should parse")
    }

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
        let style = liberty_style();
        let mesh = build_tile_mesh(&tile, coord, &style);

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
        // Budgets got a one-time bump when `build_tile_mesh` switched from
        // a hardcoded ~9-entry per-layer style table to evaluating the
        // full "liberty" style document (over 100 layers: every road/
        // admin-boundary subclass and bridge/tunnel variant, not just the
        // dozen or so classes the old table covered) -- more legitimate
        // style layers now match real tile features, so more geometry is
        // correctly emitted (this is the point: closer parity with what
        // MapLibre itself draws), not a regression. Still generous
        // headroom above current measured output, tight enough to catch a
        // regression back toward the multi-ten-MB-per-tile behaviour seen
        // before the `with_joins` fix (auto-antialiasing outlines emitting
        // full round-join discs on every polygon corner, e.g. every
        // building).
        const FIXTURES: &[(&str, &str, usize)] = &[
            ("paris_12", "fixtures/paris_12.pbf", 20_000_000),
            ("paris_14", "fixtures/paris_14.pbf", 30_000_000),
            ("london_14", "fixtures/london_14.pbf", 25_000_000),
            ("nyc_14", "fixtures/nyc_14.pbf", 25_000_000),
            ("tokyo_14", "fixtures/tokyo_14.pbf", 30_000_000),
        ];

        let style = liberty_style();
        let mut total_bytes = 0usize;
        for (name, path, budget) in FIXTURES {
            let full_path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), path);
            let bytes = std::fs::read(&full_path)
                .unwrap_or_else(|e| panic!("failed to read fixture {full_path}: {e}"));
            let tile = rgis_tiles::decode_vector_tile(&bytes).expect("decode fixture MVT");
            let coord = TileCoord { z: 14, x: 0, y: 0 };
            let mesh = build_tile_mesh(&tile, coord, &style);

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
