use geo_types::{Geometry, LineString, Point, Polygon};
use lyon_tessellation::{
    geom::point,
    math::Point as LPoint,
    path::{path::Builder, Path},
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, StrokeOptions, StrokeTessellator,
    StrokeVertex, VertexBuffers,
};
use rgis_core::Feature;

pub struct GpuMesh {
    /// Interleaved [x, y] in Web Mercator metres (EPSG:3857).
    /// The world→NDC transform is applied as a uniform in the shader.
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Tessellate all features into a polygon fill mesh (world space).
/// This mesh is viewport-independent and never needs rebuilding on pan or zoom.
pub fn tessellate_fills(features: &[Feature]) -> GpuMesh {
    let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();

    for feature in features {
        tessellate_geometry_fill(&feature.geometry, &mut buffers, &mut tessellator);
    }

    GpuMesh {
        vertices: buffers.vertices.iter().flat_map(|v| *v).collect(),
        indices: buffers.indices,
    }
}

/// Tessellate all features into a stroke mesh (world space).
/// `stroke_width` must be in metres (multiply screen pixels by the viewport resolution).
pub fn tessellate_strokes(features: &[Feature], stroke_width: f32) -> GpuMesh {
    let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let mut tessellator = StrokeTessellator::new();
    let opts = StrokeOptions::default().with_line_width(stroke_width);

    for feature in features {
        tessellate_geometry_stroke(&feature.geometry, &mut buffers, &mut tessellator, &opts);
    }

    GpuMesh {
        vertices: buffers.vertices.iter().flat_map(|v| *v).collect(),
        indices: buffers.indices,
    }
}

#[inline]
fn world_point(x: f64, y: f64) -> LPoint {
    point(x as f32, y as f32)
}

fn tessellate_geometry_fill(
    geom: &Geometry,
    buffers: &mut VertexBuffers<[f32; 2], u32>,
    tess: &mut FillTessellator,
) {
    match geom {
        Geometry::Polygon(poly) => tessellate_polygon_fill(poly, buffers, tess),
        Geometry::MultiPolygon(mp) => {
            for poly in &mp.0 {
                tessellate_polygon_fill(poly, buffers, tess);
            }
        }
        _ => {}
    }
}

fn tessellate_polygon_fill(
    poly: &Polygon,
    buffers: &mut VertexBuffers<[f32; 2], u32>,
    tess: &mut FillTessellator,
) {
    let mut builder = Path::builder();
    add_ring_to_path(&mut builder, poly.exterior());
    for interior in poly.interiors() {
        add_ring_to_path(&mut builder, interior);
    }
    let path = builder.build();

    let _ = tess.tessellate_path(
        &path,
        &FillOptions::default(),
        &mut BuffersBuilder::new(buffers, |v: FillVertex| {
            let p = v.position();
            [p.x, p.y]
        }),
    );
}

fn add_ring_to_path(builder: &mut Builder, ring: &LineString) {
    let mut coords = ring.coords();
    let Some(first) = coords.next() else { return };
    builder.begin(world_point(first.x, first.y));
    for c in coords {
        builder.line_to(world_point(c.x, c.y));
    }
    builder.end(true);
}

fn tessellate_geometry_stroke(
    geom: &Geometry,
    buffers: &mut VertexBuffers<[f32; 2], u32>,
    tess: &mut StrokeTessellator,
    opts: &StrokeOptions,
) {
    match geom {
        Geometry::LineString(ls) => tessellate_linestring_stroke(ls, buffers, tess, opts),
        Geometry::MultiLineString(mls) => {
            for ls in &mls.0 {
                tessellate_linestring_stroke(ls, buffers, tess, opts);
            }
        }
        Geometry::Polygon(poly) => {
            tessellate_linestring_stroke(poly.exterior(), buffers, tess, opts);
            for interior in poly.interiors() {
                tessellate_linestring_stroke(interior, buffers, tess, opts);
            }
        }
        Geometry::MultiPolygon(mp) => {
            for poly in &mp.0 {
                tessellate_linestring_stroke(poly.exterior(), buffers, tess, opts);
                for interior in poly.interiors() {
                    tessellate_linestring_stroke(interior, buffers, tess, opts);
                }
            }
        }
        _ => {}
    }
}

fn tessellate_linestring_stroke(
    ls: &LineString,
    buffers: &mut VertexBuffers<[f32; 2], u32>,
    tess: &mut StrokeTessellator,
    opts: &StrokeOptions,
) {
    let mut coords = ls.coords();
    let Some(first) = coords.next() else { return };
    let mut builder = Path::builder();
    builder.begin(world_point(first.x, first.y));
    for c in coords {
        builder.line_to(world_point(c.x, c.y));
    }
    builder.end(false);
    let path = builder.build();

    let _ = tess.tessellate_path(
        &path,
        opts,
        &mut BuffersBuilder::new(buffers, |v: StrokeVertex| {
            let p = v.position();
            [p.x, p.y]
        }),
    );
}

/// Build a quad mesh for point features (world space).
/// `radius` must be in metres.
pub fn build_points_mesh(features: &[Feature], radius: f32) -> GpuMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for feature in features {
        collect_points(&feature.geometry, radius, &mut vertices, &mut indices);
    }
    GpuMesh { vertices, indices }
}

fn collect_points(geom: &Geometry, radius: f32, vertices: &mut Vec<f32>, indices: &mut Vec<u32>) {
    match geom {
        Geometry::Point(p) => push_point_quad(p, radius, vertices, indices),
        Geometry::MultiPoint(mp) => {
            for p in &mp.0 {
                push_point_quad(p, radius, vertices, indices);
            }
        }
        _ => {}
    }
}

fn push_point_quad(p: &Point, radius: f32, vertices: &mut Vec<f32>, indices: &mut Vec<u32>) {
    let cx = p.x() as f32;
    let cy = p.y() as f32;
    let base = (vertices.len() / 2) as u32;
    // Corners in world space: Y+ is north (same as NDC y+), so top = cy + radius.
    vertices.extend_from_slice(&[
        cx - radius, cy + radius, // TL
        cx + radius, cy + radius, // TR
        cx + radius, cy - radius, // BR
        cx - radius, cy - radius, // BL
    ]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}
