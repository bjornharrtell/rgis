//! Converts `rgis_core` layer geometry into GPU-ready triangle meshes.
//!
//! Fills, strokes, and point markers for every visible layer are tessellated
//! with `lyon` into a single vertex/index buffer (in screen-pixel space), so
//! the whole scene can be drawn with one indexed draw call.

use bytemuck::{Pod, Zeroable};
use geo_types::{
    Coord, Geometry, Line, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
    Triangle,
};
use lyon::math::point;
use lyon::path::{Builder, Path};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, StrokeOptions, StrokeTessellator,
    StrokeVertex, VertexBuffers,
};
use rgis_core::{Layer, Viewport};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

/// A tessellated triangle mesh for the whole visible project, in screen-pixel
/// space (already offset by the map viewport's on-screen origin).
#[derive(Debug, Default, Clone)]
pub struct SceneMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl SceneMesh {
    /// Appends `other`'s triangles after this mesh's, so `other` is drawn on
    /// top (there's no depth test — draw order is paint order).
    pub fn extend(&mut self, other: SceneMesh) {
        let offset = self.vertices.len() as u32;
        self.vertices.extend(other.vertices);
        self.indices
            .extend(other.indices.into_iter().map(|i| i + offset));
    }
}

pub fn build_scene_mesh(layers: &[Layer], viewport: &Viewport) -> SceneMesh {
    build_scene_mesh_with_offset(layers, viewport, [0.0, 0.0])
}

pub fn build_scene_mesh_with_offset(
    layers: &[Layer],
    viewport: &Viewport,
    offset: [f32; 2],
) -> SceneMesh {
    let mut visible_layers: Vec<_> = layers.iter().filter(|layer| layer.visible).collect();
    visible_layers.sort_by_key(|layer| layer.z_order);

    let mut buffers: VertexBuffers<Vertex, u32> = VertexBuffers::new();
    let mut fill_tess = FillTessellator::new();
    let mut stroke_tess = StrokeTessellator::new();

    for layer in visible_layers {
        let mut ctx = LayerTessCtx {
            viewport,
            offset,
            buffers: &mut buffers,
            fill_tess: &mut fill_tess,
            stroke_tess: &mut stroke_tess,
            fill_color: color_to_array(layer.style.fill),
            stroke_color: color_to_array(layer.style.stroke),
            stroke_width: layer.style.stroke_width.max(0.1),
            point_radius: layer.style.point_radius.max(1.0),
        };
        for feature in &layer.features {
            ctx.append_geometry(&feature.geometry);
        }
    }

    SceneMesh {
        vertices: buffers.vertices,
        indices: buffers.indices,
    }
}

struct LayerTessCtx<'a> {
    viewport: &'a Viewport,
    offset: [f32; 2],
    buffers: &'a mut VertexBuffers<Vertex, u32>,
    fill_tess: &'a mut FillTessellator,
    stroke_tess: &'a mut StrokeTessellator,
    fill_color: [f32; 4],
    stroke_color: [f32; 4],
    stroke_width: f32,
    point_radius: f32,
}

impl LayerTessCtx<'_> {
    fn screen(&self, coord: Coord) -> lyon::math::Point {
        let [x, y] = self.viewport.world_to_screen(coord);
        point(x + self.offset[0], y + self.offset[1])
    }

    fn append_geometry(&mut self, geometry: &Geometry) {
        match geometry {
            Geometry::Point(p) => self.append_point(p),
            Geometry::MultiPoint(points) => self.append_multipoint(points),
            Geometry::Line(line) => self.append_line(line),
            Geometry::LineString(line_string) => self.append_linestring(line_string),
            Geometry::MultiLineString(lines) => self.append_multilinestring(lines),
            Geometry::Polygon(polygon) => self.append_polygon(polygon),
            Geometry::MultiPolygon(polygons) => self.append_multipolygon(polygons),
            Geometry::Rect(rect) => self.append_polygon(&Polygon::from(*rect)),
            Geometry::Triangle(triangle) => self.append_polygon(&triangle_to_polygon(*triangle)),
            Geometry::GeometryCollection(collection) => {
                for geometry in &collection.0 {
                    self.append_geometry(geometry);
                }
            }
        }
    }

    fn append_multipoint(&mut self, points: &MultiPoint) {
        for p in &points.0 {
            self.append_point(p);
        }
    }

    fn append_point(&mut self, p: &Point) {
        const SEGMENTS: u32 = 16;
        let center = self.screen(p.0);
        let radius = self.point_radius;

        let mut builder = Path::builder();
        for i in 0..SEGMENTS {
            let angle = (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
            let p = point(
                center.x + radius * angle.cos(),
                center.y + radius * angle.sin(),
            );
            if i == 0 {
                builder.begin(p);
            } else {
                builder.line_to(p);
            }
        }
        builder.end(true);

        self.fill_path(builder.build(), self.fill_color);
    }

    fn append_line(&mut self, line: &Line) {
        let mut builder = Path::builder();
        builder.begin(self.screen(line.start));
        builder.line_to(self.screen(line.end));
        builder.end(false);
        self.stroke_path(builder.build());
    }

    fn append_multilinestring(&mut self, lines: &MultiLineString) {
        for line in &lines.0 {
            self.append_linestring(line);
        }
    }

    fn append_linestring(&mut self, line_string: &LineString) {
        let Some(path) = self.build_open_path(line_string) else {
            return;
        };
        self.stroke_path(path);
    }

    fn append_polygon(&mut self, polygon: &Polygon) {
        let mut builder = Path::builder();
        let mut any_ring = false;
        any_ring |= self.build_ring(&mut builder, polygon.exterior());
        for ring in polygon.interiors() {
            any_ring |= self.build_ring(&mut builder, ring);
        }
        if !any_ring {
            return;
        }
        let path = builder.build();
        self.fill_path(path.clone(), self.fill_color);
        self.stroke_path(path);
    }

    fn append_multipolygon(&mut self, polygons: &MultiPolygon) {
        for polygon in &polygons.0 {
            self.append_polygon(polygon);
        }
    }

    /// Builds an open (non-closed) path from a line string, for stroking.
    fn build_open_path(&self, line_string: &LineString) -> Option<Path> {
        let mut coords = line_string.coords();
        let first = coords.next()?;
        let mut builder = Path::builder();
        builder.begin(self.screen(*first));
        for coord in coords {
            builder.line_to(self.screen(*coord));
        }
        builder.end(false);
        Some(builder.build())
    }

    /// Adds a closed ring's subpath to `builder`. Returns `true` if any
    /// geometry was added.
    fn build_ring(&self, builder: &mut Builder, ring: &LineString) -> bool {
        let mut coords = ring.coords();
        let Some(first) = coords.next() else {
            return false;
        };
        builder.begin(self.screen(*first));
        for coord in coords {
            builder.line_to(self.screen(*coord));
        }
        builder.end(true);
        true
    }

    fn fill_path(&mut self, path: Path, color: [f32; 4]) {
        let _ = self.fill_tess.tessellate_path(
            &path,
            &FillOptions::default(),
            &mut BuffersBuilder::new(self.buffers, move |vertex: FillVertex| {
                let p = vertex.position();
                Vertex {
                    position: [p.x, p.y],
                    color,
                }
            }),
        );
    }

    fn stroke_path(&mut self, path: Path) {
        let color = self.stroke_color;
        let options = StrokeOptions::default().with_line_width(self.stroke_width);
        let _ = self.stroke_tess.tessellate_path(
            &path,
            &options,
            &mut BuffersBuilder::new(self.buffers, move |vertex: StrokeVertex| {
                let p = vertex.position();
                Vertex {
                    position: [p.x, p.y],
                    color,
                }
            }),
        );
    }
}

fn triangle_to_polygon(triangle: Triangle) -> Polygon {
    Polygon::new(
        vec![triangle.v1(), triangle.v2(), triangle.v3(), triangle.v1()].into(),
        Vec::new(),
    )
}

fn color_to_array(color: rgis_core::Color) -> [f32; 4] {
    [color.r, color.g, color.b, color.a]
}
