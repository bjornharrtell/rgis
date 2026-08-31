//! Converts `rgis_core` layer geometry into GPU-ready triangle meshes.
//!
//! Fills are tessellated with `earcut` and strokes/point markers are
//! extruded by hand into quads/fans, all into a single vertex/index buffer
//! (in screen-pixel space), so the whole scene can be drawn with one
//! indexed draw call.

use bytemuck::{Pod, Zeroable};
use geo_types::{
    Coord, Geometry, Line, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
    Triangle,
};
use rearcut::Earcut;
use rgis_core::{Layer, Viewport};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

/// Number of segments used to approximate circles (point markers and round
/// line joins/caps).
const CIRCLE_SEGMENTS: u32 = 16;

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

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut earcut: Earcut = Earcut::new();
    let mut earcut_buf: Vec<u32> = Vec::new();
    let mut earcut_flat: Vec<f64> = Vec::new();

    for layer in visible_layers {
        let mut ctx = LayerTessCtx {
            viewport,
            offset,
            vertices: &mut vertices,
            indices: &mut indices,
            earcut: &mut earcut,
            earcut_buf: &mut earcut_buf,
            earcut_flat: &mut earcut_flat,
            fill_color: color_to_array(layer.style.fill),
            stroke_color: color_to_array(layer.style.stroke),
            stroke_width: layer.style.stroke_width.max(0.1),
            point_radius: layer.style.point_radius.max(1.0),
        };
        for feature in &layer.features {
            ctx.append_geometry(&feature.geometry);
        }
    }

    SceneMesh { vertices, indices }
}

struct LayerTessCtx<'a> {
    viewport: &'a Viewport,
    offset: [f32; 2],
    vertices: &'a mut Vec<Vertex>,
    indices: &'a mut Vec<u32>,
    earcut: &'a mut Earcut,
    earcut_buf: &'a mut Vec<u32>,
    earcut_flat: &'a mut Vec<f64>,
    fill_color: [f32; 4],
    stroke_color: [f32; 4],
    stroke_width: f32,
    point_radius: f32,
}

impl LayerTessCtx<'_> {
    fn screen(&self, coord: Coord) -> [f32; 2] {
        let [x, y] = self.viewport.world_to_screen(coord);
        [x + self.offset[0], y + self.offset[1]]
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
        let center = self.screen(p.0);
        let color = self.fill_color;
        self.push_disc(center, self.point_radius, color);
    }

    fn append_line(&mut self, line: &Line) {
        let points = [self.screen(line.start), self.screen(line.end)];
        self.stroke_polyline(&points, false);
    }

    fn append_multilinestring(&mut self, lines: &MultiLineString) {
        for line in &lines.0 {
            self.append_linestring(line);
        }
    }

    fn append_linestring(&mut self, line_string: &LineString) {
        let points = self.line_string_points(line_string);
        if points.len() < 2 {
            return;
        }
        self.stroke_polyline(&points, false);
    }

    fn append_polygon(&mut self, polygon: &Polygon) {
        self.fill_polygon(polygon);

        let exterior = self.ring_points(polygon.exterior());
        if exterior.len() >= 2 {
            self.stroke_polyline(&exterior, true);
        }
        for ring in polygon.interiors() {
            let ring = self.ring_points(ring);
            if ring.len() >= 2 {
                self.stroke_polyline(&ring, true);
            }
        }
    }

    fn append_multipolygon(&mut self, polygons: &MultiPolygon) {
        for polygon in &polygons.0 {
            self.append_polygon(polygon);
        }
    }

    /// Screen-space points of an open line string.
    fn line_string_points(&self, line_string: &LineString) -> Vec<[f32; 2]> {
        line_string.coords().map(|c| self.screen(*c)).collect()
    }

    /// Screen-space points of a closed ring, with the duplicated
    /// closing coordinate (present on `geo_types` rings) dropped, since
    /// `stroke_polyline`/`fill_polygon` close the loop themselves.
    fn ring_points(&self, ring: &LineString) -> Vec<[f32; 2]> {
        let mut coords: Vec<Coord> = ring.coords().copied().collect();
        if coords.len() > 1 && coords.first() == coords.last() {
            coords.pop();
        }
        coords.into_iter().map(|c| self.screen(c)).collect()
    }

    /// Triangulates a polygon (exterior + holes) with `earcut` and appends
    /// the resulting triangles using the layer's fill color.
    fn fill_polygon(&mut self, polygon: &Polygon) {
        let exterior = self.ring_points(polygon.exterior());
        if exterior.len() < 3 {
            return;
        }

        let mut data = exterior;
        let mut hole_indices: Vec<usize> = Vec::new();
        for ring in polygon.interiors() {
            let ring = self.ring_points(ring);
            if ring.len() < 3 {
                continue;
            }
            hole_indices.push(data.len());
            data.extend(ring);
        }

        self.earcut_flat.clear();
        self.earcut_flat
            .extend(data.iter().flat_map(|&[x, y]| [x as f64, y as f64]));
        self.earcut
            .earcut_into(self.earcut_flat, &hole_indices, 2, self.earcut_buf);
        if self.earcut_buf.is_empty() {
            return;
        }

        let color = self.fill_color;
        let base = self.vertices.len() as u32;
        self.vertices
            .extend(data.iter().map(|&position| Vertex { position, color }));
        self.indices
            .extend(self.earcut_buf.iter().map(|&i| base + i));
    }

    /// Extrudes a polyline into stroke-width quads, with round joins/caps
    /// (a disc at every vertex) so segments meet cleanly.
    fn stroke_polyline(&mut self, points: &[[f32; 2]], closed: bool) {
        let n = points.len();
        if n < 2 {
            return;
        }
        let half_width = self.stroke_width / 2.0;
        let color = self.stroke_color;

        let segment_count = if closed { n } else { n - 1 };
        for i in 0..segment_count {
            let a = points[i];
            let b = points[(i + 1) % n];
            self.push_segment_quad(a, b, half_width, color);
        }
        for &center in points {
            self.push_disc(center, half_width, color);
        }
    }

    fn push_segment_quad(&mut self, a: [f32; 2], b: [f32; 2], half_width: f32, color: [f32; 4]) {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < f32::EPSILON {
            return;
        }
        let nx = -dy / len * half_width;
        let ny = dx / len * half_width;

        let base = self.vertices.len() as u32;
        self.vertices.push(Vertex {
            position: [a[0] + nx, a[1] + ny],
            color,
        });
        self.vertices.push(Vertex {
            position: [a[0] - nx, a[1] - ny],
            color,
        });
        self.vertices.push(Vertex {
            position: [b[0] + nx, b[1] + ny],
            color,
        });
        self.vertices.push(Vertex {
            position: [b[0] - nx, b[1] - ny],
            color,
        });
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }

    fn push_disc(&mut self, center: [f32; 2], radius: f32, color: [f32; 4]) {
        let base = self.vertices.len() as u32;
        self.vertices.push(Vertex {
            position: center,
            color,
        });
        for i in 0..CIRCLE_SEGMENTS {
            let angle = (i as f32 / CIRCLE_SEGMENTS as f32) * std::f32::consts::TAU;
            self.vertices.push(Vertex {
                position: [
                    center[0] + radius * angle.cos(),
                    center[1] + radius * angle.sin(),
                ],
                color,
            });
        }
        for i in 0..CIRCLE_SEGMENTS {
            let next = if i + 1 == CIRCLE_SEGMENTS { 1 } else { i + 2 };
            self.indices
                .extend_from_slice(&[base, base + i + 1, base + next]);
        }
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
