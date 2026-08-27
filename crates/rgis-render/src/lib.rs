use anyhow::Error as PathError;
use geo_types::{
    Coord, Geometry, Line, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
    Triangle,
};
use gpui::{Hsla, Path, PathBuilder, Pixels, Point as GpuiPoint, Rgba, point, px};
use rgis_core::{Feature, Layer, LayerId, Viewport};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("failed to build gpui path: {0}")]
    Path(#[from] PathError),
}

#[derive(Debug)]
pub struct LayerPaths {
    pub layer_id: LayerId,
    pub fills: Vec<(Path<Pixels>, Hsla)>,
    pub strokes: Vec<(Path<Pixels>, Hsla)>,
    pub points: Vec<(Path<Pixels>, Hsla)>,
}

pub fn build_project_paths(
    layers: &[Layer],
    viewport: &Viewport,
) -> Result<Vec<LayerPaths>, RenderError> {
    build_project_paths_with_offset(layers, viewport, [0.0, 0.0])
}

pub fn build_project_paths_with_offset(
    layers: &[Layer],
    viewport: &Viewport,
    offset: [f32; 2],
) -> Result<Vec<LayerPaths>, RenderError> {
    let mut visible_layers: Vec<_> = layers.iter().filter(|layer| layer.visible).collect();
    visible_layers.sort_by_key(|layer| layer.z_order);
    visible_layers
        .into_iter()
        .map(|layer| build_layer_paths_with_offset(layer, viewport, offset))
        .collect()
}

pub fn build_layer_paths(layer: &Layer, viewport: &Viewport) -> Result<LayerPaths, RenderError> {
    build_layer_paths_with_offset(layer, viewport, [0.0, 0.0])
}

pub fn build_layer_paths_with_offset(
    layer: &Layer,
    viewport: &Viewport,
    offset: [f32; 2],
) -> Result<LayerPaths, RenderError> {
    let mut fill_builder = PathBuilder::fill();
    let mut stroke_builder = PathBuilder::stroke(px(layer.style.stroke_width.max(0.1)));
    let mut point_builder = PathBuilder::fill();

    let mut has_fill = false;
    let mut has_stroke = false;
    let mut has_points = false;
    let mut collector = PathCollector {
        viewport,
        fill_builder: &mut fill_builder,
        stroke_builder: &mut stroke_builder,
        point_builder: &mut point_builder,
        has_fill: &mut has_fill,
        has_stroke: &mut has_stroke,
        has_points: &mut has_points,
        point_radius: layer.style.point_radius.max(1.0),
        offset,
    };

    for feature in &layer.features {
        collector.append_feature(feature);
    }

    Ok(LayerPaths {
        layer_id: layer.id,
        fills: build_path_vec(has_fill, fill_builder, color_to_hsla(layer.style.fill))?,
        strokes: build_path_vec(
            has_stroke,
            stroke_builder,
            color_to_hsla(layer.style.stroke),
        )?,
        points: build_path_vec(has_points, point_builder, color_to_hsla(layer.style.fill))?,
    })
}

fn build_path_vec(
    has_geometry: bool,
    builder: PathBuilder,
    color: Hsla,
) -> Result<Vec<(Path<Pixels>, Hsla)>, RenderError> {
    if !has_geometry {
        return Ok(Vec::new());
    }
    Ok(vec![(builder.build()?, color)])
}

struct PathCollector<'a> {
    viewport: &'a Viewport,
    fill_builder: &'a mut PathBuilder,
    stroke_builder: &'a mut PathBuilder,
    point_builder: &'a mut PathBuilder,
    has_fill: &'a mut bool,
    has_stroke: &'a mut bool,
    has_points: &'a mut bool,
    point_radius: f32,
    offset: [f32; 2],
}

impl PathCollector<'_> {
    fn append_feature(&mut self, feature: &Feature) {
        self.append_geometry(&feature.geometry);
    }

    fn append_geometry(&mut self, geometry: &Geometry) {
        match geometry {
            Geometry::Point(point) => {
                append_point_circle(
                    point,
                    self.viewport,
                    self.point_builder,
                    self.point_radius,
                    self.offset,
                );
                *self.has_points = true;
            }
            Geometry::MultiPoint(points) => {
                append_multipoint(
                    points,
                    self.viewport,
                    self.point_builder,
                    self.has_points,
                    self.point_radius,
                    self.offset,
                );
            }
            Geometry::Line(line) => {
                append_line(line, self.viewport, self.stroke_builder, self.offset);
                *self.has_stroke = true;
            }
            Geometry::LineString(line_string) => {
                *self.has_stroke |= append_linestring(
                    line_string,
                    self.viewport,
                    self.stroke_builder,
                    false,
                    self.offset,
                );
            }
            Geometry::MultiLineString(lines) => {
                *self.has_stroke |=
                    append_multilinestring(lines, self.viewport, self.stroke_builder, self.offset);
            }
            Geometry::Polygon(polygon) => {
                *self.has_fill |=
                    append_polygon_fill(polygon, self.viewport, self.fill_builder, self.offset);
                *self.has_stroke |=
                    append_polygon_stroke(polygon, self.viewport, self.stroke_builder, self.offset);
            }
            Geometry::MultiPolygon(polygons) => {
                *self.has_fill |= append_multipolygon_fill(
                    polygons,
                    self.viewport,
                    self.fill_builder,
                    self.offset,
                );
                *self.has_stroke |= append_multipolygon_stroke(
                    polygons,
                    self.viewport,
                    self.stroke_builder,
                    self.offset,
                );
            }
            Geometry::Rect(rect) => {
                let polygon = Polygon::from(*rect);
                *self.has_fill |=
                    append_polygon_fill(&polygon, self.viewport, self.fill_builder, self.offset);
                *self.has_stroke |= append_polygon_stroke(
                    &polygon,
                    self.viewport,
                    self.stroke_builder,
                    self.offset,
                );
            }
            Geometry::Triangle(triangle) => {
                let polygon = triangle_to_polygon(*triangle);
                *self.has_fill |=
                    append_polygon_fill(&polygon, self.viewport, self.fill_builder, self.offset);
                *self.has_stroke |= append_polygon_stroke(
                    &polygon,
                    self.viewport,
                    self.stroke_builder,
                    self.offset,
                );
            }
            Geometry::GeometryCollection(collection) => {
                for geometry in &collection.0 {
                    self.append_geometry(geometry);
                }
            }
        }
    }
}

fn append_multipoint(
    points: &MultiPoint,
    viewport: &Viewport,
    point_builder: &mut PathBuilder,
    has_points: &mut bool,
    point_radius: f32,
    offset: [f32; 2],
) {
    for point in &points.0 {
        append_point_circle(point, viewport, point_builder, point_radius, offset);
        *has_points = true;
    }
}

fn append_line(line: &Line, viewport: &Viewport, builder: &mut PathBuilder, offset: [f32; 2]) {
    builder.move_to(screen_point(viewport, line.start, offset));
    builder.line_to(screen_point(viewport, line.end, offset));
}

fn append_multilinestring(
    lines: &MultiLineString,
    viewport: &Viewport,
    builder: &mut PathBuilder,
    offset: [f32; 2],
) -> bool {
    let mut added = false;
    for line in &lines.0 {
        added |= append_linestring(line, viewport, builder, false, offset);
    }
    added
}

fn append_polygon_fill(
    polygon: &Polygon,
    viewport: &Viewport,
    builder: &mut PathBuilder,
    offset: [f32; 2],
) -> bool {
    let mut added = append_ring(polygon.exterior(), viewport, builder, true, offset);
    for ring in polygon.interiors() {
        added |= append_ring(ring, viewport, builder, true, offset);
    }
    added
}

fn append_polygon_stroke(
    polygon: &Polygon,
    viewport: &Viewport,
    builder: &mut PathBuilder,
    offset: [f32; 2],
) -> bool {
    let mut added = append_ring(polygon.exterior(), viewport, builder, true, offset);
    for ring in polygon.interiors() {
        added |= append_ring(ring, viewport, builder, true, offset);
    }
    added
}

fn append_multipolygon_fill(
    polygons: &MultiPolygon,
    viewport: &Viewport,
    builder: &mut PathBuilder,
    offset: [f32; 2],
) -> bool {
    let mut added = false;
    for polygon in &polygons.0 {
        added |= append_polygon_fill(polygon, viewport, builder, offset);
    }
    added
}

fn append_multipolygon_stroke(
    polygons: &MultiPolygon,
    viewport: &Viewport,
    builder: &mut PathBuilder,
    offset: [f32; 2],
) -> bool {
    let mut added = false;
    for polygon in &polygons.0 {
        added |= append_polygon_stroke(polygon, viewport, builder, offset);
    }
    added
}

fn append_linestring(
    line_string: &LineString,
    viewport: &Viewport,
    builder: &mut PathBuilder,
    closed: bool,
    offset: [f32; 2],
) -> bool {
    let mut coords = line_string.coords();
    let Some(first) = coords.next() else {
        return false;
    };
    builder.move_to(screen_point(viewport, *first, offset));
    let mut point_count = 1usize;
    for coord in coords {
        builder.line_to(screen_point(viewport, *coord, offset));
        point_count += 1;
    }
    if closed && point_count > 2 {
        builder.close();
    }
    point_count > 1
}

fn append_ring(
    ring: &LineString,
    viewport: &Viewport,
    builder: &mut PathBuilder,
    close: bool,
    offset: [f32; 2],
) -> bool {
    append_linestring(ring, viewport, builder, close, offset)
}

fn append_point_circle(
    point_geometry: &Point,
    viewport: &Viewport,
    builder: &mut PathBuilder,
    radius: f32,
    offset: [f32; 2],
) {
    const SEGMENTS: usize = 16;
    let center = screen_point(viewport, point_geometry.0, offset);
    let center_x = f32::from(center.x);
    let center_y = f32::from(center.y);
    for index in 0..SEGMENTS {
        let angle = (index as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
        let point = point(
            px(center_x + radius * angle.cos()),
            px(center_y + radius * angle.sin()),
        );
        if index == 0 {
            builder.move_to(point);
        } else {
            builder.line_to(point);
        }
    }
    builder.close();
}

fn triangle_to_polygon(triangle: Triangle) -> Polygon {
    Polygon::new(
        vec![triangle.v1(), triangle.v2(), triangle.v3(), triangle.v1()].into(),
        Vec::new(),
    )
}

fn screen_point(viewport: &Viewport, coord: Coord, offset: [f32; 2]) -> GpuiPoint<Pixels> {
    let [x, y] = viewport.world_to_screen(coord);
    point(px(x + offset[0]), px(y + offset[1]))
}

fn color_to_hsla(color: rgis_core::Color) -> Hsla {
    Hsla::from(Rgba {
        r: color.r,
        g: color.g,
        b: color.b,
        a: color.a,
    })
}

#[cfg(test)]
mod tests {
    use geo_types::{Geometry, LineString, MultiPoint, Point, Polygon, polygon};
    use rgis_core::{Feature, Layer, LayerId, Viewport};

    use super::{build_layer_paths, build_project_paths};

    fn viewport() -> Viewport {
        Viewport {
            center: geo_types::Coord { x: 0.0, y: 0.0 },
            zoom: 2.0,
            width_px: 800,
            height_px: 600,
        }
    }

    #[test]
    fn builds_fill_stroke_and_point_paths() {
        let polygon = polygon![
            (x: -1000.0, y: -1000.0),
            (x: 1000.0, y: -1000.0),
            (x: 1000.0, y: 1000.0),
            (x: -1000.0, y: 1000.0),
            (x: -1000.0, y: -1000.0),
        ];
        let line = LineString::from(vec![(0.0, 0.0), (100.0, 100.0)]);
        let points = MultiPoint(vec![Point::new(10.0, 20.0), Point::new(20.0, 30.0)]);

        let layer = Layer::new(
            LayerId(1),
            "mixed",
            vec![
                Feature {
                    geometry: Geometry::Polygon(polygon),
                    properties: serde_json::Value::Null,
                },
                Feature {
                    geometry: Geometry::LineString(line),
                    properties: serde_json::Value::Null,
                },
                Feature {
                    geometry: Geometry::MultiPoint(points),
                    properties: serde_json::Value::Null,
                },
            ],
        );

        let paths = build_layer_paths(&layer, &viewport()).unwrap();
        assert_eq!(paths.fills.len(), 1);
        assert_eq!(paths.strokes.len(), 1);
        assert_eq!(paths.points.len(), 1);
    }

    #[test]
    fn filters_hidden_layers_from_project_paths() {
        let polygon = Polygon::new(
            LineString::from(vec![
                (-1.0, -1.0),
                (1.0, -1.0),
                (1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ]),
            Vec::new(),
        );
        let mut hidden = Layer::new(
            LayerId(2),
            "hidden",
            vec![Feature {
                geometry: Geometry::Polygon(polygon.clone()),
                properties: serde_json::Value::Null,
            }],
        );
        hidden.visible = false;
        let mut visible = Layer::new(
            LayerId(3),
            "visible",
            vec![Feature {
                geometry: Geometry::Polygon(polygon),
                properties: serde_json::Value::Null,
            }],
        );
        visible.z_order = 1;

        let paths = build_project_paths(&[hidden, visible], &viewport()).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].layer_id, LayerId(3));
    }
}
