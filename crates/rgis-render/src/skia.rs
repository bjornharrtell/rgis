use geo_types::{Coord, Geometry, LineString, Polygon};
use image::RgbaImage;
use rgis_core::{Layer, Viewport};
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

/// Rasterizes plain vector layers into a premultiplied-alpha-free image.
///
/// Plain layers are deliberately kept out of the wgpu tessellation path. Skia
/// handles polygon filling, antialiasing, and line joins/caps here, while the
/// resulting image is uploaded through the existing texture path.
pub fn render_vector_layers(layers: &[Layer], viewport: &Viewport) -> Option<RgbaImage> {
    let width = viewport.width_px.max(1);
    let height = viewport.height_px.max(1);
    let mut pixmap = Pixmap::new(width, height)?;

    let mut visible: Vec<_> = layers.iter().filter(|layer| layer.visible).collect();
    visible.sort_by_key(|layer| layer.z_order);
    for layer in visible {
        let fill = Color::from_rgba(
            layer.style.fill.r,
            layer.style.fill.g,
            layer.style.fill.b,
            layer.style.fill.a,
        )
        .unwrap();
        let stroke = Color::from_rgba(
            layer.style.stroke.r,
            layer.style.stroke.g,
            layer.style.stroke.b,
            layer.style.stroke.a,
        )
        .unwrap();
        for feature in &layer.features {
            draw_geometry(
                &mut pixmap,
                &feature.geometry,
                viewport,
                fill,
                stroke,
                layer.style.stroke_width.max(0.1),
                layer.style.point_radius.max(1.0),
            );
        }
    }

    // tiny-skia stores premultiplied RGBA, while the wgpu texture pipeline
    // expects straight alpha.
    let mut data = pixmap.take();
    for pixel in data.chunks_exact_mut(4) {
        let alpha = pixel[3] as u32;
        if alpha != 0 && alpha != 255 {
            for channel in &mut pixel[..3] {
                *channel = ((*channel as u32 * 255 + alpha / 2) / alpha) as u8;
            }
        }
    }
    RgbaImage::from_raw(width, height, data)
}

fn draw_geometry(
    pixmap: &mut Pixmap,
    geometry: &Geometry,
    viewport: &Viewport,
    fill: Color,
    stroke: Color,
    stroke_width: f32,
    point_radius: f32,
) {
    let mut path = PathBuilder::new();
    match geometry {
        Geometry::Point(point) => {
            let [x, y] = screen(viewport, point.0);
            path.push_circle(x, y, point_radius);
        }
        Geometry::MultiPoint(points) => {
            for point in &points.0 {
                let [x, y] = screen(viewport, point.0);
                path.push_circle(x, y, point_radius);
            }
        }
        Geometry::Line(line) => {
            move_line(&mut path, viewport, &[line.start, line.end], false);
        }
        Geometry::LineString(line) => move_linestring(&mut path, viewport, line, false),
        Geometry::MultiLineString(lines) => {
            for line in &lines.0 {
                move_linestring(&mut path, viewport, line, false);
            }
        }
        Geometry::Polygon(polygon) => move_polygon(&mut path, viewport, polygon),
        Geometry::MultiPolygon(polygons) => {
            for polygon in &polygons.0 {
                move_polygon(&mut path, viewport, polygon);
            }
        }
        Geometry::Rect(rect) => move_polygon(&mut path, viewport, &Polygon::from(*rect)),
        Geometry::Triangle(triangle) => {
            move_line(
                &mut path,
                viewport,
                &[triangle.v1(), triangle.v2(), triangle.v3(), triangle.v1()],
                true,
            );
        }
        Geometry::GeometryCollection(collection) => {
            for geometry in &collection.0 {
                draw_geometry(
                    pixmap,
                    geometry,
                    viewport,
                    fill,
                    stroke,
                    stroke_width,
                    point_radius,
                );
            }
            return;
        }
    }
    let Some(path) = path.finish() else { return };
    let mut fill_paint = Paint::default();
    fill_paint.set_color(fill);
    pixmap.fill_path(
        &path,
        &fill_paint,
        FillRule::EvenOdd,
        Transform::identity(),
        None,
    );
    let mut stroke_paint = Paint::default();
    stroke_paint.set_color(stroke);
    let stroke_style = Stroke {
        width: stroke_width,
        line_cap: tiny_skia::LineCap::Round,
        line_join: tiny_skia::LineJoin::Round,
        ..Stroke::default()
    };
    pixmap.stroke_path(
        &path,
        &stroke_paint,
        &stroke_style,
        Transform::identity(),
        None,
    );
}

fn move_polygon(path: &mut PathBuilder, viewport: &Viewport, polygon: &Polygon) {
    move_linestring(path, viewport, polygon.exterior(), true);
    for ring in polygon.interiors() {
        move_linestring(path, viewport, ring, true);
    }
}

fn move_linestring(path: &mut PathBuilder, viewport: &Viewport, line: &LineString, close: bool) {
    let coords: Vec<_> = line.coords().copied().collect();
    move_line(path, viewport, &coords, close);
}

fn move_line(path: &mut PathBuilder, viewport: &Viewport, coords: &[Coord], close: bool) {
    let Some(first) = coords.first() else { return };
    let [x, y] = screen(viewport, *first);
    path.move_to(x, y);
    for coord in &coords[1..] {
        let [x, y] = screen(viewport, *coord);
        path.line_to(x, y);
    }
    if close {
        path.close();
    }
}

fn screen(viewport: &Viewport, coord: Coord) -> [f32; 2] {
    viewport.world_to_screen(coord)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Geometry, Point};
    use rgis_core::{Feature, Layer, LayerId};

    #[test]
    fn rasterizes_plain_point_without_tessellating() {
        let mut viewport = Viewport::default();
        viewport.width_px = 32;
        viewport.height_px = 32;
        let layer = Layer::new(
            LayerId(1),
            "points",
            vec![Feature {
                geometry: Geometry::Point(Point::new(0.0, 0.0)),
                properties: Default::default(),
            }],
        );
        let image = render_vector_layers(&[layer], &viewport).unwrap();
        assert!(image.pixels().any(|pixel| pixel[3] != 0));
    }
}
