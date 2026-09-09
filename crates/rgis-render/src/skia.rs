use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use geo_types::{Coord, Geometry, LineString, Polygon};
use image::RgbaImage;
use rgis_core::{Layer, Viewport};
use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

/// Stable texture key for the single composited user-vector image.
pub const VECTOR_TEXTURE_KEY: u64 = u64::MAX;

/// A cached vector image and the key used by the GPU texture cache.
#[derive(Clone)]
pub struct CachedVectorImage {
    pub key: u64,
    pub image: Arc<RgbaImage>,
    /// Screen-space destination rectangle. During a pan preview this can
    /// extend beyond the viewport while the cached image is repositioned.
    pub rect: [f32; 4],
}

/// Retains the last rendered vector image.
///
/// Map callbacks run for every window repaint, not just after map changes.
/// Keeping this cache outside the rasterizer avoids rebuilding the complete
/// layer image while the rest of the map is fetching or drawing tiles.
#[derive(Default)]
pub struct VectorRenderCache {
    layer_key: Option<u64>,
    viewport_key: Option<u64>,
    viewport: Option<Viewport>,
    image: Option<Arc<RgbaImage>>,
    layers: Option<Arc<Vec<Layer>>>,
}

impl VectorRenderCache {
    pub fn render(&mut self, layers: &[Layer], viewport: &Viewport) -> Option<CachedVectorImage> {
        self.render_with_preview(layers, viewport, false)
    }

    /// Returns a transformed preview of the last exact image while the
    /// viewport is being dragged. The next non-preview call renders the new
    /// viewport.
    pub fn render_with_preview(
        &mut self,
        layers: &[Layer],
        viewport: &Viewport,
        preview: bool,
    ) -> Option<CachedVectorImage> {
        self.snapshot_layers(layers);
        let layer_key = self.layer_key?;
        let viewport_key = viewport_key(viewport);
        if self.image.is_none() && preview {
            return None;
        }
        if self.image.is_none() || (!preview && self.viewport_key != Some(viewport_key)) {
            let image = Arc::new(render_vector_layers(layers, viewport)?);
            self.layer_key = Some(layer_key);
            self.viewport_key = Some(viewport_key);
            self.viewport = Some(viewport.clone());
            self.image = Some(image);
        }

        let image = self.image.as_ref()?;
        let cached_viewport = self.viewport.as_ref()?;
        let rect = if self.viewport_key == Some(viewport_key) {
            [
                0.0,
                0.0,
                viewport.width_px as f32,
                viewport.height_px as f32,
            ]
        } else {
            transformed_image_rect(cached_viewport, viewport, image)
        };
        Some(CachedVectorImage {
            key: VECTOR_TEXTURE_KEY,
            image: Arc::clone(image),
            rect,
        })
    }

    /// Snapshots layer geometry once so background viewport renders can reuse
    /// the same immutable data without cloning it for every interaction.
    pub fn snapshot_layers(&mut self, layers: &[Layer]) -> Arc<Vec<Layer>> {
        let layer_key = layer_render_key(layers);
        if self.layer_key != Some(layer_key) || self.layers.is_none() {
            self.layer_key = Some(layer_key);
            self.viewport_key = None;
            self.viewport = None;
            self.image = None;
            self.layers = Some(Arc::new(layers.to_vec()));
        }
        Arc::clone(self.layers.as_ref().expect("layer snapshot initialized"))
    }

    pub fn layer_key(&self) -> Option<u64> {
        self.layer_key
    }

    pub fn needs_render(&self, viewport: &Viewport) -> bool {
        self.image.is_none() || self.viewport_key != Some(viewport_key(viewport))
    }

    pub fn install_rendered(&mut self, viewport: &Viewport, image: RgbaImage) {
        self.viewport_key = Some(viewport_key(viewport));
        self.viewport = Some(viewport.clone());
        self.image = Some(Arc::new(image));
    }
}

/// Rasterizes plain vector layers into a premultiplied-alpha-free image.
///
/// Plain layers are deliberately kept out of the wgpu tessellation path. Skia
/// handles polygon filling, antialiasing, and line joins/caps here, while the
/// resulting image is uploaded through the existing texture path.
pub fn render_vector_layers(layers: &[Layer], viewport: &Viewport) -> Option<RgbaImage> {
    render_vector_layers_cancellable(layers, viewport, &AtomicU64::new(0), 0)
}

/// Rasterizes vector layers and returns `None` if `generation` is no longer
/// current. The checks are intentionally between features and scanline
/// conversion chunks so a superseded viewport render can stop promptly.
pub fn render_vector_layers_cancellable(
    layers: &[Layer],
    viewport: &Viewport,
    generation_counter: &AtomicU64,
    generation: u64,
) -> Option<RgbaImage> {
    let cancelled = || generation_counter.load(Ordering::Acquire) != generation;
    if cancelled() {
        return None;
    }
    let width = viewport.width_px.max(1);
    let height = viewport.height_px.max(1);
    let mut pixmap = Pixmap::new(width, height)?;
    let visible_bounds = viewport_bounds(viewport);
    let mut visible: Vec<_> = layers.iter().filter(|layer| layer.visible).collect();
    visible.sort_by_key(|layer| layer.z_order);
    for layer in visible {
        if cancelled() {
            return None;
        }
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
        if let Some(bounds) = layer.bounds
            && !bounds_intersect(bounds, visible_bounds)
        {
            continue;
        }
        for (index, feature) in layer.features.iter().enumerate() {
            if cancelled() {
                return None;
            }
            if let Some(Some(bounds)) = layer.feature_bounds.get(index)
                && !bounds_intersect(*bounds, visible_bounds)
            {
                continue;
            }
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
    for (index, pixel) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        if index % 4096 == 0 && cancelled() {
            return None;
        }
        let alpha = pixel[3] as u32;
        if alpha != 0 && alpha != 255 {
            for channel in &mut pixel[..3] {
                *channel = ((*channel as u32 * 255 + alpha / 2) / alpha) as u8;
            }
        }
    }
    RgbaImage::from_raw(width, height, data)
}

fn viewport_bounds(viewport: &Viewport) -> rgis_core::Bounds {
    let half_width = viewport.width_px as f64 * viewport.resolution() * 0.5;
    let half_height = viewport.height_px as f64 * viewport.resolution() * 0.5;
    rgis_core::Bounds {
        min_x: viewport.center.x - half_width,
        min_y: viewport.center.y - half_height,
        max_x: viewport.center.x + half_width,
        max_y: viewport.center.y + half_height,
    }
}

fn bounds_intersect(a: rgis_core::Bounds, b: rgis_core::Bounds) -> bool {
    a.min_x <= b.max_x && a.max_x >= b.min_x && a.min_y <= b.max_y && a.max_y >= b.min_y
}

fn layer_render_key(layers: &[Layer]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for layer in layers {
        layer.id.hash(&mut hasher);
        layer.visible.hash(&mut hasher);
        layer.z_order.hash(&mut hasher);
        layer.features.len().hash(&mut hasher);
        layer.style.fill.r.to_bits().hash(&mut hasher);
        layer.style.fill.g.to_bits().hash(&mut hasher);
        layer.style.fill.b.to_bits().hash(&mut hasher);
        layer.style.fill.a.to_bits().hash(&mut hasher);
        layer.style.stroke.r.to_bits().hash(&mut hasher);
        layer.style.stroke.g.to_bits().hash(&mut hasher);
        layer.style.stroke.b.to_bits().hash(&mut hasher);
        layer.style.stroke.a.to_bits().hash(&mut hasher);
        layer.style.stroke_width.to_bits().hash(&mut hasher);
        layer.style.point_radius.to_bits().hash(&mut hasher);
    }

    hasher.finish() | (1 << 63)
}

fn viewport_key(viewport: &Viewport) -> u64 {
    let mut hasher = DefaultHasher::new();
    viewport.center.x.to_bits().hash(&mut hasher);
    viewport.center.y.to_bits().hash(&mut hasher);
    viewport.zoom.to_bits().hash(&mut hasher);
    viewport.width_px.hash(&mut hasher);
    viewport.height_px.hash(&mut hasher);
    hasher.finish()
}

fn transformed_image_rect(
    cached_viewport: &Viewport,
    viewport: &Viewport,
    image: &RgbaImage,
) -> [f32; 4] {
    let scale = cached_viewport.resolution() / viewport.resolution();
    let width = image.width() as f64 * scale;
    let height = image.height() as f64 * scale;
    let offset_x = (cached_viewport.center.x - viewport.center.x) / viewport.resolution();
    let offset_y = (viewport.center.y - cached_viewport.center.y) / viewport.resolution();
    [
        (viewport.width_px as f64 * 0.5 + offset_x - width * 0.5) as f32,
        (viewport.height_px as f64 * 0.5 + offset_y - height * 0.5) as f32,
        width as f32,
        height as f32,
    ]
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
        Geometry::Rect(rect) => {
            move_polygon(&mut path, viewport, &Polygon::from(*rect));
        }
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
    let mut coords = line.coords();
    let Some(first) = coords.next() else { return };
    let [x, y] = screen(viewport, *first);
    path.move_to(x, y);
    for coord in coords {
        let [x, y] = screen(viewport, *coord);
        path.line_to(x, y);
    }
    if close {
        path.close();
    }
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
        let viewport = Viewport {
            width_px: 32,
            height_px: 32,
            ..Viewport::default()
        };
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

    #[test]
    fn vector_render_cache_reuses_unchanged_image() {
        let viewport = Viewport {
            width_px: 32,
            height_px: 32,
            ..Viewport::default()
        };
        let layer = Layer::new(
            LayerId(1),
            "points",
            vec![Feature {
                geometry: Geometry::Point(Point::new(0.0, 0.0)),
                properties: Default::default(),
            }],
        );
        let mut cache = VectorRenderCache::default();
        let first = cache
            .render(std::slice::from_ref(&layer), &viewport)
            .unwrap();
        let second = cache.render(&[layer], &viewport).unwrap();
        assert!(Arc::ptr_eq(&first.image, &second.image));
        assert_eq!(first.key, second.key);
        assert_eq!(first.rect, [0.0, 0.0, 32.0, 32.0]);
    }

    #[test]
    fn vector_render_cache_transforms_image_during_preview() {
        let viewport = Viewport {
            width_px: 32,
            height_px: 32,
            ..Viewport::default()
        };
        let layer = Layer::new(
            LayerId(1),
            "points",
            vec![Feature {
                geometry: Geometry::Point(Point::new(0.0, 0.0)),
                properties: Default::default(),
            }],
        );
        let mut cache = VectorRenderCache::default();
        let first = cache
            .render(std::slice::from_ref(&layer), &viewport)
            .unwrap();
        let mut panned = viewport;
        panned.pan(8.0, 0.0);
        let preview = cache
            .render_with_preview(std::slice::from_ref(&layer), &panned, true)
            .unwrap();
        assert!(Arc::ptr_eq(&first.image, &preview.image));
        assert_eq!(preview.rect[2..], [32.0, 32.0]);
        assert_eq!(preview.rect[0], 8.0);
        let exact = cache.render(std::slice::from_ref(&layer), &panned).unwrap();
        assert!(!Arc::ptr_eq(&first.image, &exact.image));
        assert_eq!(exact.rect, [0.0, 0.0, 32.0, 32.0]);
    }
}
