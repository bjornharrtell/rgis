use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use rgis_render::{StyleSheet, TileDraw};
use rgis_tiles::{
    StyleRasterSource, TileCoord, TileFetcher, tile_screen_rect, visible_tiles_for_zoom,
};

const RASTER_TILE_CACHE_SIZE: usize = 256;

pub type RasterFetchers = HashMap<String, Arc<TileFetcher>>;
pub type RasterTileCaches = HashMap<String, LruCache<TileCoord, Arc<image::RgbaImage>>>;

pub fn fetchers_for_style(style: &StyleSheet) -> RasterFetchers {
    let mut fetchers = HashMap::new();
    for layer in style.layers_of_kind("raster") {
        let Some(source_id) = &layer.source else {
            continue;
        };
        if fetchers.contains_key(source_id) {
            continue;
        }
        let Some(source) = style.sources.get(source_id) else {
            continue;
        };
        let Some(template) = source.tiles.as_ref().and_then(|tiles| tiles.first()) else {
            continue;
        };
        let raster_source = StyleRasterSource::new(
            template.clone(),
            source.maxzoom.unwrap_or(22.0) as u8,
            source.tile_size.unwrap_or(256),
        );
        fetchers.insert(source_id.clone(), Arc::new(TileFetcher::new(raster_source)));
    }
    fetchers
}

pub fn drain_fetchers(fetchers: &RasterFetchers, caches: &mut RasterTileCaches) {
    for (source_id, fetcher) in fetchers {
        let cache = caches
            .entry(source_id.clone())
            .or_insert_with(|| LruCache::new(NonZeroUsize::new(RASTER_TILE_CACHE_SIZE).unwrap()));
        while let Ok(ready) = fetcher.receiver.try_recv() {
            cache.put(ready.coord, ready.image);
        }
    }
}

pub fn tile_draw_key(source_id: &str, coord: TileCoord) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source_id.hash(&mut hasher);
    coord.z.hash(&mut hasher);
    coord.x.hash(&mut hasher);
    coord.y.hash(&mut hasher);
    hasher.finish()
}

pub fn ancestor_coords(coord: TileCoord) -> impl Iterator<Item = TileCoord> {
    let mut ancestors = Vec::new();
    let (mut z, mut x, mut y) = (coord.z, coord.x, coord.y);
    while z > 0 {
        z -= 1;
        x /= 2;
        y /= 2;
        ancestors.push(TileCoord { z, x, y });
    }
    ancestors.into_iter()
}

pub fn collect_draws(
    style: &StyleSheet,
    viewport: &rgis_core::Viewport,
    fetchers: &RasterFetchers,
    caches: &mut RasterTileCaches,
) -> Vec<TileDraw> {
    let mut draws = Vec::new();
    for layer in style.layers_of_kind("raster") {
        if !layer.matches_zoom(viewport.zoom) {
            continue;
        }
        let Some(source_id) = &layer.source else {
            continue;
        };
        let Some(fetcher) = fetchers.get(source_id) else {
            continue;
        };
        let coords = visible_tiles_for_zoom(viewport, fetcher.max_zoom());
        let eval_ctx = rgis_render::EvalContext::new(viewport.zoom);
        let opacity = layer.paint("raster-opacity").eval_f64(&eval_ctx, 1.0) as f32;
        let cache = caches
            .entry(source_id.clone())
            .or_insert_with(|| LruCache::new(NonZeroUsize::new(RASTER_TILE_CACHE_SIZE).unwrap()));
        for coord in coords {
            let image = cache.get(&coord).cloned();
            if let Some(image) = image {
                draws.push(TileDraw {
                    key: tile_draw_key(source_id, coord),
                    rect: tile_screen_rect(coord, viewport),
                    rgba: image,
                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                    opacity,
                });
                continue;
            }

            fetcher.request(coord);
            let mut ancestor = coord;
            while ancestor.z > 0 {
                ancestor = TileCoord {
                    z: ancestor.z - 1,
                    x: ancestor.x / 2,
                    y: ancestor.y / 2,
                };
                if let Some(image) = cache.get(&ancestor).cloned() {
                    let dz = coord.z - ancestor.z;
                    let divisor = 1_u32 << dz;
                    let child_x = coord.x % divisor;
                    let child_y = coord.y % divisor;
                    let min_u = child_x as f32 / divisor as f32;
                    let min_v = child_y as f32 / divisor as f32;
                    let max_u = (child_x + 1) as f32 / divisor as f32;
                    let max_v = (child_y + 1) as f32 / divisor as f32;
                    draws.push(TileDraw {
                        key: tile_draw_key(source_id, ancestor),
                        rect: tile_screen_rect(coord, viewport),
                        rgba: image,
                        uv_rect: [min_u, min_v, max_u, max_v],
                        opacity,
                    });
                    break;
                }
                fetcher.request(ancestor);
            }
        }
    }
    draws
}
