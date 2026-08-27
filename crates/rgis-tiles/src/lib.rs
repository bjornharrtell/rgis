use std::sync::{Arc, Mutex};

use async_channel::{Receiver, Sender};
use image::RgbaImage;
use lru::LruCache;
use thiserror::Error;

mod vector;
pub use vector::{
    OPENFREEMAP_ATTRIBUTION, OPENFREEMAP_MAX_ZOOM, PropertyValue, VectorFeature, VectorTile,
    VectorTileFetcher, VectorTileLayer, VectorTileReady, decode_vector_tile,
};

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

pub struct Tile {
    pub coord: TileCoord,
    pub image: RgbaImage,
}

#[derive(Debug, Error)]
pub enum TileError {
    #[error("network error: {0}")]
    Network(String),
    #[error("image decode error: {0}")]
    Image(#[from] image::ImageError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("vector tile decode error: {0}")]
    Mvt(String),
}

// ── TileSource trait ──────────────────────────────────────────────────────────

pub trait TileSource: Send + Sync + 'static {
    fn url(&self, coord: TileCoord) -> String;
    fn attribution(&self) -> &str;
    fn max_zoom(&self) -> u8;
    fn tile_size_px(&self) -> u32 {
        256
    }
}

pub struct OsmTileSource;

impl TileSource for OsmTileSource {
    fn url(&self, c: TileCoord) -> String {
        let sub = b"abc"[(c.x as usize + c.y as usize) % 3] as char;
        format!(
            "https://{sub}.tile.openstreetmap.org/{}/{}/{}.png",
            c.z, c.x, c.y
        )
    }
    fn attribution(&self) -> &str {
        "\u{00A9} OpenStreetMap contributors"
    }
    fn max_zoom(&self) -> u8 {
        19
    }
}

// ── Disk cache helpers (native only; the browser has no filesystem) ─────────

#[cfg(not(target_arch = "wasm32"))]
mod disk_cache {
    use super::TileCoord;
    use directories::ProjectDirs;
    use image::RgbaImage;
    use std::path::PathBuf;

    fn cache_dir() -> Option<PathBuf> {
        ProjectDirs::from("rs", "", "rgis").map(|d| d.cache_dir().join("tiles"))
    }

    fn disk_path(coord: TileCoord) -> Option<PathBuf> {
        cache_dir().map(|d| {
            d.join(coord.z.to_string())
                .join(coord.x.to_string())
                .join(format!("{}.png", coord.y))
        })
    }

    pub fn read(coord: TileCoord) -> Option<RgbaImage> {
        let path = disk_path(coord)?;
        let bytes = std::fs::read(&path).ok()?;
        image::load_from_memory(&bytes).ok().map(|i| i.to_rgba8())
    }

    pub fn write(coord: TileCoord, bytes: &[u8]) {
        if let Some(path) = disk_path(coord) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&path, bytes);
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod disk_cache {
    use super::TileCoord;
    use image::RgbaImage;

    pub fn read(_coord: TileCoord) -> Option<RgbaImage> {
        None
    }

    pub fn write(_coord: TileCoord, _bytes: &[u8]) {}
}

// ── TileCache (in-memory LRU) ─────────────────────────────────────────────────

const MEMORY_CACHE_SIZE: usize = 256;

pub struct TileCache {
    lru: LruCache<TileCoord, Arc<RgbaImage>>,
}

impl TileCache {
    pub fn new() -> Self {
        Self {
            lru: LruCache::new(std::num::NonZeroUsize::new(MEMORY_CACHE_SIZE).unwrap()),
        }
    }

    pub fn get(&mut self, coord: TileCoord) -> Option<Arc<RgbaImage>> {
        self.lru.get(&coord).cloned()
    }

    pub fn insert(&mut self, coord: TileCoord, img: RgbaImage) -> Arc<RgbaImage> {
        let arc = Arc::new(img);
        self.lru.put(coord, Arc::clone(&arc));
        arc
    }
}

impl Default for TileCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── TileFetcher ───────────────────────────────────────────────────────────────

pub struct TileReady {
    pub coord: TileCoord,
    pub image: Arc<RgbaImage>,
}

pub struct TileFetcher {
    cache: Arc<Mutex<TileCache>>,
    source: Arc<dyn TileSource>,
    sender: Sender<TileReady>,
    pub receiver: Receiver<TileReady>,
}

impl TileFetcher {
    pub fn new(source: impl TileSource) -> Self {
        let (sender, receiver) = async_channel::bounded(256);
        Self {
            cache: Arc::new(Mutex::new(TileCache::new())),
            source: Arc::new(source),
            sender,
            receiver,
        }
    }

    pub fn attribution(&self) -> &str {
        self.source.attribution()
    }

    /// Request a tile. Delivery is asynchronous: the result (if any) shows up
    /// on `self.receiver`. Works identically on native (background thread,
    /// via `ehttp`'s `ureq` backend) and wasm32 (browser `fetch`).
    pub fn request(&self, coord: TileCoord) {
        if let Some(img) = self.cache.lock().unwrap().get(coord) {
            let _ = self.sender.try_send(TileReady { coord, image: img });
            return;
        }

        if let Some(img) = disk_cache::read(coord) {
            let arc = self.cache.lock().unwrap().insert(coord, img);
            let _ = self.sender.try_send(TileReady { coord, image: arc });
            return;
        }

        let cache = Arc::clone(&self.cache);
        let sender = self.sender.clone();
        let request = ehttp::Request::get(self.source.url(coord));

        ehttp::fetch(request, move |result: ehttp::Result<ehttp::Response>| {
            let Ok(response) = result else { return };
            if !response.ok {
                return;
            }
            let Ok(img) = image::load_from_memory(&response.bytes) else {
                return;
            };
            let rgba = img.to_rgba8();

            disk_cache::write(coord, &response.bytes);

            let arc = cache.lock().unwrap().insert(coord, rgba);
            let _ = sender.try_send(TileReady { coord, image: arc });
        });
    }
}

// ── Viewport -> visible tile coords ──────────────────────────────────────────

use rgis_core::{EARTH_HALF_CIRC, Viewport};

pub fn visible_tiles(viewport: &Viewport, source: &dyn TileSource) -> Vec<TileCoord> {
    visible_tiles_for_zoom(viewport, source.max_zoom())
}

/// Like [`visible_tiles`], but for sources (e.g. [`VectorTileFetcher`]) that
/// don't implement [`TileSource`].
pub fn visible_tiles_for_zoom(viewport: &Viewport, max_zoom: u8) -> Vec<TileCoord> {
    let z = (viewport.zoom.floor() as u8).min(max_zoom);
    let n = 2_u32.pow(z as u32) as f64;

    let merc_to_tile = |mx: f64, my: f64| -> (f64, f64) {
        let tx = (mx + EARTH_HALF_CIRC) / (2.0 * EARTH_HALF_CIRC) * n;
        let ty = (1.0 - (my + EARTH_HALF_CIRC) / (2.0 * EARTH_HALF_CIRC)) * n;
        (tx, ty)
    };

    let res = viewport.resolution();
    let half_w = viewport.width_px as f64 * res * 0.5;
    let half_h = viewport.height_px as f64 * res * 0.5;

    let cx = viewport.center.x;
    let cy = viewport.center.y;

    let (tx_min, ty_min) = merc_to_tile(cx - half_w, cy + half_h);
    let (tx_max, ty_max) = merc_to_tile(cx + half_w, cy - half_h);

    let x0 = (tx_min.floor() as i64).max(0) as u32;
    let x1 = (tx_max.ceil() as i64).min(n as i64 - 1).max(0) as u32;
    let y0 = (ty_min.floor() as i64).max(0) as u32;
    let y1 = (ty_max.ceil() as i64).min(n as i64 - 1).max(0) as u32;

    let mut tiles = Vec::new();
    for x in x0..=x1 {
        for y in y0..=y1 {
            tiles.push(TileCoord { z, x, y });
        }
    }
    tiles
}

pub fn tile_screen_rect(coord: TileCoord, viewport: &Viewport) -> [f32; 4] {
    let n = 2_u32.pow(coord.z as u32) as f64;
    let tile_merc_size = 2.0 * EARTH_HALF_CIRC / n;

    let mx = coord.x as f64 * tile_merc_size - EARTH_HALF_CIRC;
    let my = EARTH_HALF_CIRC - (coord.y + 1) as f64 * tile_merc_size;

    let [sx0, sy0] = viewport.world_to_screen(geo_types::Coord {
        x: mx,
        y: my + tile_merc_size,
    });
    let [sx1, sy1] = viewport.world_to_screen(geo_types::Coord {
        x: mx + tile_merc_size,
        y: my,
    });

    [sx0, sy0, sx1 - sx0, sy1 - sy0]
}
