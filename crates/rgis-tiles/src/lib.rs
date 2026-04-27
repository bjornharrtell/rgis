use std::{
    path::PathBuf,
    sync::Arc,
};

use async_channel::{Receiver, Sender};
use directories::ProjectDirs;
use image::RgbaImage;
use lru::LruCache;
use thiserror::Error;
use tokio::sync::Mutex;

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
    Network(#[from] reqwest::Error),
    #[error("image decode error: {0}")]
    Image(#[from] image::ImageError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
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
        format!("https://{sub}.tile.openstreetmap.org/{}/{}/{}.png", c.z, c.x, c.y)
    }
    fn attribution(&self) -> &str {
        "\u{00A9} OpenStreetMap contributors"
    }
    fn max_zoom(&self) -> u8 {
        19
    }
}

// ── Disk cache helpers ────────────────────────────────────────────────────────

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

async fn read_disk(coord: TileCoord) -> Option<RgbaImage> {
    let path = disk_path(coord)?;
    let bytes = tokio::fs::read(&path).await.ok()?;
    image::load_from_memory(&bytes).ok().map(|i| i.to_rgba8())
}

async fn write_disk(coord: TileCoord, bytes: &[u8]) {
    if let Some(path) = disk_path(coord) {
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let _ = tokio::fs::write(&path, bytes).await;
    }
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
    client: reqwest::Client,
}

impl TileFetcher {
    pub fn new(source: impl TileSource) -> Self {
        let (sender, receiver) = async_channel::bounded(256);
        let client = reqwest::Client::builder()
            .user_agent("rgis/0.1 (https://github.com/yourname/rgis)")
            .build()
            .expect("failed to build reqwest client");
        Self {
            cache: Arc::new(Mutex::new(TileCache::new())),
            source: Arc::new(source),
            sender,
            receiver,
            client,
        }
    }

    pub fn attribution(&self) -> &str {
        self.source.attribution()
    }

    pub fn request(&self, coord: TileCoord) {
        let cache = Arc::clone(&self.cache);
        let source = Arc::clone(&self.source);
        let sender = self.sender.clone();
        let client = self.client.clone();

        tokio::spawn(async move {
            {
                let mut c = cache.lock().await;
                if let Some(img) = c.get(coord) {
                    let _ = sender.send(TileReady { coord, image: img }).await;
                    return;
                }
            }

            if let Some(img) = read_disk(coord).await {
                let arc = cache.lock().await.insert(coord, img);
                let _ = sender.send(TileReady { coord, image: arc }).await;
                return;
            }

            let url = source.url(coord);
            let Ok(resp) = client.get(&url).send().await else { return };
            if !resp.status().is_success() {
                return;
            }
            let Ok(bytes) = resp.bytes().await else { return };

            let Ok(img) = image::load_from_memory(&bytes) else { return };
            let rgba = img.to_rgba8();

            write_disk(coord, &bytes).await;

            let arc = cache.lock().await.insert(coord, rgba);
            let _ = sender.send(TileReady { coord, image: arc }).await;
        });
    }
}

// ── Viewport -> visible tile coords ──────────────────────────────────────────

use rgis_core::{Viewport, EARTH_HALF_CIRC};

pub fn visible_tiles(viewport: &Viewport, source: &dyn TileSource) -> Vec<TileCoord> {
    let z = (viewport.zoom.floor() as u8).min(source.max_zoom());
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

    let [sx0, sy0] = viewport.world_to_screen(geo_types::Coord { x: mx, y: my + tile_merc_size });
    let [sx1, sy1] = viewport.world_to_screen(geo_types::Coord { x: mx + tile_merc_size, y: my });

    [sx0, sy0, sx1 - sx0, sy1 - sy0]
}
