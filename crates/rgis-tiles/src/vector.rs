//! Mapbox Vector Tile (MVT) decoding and fetching, used to render the
//! [OpenFreeMap](https://openfreemap.org/) vector basemap.
//!
//! OpenFreeMap (like any MapLibre-style vector tile source) publishes a
//! [TileJSON](https://github.com/mapbox/tilejson-spec) document at a fixed
//! URL whose `tiles` template embeds a data-version path segment that
//! changes over time, so the actual `{z}/{x}/{y}.pbf` template is resolved
//! once at startup rather than hard-coded.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use async_channel::{Receiver, Sender};
use fast_mvt::{MvtReaderRef, MvtValueRef};
use lru::LruCache;

use crate::{TileCoord, TileError};

// ── Decoded tile types ───────────────────────────────────────────────────────

/// A decoded MVT feature property value.
#[derive(Debug, Clone)]
pub enum PropertyValue {
    String(String),
    Number(f64),
    Bool(bool),
}

/// A single decoded feature, with geometry in tile-local coordinates
/// (`[0, layer.extent]`, Y axis pointing down) rather than a real-world CRS.
#[derive(Debug, Clone)]
pub struct VectorFeature {
    pub geometry: geo_types::Geometry<i32>,
    pub properties: HashMap<String, PropertyValue>,
}

impl VectorFeature {
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.properties.get(key) {
            Some(PropertyValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_number(&self, key: &str) -> Option<f64> {
        match self.properties.get(key) {
            Some(PropertyValue::Number(n)) => Some(*n),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorTileLayer {
    pub name: String,
    /// Size of the tile in local coordinate units (commonly 4096).
    pub extent: u32,
    pub features: Vec<VectorFeature>,
}

#[derive(Debug, Clone, Default)]
pub struct VectorTile {
    pub layers: Vec<VectorTileLayer>,
}

/// Decodes a raw `.pbf` Mapbox Vector Tile payload.
pub fn decode_vector_tile(bytes: &[u8]) -> Result<VectorTile, TileError> {
    let reader = MvtReaderRef::new(bytes).map_err(|e| TileError::Mvt(e.to_string()))?;
    let mut layers = Vec::new();
    for layer in reader.layers() {
        let mut features = Vec::with_capacity(layer.feature_count());
        for feature in layer.features() {
            let Ok(geometry) = feature.geometry() else {
                continue;
            };
            let mut properties = HashMap::new();
            if let Ok(props) = feature.properties_vec() {
                for (key, value) in props {
                    properties.insert(key.to_string(), convert_value(value));
                }
            }
            features.push(VectorFeature {
                geometry,
                properties,
            });
        }
        layers.push(VectorTileLayer {
            name: layer.name().to_string(),
            extent: layer.extent(),
            features,
        });
    }
    Ok(VectorTile { layers })
}

fn convert_value(value: MvtValueRef) -> PropertyValue {
    match value {
        MvtValueRef::String(s) => PropertyValue::String(s.to_string()),
        MvtValueRef::Float(f) => PropertyValue::Number(f as f64),
        MvtValueRef::Double(d) => PropertyValue::Number(d),
        MvtValueRef::Int(i) | MvtValueRef::SInt(i) => PropertyValue::Number(i as f64),
        MvtValueRef::UInt(u) => PropertyValue::Number(u as f64),
        MvtValueRef::Bool(b) => PropertyValue::Bool(b),
        MvtValueRef::Null => PropertyValue::String(String::new()),
    }
}

// ── Disk cache helpers (native only; the browser has no filesystem) ─────────

#[cfg(not(target_arch = "wasm32"))]
mod disk_cache {
    use super::TileCoord;
    use directories::ProjectDirs;
    use std::path::PathBuf;

    fn cache_dir() -> Option<PathBuf> {
        ProjectDirs::from("rs", "", "rgis").map(|d| d.cache_dir().join("vector-tiles"))
    }

    fn disk_path(coord: TileCoord) -> Option<PathBuf> {
        cache_dir().map(|d| {
            d.join(coord.z.to_string())
                .join(coord.x.to_string())
                .join(format!("{}.pbf", coord.y))
        })
    }

    pub fn read(coord: TileCoord) -> Option<Vec<u8>> {
        std::fs::read(disk_path(coord)?).ok()
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

    pub fn read(_coord: TileCoord) -> Option<Vec<u8>> {
        None
    }

    pub fn write(_coord: TileCoord, _bytes: &[u8]) {}
}

// ── VectorTileFetcher ─────────────────────────────────────────────────────────

const OPENFREEMAP_TILEJSON_URL: &str = "https://tiles.openfreemap.org/planet";
pub const OPENFREEMAP_ATTRIBUTION: &str =
    "OpenFreeMap \u{00A9} OpenMapTiles Data from OpenStreetMap contributors";
pub const OPENFREEMAP_MAX_ZOOM: u8 = 14;

const MEMORY_CACHE_SIZE: usize = 128;

pub struct VectorTileReady {
    pub coord: TileCoord,
    pub tile: Arc<VectorTile>,
}

/// A tile whose bytes have been fetched (from the network) but not yet
/// decoded. MVT decoding parses every feature's geometry and properties out
/// of the protobuf payload, which can be substantial CPU work for a busy
/// tile -- deferring it here (rather than doing it inline in the network
/// completion callback) lets callers decode it as part of their own
/// throttled/backgrounded work, instead of it running unconditionally and
/// synchronously the moment a response arrives (which, on wasm, is on the
/// browser's single JS thread, with no way to time-slice it).
pub struct VectorTileFetched {
    pub coord: TileCoord,
    pub bytes: Vec<u8>,
}

/// The `{z}/{x}/{y}` tile URL template, resolved asynchronously from a
/// TileJSON document. Requests made before it resolves are queued.
enum TemplateState {
    Loading(Vec<TileCoord>),
    Ready(String),
    /// The TileJSON fetch failed; give up silently, matching the raster
    /// fetcher's "ignore failures" behaviour.
    Failed,
}

pub struct VectorTileFetcher {
    cache: Arc<Mutex<LruCache<TileCoord, Arc<VectorTile>>>>,
    sender: Sender<VectorTileReady>,
    pub receiver: Receiver<VectorTileReady>,
    raw_sender: Sender<VectorTileFetched>,
    pub raw_receiver: Receiver<VectorTileFetched>,
    template: Arc<Mutex<TemplateState>>,
}

impl VectorTileFetcher {
    /// Creates a fetcher for OpenFreeMap's public vector tile service,
    /// kicking off the TileJSON resolution in the background.
    pub fn new_openfreemap() -> Arc<Self> {
        let (sender, receiver) = async_channel::bounded(256);
        let (raw_sender, raw_receiver) = async_channel::bounded(256);
        let fetcher = Arc::new(Self {
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(MEMORY_CACHE_SIZE).unwrap(),
            ))),
            sender,
            receiver,
            raw_sender,
            raw_receiver,
            template: Arc::new(Mutex::new(TemplateState::Loading(Vec::new()))),
        });

        let callback_fetcher = Arc::clone(&fetcher);
        let request = ehttp::Request::get(OPENFREEMAP_TILEJSON_URL);
        ehttp::fetch(request, move |result: ehttp::Result<ehttp::Response>| {
            let template = result.ok().filter(|r| r.ok).and_then(|r| {
                let json: serde_json::Value = serde_json::from_slice(&r.bytes).ok()?;
                json.get("tiles")?.get(0)?.as_str().map(|s| s.to_string())
            });
            callback_fetcher.resolve_template(template);
        });

        fetcher
    }

    fn resolve_template(&self, template: Option<String>) {
        let pending = {
            let mut state = self.template.lock().unwrap();
            let previous = std::mem::replace(
                &mut *state,
                match &template {
                    Some(url) => TemplateState::Ready(url.clone()),
                    None => TemplateState::Failed,
                },
            );
            match previous {
                TemplateState::Loading(coords) => coords,
                _ => Vec::new(),
            }
        };
        for coord in pending {
            self.request(coord);
        }
    }

    /// Request a tile. Delivery is asynchronous: the result (if any) shows
    /// up on `self.receiver`.
    pub fn request(&self, coord: TileCoord) {
        if let Some(tile) = self.cache.lock().unwrap().get(&coord).cloned() {
            let _ = self.sender.try_send(VectorTileReady { coord, tile });
            return;
        }

        if let Some(bytes) = disk_cache::read(coord)
            && let Ok(tile) = decode_vector_tile(&bytes)
        {
            let arc = self.cache_insert(coord, tile);
            let _ = self.sender.try_send(VectorTileReady { coord, tile: arc });
            return;
        }

        let template = {
            let mut state = self.template.lock().unwrap();
            match &mut *state {
                TemplateState::Ready(template) => Some(template.clone()),
                TemplateState::Loading(pending) => {
                    pending.push(coord);
                    None
                }
                TemplateState::Failed => None,
            }
        };
        if let Some(template) = template {
            self.fetch_url(coord, tile_url(&template, coord));
        }
    }

    fn cache_insert(&self, coord: TileCoord, tile: VectorTile) -> Arc<VectorTile> {
        let arc = Arc::new(tile);
        self.cache.lock().unwrap().put(coord, Arc::clone(&arc));
        arc
    }

    /// Decodes raw MVT bytes (as delivered via `raw_receiver`) and stores
    /// the result in the decoded-tile cache, so a later `request()` for the
    /// same coord can skip the network+decode entirely. Meant to be called
    /// from whatever backgrounded/throttled job the caller uses to consume
    /// `raw_receiver`, not inline on a UI/main thread.
    pub fn decode_and_cache(
        &self,
        coord: TileCoord,
        bytes: &[u8],
    ) -> Result<Arc<VectorTile>, TileError> {
        let tile = decode_vector_tile(bytes)?;
        Ok(self.cache_insert(coord, tile))
    }

    fn fetch_url(&self, coord: TileCoord, url: String) {
        let raw_sender = self.raw_sender.clone();
        let request = ehttp::Request::get(url);

        ehttp::fetch(request, move |result: ehttp::Result<ehttp::Response>| {
            let Ok(response) = result else { return };
            if !response.ok {
                return;
            }
            disk_cache::write(coord, &response.bytes);
            let _ = raw_sender.try_send(VectorTileFetched {
                coord,
                bytes: response.bytes,
            });
        });
    }
}

fn tile_url(template: &str, coord: TileCoord) -> String {
    template
        .replace("{z}", &coord.z.to_string())
        .replace("{x}", &coord.x.to_string())
        .replace("{y}", &coord.y.to_string())
}
