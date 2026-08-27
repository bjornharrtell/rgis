use geo::MapCoords;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("unsupported file extension: {0}")]
    UnsupportedFormat(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("geojson error: {0}")]
    GeoJson(String),
    #[error("shapefile error: {0}")]
    Shapefile(String),
    #[error("flatgeobuf error: {0}")]
    FlatGeobuf(String),
}

mod flatgeobuf_reader;
mod geojson_reader;
mod shapefile_reader;

pub use flatgeobuf_reader::load_flatgeobuf;
pub use geojson_reader::load_geojson;
pub use shapefile_reader::load_shapefile;

use rgis_core::Feature;

pub struct LoadedLayer {
    pub name: String,
    pub features: Vec<Feature>,
}

/// Load a layer from a file path, dispatching on file extension. Native-only:
/// shapefiles need their `.shx`/`.dbf` sibling files, which only a real
/// filesystem path (not in-memory bytes) can resolve.
pub fn load_path(path: impl AsRef<std::path::Path>) -> Result<LoadedLayer, IoError> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let mut loaded = match ext.as_str() {
        "geojson" | "json" => load_geojson(path)?,
        "shp" => load_shapefile(path)?,
        "fgb" => load_flatgeobuf(path)?,
        other => return Err(IoError::UnsupportedFormat(other.to_owned())),
    };
    loaded.features = reproject_features(loaded.features);
    Ok(loaded)
}

/// Load a layer from an in-memory byte buffer, dispatching on the file name's
/// extension. Works on both native and wasm32 (e.g. bytes handed back by a
/// browser file picker). Shapefiles are not supported here since they need
/// sibling `.shx`/`.dbf` files; use [`load_path`] on native for those.
pub fn load_bytes(name: &str, bytes: &[u8]) -> Result<LoadedLayer, IoError> {
    let ext = std::path::Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let mut loaded = match ext.as_str() {
        "geojson" | "json" => geojson_reader::load_geojson_bytes(name, bytes)?,
        "fgb" => flatgeobuf_reader::load_flatgeobuf_bytes(name, bytes)?,
        other => return Err(IoError::UnsupportedFormat(other.to_owned())),
    };
    loaded.features = reproject_features(loaded.features);
    Ok(loaded)
}

/// Reproject all feature geometries from WGS-84 (lon/lat degrees) to
/// Web Mercator (EPSG:3857, metres). GeoJSON is always WGS-84 by spec;
/// Shapefiles and FlatGeobuf are assumed WGS-84 unless a CRS is detected.
fn reproject_features(features: Vec<Feature>) -> Vec<Feature> {
    features
        .into_iter()
        .map(|f| Feature {
            geometry: f
                .geometry
                .map_coords(|c| rgis_core::lonlat_to_mercator(c.x, c.y)),
            properties: f.properties,
        })
        .collect()
}
