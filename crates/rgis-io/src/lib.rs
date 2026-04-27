use thiserror::Error;
use geo::MapCoords;

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

mod geojson_reader;
mod flatgeobuf_reader;
mod shapefile_reader;

pub use geojson_reader::load_geojson;
pub use flatgeobuf_reader::load_flatgeobuf;
pub use shapefile_reader::load_shapefile;

use rgis_core::Feature;

pub struct LoadedLayer {
    pub name: String,
    pub features: Vec<Feature>,
}

/// Load a layer from any supported file path, dispatching on file extension.
pub async fn load(path: impl AsRef<std::path::Path>) -> Result<LoadedLayer, IoError> {
    let path = path.as_ref().to_owned();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "geojson" | "json" => {
            tokio::task::spawn_blocking(move || load_geojson(&path).map(|mut l| {
                l.features = reproject_features(l.features);
                l
            }))
            .await
            .unwrap()
        }
        "shp" => {
            tokio::task::spawn_blocking(move || load_shapefile(&path).map(|mut l| {
                l.features = reproject_features(l.features);
                l
            }))
            .await
            .unwrap()
        }
        "fgb" => {
            tokio::task::spawn_blocking(move || load_flatgeobuf(&path).map(|mut l| {
                l.features = reproject_features(l.features);
                l
            }))
            .await
            .unwrap()
        }
        other => Err(IoError::UnsupportedFormat(other.to_owned())),
    }
}

/// Reproject all feature geometries from WGS-84 (lon/lat degrees) to
/// Web Mercator (EPSG:3857, metres). GeoJSON is always WGS-84 by spec;
/// Shapefiles and FlatGeobuf are assumed WGS-84 unless a CRS is detected.
fn reproject_features(features: Vec<Feature>) -> Vec<Feature> {
    features
        .into_iter()
        .map(|f| Feature {
            geometry: f.geometry.map_coords(|c| {
                rgis_core::lonlat_to_mercator(c.x, c.y)
            }),
            properties: f.properties,
        })
        .collect()
}
