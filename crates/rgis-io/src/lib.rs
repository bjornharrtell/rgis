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
    /// EPSG code of the source CRS, if known and not WGS-84. `None` means
    /// the coordinates are assumed to already be WGS-84 (lon/lat degrees).
    pub epsg: Option<u16>,
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
    loaded.features = reproject_features(loaded.features, loaded.epsg)?;
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
    loaded.features = reproject_features(loaded.features, loaded.epsg)?;
    Ok(loaded)
}

/// Reproject all feature geometries to Web Mercator (EPSG:3857, metres).
/// If `epsg` names a non-WGS-84 source CRS (e.g. detected from a GeoJSON
/// `crs` member), geometries are reprojected from that CRS directly.
/// Otherwise coordinates are assumed to be WGS-84 (lon/lat degrees); GeoJSON
/// is always WGS-84 by spec, and Shapefiles/FlatGeobuf are assumed WGS-84
/// unless a CRS is detected. Out-of-range WGS-84 coordinates (e.g. a
/// projected CRS mistakenly treated as WGS-84) are rejected rather than
/// silently reprojected into non-finite mercator values.
fn reproject_features(features: Vec<Feature>, epsg: Option<u16>) -> Result<Vec<Feature>, IoError> {
    features
        .into_iter()
        .map(|f| {
            let geometry = match epsg {
                Some(code) => rgis_core::reproject_geometry_to_mercator(code, f.geometry)
                    .map_err(IoError::GeoJson)?,
                None => f.geometry.try_map_coords(|c| {
                    if !(-180.0..=180.0).contains(&c.x) || !(-90.0..=90.0).contains(&c.y) {
                        return Err(IoError::GeoJson(format!(
                            "coordinate ({}, {}) is out of WGS-84 lon/lat range; \
                             the source data may use a projected CRS",
                            c.x, c.y
                        )));
                    }
                    Ok(rgis_core::lonlat_to_mercator(c.x, c.y))
                })?,
            };
            Ok(Feature {
                geometry,
                properties: f.properties,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::Geometry;

    #[test]
    fn load_geojson_bytes_with_projected_crs() {
        // Real-world sample: a Danish dataset in EPSG:25832 (UTM 32N) with an
        // explicit `crs` member, which previously produced non-finite
        // Web Mercator coordinates and crashed the tessellator.
        let geojson = r#"{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[569290,6287398.8],[569188,6287422],[569151.2,6287397.2],[569180.8,6287364],[569154.4,6287340.4],[569110.8,6287184.4],[569132.4,6287178],[569254,6287326.4],[569287.6,6287368],[569290,6287398.8]]]},"properties":{"lokalitet_id":250417}}],"crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:EPSG::25832"}}}"#;
        let loaded = load_bytes("layer.geojson", geojson.as_bytes()).unwrap();
        assert_eq!(loaded.features.len(), 1);
        let Geometry::Polygon(poly) = &loaded.features[0].geometry else {
            panic!("expected polygon");
        };
        for c in poly.exterior().coords() {
            assert!(c.x.is_finite() && c.y.is_finite());
        }
    }
}
