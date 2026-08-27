use std::path::Path;

use rgis_core::Feature;
use serde_json::Value;

use crate::{IoError, LoadedLayer};

pub fn load_geojson(path: &Path) -> Result<LoadedLayer, IoError> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("layer")
        .to_owned();
    let raw = std::fs::read_to_string(path)?;
    parse_geojson(name, &raw)
}

pub fn load_geojson_bytes(name: &str, bytes: &[u8]) -> Result<LoadedLayer, IoError> {
    let name = Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("layer")
        .to_owned();
    let raw = std::str::from_utf8(bytes).map_err(|e| IoError::GeoJson(e.to_string()))?;
    parse_geojson(name, raw)
}

fn parse_geojson(name: String, raw: &str) -> Result<LoadedLayer, IoError> {
    let fc: geojson::GeoJson = raw
        .parse::<geojson::GeoJson>()
        .map_err(|e| IoError::GeoJson(e.to_string()))?;

    let epsg = extract_epsg(&fc)?;

    let collection = match fc {
        geojson::GeoJson::FeatureCollection(fc) => fc,
        geojson::GeoJson::Feature(f) => geojson::FeatureCollection {
            bbox: None,
            features: vec![f],
            foreign_members: None,
        },
        geojson::GeoJson::Geometry(g) => geojson::FeatureCollection {
            bbox: None,
            features: vec![geojson::Feature {
                bbox: None,
                geometry: Some(g),
                id: None,
                properties: None,
                foreign_members: None,
            }],
            foreign_members: None,
        },
    };

    let mut features = Vec::with_capacity(collection.features.len());
    for f in collection.features {
        let Some(geom_raw) = f.geometry else { continue };
        let geo_geom: geo_types::Geometry = (&geom_raw)
            .try_into()
            .map_err(|e: geojson::Error| IoError::GeoJson(e.to_string()))?;
        let properties = f.properties.map(Value::Object).unwrap_or(Value::Null);
        features.push(Feature {
            geometry: geo_geom,
            properties,
        });
    }

    Ok(LoadedLayer {
        name,
        features,
        epsg,
    })
}

/// GeoJSON coordinates are WGS-84 (lon/lat degrees) per RFC 7946, but the
/// deprecated top-level `crs` member is still produced by some tools (e.g.
/// exports in a projected CRS such as EPSG:25832). Extract the EPSG code so
/// the caller can reproject properly instead of treating those coordinates
/// as if they were already lon/lat, which previously crashed the tessellator.
fn extract_epsg(gj: &geojson::GeoJson) -> Result<Option<u16>, IoError> {
    let foreign_members = match gj {
        geojson::GeoJson::FeatureCollection(fc) => fc.foreign_members.as_ref(),
        geojson::GeoJson::Feature(f) => f.foreign_members.as_ref(),
        geojson::GeoJson::Geometry(g) => g.foreign_members.as_ref(),
    };
    let Some(crs_name) = foreign_members
        .and_then(|m| m.get("crs"))
        .and_then(|crs| crs.get("properties"))
        .and_then(|props| props.get("name"))
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    if crs_name.contains("CRS84") || crs_name.contains("4326") {
        return Ok(None);
    }
    let code = crs_name
        .rsplit(':')
        .next()
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| IoError::GeoJson(format!("unrecognized CRS \"{crs_name}\"")))?;
    Ok(Some(code))
}
