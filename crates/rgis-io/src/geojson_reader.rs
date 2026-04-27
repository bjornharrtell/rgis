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
    let fc: geojson::GeoJson = raw
        .parse::<geojson::GeoJson>()
        .map_err(|e| IoError::GeoJson(e.to_string()))?;

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
        let properties = f
            .properties
            .map(Value::Object)
            .unwrap_or(Value::Null);
        features.push(Feature {
            geometry: geo_geom,
            properties,
        });
    }

    Ok(LoadedLayer { name, features })
}
