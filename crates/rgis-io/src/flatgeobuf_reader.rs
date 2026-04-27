use std::path::Path;

use flatgeobuf::{FallibleStreamingIterator, FgbReader};
use geo_types::Geometry;
use geozero::ToGeo;
use rgis_core::Feature;
use serde_json::Value;

use crate::{IoError, LoadedLayer};

pub fn load_flatgeobuf(path: &Path) -> Result<LoadedLayer, IoError> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("layer")
        .to_owned();

    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut fgb = FgbReader::open(&mut reader)
        .map_err(|e| IoError::FlatGeobuf(e.to_string()))?
        .select_all()
        .map_err(|e| IoError::FlatGeobuf(e.to_string()))?;

    let mut features = Vec::new();

    while let Some(feature) = fgb
        .next()
        .map_err(|e: flatgeobuf::Error| IoError::FlatGeobuf(e.to_string()))?
    {
        let geometry: Geometry<f64> = feature
            .to_geo()
            .map_err(|e| IoError::FlatGeobuf(e.to_string()))?;

        features.push(Feature {
            geometry,
            // Properties decoding deferred to attribute-table milestone.
            properties: Value::Null,
        });
    }

    Ok(LoadedLayer { name, features })
}
