use std::path::Path;

use geo_types::Geometry;
use rgis_core::Feature;
use serde_json::Value;
use shapefile::dbase::FieldValue;

use crate::{IoError, LoadedLayer};

pub fn load_shapefile(path: &Path) -> Result<LoadedLayer, IoError> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("layer")
        .to_owned();

    let mut reader = shapefile::Reader::from_path(path)
        .map_err(|e| IoError::Shapefile(e.to_string()))?;

    let mut features = Vec::new();

    for result in reader.iter_shapes_and_records() {
        let (shape, record) = result.map_err(|e| IoError::Shapefile(e.to_string()))?;

        let geometry: Option<Geometry> = shape_to_geo(shape);
        let Some(geometry) = geometry else { continue };

        let mut map = serde_json::Map::new();
        for (name, value) in record {
            map.insert(name.to_string(), field_value_to_json(value));
        }

        features.push(Feature {
            geometry,
            properties: Value::Object(map),
        });
    }

    Ok(LoadedLayer { name, features })
}

fn field_value_to_json(v: FieldValue) -> Value {
    match v {
        FieldValue::Character(Some(s)) => Value::String(s),
        FieldValue::Numeric(Some(n)) => {
            serde_json::Number::from_f64(n).map(Value::Number).unwrap_or(Value::Null)
        }
        FieldValue::Float(Some(f)) => {
            serde_json::Number::from_f64(f as f64).map(Value::Number).unwrap_or(Value::Null)
        }
        FieldValue::Integer(n) => Value::Number(n.into()),
        FieldValue::Logical(Some(b)) => Value::Bool(b),
        _ => Value::Null,
    }
}

fn shape_to_geo(shape: shapefile::Shape) -> Option<Geometry> {
    use shapefile::Shape::*;
    match shape {
        Point(p) => Some(Geometry::Point(geo_types::Point::new(p.x, p.y))),
        PointM(p) => Some(Geometry::Point(geo_types::Point::new(p.x, p.y))),
        PointZ(p) => Some(Geometry::Point(geo_types::Point::new(p.x, p.y))),
        Multipoint(mp) => {
            let points = mp
                .points()
                .iter()
                .map(|p| geo_types::Point::new(p.x, p.y))
                .collect();
            Some(Geometry::MultiPoint(geo_types::MultiPoint(points)))
        }
        Polyline(pl) => {
            let lines: Vec<geo_types::LineString> = pl
                .parts()
                .iter()
                .map(|part| {
                    geo_types::LineString(
                        part.iter()
                            .map(|p| geo_types::Coord { x: p.x, y: p.y })
                            .collect(),
                    )
                })
                .collect();
            if lines.len() == 1 {
                Some(Geometry::LineString(lines.into_iter().next().unwrap()))
            } else {
                Some(Geometry::MultiLineString(geo_types::MultiLineString(lines)))
            }
        }
        Polygon(poly) => {
            // shapefile rings: exterior ring first, then interior rings.
            // Multiple exterior rings produce a MultiPolygon.
            let rings: Vec<geo_types::LineString> = poly
                .rings()
                .iter()
                .map(|r| {
                    geo_types::LineString(
                        r.points()
                            .iter()
                            .map(|p| geo_types::Coord { x: p.x, y: p.y })
                            .collect(),
                    )
                })
                .collect();

            if rings.is_empty() {
                return None;
            }

            // Simple heuristic: treat first ring as exterior, rest as holes.
            let exterior = rings[0].clone();
            let interiors = rings[1..].to_vec();
            Some(Geometry::Polygon(geo_types::Polygon::new(
                exterior, interiors,
            )))
        }
        NullShape => None,
        _ => None,
    }
}
