//! Runtime parser and evaluator for the MapLibre/Mapbox style spec
//! (<https://maplibre.org/maplibre-style-spec/>), enabling `rgis-render` to
//! render any compliant style document -- rather than one hardcoded
//! approximation baked into Rust `match` arms -- and to switch styles live
//! by swapping a [`sheet::StyleSheet`] at runtime.

pub mod expr;
pub mod filter;
pub mod sheet;
pub mod value;

pub use expr::{EvalContext, FeatureProperties};
pub use sheet::{Layer, Source, StyleSheet};
pub use value::{Color, Value};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_liberty_style() {
        let json = include_str!("../fixtures/liberty.json");
        let style = StyleSheet::parse(json).expect("liberty style should parse");
        assert_eq!(style.version, 8);
        assert!(
            style.layers.len() > 50,
            "expected many layers, got {}",
            style.layers.len()
        );
        assert!(style.sources.contains_key("openmaptiles"));

        let background = style.layers.iter().find(|l| l.id == "background").unwrap();
        let ctx = EvalContext::new(10.0);
        let color = background
            .paint("background-color")
            .eval_color(&ctx, Color([1.0, 1.0, 1.0, 1.0]));
        // "#f8f4f0"
        assert!((color.0[0] - 0.973).abs() < 0.01);
    }

    #[test]
    fn evaluates_zoom_interpolated_fill_color() {
        let json = include_str!("../fixtures/liberty.json");
        let style = StyleSheet::parse(json).unwrap();
        let layer = style
            .layers
            .iter()
            .find(|l| l.id == "landuse_residential")
            .unwrap();
        let ctx9 = EvalContext::new(9.0);
        let ctx12 = EvalContext::new(12.0);
        let c9 = layer
            .paint("fill-color")
            .eval_color(&ctx9, Color::TRANSPARENT);
        let c12 = layer
            .paint("fill-color")
            .eval_color(&ctx12, Color::TRANSPARENT);
        assert_ne!(
            c9.0, c12.0,
            "color should change across the interpolation stops"
        );
    }

    #[test]
    fn evaluates_filter_against_feature() {
        use std::collections::HashMap;
        struct Feat(HashMap<String, Value>);
        impl FeatureProperties for Feat {
            fn get_property(&self, key: &str) -> Option<Value> {
                self.0.get(key).cloned()
            }
        }

        let json = include_str!("../fixtures/liberty.json");
        let style = StyleSheet::parse(json).unwrap();
        let layer = style
            .layers
            .iter()
            .find(|l| l.id == "landcover_wood")
            .unwrap();

        let mut props = HashMap::new();
        props.insert("class".to_string(), Value::String("wood".into()));
        let wood = Feat(props);
        assert!(layer.matches_feature(&wood, 10.0));

        let mut props2 = HashMap::new();
        props2.insert("class".to_string(), Value::String("grass".into()));
        let grass = Feat(props2);
        assert!(!layer.matches_feature(&grass, 10.0));
    }

    /// Regression test for a real parsing bug: a plain JSON array whose
    /// first element is a string (e.g. `text-font: ["Noto Sans Italic"]`)
    /// used to be mis-parsed as a call to an unrecognized expression
    /// operator named `"Noto Sans Italic"`, which failed to parse at all
    /// and silently discarded the whole property -- so every symbol
    /// layer's `text-font` (italic region/water labels, bold country
    /// names, etc.) always evaluated as unset, regardless of what the
    /// style actually specified.
    #[test]
    fn text_font_array_parses_as_a_literal_fontstack_list() {
        let json = include_str!("../fixtures/liberty.json");
        let style = StyleSheet::parse(json).unwrap();
        let ctx = EvalContext::new(10.0);

        let bold_layer = style
            .layers
            .iter()
            .find(|l| l.id == "label_country_1")
            .unwrap();
        assert_eq!(
            bold_layer.layout("text-font").eval_string(&ctx).as_deref(),
            Some("Noto Sans Bold")
        );

        let italic_layer = style.layers.iter().find(|l| l.id == "label_state").unwrap();
        assert_eq!(
            italic_layer
                .layout("text-font")
                .eval_string(&ctx)
                .as_deref(),
            Some("Noto Sans Italic")
        );
    }
}
