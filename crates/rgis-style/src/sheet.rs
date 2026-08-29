//! Typed representation of a MapLibre/Mapbox style document
//! (<https://maplibre.org/maplibre-style-spec/>), parsed once from JSON at
//! load time (or on a live style switch) with every layer's `filter`,
//! `paint`, and `layout` properties pre-compiled into [`crate::expr::Expr`]
//! trees, so per-feature/per-frame evaluation only walks already-parsed
//! expressions rather than re-parsing JSON.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value as Json;

use crate::expr::{EvalContext, Expr, FeatureProperties, parse, parse_legacy_stops};
use crate::filter::parse_filter;
use crate::value::{Color, Value};

#[derive(Debug, thiserror::Error)]
pub enum StyleError {
    #[error("invalid style JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// A layer property (paint or layout field) compiled from JSON: either an
/// expression/legacy-stop-function tree, or nothing (property absent, use
/// the spec's documented default).
#[derive(Debug, Clone, Default)]
pub struct Prop(pub Option<Expr>);

impl Prop {
    fn parse(json: Option<&Json>) -> Self {
        let Some(json) = json else { return Prop(None) };
        // Legacy `{"stops": [...]}` objects aren't valid expression JSON
        // (`parse` would hit the `Json::Object` literal fallback and
        // silently produce `Value::Null`), so try that form first.
        if json.is_object()
            && let Some(expr) = parse_legacy_stops(json)
        {
            return Prop(Some(expr));
        }
        match parse(json) {
            Ok(expr) => Prop(Some(expr)),
            Err(_) => Prop(None),
        }
    }

    pub fn eval(&self, ctx: &EvalContext) -> Option<Value> {
        self.0.as_ref().map(|e| e.eval(ctx))
    }

    pub fn eval_f64(&self, ctx: &EvalContext, default: f64) -> f64 {
        self.eval(ctx).and_then(|v| v.as_f64()).unwrap_or(default)
    }

    pub fn eval_color(&self, ctx: &EvalContext, default: Color) -> Color {
        self.eval(ctx).and_then(|v| v.as_color()).unwrap_or(default)
    }

    pub fn eval_string(&self, ctx: &EvalContext) -> Option<String> {
        self.eval(ctx).map(|v| v.to_display_string())
    }

    pub fn eval_bool(&self, ctx: &EvalContext, default: bool) -> bool {
        self.eval(ctx).map(|v| v.as_bool()).unwrap_or(default)
    }
}

/// A compiled style layer. `paint`/`layout` are kept as raw property maps
/// (`serde_json::Value` compiled lazily via [`Layer::paint`]/[`Layer::layout`])
/// rather than a fixed struct per layer type, since paint/layout property
/// sets differ per `type` and the style spec keeps growing them --  this
/// avoids re-deriving a huge struct hierarchy that must track the spec
/// exactly to parse successfully.
#[derive(Debug, Clone)]
pub struct Layer {
    pub id: String,
    pub kind: String,
    pub source: Option<String>,
    pub source_layer: Option<String>,
    pub minzoom: f64,
    pub maxzoom: f64,
    pub filter: Option<Expr>,
    paint: HashMap<String, Prop>,
    layout: HashMap<String, Prop>,
}

impl Layer {
    pub fn paint(&self, key: &str) -> &Prop {
        static EMPTY: Prop = Prop(None);
        self.paint.get(key).unwrap_or(&EMPTY)
    }

    pub fn layout(&self, key: &str) -> &Prop {
        static EMPTY: Prop = Prop(None);
        self.layout.get(key).unwrap_or(&EMPTY)
    }

    /// Whether this layer applies at `zoom` (style-spec semantics: `[minzoom,
    /// maxzoom)`, i.e. `maxzoom` itself is excluded).
    pub fn matches_zoom(&self, zoom: f64) -> bool {
        zoom >= self.minzoom && zoom < self.maxzoom
    }

    /// Whether `feature` passes this layer's `filter` (a layer with no
    /// filter matches every feature on its source-layer).
    pub fn matches_feature(&self, feature: &dyn FeatureProperties, zoom: f64) -> bool {
        match &self.filter {
            None => true,
            Some(expr) => expr
                .eval(&EvalContext::with_feature(zoom, feature))
                .as_bool(),
        }
    }
}

fn parse_prop_map(json: Option<&Json>) -> HashMap<String, Prop> {
    let Some(Json::Object(map)) = json else {
        return HashMap::new();
    };
    map.iter()
        .map(|(k, v)| (k.clone(), Prop::parse(Some(v))))
        .collect()
}

#[derive(Debug, Deserialize)]
struct RawLayer {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    source: Option<String>,
    #[serde(rename = "source-layer")]
    source_layer: Option<String>,
    minzoom: Option<f64>,
    maxzoom: Option<f64>,
    filter: Option<Json>,
    paint: Option<Json>,
    layout: Option<Json>,
}

impl From<RawLayer> for Layer {
    fn from(raw: RawLayer) -> Self {
        Layer {
            id: raw.id,
            kind: raw.kind,
            source: raw.source,
            source_layer: raw.source_layer,
            minzoom: raw.minzoom.unwrap_or(0.0),
            maxzoom: raw.maxzoom.unwrap_or(24.0),
            filter: raw.filter.as_ref().and_then(|f| parse_filter(f).ok()),
            paint: parse_prop_map(raw.paint.as_ref()),
            layout: parse_prop_map(raw.layout.as_ref()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Source {
    #[serde(rename = "type")]
    pub kind: String,
    pub url: Option<String>,
    pub tiles: Option<Vec<String>>,
    #[serde(default)]
    pub tile_size: Option<u32>,
    pub minzoom: Option<f64>,
    pub maxzoom: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawStyleSheet {
    version: u32,
    #[serde(default)]
    sources: HashMap<String, Source>,
    sprite: Option<String>,
    glyphs: Option<String>,
    layers: Vec<RawLayer>,
}

/// A fully parsed, ready-to-evaluate style document. Cheap to swap at
/// runtime (`Arc<StyleSheet>` held by the renderer, replaced wholesale on a
/// live style change) since all the JSON parsing/expression compilation
/// happens once here rather than per-frame.
#[derive(Debug)]
pub struct StyleSheet {
    pub version: u32,
    pub sources: HashMap<String, Source>,
    pub sprite: Option<String>,
    pub glyphs: Option<String>,
    pub layers: Vec<Layer>,
}

impl StyleSheet {
    pub fn parse(json: &str) -> Result<StyleSheet, StyleError> {
        let raw: RawStyleSheet = serde_json::from_str(json)?;
        Ok(StyleSheet {
            version: raw.version,
            sources: raw.sources,
            sprite: raw.sprite,
            glyphs: raw.glyphs,
            layers: raw.layers.into_iter().map(Layer::from).collect(),
        })
    }

    /// Layers in style (bottom-to-top paint) order whose `source-layer`
    /// matches `source_layer_name` and whose `type` is one of `kinds`.
    pub fn layers_for<'a>(
        &'a self,
        source_layer_name: &'a str,
        kinds: &'a [&str],
    ) -> impl Iterator<Item = &'a Layer> {
        self.layers.iter().filter(move |l| {
            kinds.contains(&l.kind.as_str()) && l.source_layer.as_deref() == Some(source_layer_name)
        })
    }

    pub fn layers_of_kind<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Layer> {
        self.layers.iter().filter(move |l| l.kind == kind)
    }
}
