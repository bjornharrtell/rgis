//! Parser and evaluator for the MapLibre/Mapbox style-spec expression
//! language (<https://maplibre.org/maplibre-style-spec/expressions/>),
//! plus support for the older (pre-expression) "stop function" paint/
//! layout property form (`{"stops": [[zoom, value], ...], "base": ...}`)
//! that some style JSON (and legacy converters) still emit.

use serde_json::Value as Json;

use crate::value::{Color, Value};

/// Anything an expression can read `["get", key]`/`["has", key]` from.
/// Implemented by the host crate's own feature type (e.g. `rgis-tiles`'s
/// `VectorFeature`) so this crate doesn't need to depend on it.
pub trait FeatureProperties {
    fn get_property(&self, key: &str) -> Option<Value>;

    /// The feature's geometry type, one of `"Point"`, `"LineString"`, or
    /// `"Polygon"` (multi-geometries collapse to the same category), for
    /// the style spec's `["geometry-type"]` expression. Defaults to `None`
    /// (`geometry-type` evaluates to `Null`) so existing implementers
    /// (test fixtures, anything with no real geometry) don't need to
    /// implement this to keep compiling.
    fn geometry_type(&self) -> Option<&str> {
        None
    }
}

/// Everything an expression needs to evaluate: the current zoom level and
/// (optionally) a feature to pull `get`/`has` properties from -- `None` for
/// zoom-only contexts like `background-color`/`raster-opacity`, which have
/// no associated feature.
pub struct EvalContext<'a> {
    pub zoom: f64,
    pub feature: Option<&'a dyn FeatureProperties>,
}

impl<'a> EvalContext<'a> {
    pub fn new(zoom: f64) -> Self {
        EvalContext {
            zoom,
            feature: None,
        }
    }

    pub fn with_feature(zoom: f64, feature: &'a dyn FeatureProperties) -> Self {
        EvalContext {
            zoom,
            feature: Some(feature),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Interpolation {
    Linear,
    Exponential(f64),
    CubicBezier(f64, f64, f64, f64),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Value),
    Get(String),
    Has(String),
    NotHas(String),
    Zoom,
    /// `["geometry-type"]` -- the current feature's geometry type as one
    /// of the style spec's three values (`"Point"`, `"LineString"`,
    /// `"Polygon"`; multi-geometries collapse into the same category as
    /// their single-geometry counterpart, matching MapLibre's own
    /// behavior). `Null` outside a feature context (e.g. `background-*`
    /// properties, which have no associated feature).
    GeometryType,
    /// `["interpolate", [type], input, stop1, val1, stop2, val2, ...]`
    Interpolate {
        interpolation: Interpolation,
        input: Box<Expr>,
        stops: Vec<(f64, Expr)>,
    },
    /// `["step", input, val0, stop1, val1, ...]`
    Step {
        input: Box<Expr>,
        default: Box<Expr>,
        stops: Vec<(f64, Expr)>,
    },
    /// `["match", input, label1, out1, label2, out2, ..., fallback]`
    Match {
        input: Box<Expr>,
        cases: Vec<(Vec<Value>, Expr)>,
        fallback: Box<Expr>,
    },
    /// `["case", cond1, out1, cond2, out2, ..., fallback]`
    Case {
        branches: Vec<(Expr, Expr)>,
        fallback: Box<Expr>,
    },
    All(Vec<Expr>),
    Any(Vec<Expr>),
    Not(Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    In(Box<Expr>, Vec<Expr>),
    Coalesce(Vec<Expr>),
    Concat(Vec<Expr>),
    ToString(Box<Expr>),
    ToNumber(Box<Expr>),
    Length(Box<Expr>),
    At(Box<Expr>, Box<Expr>),
}

#[derive(Debug, thiserror::Error)]
pub enum ExprError {
    #[error("unsupported expression operator: {0}")]
    UnsupportedOperator(String),
    #[error("malformed expression: {0}")]
    Malformed(String),
}

fn json_to_value(json: &Json) -> Value {
    match json {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) => Value::Number(n.as_f64().unwrap_or(0.0)),
        Json::String(s) => Value::String(s.clone()),
        Json::Array(a) => Value::Array(a.iter().map(json_to_value).collect()),
        Json::Object(_) => Value::Null,
    }
}

/// Parses a raw JSON paint/layout/filter value into an [`Expr`] tree.
/// Non-array, non-`["expr", ...]` values (plain strings/numbers/bools) are
/// treated as literals, matching how the style spec allows a bare constant
/// wherever an expression is accepted.
pub fn parse(json: &Json) -> Result<Expr, ExprError> {
    let Json::Array(arr) = json else {
        return Ok(Expr::Literal(json_to_value(json)));
    };
    let Some(Json::String(op)) = arr.first() else {
        // A plain JSON array with no leading operator string is a literal
        // array value (e.g. `line-dasharray: [1, 1.5]`), not an expression.
        return Ok(Expr::Literal(json_to_value(json)));
    };
    let args = &arr[1..];
    let parse_at = |i: usize| -> Result<Expr, ExprError> {
        parse(
            args.get(i)
                .ok_or_else(|| ExprError::Malformed(format!("{op}: missing arg {i}")))?,
        )
    };
    let parse_all =
        |from: usize| -> Result<Vec<Expr>, ExprError> { args[from..].iter().map(parse).collect() };

    Ok(match op.as_str() {
        "literal" => Expr::Literal(json_to_value(args.first().unwrap_or(&Json::Null))),
        "get" => Expr::Get(
            args.first()
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
        "has" => Expr::Has(
            args.first()
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
        "!has" => Expr::NotHas(
            args.first()
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
        "zoom" => Expr::Zoom,
        "geometry-type" => Expr::GeometryType,
        "interpolate" => {
            let interp_spec = args
                .first()
                .ok_or_else(|| ExprError::Malformed("interpolate: missing type".into()))?;
            let interpolation = match interp_spec {
                Json::Array(a) => match a.first().and_then(Json::as_str) {
                    Some("linear") => Interpolation::Linear,
                    Some("exponential") => {
                        Interpolation::Exponential(a.get(1).and_then(Json::as_f64).unwrap_or(1.0))
                    }
                    Some("cubic-bezier") => Interpolation::CubicBezier(
                        a.get(1).and_then(Json::as_f64).unwrap_or(0.0),
                        a.get(2).and_then(Json::as_f64).unwrap_or(0.0),
                        a.get(3).and_then(Json::as_f64).unwrap_or(1.0),
                        a.get(4).and_then(Json::as_f64).unwrap_or(1.0),
                    ),
                    _ => Interpolation::Linear,
                },
                _ => Interpolation::Linear,
            };
            let input = Box::new(parse_at(1)?);
            let mut stops = Vec::new();
            let mut i = 2;
            while i + 1 < args.len() {
                let stop = args[i].as_f64().ok_or_else(|| {
                    ExprError::Malformed("interpolate: stop must be a number".into())
                })?;
                stops.push((stop, parse(&args[i + 1])?));
                i += 2;
            }
            Expr::Interpolate {
                interpolation,
                input,
                stops,
            }
        }
        "step" => {
            let input = Box::new(parse_at(0)?);
            let default = Box::new(parse_at(1)?);
            let mut stops = Vec::new();
            let mut i = 2;
            while i + 1 < args.len() {
                let stop = args[i]
                    .as_f64()
                    .ok_or_else(|| ExprError::Malformed("step: stop must be a number".into()))?;
                stops.push((stop, parse(&args[i + 1])?));
                i += 2;
            }
            Expr::Step {
                input,
                default,
                stops,
            }
        }
        "match" => {
            let input = Box::new(parse_at(0)?);
            let mut cases = Vec::new();
            let mut i = 1;
            while i + 1 < args.len() {
                let labels = match &args[i] {
                    Json::Array(a) => a.iter().map(json_to_value).collect(),
                    other => vec![json_to_value(other)],
                };
                cases.push((labels, parse(&args[i + 1])?));
                i += 2;
            }
            // Odd trailing element (after full label/output pairs) is the
            // fallback; a well-formed `match` always has one.
            let fallback = if i < args.len() {
                Box::new(parse(&args[i])?)
            } else {
                Box::new(Expr::Literal(Value::Null))
            };
            Expr::Match {
                input,
                cases,
                fallback,
            }
        }
        "case" => {
            let mut branches = Vec::new();
            let mut i = 0;
            while i + 1 < args.len() {
                branches.push((parse(&args[i])?, parse(&args[i + 1])?));
                i += 2;
            }
            let fallback = if i < args.len() {
                Box::new(parse(&args[i])?)
            } else {
                Box::new(Expr::Literal(Value::Null))
            };
            Expr::Case { branches, fallback }
        }
        "all" => Expr::All(parse_all(0)?),
        "any" => Expr::Any(parse_all(0)?),
        "!" => Expr::Not(Box::new(parse_at(0)?)),
        "==" => Expr::Eq(Box::new(parse_at(0)?), Box::new(parse_at(1)?)),
        "!=" => Expr::Ne(Box::new(parse_at(0)?), Box::new(parse_at(1)?)),
        "<" => Expr::Lt(Box::new(parse_at(0)?), Box::new(parse_at(1)?)),
        "<=" => Expr::Le(Box::new(parse_at(0)?), Box::new(parse_at(1)?)),
        ">" => Expr::Gt(Box::new(parse_at(0)?), Box::new(parse_at(1)?)),
        ">=" => Expr::Ge(Box::new(parse_at(0)?), Box::new(parse_at(1)?)),
        "in" => {
            let needle = Box::new(parse_at(0)?);
            // The spec's own `in` form is `["in", needle, haystack_array]`
            // (2 args); legacy filters (normalized by `filter::parse_filter`)
            // instead spell it variadically as `["in", needle, v1, v2, ...]`.
            // Support both.
            let haystack = if args.len() == 2 {
                match &args[1] {
                    Json::Array(items) => items.iter().map(parse).collect::<Result<Vec<_>, _>>()?,
                    other => vec![parse(other)?],
                }
            } else {
                parse_all(1)?
            };
            Expr::In(needle, haystack)
        }
        "coalesce" => Expr::Coalesce(parse_all(0)?),
        "concat" => Expr::Concat(parse_all(0)?),
        "to-string" => Expr::ToString(Box::new(parse_at(0)?)),
        "to-number" => Expr::ToNumber(Box::new(parse_at(0)?)),
        "length" => Expr::Length(Box::new(parse_at(0)?)),
        "at" => Expr::At(Box::new(parse_at(0)?), Box::new(parse_at(1)?)),
        other => return Err(ExprError::UnsupportedOperator(other.to_string())),
    })
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Null, Value::Null) => true,
        // The style spec compares across compatible types by string/number
        // coercion (e.g. matching a numeric tag value against a `match`
        // label written as a string in the style JSON).
        _ => a.to_display_string() == b.to_display_string(),
    }
}

fn interpolate_numbers(
    interpolation: &Interpolation,
    t: f64,
    lo: f64,
    hi: f64,
    out_lo: f64,
    out_hi: f64,
) -> f64 {
    let f = match interpolation {
        Interpolation::Linear => {
            if hi > lo {
                (t - lo) / (hi - lo)
            } else {
                0.0
            }
        }
        Interpolation::Exponential(base) => {
            if hi <= lo {
                0.0
            } else if (*base - 1.0).abs() < 1e-6 {
                (t - lo) / (hi - lo)
            } else {
                let range = hi - lo;
                (base.powf(t - lo) - 1.0) / (base.powf(range) - 1.0)
            }
        }
        Interpolation::CubicBezier(x1, y1, x2, y2) => {
            let lin = if hi > lo { (t - lo) / (hi - lo) } else { 0.0 };
            cubic_bezier_ease(*x1, *y1, *x2, *y2, lin)
        }
    };
    out_lo + (out_hi - out_lo) * f.clamp(0.0, 1.0)
}

/// Solves a CSS-style cubic-bezier timing function (control points fixed at
/// `(0,0)` and `(1,1)`) for the output `y` at parametric progress `x_target`
/// via bisection on the bezier's own parameter `u`, since the spec defines
/// `cubic-bezier` interpolation exactly the same way CSS transition timing
/// functions work.
fn cubic_bezier_ease(x1: f64, y1: f64, x2: f64, y2: f64, x_target: f64) -> f64 {
    let bezier = |u: f64, p1: f64, p2: f64| -> f64 {
        let mu = 1.0 - u;
        3.0 * mu * mu * u * p1 + 3.0 * mu * u * u * p2 + u * u * u
    };
    let mut lo = 0.0;
    let mut hi = 1.0;
    let mut u = x_target;
    for _ in 0..20 {
        let x = bezier(u, x1, x2);
        if (x - x_target).abs() < 1e-6 {
            break;
        }
        if x < x_target {
            lo = u;
        } else {
            hi = u;
        }
        u = (lo + hi) / 2.0;
    }
    bezier(u, y1, y2)
}

fn interpolate_colors(
    interpolation: &Interpolation,
    t: f64,
    lo: f64,
    hi: f64,
    out_lo: Color,
    out_hi: Color,
) -> Color {
    let mut result = [0.0f32; 4];
    for (i, channel) in result.iter_mut().enumerate() {
        *channel = interpolate_numbers(
            interpolation,
            t,
            lo,
            hi,
            out_lo.0[i] as f64,
            out_hi.0[i] as f64,
        ) as f32;
    }
    Color(result)
}

impl Expr {
    pub fn eval(&self, ctx: &EvalContext) -> Value {
        match self {
            Expr::Literal(v) => v.clone(),
            Expr::Get(key) => ctx
                .feature
                .and_then(|f| f.get_property(key))
                .unwrap_or(Value::Null),
            Expr::Has(key) => Value::Bool(ctx.feature.and_then(|f| f.get_property(key)).is_some()),
            Expr::NotHas(key) => {
                Value::Bool(ctx.feature.and_then(|f| f.get_property(key)).is_none())
            }
            Expr::Zoom => Value::Number(ctx.zoom),
            Expr::GeometryType => ctx
                .feature
                .and_then(|f| f.geometry_type())
                .map(|s| Value::String(s.to_string()))
                .unwrap_or(Value::Null),
            Expr::Interpolate {
                interpolation,
                input,
                stops,
            } => {
                let t = input.eval(ctx).as_f64().unwrap_or(0.0);
                eval_interpolate(interpolation, t, stops, ctx)
            }
            Expr::Step {
                input,
                default,
                stops,
            } => {
                let t = input.eval(ctx).as_f64().unwrap_or(0.0);
                let mut result = default.eval(ctx);
                for (stop, expr) in stops {
                    if t >= *stop {
                        result = expr.eval(ctx);
                    } else {
                        break;
                    }
                }
                result
            }
            Expr::Match {
                input,
                cases,
                fallback,
            } => {
                let v = input.eval(ctx);
                for (labels, out) in cases {
                    if labels.iter().any(|l| values_equal(l, &v)) {
                        return out.eval(ctx);
                    }
                }
                fallback.eval(ctx)
            }
            Expr::Case { branches, fallback } => {
                for (cond, out) in branches {
                    if cond.eval(ctx).as_bool() {
                        return out.eval(ctx);
                    }
                }
                fallback.eval(ctx)
            }
            Expr::All(exprs) => Value::Bool(exprs.iter().all(|e| e.eval(ctx).as_bool())),
            Expr::Any(exprs) => Value::Bool(exprs.iter().any(|e| e.eval(ctx).as_bool())),
            Expr::Not(e) => Value::Bool(!e.eval(ctx).as_bool()),
            Expr::Eq(a, b) => Value::Bool(values_equal(&a.eval(ctx), &b.eval(ctx))),
            Expr::Ne(a, b) => Value::Bool(!values_equal(&a.eval(ctx), &b.eval(ctx))),
            Expr::Lt(a, b) => {
                Value::Bool(cmp_values(&a.eval(ctx), &b.eval(ctx)).is_some_and(|o| o.is_lt()))
            }
            Expr::Le(a, b) => {
                Value::Bool(cmp_values(&a.eval(ctx), &b.eval(ctx)).is_some_and(|o| o.is_le()))
            }
            Expr::Gt(a, b) => {
                Value::Bool(cmp_values(&a.eval(ctx), &b.eval(ctx)).is_some_and(|o| o.is_gt()))
            }
            Expr::Ge(a, b) => {
                Value::Bool(cmp_values(&a.eval(ctx), &b.eval(ctx)).is_some_and(|o| o.is_ge()))
            }
            Expr::In(needle, haystack) => {
                let v = needle.eval(ctx);
                Value::Bool(haystack.iter().any(|e| values_equal(&e.eval(ctx), &v)))
            }
            Expr::Coalesce(exprs) => {
                for e in exprs {
                    let v = e.eval(ctx);
                    if v != Value::Null {
                        return v;
                    }
                }
                Value::Null
            }
            Expr::Concat(exprs) => Value::String(
                exprs
                    .iter()
                    .map(|e| e.eval(ctx).to_display_string())
                    .collect(),
            ),
            Expr::ToString(e) => Value::String(e.eval(ctx).to_display_string()),
            Expr::ToNumber(e) => e
                .eval(ctx)
                .as_f64()
                .map(Value::Number)
                .unwrap_or(Value::Null),
            Expr::Length(e) => match e.eval(ctx) {
                Value::String(s) => Value::Number(s.chars().count() as f64),
                Value::Array(a) => Value::Number(a.len() as f64),
                _ => Value::Null,
            },
            Expr::At(idx, arr) => {
                let i = idx.eval(ctx).as_f64().unwrap_or(0.0) as usize;
                match arr.eval(ctx) {
                    Value::Array(a) => a.into_iter().nth(i).unwrap_or(Value::Null),
                    _ => Value::Null,
                }
            }
        }
    }
}

fn cmp_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x.partial_cmp(&y),
        _ => a.to_display_string().partial_cmp(&b.to_display_string()),
    }
}

fn eval_interpolate(
    interpolation: &Interpolation,
    t: f64,
    stops: &[(f64, Expr)],
    ctx: &EvalContext,
) -> Value {
    if stops.is_empty() {
        return Value::Null;
    }
    if t <= stops[0].0 {
        return stops[0].1.eval(ctx);
    }
    if t >= stops[stops.len() - 1].0 {
        return stops[stops.len() - 1].1.eval(ctx);
    }
    for w in stops.windows(2) {
        let (lo, lo_expr) = &w[0];
        let (hi, hi_expr) = &w[1];
        if t >= *lo && t <= *hi {
            let lo_val = lo_expr.eval(ctx);
            let hi_val = hi_expr.eval(ctx);
            return interpolate_pair(interpolation, t, *lo, *hi, &lo_val, &hi_val);
        }
    }
    stops[stops.len() - 1].1.eval(ctx)
}

fn interpolate_pair(
    interpolation: &Interpolation,
    t: f64,
    lo: f64,
    hi: f64,
    lo_val: &Value,
    hi_val: &Value,
) -> Value {
    match (lo_val, hi_val) {
        (Value::Number(a), Value::Number(b)) => {
            Value::Number(interpolate_numbers(interpolation, t, lo, hi, *a, *b))
        }
        _ => {
            // Colors are the only other interpolatable type the style spec
            // defines; string colors (`"#fff"`) are coerced via
            // `as_color`. Anything else (e.g. mismatched types) just steps
            // rather than interpolating.
            match (lo_val.as_color(), hi_val.as_color()) {
                (Some(a), Some(b)) => {
                    Value::Color(interpolate_colors(interpolation, t, lo, hi, a, b))
                }
                _ => {
                    if t - lo <= hi - t {
                        lo_val.clone()
                    } else {
                        hi_val.clone()
                    }
                }
            }
        }
    }
}

/// Parses the legacy (pre-expression) "stop function" property form:
/// `{"stops": [[zoom, value], ...], "base": exponential_base, "property":
/// ..., "type": ...}`. Only the plain zoom-keyed (no `property`/data-driven)
/// form is supported -- the only one still found in real style JSON that
/// isn't better expressed as an `interpolate` expression.
pub fn parse_legacy_stops(json: &Json) -> Option<Expr> {
    let obj = json.as_object()?;
    let stops_json = obj.get("stops")?.as_array()?;
    let base = obj.get("base").and_then(Json::as_f64).unwrap_or(1.0);
    let mut stops = Vec::new();
    for pair in stops_json {
        let pair = pair.as_array()?;
        let zoom = pair.first()?.as_f64()?;
        let value = parse(pair.get(1)?).ok()?;
        stops.push((zoom, value));
    }
    Some(Expr::Interpolate {
        interpolation: Interpolation::Exponential(base),
        input: Box::new(Expr::Zoom),
        stops,
    })
}
