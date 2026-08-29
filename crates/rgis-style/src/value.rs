//! Runtime value type produced by evaluating style-spec expressions
//! (<https://maplibre.org/maplibre-style-spec/expressions/>), plus CSS
//! color-string parsing (`#rgb`, `#rrggbbaa`, `rgb()`, `rgba()`, `hsl()`,
//! `hsla()`) since MapLibre paint properties accept colors in any of those
//! forms as plain strings.

use std::fmt;

/// An RGBA color, straight (non-premultiplied) alpha, each channel in
/// `[0.0, 1.0]` -- the same representation `rgis-render` already uses for
/// vertex colors, so evaluated paint colors can be handed straight to the
/// mesh builders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color(pub [f32; 4]);

impl Color {
    pub const TRANSPARENT: Color = Color([0.0, 0.0, 0.0, 0.0]);

    pub fn to_array(self) -> [f32; 4] {
        self.0
    }

    /// Parses any CSS color string accepted by the style spec. Returns
    /// `None` (rather than a default color) on unrecognized input so
    /// callers can fall back to a documented default instead of silently
    /// rendering the wrong color.
    pub fn parse(s: &str) -> Option<Color> {
        let s = s.trim();
        if let Some(hex) = s.strip_prefix('#') {
            return parse_hex(hex);
        }
        if let Some(inner) = s.strip_prefix("rgba(").and_then(|s| s.strip_suffix(')')) {
            return parse_rgb_components(inner, true);
        }
        if let Some(inner) = s.strip_prefix("rgb(").and_then(|s| s.strip_suffix(')')) {
            return parse_rgb_components(inner, false);
        }
        if let Some(inner) = s.strip_prefix("hsla(").and_then(|s| s.strip_suffix(')')) {
            return parse_hsl_components(inner, true);
        }
        if let Some(inner) = s.strip_prefix("hsl(").and_then(|s| s.strip_suffix(')')) {
            return parse_hsl_components(inner, false);
        }
        match s {
            "transparent" => Some(Color::TRANSPARENT),
            "black" => Some(Color([0.0, 0.0, 0.0, 1.0])),
            "white" => Some(Color([1.0, 1.0, 1.0, 1.0])),
            "red" => Some(Color([1.0, 0.0, 0.0, 1.0])),
            "green" => Some(Color([0.0, 0.5019608, 0.0, 1.0])),
            "blue" => Some(Color([0.0, 0.0, 1.0, 1.0])),
            _ => None,
        }
    }
}

fn parse_hex(hex: &str) -> Option<Color> {
    let expand = |c: char| -> Option<u8> { u8::from_str_radix(&format!("{c}{c}"), 16).ok() };
    let byte = |s: &str| -> Option<u8> { u8::from_str_radix(s, 16).ok() };
    match hex.len() {
        3 => {
            let mut cs = hex.chars();
            let r = expand(cs.next()?)?;
            let g = expand(cs.next()?)?;
            let b = expand(cs.next()?)?;
            Some(Color([
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                1.0,
            ]))
        }
        4 => {
            let mut cs = hex.chars();
            let r = expand(cs.next()?)?;
            let g = expand(cs.next()?)?;
            let b = expand(cs.next()?)?;
            let a = expand(cs.next()?)?;
            Some(Color([
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ]))
        }
        6 => {
            let r = byte(&hex[0..2])?;
            let g = byte(&hex[2..4])?;
            let b = byte(&hex[4..6])?;
            Some(Color([
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                1.0,
            ]))
        }
        8 => {
            let r = byte(&hex[0..2])?;
            let g = byte(&hex[2..4])?;
            let b = byte(&hex[4..6])?;
            let a = byte(&hex[6..8])?;
            Some(Color([
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ]))
        }
        _ => None,
    }
}

fn parse_component(s: &str) -> Option<f32> {
    let s = s.trim();
    if let Some(pct) = s.strip_suffix('%') {
        Some(pct.trim().parse::<f32>().ok()? / 100.0 * 255.0)
    } else {
        s.parse::<f32>().ok()
    }
}

fn parse_rgb_components(inner: &str, has_alpha: bool) -> Option<Color> {
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    let expected = if has_alpha { 4 } else { 3 };
    if parts.len() != expected {
        return None;
    }
    let r = parse_component(parts[0])? / 255.0;
    let g = parse_component(parts[1])? / 255.0;
    let b = parse_component(parts[2])? / 255.0;
    let a = if has_alpha {
        parts[3].trim().parse::<f32>().ok()?
    } else {
        1.0
    };
    Some(Color([r, g, b, a]))
}

fn parse_hsl_components(inner: &str, has_alpha: bool) -> Option<Color> {
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    let expected = if has_alpha { 4 } else { 3 };
    if parts.len() != expected {
        return None;
    }
    let h = parts[0].trim_end_matches("deg").parse::<f32>().ok()?;
    let s = parts[1].trim_end_matches('%').parse::<f32>().ok()? / 100.0;
    let l = parts[2].trim_end_matches('%').parse::<f32>().ok()? / 100.0;
    let a = if has_alpha {
        parts[3].trim().parse::<f32>().ok()?
    } else {
        1.0
    };
    let (r, g, b) = hsl_to_rgb(h, s, l);
    Some(Color([r, g, b, a]))
}

/// `h` in degrees `[0, 360)`, `s`/`l` in `[0, 1]`.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h = (h % 360.0 + 360.0) % 360.0 / 360.0;
    let hue_to_rgb = |p: f32, q: f32, mut t: f32| -> f32 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 1.0 / 2.0 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    };
    (
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    )
}

/// The result of evaluating a style-spec expression: a small dynamically
/// typed value, mirroring the expression language's own runtime types
/// (booleans, numbers, strings, colors, arrays; no distinct object type is
/// needed for the subset of the spec this renderer evaluates).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Color(Color),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Null => false,
            Value::Number(n) => *n != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Color(_) => true,
        }
    }

    /// Coerces to a color: strings are parsed as CSS colors (returning
    /// `None` if unparseable, matching how the spec treats malformed color
    /// literals as evaluation errors).
    pub fn as_color(&self) -> Option<Color> {
        match self {
            Value::Color(c) => Some(*c),
            Value::String(s) => Color::parse(s),
            _ => None,
        }
    }

    pub fn to_display_string(&self) -> String {
        match self {
            Value::Null => String::new(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    format!("{}", *n as i64)
                } else {
                    n.to_string()
                }
            }
            Value::String(s) => s.clone(),
            Value::Color(c) => format!("{c:?}"),
            Value::Array(a) => a
                .iter()
                .map(Value::to_display_string)
                .collect::<Vec<_>>()
                .join(","),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_display_string())
    }
}

impl PartialEq<f64> for Value {
    fn eq(&self, other: &f64) -> bool {
        self.as_f64() == Some(*other)
    }
}
