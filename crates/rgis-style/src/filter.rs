//! Layer `filter` evaluation. The style spec has two filter dialects that
//! can both appear in real style JSON:
//!
//! - **legacy filters**: `["==", "class", "residential"]`, `["in", "class",
//!   "a", "b"]`, `["all", f1, f2, ...]`, `["has", "class"]` -- the *key* is a
//!   bare string, implicitly meaning "this feature's property".
//! - **expression filters**: `["==", ["get", "class"], "residential"]` --
//!   ordinary expressions, evaluated for truthiness.
//!
//! [`parse_filter`] normalizes the legacy dialect into the expression form
//! (turning bare-string keys into `["get", key]`) and then defers to
//! [`crate::expr::parse`], so [`crate::expr::Expr`] only has to implement
//! one evaluation path.

use serde_json::{Value as Json, json};

use crate::expr::{Expr, ExprError, parse};

const COMPARISON_OPS: &[&str] = &["==", "!=", "<", "<=", ">", ">="];

fn normalize(json_val: &Json) -> Json {
    let Json::Array(arr) = json_val else {
        return json_val.clone();
    };
    let Some(op) = arr.first().and_then(Json::as_str) else {
        return json_val.clone();
    };

    if op == "all" || op == "any" {
        let sub: Vec<Json> = arr[1..].iter().map(normalize).collect();
        let mut out = vec![json!(op)];
        out.extend(sub);
        return Json::Array(out);
    }
    if op == "none" {
        let sub: Vec<Json> = arr[1..].iter().map(normalize).collect();
        let mut any = vec![json!("any")];
        any.extend(sub);
        return json!(["!", Json::Array(any)]);
    }
    if COMPARISON_OPS.contains(&op) && arr.len() == 3 && arr[1].is_string() {
        return json!([op, ["get", arr[1].clone()], arr[2].clone()]);
    }
    if (op == "in" || op == "!in") && arr.len() >= 2 && arr[1].is_string() {
        let key = arr[1].clone();
        let mut in_expr = vec![json!("in"), json!(["get", key])];
        in_expr.extend(arr[2..].iter().cloned());
        let in_expr = Json::Array(in_expr);
        return if op == "!in" {
            json!(["!", in_expr])
        } else {
            in_expr
        };
    }
    // `has`/`!has` take a bare key string in both dialects; anything else
    // is assumed to already be a well-formed expression.
    json_val.clone()
}

/// Parses a layer's `filter` JSON (either dialect) into an evaluable
/// [`Expr`]. A missing filter (`None`) means "always matches"; callers
/// should treat that case separately rather than calling this with `None`.
pub fn parse_filter(filter_json: &Json) -> Result<Expr, ExprError> {
    parse(&normalize(filter_json))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{EvalContext, FeatureProperties};
    use crate::value::Value;
    use std::collections::HashMap;

    struct Feat(HashMap<String, Value>);
    impl FeatureProperties for Feat {
        fn get_property(&self, key: &str) -> Option<Value> {
            self.0.get(key).cloned()
        }
    }

    fn feat(pairs: &[(&str, Value)]) -> Feat {
        Feat(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn legacy_eq_filter() {
        let f = parse_filter(&json!(["==", "class", "wood"])).unwrap();
        let feature = feat(&[("class", Value::String("wood".into()))]);
        let ctx = EvalContext::with_feature(10.0, &feature);
        assert!(f.eval(&ctx).as_bool());

        let feature2 = feat(&[("class", Value::String("grass".into()))]);
        let ctx2 = EvalContext::with_feature(10.0, &feature2);
        assert!(!f.eval(&ctx2).as_bool());
    }

    #[test]
    fn legacy_in_filter() {
        let f = parse_filter(&json!(["in", "class", "city", "town"])).unwrap();
        let feature = feat(&[("class", Value::String("town".into()))]);
        let ctx = EvalContext::with_feature(0.0, &feature);
        assert!(f.eval(&ctx).as_bool());
    }

    #[test]
    fn expression_filter() {
        let f = parse_filter(&json!(["all", ["==", ["get", "class"], "wood"]])).unwrap();
        let feature = feat(&[("class", Value::String("wood".into()))]);
        let ctx = EvalContext::with_feature(0.0, &feature);
        assert!(f.eval(&ctx).as_bool());
    }

    #[test]
    fn none_filter() {
        let f = parse_filter(&json!(["none", ["==", "class", "wood"]])).unwrap();
        let feature = feat(&[("class", Value::String("wood".into()))]);
        let ctx = EvalContext::with_feature(0.0, &feature);
        assert!(!f.eval(&ctx).as_bool());
    }
}
