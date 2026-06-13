//! JSON geometry documents.
//!
//! Accepted top-level forms:
//! - an array of geometry objects: `[ {...}, {...} ]`
//! - an object with a `geometries`, `objects`, `shapes` or `items` array
//!
//! Each geometry object is tolerant about field spellings:
//!
//! ```json
//! {
//!   "shape": "cube",            // or "type"; default "point"
//!   "pos": [1, 2, 3],           // or "position", or "x"/"y"/"z" fields
//!   "size": 2.0,                // or "radius" | "scale": number or [sx,sy,sz]
//!   "rotation": [0, 45, 0],     // degrees, optional
//!   "color": "#ff8800",         // or [r,g,b] (0–1 or 0–255), optional
//!   "label": "my box"           // or "name", optional
//! }
//! ```

use serde_json::Value;

use crate::scene::GeometryType;

use super::{
    normalize_rgb, parse_color, parse_shape, GeometryError, GeometryRecord, Result, POINT_SIZE,
};

/// Parse a JSON string into geometry records.
pub fn parse_json(text: &str, default_color: [f32; 3]) -> Result<Vec<GeometryRecord>> {
    let value: Value = serde_json::from_str(text)
        .map_err(|e| GeometryError::Format(format!("invalid JSON: {}", e)))?;

    let items: &Vec<Value> = match &value {
        Value::Array(items) => items,
        Value::Object(map) => ["geometries", "objects", "shapes", "items"]
            .iter()
            .find_map(|k| map.get(*k).and_then(|v| v.as_array()))
            .ok_or_else(|| {
                GeometryError::Format(
                    "expected a JSON array or an object with a 'geometries' array".into(),
                )
            })?,
        _ => {
            return Err(GeometryError::Format(
                "expected a JSON array of geometry objects".into(),
            ))
        }
    };

    let mut records = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        records.push(parse_object(item, i + 1, default_color)?);
    }
    Ok(records)
}

fn parse_object(item: &Value, n: usize, default_color: [f32; 3]) -> Result<GeometryRecord> {
    let err = |message: String| GeometryError::Parse { line: n, message };
    let obj = item
        .as_object()
        .ok_or_else(|| err("geometry entry is not an object".into()))?;

    let get = |keys: &[&str]| keys.iter().find_map(|k| obj.get(*k));

    let (shape, is_point) = match get(&["shape", "type"]).and_then(|v| v.as_str()) {
        Some(s) => parse_shape(s).ok_or_else(|| err(format!("unknown shape '{}'", s)))?,
        None => (GeometryType::Sphere, true),
    };

    let position = match get(&["pos", "position"]) {
        Some(v) => triple(v).ok_or_else(|| err("'pos' must be an array of 3 numbers".into()))?,
        None => {
            let coord = |k: &str| get(&[k]).and_then(Value::as_f64).map(|v| v as f32);
            match (coord("x"), coord("y"), coord("z")) {
                (Some(x), Some(y), Some(z)) => [x, y, z],
                _ => return Err(err("missing position ('pos' array or x/y/z fields)".into())),
            }
        }
    };

    let mut record = GeometryRecord::new(shape, position);
    record.color = default_color;
    if is_point {
        record.scale = [POINT_SIZE; 3];
    }

    if let Some(v) = get(&["size", "radius", "scale"]) {
        record.scale = match v {
            Value::Number(s) => [s.as_f64().unwrap_or(1.0) as f32; 3],
            other => triple(other)
                .ok_or_else(|| err("'scale' must be a number or [sx, sy, sz]".into()))?,
        };
    }
    if let Some(v) = get(&["rotation", "rot"]) {
        record.rotation =
            triple(v).ok_or_else(|| err("'rotation' must be [rx, ry, rz] degrees".into()))?;
    }
    if let Some(v) = get(&["color", "colour"]) {
        record.color = match v {
            Value::String(s) => {
                parse_color(s).ok_or_else(|| err(format!("bad color '{}'", s)))?
            }
            other => triple(other)
                .map(normalize_rgb)
                .ok_or_else(|| err("'color' must be \"#hex\" or [r, g, b]".into()))?,
        };
    }
    record.label = get(&["label", "name"])
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Ok(record)
}

fn triple(v: &Value) -> Option<[f32; 3]> {
    let arr = v.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    let mut out = [0.0f32; 3];
    for (slot, item) in out.iter_mut().zip(arr) {
        *slot = item.as_f64()? as f32;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::DEFAULT_COLOR;

    #[test]
    fn array_and_wrapped_forms_parse() {
        let arr =
            r##"[{"shape":"cube","pos":[1,2,3],"size":2,"color":"#ff0000","label":"a"}]"##;
        let recs = parse_json(arr, DEFAULT_COLOR).unwrap();
        assert_eq!(recs[0].shape, GeometryType::Cube);
        assert_eq!(recs[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(recs[0].scale, [2.0; 3]);
        assert_eq!(recs[0].color, [1.0, 0.0, 0.0]);
        assert_eq!(recs[0].label.as_deref(), Some("a"));

        let wrapped = r#"{"geometries":[{"type":"plane","x":1,"y":0,"z":0,
            "scale":[4,4,1],"rotation":[0,0,90],"color":[255,128,0],"name":"floor"}]}"#;
        let recs = parse_json(wrapped, DEFAULT_COLOR).unwrap();
        assert_eq!(recs[0].shape, GeometryType::Plane);
        assert_eq!(recs[0].scale, [4.0, 4.0, 1.0]);
        assert_eq!(recs[0].rotation, [0.0, 0.0, 90.0]);
        assert_eq!(recs[0].color[0], 1.0);
    }

    #[test]
    fn shapeless_entries_become_points() {
        let recs = parse_json(r#"[{"x":1,"y":2,"z":3}]"#, DEFAULT_COLOR).unwrap();
        assert_eq!(recs[0].shape, GeometryType::Sphere);
        assert_eq!(recs[0].scale, [POINT_SIZE; 3]);
    }

    #[test]
    fn malformed_documents_report_clear_errors() {
        assert!(parse_json("not json", DEFAULT_COLOR).is_err());
        assert!(parse_json("42", DEFAULT_COLOR).is_err());
        assert!(parse_json(r#"{"other": []}"#, DEFAULT_COLOR).is_err());
        assert!(parse_json("[42]", DEFAULT_COLOR).is_err());
        assert!(parse_json(r#"[{"shape":"cube"}]"#, DEFAULT_COLOR).is_err());
        assert!(parse_json(r#"[{"shape":"hex","pos":[0,0,0]}]"#, DEFAULT_COLOR).is_err());
        assert!(parse_json(r#"[{"pos":[1,2],"shape":"cube"}]"#, DEFAULT_COLOR).is_err());
        let err = parse_json(
            r##"[{"shape":"cube","pos":[0,0,0],"color":"#xx"}]"##,
            DEFAULT_COLOR,
        )
        .unwrap_err();
        assert!(err.to_string().contains("line 1"));
    }
}
