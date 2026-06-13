//! Geometry import dispatch: one entry point per "world" the data can come
//! from, plus extension-based routing for files and auto-detection for
//! pasted strings.

use std::path::Path;

use super::{parser, table, GeometryError, GeometryLayer, GeometryRecord, Result};

/// Import a geometry file, dispatching on its extension:
///
/// | Extension | Parser |
/// |-----------|--------|
/// | `.csv` | header-mapped table ([`table::from_csv`]) |
/// | `.xlsx` `.xlsm` `.xls` `.ods` | first Excel sheet ([`table::from_excel`]) |
/// | `.json` | geometry objects ([`json::parse_json`](super::json::parse_json)) |
/// | `.xyz` | point cloud text ([`parser::parse_xyz`]) |
/// | `.txt` `.geo` `.dsl` | auto-detected DSL or XYZ ([`parser::parse_auto`]) |
pub fn load_path(path: &Path, default_color: [f32; 3]) -> Result<Vec<GeometryRecord>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => table::from_csv(path, default_color),
        "xlsx" | "xlsm" | "xls" | "ods" => table::from_excel(path, default_color),
        "json" => super::json::parse_json(&std::fs::read_to_string(path)?, default_color),
        "xyz" => parser::parse_xyz(&std::fs::read_to_string(path)?, default_color),
        "txt" | "geo" | "dsl" => {
            parser::parse_auto(&std::fs::read_to_string(path)?, default_color)
        }
        other => Err(GeometryError::Unsupported(format!(
            "unknown geometry extension '{}' (expected csv, xlsx, xls, ods, json, xyz, txt)",
            other
        ))),
    }
}

/// Import a pasted string: JSON if it looks like JSON, otherwise DSL/XYZ.
pub fn load_string(text: &str, default_color: [f32; 3]) -> Result<Vec<GeometryRecord>> {
    let trimmed = text.trim_start();
    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        super::json::parse_json(text, default_color)
    } else {
        parser::parse_auto(text, default_color)
    }
}

/// Build a named layer from a file (layer name = file stem).
pub fn layer_from_path(path: &Path, default_color: [f32; 3]) -> Result<GeometryLayer> {
    let records = load_path(path, default_color)?;
    if records.is_empty() {
        return Err(GeometryError::Format("file contains no geometries".into()));
    }
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("layer")
        .to_string();
    Ok(GeometryLayer::new(name, records))
}

/// Build a named layer from pasted text.
pub fn layer_from_string(
    text: &str,
    name: impl Into<String>,
    default_color: [f32; 3],
) -> Result<GeometryLayer> {
    let records = load_string(text, default_color)?;
    if records.is_empty() {
        return Err(GeometryError::Format("text contains no geometries".into()));
    }
    Ok(GeometryLayer::new(name, records))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::DEFAULT_COLOR;

    #[test]
    fn string_loader_routes_json_dsl_and_xyz() {
        let json = load_string(r#"[{"shape":"cube","pos":[0,0,0]}]"#, DEFAULT_COLOR).unwrap();
        assert_eq!(json.len(), 1);
        let dsl = load_string("sphere 1 1 1 0.5", DEFAULT_COLOR).unwrap();
        assert_eq!(dsl.len(), 1);
        let xyz = load_string("1 2 3\n4 5 6", DEFAULT_COLOR).unwrap();
        assert_eq!(xyz.len(), 2);
    }

    #[test]
    fn empty_inputs_are_rejected_with_a_message() {
        let err = layer_from_string("# nothing here", "l", DEFAULT_COLOR).unwrap_err();
        assert!(err.to_string().contains("no geometries"));
    }
}
