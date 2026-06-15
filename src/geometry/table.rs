//! Header-mapped geometry tables: one row = one geometry.
//!
//! CSV files and Excel sheets (xlsx/xls/ods via `calamine`) both funnel
//! through [`RecordTable`], so column resolution behaves identically for
//! every tabular source. Recognized columns (case-insensitive):
//!
//! | Purpose | Accepted headers |
//! |---------|------------------|
//! | shape   | `shape`, `type`, `geometry`, `geom`, `kind` (optional → sphere) |
//! | position| `x`/`y`/`z`, `px`/`py`/`pz`, `pos_x`/`pos_y`/`pos_z` |
//! | uniform scale | `size`, `radius`, `scale` |
//! | per-axis scale | `sx`/`sy`/`sz`, `scale_x`/`scale_y`/`scale_z` |
//! | rotation (deg) | `rx`/`ry`/`rz`, `rot_x`/`rot_y`/`rot_z` |
//! | color   | `color`/`colour` (hex or `r,g,b`) or `r`/`g`/`b`, `red`/`green`/`blue` |
//! | label   | `label`, `name`, `id`, `tag` |

use std::path::Path;

use crate::scene::GeometryType;

use super::{
    normalize_rgb, parse_color, parse_shape, GeometryError, GeometryRecord, Result, POINT_SIZE,
};

/// A parsed-but-untyped table: headers + string cells. CSV and Excel both
/// reduce to this before column mapping.
pub struct RecordTable {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// Find a column index by any of several case-insensitive names.
fn col(headers: &[String], names: &[&str]) -> Option<usize> {
    headers
        .iter()
        .position(|h| names.iter().any(|n| h.trim().eq_ignore_ascii_case(n)))
}

fn cell(row: &[String], idx: Option<usize>) -> Option<&str> {
    idx.and_then(|i| row.get(i)).map(|s| s.trim()).filter(|s| !s.is_empty())
}

fn num(row: &[String], idx: Option<usize>, what: &str, line: usize) -> Result<Option<f32>> {
    match cell(row, idx) {
        None => Ok(None),
        Some(s) => s.parse::<f32>().map(Some).map_err(|_| GeometryError::Parse {
            line,
            message: format!("'{}' is not a number for {}", s, what),
        }),
    }
}

impl RecordTable {
    /// Map every row to a [`GeometryRecord`].
    pub fn to_records(&self, default_color: [f32; 3]) -> Result<Vec<GeometryRecord>> {
        let h = &self.headers;
        let c_shape = col(h, &["shape", "type", "geometry", "geom", "kind"]);
        let c_x = col(h, &["x", "px", "pos_x", "posx"]);
        let c_y = col(h, &["y", "py", "pos_y", "posy"]);
        let c_z = col(h, &["z", "pz", "pos_z", "posz"]);
        let c_size = col(h, &["size", "radius", "scale"]);
        let c_sx = col(h, &["sx", "scale_x"]);
        let c_sy = col(h, &["sy", "scale_y"]);
        let c_sz = col(h, &["sz", "scale_z"]);
        let c_rx = col(h, &["rx", "rot_x"]);
        let c_ry = col(h, &["ry", "rot_y"]);
        let c_rz = col(h, &["rz", "rot_z"]);
        let c_color = col(h, &["color", "colour", "col"]);
        let c_r = col(h, &["r", "red"]);
        let c_g = col(h, &["g", "green"]);
        let c_b = col(h, &["b", "blue"]);
        let c_label = col(h, &["label", "name", "id", "tag"]);

        if c_x.is_none() || c_y.is_none() || c_z.is_none() {
            return Err(GeometryError::Format(
                "table needs x, y and z columns (see docs/GEOMETRY_IMPORT.md)".into(),
            ));
        }

        let mut records = Vec::with_capacity(self.rows.len());
        for (i, row) in self.rows.iter().enumerate() {
            let line = i + 2; // 1-based + header row
            let (shape, is_point) = match cell(row, c_shape) {
                Some(s) => parse_shape(s).ok_or_else(|| GeometryError::Parse {
                    line,
                    message: format!("unknown shape '{}'", s),
                })?,
                // Shape column optional: bare coordinate tables become points.
                None => (GeometryType::Sphere, true),
            };
            let pos = [
                num(row, c_x, "x", line)?.unwrap_or(0.0),
                num(row, c_y, "y", line)?.unwrap_or(0.0),
                num(row, c_z, "z", line)?.unwrap_or(0.0),
            ];
            let mut record = GeometryRecord::new(shape, pos);
            record.color = default_color;
            if is_point {
                record.scale = [POINT_SIZE; 3];
            }

            if let Some(s) = num(row, c_size, "size", line)? {
                record.scale = [s; 3];
            }
            if let (Some(sx), Some(sy), Some(sz)) = (
                num(row, c_sx, "sx", line)?,
                num(row, c_sy, "sy", line)?,
                num(row, c_sz, "sz", line)?,
            ) {
                record.scale = [sx, sy, sz];
            }
            record.rotation = [
                num(row, c_rx, "rx", line)?.unwrap_or(0.0),
                num(row, c_ry, "ry", line)?.unwrap_or(0.0),
                num(row, c_rz, "rz", line)?.unwrap_or(0.0),
            ];

            if let Some(c) = cell(row, c_color) {
                record.color = parse_color(c).ok_or_else(|| GeometryError::Parse {
                    line,
                    message: format!("bad color '{}'", c),
                })?;
            } else if let (Some(r), Some(g), Some(b)) = (
                num(row, c_r, "r", line)?,
                num(row, c_g, "g", line)?,
                num(row, c_b, "b", line)?,
            ) {
                record.color = normalize_rgb([r, g, b]);
            }

            record.label = cell(row, c_label).map(|s| s.to_string());
            records.push(record);
        }
        Ok(records)
    }
}

/// Read a geometry CSV into records (streamed through the `csv` reader).
pub fn from_csv(path: &Path, default_color: [f32; 3]) -> Result<Vec<GeometryRecord>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| GeometryError::Format(format!("CSV open failed: {}", e)))?;
    let headers = reader
        .headers()
        .map_err(|e| GeometryError::Format(e.to_string()))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut rows = Vec::new();
    for rec in reader.records() {
        let rec = rec.map_err(|e| GeometryError::Format(e.to_string()))?;
        rows.push(rec.iter().map(|s| s.to_string()).collect());
    }
    RecordTable { headers, rows }.to_records(default_color)
}

/// Read the first sheet of an Excel workbook (xlsx/xlsm/xls/ods).
pub fn from_excel(path: &Path, default_color: [f32; 3]) -> Result<Vec<GeometryRecord>> {
    use crate::util::excel_cell_to_string as cell_to_string;
    use calamine::{open_workbook_auto, Reader};

    let mut workbook = open_workbook_auto(path)
        .map_err(|e| GeometryError::Format(format!("Excel open failed: {}", e)))?;
    let sheet_name = workbook
        .sheet_names()
        .first()
        .cloned()
        .ok_or_else(|| GeometryError::Format("workbook has no sheets".into()))?;
    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| GeometryError::Format(format!("sheet '{}': {}", sheet_name, e)))?;

    let mut rows_iter = range.rows();
    let headers: Vec<String> = rows_iter
        .next()
        .ok_or_else(|| GeometryError::Format("sheet is empty".into()))?
        .iter()
        .map(cell_to_string)
        .collect();
    let rows: Vec<Vec<String>> = rows_iter
        .map(|row| row.iter().map(cell_to_string).collect::<Vec<String>>())
        .filter(|row: &Vec<String>| row.iter().any(|c| !c.is_empty()))
        .collect();

    RecordTable { headers, rows }.to_records(default_color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::DEFAULT_COLOR;

    fn table(headers: &[&str], rows: &[&[&str]]) -> RecordTable {
        RecordTable {
            headers: headers.iter().map(|s| s.to_string()).collect(),
            rows: rows
                .iter()
                .map(|r| r.iter().map(|s| s.to_string()).collect())
                .collect(),
        }
    }

    #[test]
    fn full_header_set_maps_every_field() {
        let t = table(
            &["Type", "x", "y", "z", "size", "rx", "ry", "rz", "color", "name"],
            &[&["cube", "1", "2", "3", "2.5", "0", "45", "0", "#ff0000", "boxy"]],
        );
        let recs = t.to_records(DEFAULT_COLOR).unwrap();
        assert_eq!(recs[0].shape, GeometryType::Cube);
        assert_eq!(recs[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(recs[0].scale, [2.5; 3]);
        assert_eq!(recs[0].rotation, [0.0, 45.0, 0.0]);
        assert_eq!(recs[0].color, [1.0, 0.0, 0.0]);
        assert_eq!(recs[0].label.as_deref(), Some("boxy"));
    }

    #[test]
    fn rgb_columns_and_per_axis_scale() {
        let t = table(
            &["shape", "x", "y", "z", "sx", "sy", "sz", "red", "green", "blue"],
            &[&["plane", "0", "0", "0", "4", "1", "2", "255", "0", "0"]],
        );
        let r = &t.to_records(DEFAULT_COLOR).unwrap()[0];
        assert_eq!(r.scale, [4.0, 1.0, 2.0]);
        assert_eq!(r.color, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn missing_shape_column_defaults_to_points() {
        let t = table(&["x", "y", "z"], &[&["1", "2", "3"], &["4", "5", "6"]]);
        let recs = t.to_records(DEFAULT_COLOR).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].shape, GeometryType::Sphere);
        assert_eq!(recs[0].scale, [POINT_SIZE; 3]);
    }

    #[test]
    fn errors_carry_row_numbers() {
        let t = table(&["shape", "x", "y", "z"], &[&["cube", "1", "oops", "3"]]);
        let err = t.to_records(DEFAULT_COLOR).unwrap_err();
        assert!(err.to_string().contains("line 2"));

        let t = table(&["shape", "x", "y", "z"], &[&["pyramid", "1", "2", "3"]]);
        assert!(t.to_records(DEFAULT_COLOR).is_err());

        let t = table(&["a", "b"], &[]);
        let err = t.to_records(DEFAULT_COLOR).unwrap_err();
        assert!(err.to_string().contains("x, y and z"));
    }
}
