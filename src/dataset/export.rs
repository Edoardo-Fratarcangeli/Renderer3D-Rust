//! Export of a (filtered) row subset to CSV, streaming row by row.
//!
//! The output re-imports through the CSV loader to exactly the same
//! feature matrix and labels (verified by the integration tests).

use std::io::Write;
use std::path::Path;

use super::{Dataset, DatasetError, Result};

/// Write the given rows (feature columns + label) to a CSV file.
/// Rows are streamed: memory stays O(n_cols) regardless of subset size.
pub fn export_csv(dataset: &Dataset, rows: &[u32], path: &Path) -> Result<usize> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    // Header
    let mut header = dataset.metadata.column_names.join(",");
    header.push_str(",label\n");
    w.write_all(header.as_bytes())?;

    let mut buf = Vec::new();
    let mut line = String::new();
    for &row in rows {
        if row as usize >= dataset.n_rows() {
            return Err(DatasetError::Format(format!(
                "export row {} out of range",
                row
            )));
        }
        dataset.row(row as usize, &mut buf);
        line.clear();
        for (i, v) in buf.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push_str(&format_f32(*v));
        }
        line.push(',');
        line.push_str(&escape_csv(dataset.label_name(row as usize)));
        line.push('\n');
        w.write_all(line.as_bytes())?;
    }
    w.flush()?;
    Ok(rows.len())
}

fn format_f32(v: f32) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

fn escape_csv(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_are_formatted_without_decimals() {
        assert_eq!(format_f32(3.0), "3");
        assert_eq!(format_f32(-7.0), "-7");
        assert_eq!(format_f32(0.0), "0");
        assert_eq!(format_f32(2.5), "2.5");
        assert_eq!(format_f32(-0.125), "-0.125");
        // Huge magnitudes fall back to float formatting.
        assert!(format_f32(1e20).contains("100000000000000000000"));
    }

    #[test]
    fn csv_escaping_quotes_only_when_needed() {
        assert_eq!(escape_csv("plain"), "plain");
        assert_eq!(escape_csv("with,comma"), "\"with,comma\"");
        assert_eq!(escape_csv("with\"quote"), "\"with\"\"quote\"");
        assert_eq!(escape_csv("multi\nline"), "\"multi\nline\"");
    }
}
