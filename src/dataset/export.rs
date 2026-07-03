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

    // Header — column names come from imported files, so they need the
    // same escaping as any other untrusted cell.
    let mut header = dataset
        .metadata
        .column_names
        .iter()
        .map(|c| escape_csv(c))
        .collect::<Vec<_>>()
        .join(",");
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
    // Formula-injection guard (CWE-1236): cells starting with = + - @ TAB CR
    // are evaluated as formulas by Excel/LibreOffice. Labels and column
    // names come from untrusted sources (imported files, the TCP stream),
    // so neutralise them with the standard leading-apostrophe convention.
    let needs_guard = s
        .chars()
        .next()
        .is_some_and(|c| matches!(c, '=' | '+' | '-' | '@' | '\t' | '\r'));
    let guarded = if needs_guard {
        format!("'{}", s)
    } else {
        s.to_string()
    };
    if needs_guard || guarded.contains([',', '"', '\n']) {
        format!("\"{}\"", guarded.replace('"', "\"\""))
    } else {
        guarded
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

    #[test]
    fn csv_formula_injection_is_neutralised() {
        assert_eq!(escape_csv("=1+1"), "\"'=1+1\"");
        assert_eq!(escape_csv("=HYPERLINK(\"http://evil\")"),
                   "\"'=HYPERLINK(\"\"http://evil\"\")\"");
        assert_eq!(escape_csv("+sum"), "\"'+sum\"");
        assert_eq!(escape_csv("-neg"), "\"'-neg\"");
        assert_eq!(escape_csv("@cmd"), "\"'@cmd\"");
        assert_eq!(escape_csv("\tindent"), "\"'\tindent\"");
        // Non-leading special chars are untouched.
        assert_eq!(escape_csv("a=b"), "a=b");
    }
}
