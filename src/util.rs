//! Small cross-module helpers shared by otherwise-independent subsystems.

/// Render a calamine spreadsheet cell as a trimmed string.
///
/// Shared by the dataset loader ([`crate::dataset::loader`]) and the geometry
/// table loader ([`crate::geometry::table`]) so the Excel cell-formatting rule
/// lives in exactly one place. Integer-valued floats are emitted without the
/// trailing `.0` that spreadsheets would otherwise introduce.
pub(crate) fn excel_cell_to_string(cell: &calamine::Data) -> String {
    use calamine::Data;
    match cell {
        Data::Empty => String::new(),
        Data::Float(f) if f.fract() == 0.0 && f.abs() < 1e15 => format!("{}", *f as i64),
        other => other.to_string().trim().to_string(),
    }
}
