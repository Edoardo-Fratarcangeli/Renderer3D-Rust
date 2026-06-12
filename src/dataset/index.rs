//! Label index + filter/search evaluation.
//!
//! The index is persisted to disk as JSON so a re-import of the same file
//! skips the indexing pass. [`SearchQuery`] implements the small query
//! grammar used by the search panel; [`apply_filter`] resolves a
//! [`FilterSpec`] (label toggles + query) into the matching row set.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

use super::{Dataset, DatasetError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetIndex {
    /// label id -> sorted row indices carrying that label.
    pub label_rows: Vec<Vec<u32>>,
    pub n_rows: usize,
}

impl DatasetIndex {
    pub fn build(labels: &[u32], n_labels: usize) -> Self {
        let mut label_rows = vec![Vec::new(); n_labels];
        for (row, &l) in labels.iter().enumerate() {
            label_rows[l as usize].push(row as u32);
        }
        Self {
            label_rows,
            n_rows: labels.len(),
        }
    }

    pub fn count(&self, label: u32) -> usize {
        self.label_rows
            .get(label as usize)
            .map(|r| r.len())
            .unwrap_or(0)
    }

    /// Rows whose label is in `enabled`, in ascending row order.
    pub fn rows_for_labels(&self, enabled: &HashSet<u32>) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .label_rows
            .iter()
            .enumerate()
            .filter(|(l, _)| enabled.contains(&(*l as u32)))
            .flat_map(|(_, rows)| rows.iter().copied())
            .collect();
        out.sort_unstable();
        out
    }

    pub fn save_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json =
            serde_json::to_string(self).map_err(|e| DatasetError::Format(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(|e| DatasetError::Format(e.to_string()))
    }
}

/// A parsed search query.
///
/// Supported syntax (case-insensitive):
/// - `row:<n>`            exact row index
/// - `c<i> <op> <value>`  numeric predicate on feature column i, op in < > <= >= = !=
/// - anything else        substring match on the label name
#[derive(Debug, Clone, PartialEq)]
pub enum SearchQuery {
    All,
    Row(u32),
    Column { col: usize, op: CmpOp, value: f32 },
    LabelSubstring(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

impl CmpOp {
    fn eval(&self, a: f32, b: f32) -> bool {
        match self {
            CmpOp::Lt => a < b,
            CmpOp::Gt => a > b,
            CmpOp::Le => a <= b,
            CmpOp::Ge => a >= b,
            CmpOp::Eq => (a - b).abs() < 1e-6,
            CmpOp::Ne => (a - b).abs() >= 1e-6,
        }
    }
}

impl SearchQuery {
    pub fn parse(text: &str) -> Result<Self> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(SearchQuery::All);
        }
        if let Some(rest) = text.strip_prefix("row:") {
            let n = rest
                .trim()
                .parse::<u32>()
                .map_err(|_| DatasetError::InvalidQuery(format!("bad row number '{}'", rest)))?;
            return Ok(SearchQuery::Row(n));
        }
        if text.starts_with('c') || text.starts_with('C') {
            // try "c<idx> <op> <value>"
            for op_str in ["<=", ">=", "!=", "<", ">", "="] {
                if let Some(pos) = text.find(op_str) {
                    let col_part = text[1..pos].trim();
                    let val_part = text[pos + op_str.len()..].trim();
                    if let (Ok(col), Ok(value)) =
                        (col_part.parse::<usize>(), val_part.parse::<f32>())
                    {
                        let op = match op_str {
                            "<" => CmpOp::Lt,
                            ">" => CmpOp::Gt,
                            "<=" => CmpOp::Le,
                            ">=" => CmpOp::Ge,
                            "=" => CmpOp::Eq,
                            "!=" => CmpOp::Ne,
                            _ => unreachable!(),
                        };
                        return Ok(SearchQuery::Column { col, op, value });
                    }
                }
            }
        }
        Ok(SearchQuery::LabelSubstring(text.to_ascii_lowercase()))
    }

    /// Evaluate against one row; `features` must hold the row when the query
    /// is a column predicate (pass an empty slice otherwise).
    pub fn matches(&self, dataset: &Dataset, row: u32, features: &[f32]) -> bool {
        match self {
            SearchQuery::All => true,
            SearchQuery::Row(n) => row == *n,
            SearchQuery::Column { col, op, value } => features
                .get(*col)
                .map(|v| op.eval(*v, *value))
                .unwrap_or(false),
            SearchQuery::LabelSubstring(s) => dataset
                .label_name(row as usize)
                .to_ascii_lowercase()
                .contains(s.as_str()),
        }
    }

    pub fn needs_features(&self) -> bool {
        matches!(self, SearchQuery::Column { .. })
    }
}

/// Filter specification combining label toggles with a search query.
#[derive(Debug, Clone)]
pub struct FilterSpec {
    pub enabled_labels: HashSet<u32>,
    pub query: SearchQuery,
}

impl FilterSpec {
    pub fn all_labels(n_labels: usize) -> Self {
        Self {
            enabled_labels: (0..n_labels as u32).collect(),
            query: SearchQuery::All,
        }
    }
}

/// Resolve a FilterSpec to the matching row set (ascending order).
pub fn apply_filter(dataset: &Dataset, index: &DatasetIndex, spec: &FilterSpec) -> Vec<u32> {
    let candidate_rows = index.rows_for_labels(&spec.enabled_labels);
    if spec.query == SearchQuery::All {
        return candidate_rows;
    }
    let mut buf = Vec::new();
    let needs_features = spec.query.needs_features();
    candidate_rows
        .into_iter()
        .filter(|&row| {
            if needs_features {
                dataset.row(row as usize, &mut buf);
            }
            spec.query.matches(dataset, row, &buf)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmp_op_evaluation_covers_all_operators() {
        assert!(CmpOp::Lt.eval(1.0, 2.0) && !CmpOp::Lt.eval(2.0, 1.0));
        assert!(CmpOp::Gt.eval(2.0, 1.0) && !CmpOp::Gt.eval(1.0, 2.0));
        assert!(CmpOp::Le.eval(2.0, 2.0) && !CmpOp::Le.eval(3.0, 2.0));
        assert!(CmpOp::Ge.eval(2.0, 2.0) && !CmpOp::Ge.eval(1.0, 2.0));
        assert!(CmpOp::Eq.eval(2.0, 2.0) && !CmpOp::Eq.eval(2.1, 2.0));
        assert!(CmpOp::Ne.eval(2.1, 2.0) && !CmpOp::Ne.eval(2.0, 2.0));
    }

    #[test]
    fn malformed_column_queries_fall_back_to_substring() {
        // Not parseable as a column predicate -> label substring.
        assert!(matches!(
            SearchQuery::parse("cat > dog").unwrap(),
            SearchQuery::LabelSubstring(_)
        ));
        assert!(matches!(
            SearchQuery::parse("c > 1").unwrap(),
            SearchQuery::LabelSubstring(_)
        ));
        // Uppercase column prefix works.
        assert!(matches!(
            SearchQuery::parse("C2 <= 0.5").unwrap(),
            SearchQuery::Column { col: 2, op: CmpOp::Le, .. }
        ));
    }

    #[test]
    fn needs_features_only_for_column_queries() {
        assert!(SearchQuery::parse("c0 > 1").unwrap().needs_features());
        assert!(!SearchQuery::parse("row:1").unwrap().needs_features());
        assert!(!SearchQuery::parse("abc").unwrap().needs_features());
        assert!(!SearchQuery::All.needs_features());
    }

    #[test]
    fn rows_for_labels_ignores_unknown_ids() {
        let idx = DatasetIndex::build(&[0, 1, 0], 2);
        let rows = idx.rows_for_labels(&HashSet::from([1, 99]));
        assert_eq!(rows, vec![1]);
        assert_eq!(idx.count(99), 0);
    }
}
