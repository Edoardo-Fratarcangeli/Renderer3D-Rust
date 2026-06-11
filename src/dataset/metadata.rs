// Dataset metadata, persisted as JSON next to the projection/index caches.

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::Result;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LabelStat {
    pub name: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetMetadata {
    pub name: String,
    /// Source path; empty for generated/builtin datasets.
    pub source_path: String,
    /// "npy" | "npz" | "csv" | "idx" | "parquet" | "builtin"
    pub format: String,
    pub n_rows: usize,
    pub n_cols: usize,
    pub column_names: Vec<String>,
    /// Name of the column the labels came from, if any.
    pub label_column: Option<String>,
    pub labels: Vec<LabelStat>,
    /// Whether the feature matrix is memory mapped rather than RAM-resident.
    pub memory_mapped: bool,
    /// Unix timestamp (seconds) of creation.
    pub created_at: u64,
}

impl DatasetMetadata {
    pub fn new(name: &str, format: &str, n_rows: usize, n_cols: usize) -> Self {
        Self {
            name: name.to_string(),
            source_path: String::new(),
            format: format.to_string(),
            n_rows,
            n_cols,
            column_names: (0..n_cols).map(|i| format!("f{}", i)).collect(),
            label_column: None,
            labels: Vec::new(),
            memory_mapped: false,
            created_at: now_unix(),
        }
    }

    /// Fill `labels` stats from per-row label ids and the vocabulary.
    pub fn set_label_stats(&mut self, labels: &[u32], names: &[String]) {
        let mut counts = vec![0usize; names.len()];
        for &l in labels {
            counts[l as usize] += 1;
        }
        self.labels = names
            .iter()
            .zip(counts)
            .map(|(name, count)| LabelStat {
                name: name.clone(),
                count,
            })
            .collect();
    }

    pub fn save_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| super::DatasetError::Format(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(|e| super::DatasetError::Format(e.to_string()))
    }
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
