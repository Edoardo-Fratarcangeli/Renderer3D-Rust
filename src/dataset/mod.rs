//! Dataset domain: loading, metadata, indexing, preprocessing and export of
//! ML datasets.
//!
//! This module is pure logic: it must never depend on egui/wgpu so it stays
//! unit-testable and reusable outside the UI.
//!
//! - [`loader`] — multi-format import (NPY/NPZ/CSV/IDX, optional Parquet)
//! - [`metadata`] — JSON-persisted dataset description
//! - [`index`] — label index + filter/search evaluation
//! - [`preprocessor`] — 3D projection (PCA) with a disk cache
//! - [`export`] — CSV export of filtered subsets
//! - [`builtin`] — deterministic synthetic benchmark generators

pub mod builtin;
pub mod export;
pub mod index;
pub mod loader;
pub mod metadata;
pub mod preprocessor;
pub mod stream;

use std::fmt;

pub use index::DatasetIndex;
pub use metadata::DatasetMetadata;

/// Errors produced by the dataset layer.
#[derive(Debug)]
pub enum DatasetError {
    Io(std::io::Error),
    Format(String),
    Unsupported(String),
    InvalidQuery(String),
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetError::Io(e) => write!(f, "I/O error: {}", e),
            DatasetError::Format(m) => write!(f, "format error: {}", m),
            DatasetError::Unsupported(m) => write!(f, "unsupported: {}", m),
            DatasetError::InvalidQuery(m) => write!(f, "invalid query: {}", m),
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<std::io::Error> for DatasetError {
    fn from(e: std::io::Error) -> Self {
        DatasetError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, DatasetError>;

/// Supported numeric element types for memory-mapped sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    F32,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}

impl ElemType {
    pub fn size(&self) -> usize {
        match self {
            ElemType::I8 | ElemType::U8 => 1,
            ElemType::I16 | ElemType::U16 => 2,
            ElemType::F32 | ElemType::I32 | ElemType::U32 => 4,
            ElemType::F64 | ElemType::I64 | ElemType::U64 => 8,
        }
    }

    /// Decode one element at byte offset `off` (little endian) as f32.
    pub fn read_f32(&self, bytes: &[u8], off: usize) -> f32 {
        macro_rules! rd {
            ($t:ty, $n:expr) => {{
                let mut buf = [0u8; $n];
                buf.copy_from_slice(&bytes[off..off + $n]);
                <$t>::from_le_bytes(buf) as f32
            }};
        }
        match self {
            ElemType::F32 => rd!(f32, 4),
            ElemType::F64 => rd!(f64, 8),
            ElemType::I8 => rd!(i8, 1),
            ElemType::U8 => rd!(u8, 1),
            ElemType::I16 => rd!(i16, 2),
            ElemType::U16 => rd!(u16, 2),
            ElemType::I32 => rd!(i32, 4),
            ElemType::U32 => rd!(u32, 4),
            ElemType::I64 => rd!(i64, 8),
            ElemType::U64 => rd!(u64, 8),
        }
    }
}

/// Backing storage of the feature matrix. Large on-disk files are memory
/// mapped and decoded row by row instead of being copied into RAM.
pub enum FeatureSource {
    /// Row-major `n_rows * n_cols` matrix fully in memory (small datasets).
    InMemory(Vec<f32>),
    /// Memory-mapped raw buffer (NPY / IDX payload). Rows are decoded lazily.
    Mmap {
        map: memmap2::Mmap,
        /// Byte offset of the first data element inside the mapping.
        data_offset: usize,
        elem: ElemType,
    },
}

impl FeatureSource {
    pub fn is_memory_mapped(&self) -> bool {
        matches!(self, FeatureSource::Mmap { .. })
    }
}

/// A loaded dataset: feature matrix + per-row label ids + label vocabulary.
pub struct Dataset {
    pub metadata: DatasetMetadata,
    pub source: FeatureSource,
    /// Label id for every row (index into `label_names`).
    pub labels: Vec<u32>,
    /// Label vocabulary: id -> human readable name.
    pub label_names: Vec<String>,
}

impl fmt::Debug for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dataset")
            .field("name", &self.metadata.name)
            .field("n_rows", &self.metadata.n_rows)
            .field("n_cols", &self.metadata.n_cols)
            .field("labels", &self.label_names)
            .field("memory_mapped", &self.source.is_memory_mapped())
            .finish()
    }
}

impl Dataset {
    pub fn n_rows(&self) -> usize {
        self.metadata.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.metadata.n_cols
    }

    /// Decode row `i` into `out` (cleared first). Panics if out of range.
    pub fn row(&self, i: usize, out: &mut Vec<f32>) {
        assert!(i < self.n_rows(), "row {} out of range", i);
        out.clear();
        let d = self.n_cols();
        match &self.source {
            FeatureSource::InMemory(data) => {
                out.extend_from_slice(&data[i * d..(i + 1) * d]);
            }
            FeatureSource::Mmap {
                map,
                data_offset,
                elem,
            } => {
                let es = elem.size();
                let base = data_offset + i * d * es;
                for c in 0..d {
                    out.push(elem.read_f32(map, base + c * es));
                }
            }
        }
    }

    /// Single cell access (used by the table viewer).
    pub fn value(&self, row: usize, col: usize) -> f32 {
        let d = self.n_cols();
        assert!(row < self.n_rows() && col < d);
        match &self.source {
            FeatureSource::InMemory(data) => data[row * d + col],
            FeatureSource::Mmap {
                map,
                data_offset,
                elem,
            } => {
                let es = elem.size();
                elem.read_f32(map, data_offset + (row * d + col) * es)
            }
        }
    }

    /// Name of a row's label.
    pub fn label_name(&self, row: usize) -> &str {
        &self.label_names[self.labels[row] as usize]
    }
}

/// Stable 64-bit content key for cache filenames (FNV-1a, no extra deps).
pub fn fnv1a64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for part in parts {
        for &b in *part {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elem_type_sizes() {
        assert_eq!(ElemType::I8.size(), 1);
        assert_eq!(ElemType::U8.size(), 1);
        assert_eq!(ElemType::I16.size(), 2);
        assert_eq!(ElemType::U16.size(), 2);
        assert_eq!(ElemType::F32.size(), 4);
        assert_eq!(ElemType::I32.size(), 4);
        assert_eq!(ElemType::U32.size(), 4);
        assert_eq!(ElemType::F64.size(), 8);
        assert_eq!(ElemType::I64.size(), 8);
        assert_eq!(ElemType::U64.size(), 8);
    }

    #[test]
    fn elem_type_decodes_every_variant() {
        let check = |elem: ElemType, bytes: &[u8], expected: f32| {
            assert_eq!(elem.read_f32(bytes, 0), expected, "{:?}", elem);
        };
        check(ElemType::F32, &1.5f32.to_le_bytes(), 1.5);
        check(ElemType::F64, &(-2.25f64).to_le_bytes(), -2.25);
        check(ElemType::I8, &(-7i8).to_le_bytes(), -7.0);
        check(ElemType::U8, &200u8.to_le_bytes(), 200.0);
        check(ElemType::I16, &(-300i16).to_le_bytes(), -300.0);
        check(ElemType::U16, &60000u16.to_le_bytes(), 60000.0);
        check(ElemType::I32, &(-100000i32).to_le_bytes(), -100000.0);
        check(ElemType::U32, &100000u32.to_le_bytes(), 100000.0);
        check(ElemType::I64, &(-5i64).to_le_bytes(), -5.0);
        check(ElemType::U64, &5u64.to_le_bytes(), 5.0);
    }

    #[test]
    fn elem_type_reads_at_offset() {
        let mut bytes = vec![0u8; 8];
        bytes[4..8].copy_from_slice(&42.0f32.to_le_bytes());
        assert_eq!(ElemType::F32.read_f32(&bytes, 4), 42.0);
    }

    #[test]
    fn fnv1a64_is_stable_and_order_sensitive() {
        let a = fnv1a64(&[b"hello", b"world"]);
        assert_eq!(a, fnv1a64(&[b"hello", b"world"]));
        assert_ne!(a, fnv1a64(&[b"world", b"hello"]));
        assert_ne!(a, fnv1a64(&[b"helloworldX"]));
        // Concatenation boundary does not matter (it hashes the byte stream).
        assert_eq!(a, fnv1a64(&[b"helloworld"]));
    }

    #[test]
    fn error_display_variants() {
        let io: DatasetError = std::io::Error::other("boom").into();
        assert!(io.to_string().contains("I/O error"));
        assert!(DatasetError::Format("x".into()).to_string().contains("format error"));
        assert!(DatasetError::Unsupported("y".into())
            .to_string()
            .contains("unsupported"));
        assert!(DatasetError::InvalidQuery("z".into())
            .to_string()
            .contains("invalid query"));
    }

    #[test]
    fn dataset_debug_is_compact() {
        let ds = Dataset {
            metadata: DatasetMetadata::new("dbg", "builtin", 2, 2),
            source: FeatureSource::InMemory(vec![1.0, 2.0, 3.0, 4.0]),
            labels: vec![0, 0],
            label_names: vec!["only".into()],
        };
        let s = format!("{:?}", ds);
        assert!(s.contains("dbg") && s.contains("n_rows"));
        // The raw data must not be dumped.
        assert!(!s.contains("1.0"));
    }
}
