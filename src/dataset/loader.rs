//! Multi-format dataset loader.
//!
//! - NPY: memory mapped, rows decoded lazily (no full RAM copy).
//! - NPZ: zip of NPY entries, stream-decompressed (features + optional labels).
//! - CSV: streamed record by record through the `csv` reader.
//! - IDX (MNIST-style): memory mapped, sibling label file auto-detected.
//! - Parquet: behind the optional `parquet-support` feature.
//!
//! Entry point is [`load`], which dispatches on the file extension and
//! returns a fully described [`Dataset`].

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::{
    metadata::DatasetMetadata, Dataset, DatasetError, ElemType, FeatureSource, Result,
};

/// Options controlling the import.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Hard cap on imported rows (None = all rows).
    pub max_rows: Option<usize>,
    /// CSV label column name; if None it is auto-detected.
    pub label_column: Option<String>,
}

/// Load a dataset, dispatching on the file extension.
pub fn load(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "npy" => load_npy(path, opts),
        "npz" => load_npz(path, opts),
        "csv" => load_csv(path, opts),
        "xlsx" | "xlsm" | "xls" | "ods" => load_excel(path, opts),
        "idx" | "idx3-ubyte" | "idx1-ubyte" | "ubyte" => load_idx(path, opts),
        #[cfg(feature = "parquet-support")]
        "parquet" => load_parquet(path, opts),
        #[cfg(not(feature = "parquet-support"))]
        "parquet" => Err(DatasetError::Unsupported(
            "parquet support not compiled in (enable feature `parquet-support`)".into(),
        )),
        other => Err(DatasetError::Unsupported(format!(
            "unknown dataset extension '{}'",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// NPY
// ---------------------------------------------------------------------------

pub struct NpyHeader {
    pub elem: ElemType,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
    /// Byte offset of the first data element.
    pub data_offset: usize,
}

/// Parse an NPY header from the start of `bytes`.
pub fn parse_npy_header(bytes: &[u8]) -> Result<NpyHeader> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(DatasetError::Format("not an NPY file (bad magic)".into()));
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(DatasetError::Format("truncated NPY header".into()));
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        v => {
            return Err(DatasetError::Format(format!(
                "unsupported NPY version {}",
                v
            )))
        }
    };
    if bytes.len() < header_start + header_len {
        return Err(DatasetError::Format("truncated NPY header".into()));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|_| DatasetError::Format("NPY header is not UTF-8".into()))?;

    let descr = extract_py_str(header, "descr")
        .ok_or_else(|| DatasetError::Format("NPY header missing 'descr'".into()))?;
    let elem = parse_descr(&descr)?;
    let fortran_order = header
        .split("fortran_order")
        .nth(1)
        .map(|s| s.trim_start_matches(['\'', ':', ' ']).starts_with("True"))
        .unwrap_or(false);
    let shape = extract_shape(header)?;

    Ok(NpyHeader {
        elem,
        fortran_order,
        shape,
        data_offset: header_start + header_len,
    })
}

fn extract_py_str(header: &str, key: &str) -> Option<String> {
    let pos = header.find(&format!("'{}'", key))?;
    let rest = &header[pos + key.len() + 2..];
    let q1 = rest.find('\'')?;
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('\'')?;
    Some(rest[..q2].to_string())
}

fn extract_shape(header: &str) -> Result<Vec<usize>> {
    let pos = header
        .find("'shape'")
        .ok_or_else(|| DatasetError::Format("NPY header missing 'shape'".into()))?;
    let rest = &header[pos..];
    let open = rest
        .find('(')
        .ok_or_else(|| DatasetError::Format("malformed shape tuple".into()))?;
    let close = rest[open..]
        .find(')')
        .ok_or_else(|| DatasetError::Format("malformed shape tuple".into()))?;
    let inner = &rest[open + 1..open + close];
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        shape.push(
            part.parse::<usize>()
                .map_err(|_| DatasetError::Format(format!("bad shape value '{}'", part)))?,
        );
    }
    Ok(shape)
}

fn parse_descr(descr: &str) -> Result<ElemType> {
    // descr like "<f4", "|u1", "<i8"; big endian (">") is rejected.
    let bytes = descr.as_bytes();
    if bytes.is_empty() {
        return Err(DatasetError::Format("empty dtype descr".into()));
    }
    let (order, kind) = if bytes[0] == b'<' || bytes[0] == b'|' || bytes[0] == b'=' {
        (bytes[0], &descr[1..])
    } else if bytes[0] == b'>' {
        return Err(DatasetError::Unsupported(
            "big-endian NPY arrays are not supported".into(),
        ));
    } else {
        (b'<', descr)
    };
    let _ = order;
    let elem = match kind {
        "f4" => ElemType::F32,
        "f8" => ElemType::F64,
        "i1" => ElemType::I8,
        "u1" => ElemType::U8,
        "i2" => ElemType::I16,
        "u2" => ElemType::U16,
        "i4" => ElemType::I32,
        "u4" => ElemType::U32,
        "i8" => ElemType::I64,
        "u8" => ElemType::U64,
        other => {
            return Err(DatasetError::Unsupported(format!(
                "unsupported NPY dtype '{}'",
                other
            )))
        }
    };
    Ok(elem)
}

/// Auto-detect the label column from header names.
///
/// Strong names ("label", "target", "class", ...) take precedence over the
/// weak name "y", which often collides with a coordinate column.
pub fn detect_label_column(headers: &[String]) -> Option<usize> {
    let strong = headers.iter().position(|h| {
        matches!(
            h.to_ascii_lowercase().as_str(),
            "label" | "labels" | "target" | "class" | "species" | "category"
        )
    });
    strong.or_else(|| headers.iter().position(|h| h.eq_ignore_ascii_case("y")))
}

/// Flatten a shape into (rows, cols): trailing dims are folded into columns.
fn shape_to_2d(shape: &[usize]) -> Result<(usize, usize)> {
    match shape.len() {
        0 => Err(DatasetError::Format("scalar NPY array".into())),
        1 => Ok((shape[0], 1)),
        _ => Ok((shape[0], shape[1..].iter().product())),
    }
}

fn load_npy(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    let file = std::fs::File::open(path)?;
    // SAFETY: the file is opened read-only and the mapping is kept alive by
    // the returned Dataset for as long as rows are decoded from it.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let header = parse_npy_header(&map)?;
    if header.fortran_order {
        return Err(DatasetError::Unsupported(
            "fortran-order NPY arrays are not supported".into(),
        ));
    }
    let (mut n_rows, n_cols) = shape_to_2d(&header.shape)?;
    if let Some(cap) = opts.max_rows {
        n_rows = n_rows.min(cap);
    }
    let expected = header.data_offset + n_rows * n_cols * header.elem.size();
    if map.len() < expected {
        return Err(DatasetError::Format("NPY payload shorter than shape".into()));
    }

    let name = file_stem(path);
    let mut metadata = DatasetMetadata::new(&name, "npy", n_rows, n_cols);
    metadata.source_path = path.to_string_lossy().to_string();
    metadata.memory_mapped = true;

    // No labels inside a bare NPY: look for a sibling "<stem>_labels.npy".
    let (labels, label_names, label_column) = sibling_npy_labels(path, n_rows)?;
    metadata.label_column = label_column;
    metadata.set_label_stats(&labels, &label_names);

    Ok(Dataset {
        metadata,
        source: FeatureSource::Mmap {
            map,
            data_offset: header.data_offset,
            elem: header.elem,
        },
        labels,
        label_names,
    })
}

/// Look for `<stem>_labels.npy` or `<stem>.labels.npy` next to a feature file.
fn sibling_npy_labels(path: &Path, n_rows: usize) -> Result<(Vec<u32>, Vec<String>, Option<String>)> {
    let stem = file_stem(path);
    let dir = path.parent().unwrap_or(Path::new("."));
    let candidates = [
        dir.join(format!("{}_labels.npy", stem)),
        dir.join(format!("{}.labels.npy", stem)),
    ];
    for cand in &candidates {
        if cand.is_file() {
            let raw = std::fs::read(cand)?;
            let values = decode_npy_as_f32(&raw)?;
            if values.len() < n_rows {
                return Err(DatasetError::Format(format!(
                    "label file {} has {} entries, expected at least {}",
                    cand.display(),
                    values.len(),
                    n_rows
                )));
            }
            let (labels, names) = labels_from_numeric(&values[..n_rows]);
            return Ok((labels, names, Some(file_stem(cand))));
        }
    }
    Ok(unlabeled(n_rows))
}

/// Decode a whole (small) NPY buffer to f32 values, flattened.
pub fn decode_npy_as_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let header = parse_npy_header(bytes)?;
    if header.fortran_order {
        return Err(DatasetError::Unsupported(
            "fortran-order NPY arrays are not supported".into(),
        ));
    }
    let count: usize = header.shape.iter().product::<usize>().max(
        if header.shape.is_empty() { 1 } else { 0 },
    );
    let es = header.elem.size();
    if bytes.len() < header.data_offset + count * es {
        return Err(DatasetError::Format("NPY payload shorter than shape".into()));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(header.elem.read_f32(bytes, header.data_offset + i * es));
    }
    Ok(out)
}

/// Map numeric label values to dense ids with stable, sorted vocabulary.
pub fn labels_from_numeric(values: &[f32]) -> (Vec<u32>, Vec<String>) {
    let mut uniq: Vec<i64> = values.iter().map(|v| *v as i64).collect();
    uniq.sort_unstable();
    uniq.dedup();
    let id_of: HashMap<i64, u32> = uniq
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i as u32))
        .collect();
    let labels = values.iter().map(|v| id_of[&(*v as i64)]).collect();
    let names = uniq.iter().map(|v| v.to_string()).collect();
    (labels, names)
}

fn unlabeled(n_rows: usize) -> (Vec<u32>, Vec<String>, Option<String>) {
    (vec![0; n_rows], vec!["unlabeled".to_string()], None)
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_string()
}

// ---------------------------------------------------------------------------
// NPZ
// ---------------------------------------------------------------------------

fn load_npz(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .map_err(|e| DatasetError::Format(format!("bad NPZ archive: {}", e)))?;

    // Find the feature entry (first array with rank >= 2, or the largest)
    // and an optional label entry (rank-1 array named y/labels/target).
    let mut feature_entry: Option<String> = None;
    let mut label_entry: Option<String> = None;
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
        .collect();
    for name in &names {
        let stem = name.trim_end_matches(".npy").to_ascii_lowercase();
        if matches!(stem.as_str(), "y" | "labels" | "label" | "target" | "y_train" | "y_test") {
            label_entry.get_or_insert_with(|| name.clone());
        } else {
            feature_entry.get_or_insert_with(|| name.clone());
        }
    }
    let feature_entry = feature_entry
        .ok_or_else(|| DatasetError::Format("NPZ contains no feature array".into()))?;

    let read_entry = |archive: &mut zip::ZipArchive<std::io::BufReader<std::fs::File>>,
                      name: &str|
     -> Result<Vec<u8>> {
        let mut entry = archive
            .by_name(name)
            .map_err(|e| DatasetError::Format(format!("NPZ entry '{}': {}", name, e)))?;
        // Stream-decompress: the compressed archive is never fully buffered.
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        Ok(buf)
    };

    let feat_raw = read_entry(&mut archive, &feature_entry)?;
    let header = parse_npy_header(&feat_raw)?;
    let (mut n_rows, n_cols) = shape_to_2d(&header.shape)?;
    if let Some(cap) = opts.max_rows {
        n_rows = n_rows.min(cap);
    }
    let es = header.elem.size();
    let mut data = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows * n_cols {
        data.push(header.elem.read_f32(&feat_raw, header.data_offset + i * es));
    }
    drop(feat_raw);

    let (labels, label_names, label_column) = if let Some(label_name) = &label_entry {
        let raw = read_entry(&mut archive, label_name)?;
        let values = decode_npy_as_f32(&raw)?;
        if values.len() < n_rows {
            return Err(DatasetError::Format("NPZ label array shorter than rows".into()));
        }
        let (l, n) = labels_from_numeric(&values[..n_rows]);
        (l, n, Some(label_name.trim_end_matches(".npy").to_string()))
    } else {
        unlabeled(n_rows)
    };

    let mut metadata = DatasetMetadata::new(&file_stem(path), "npz", n_rows, n_cols);
    metadata.source_path = path.to_string_lossy().to_string();
    metadata.label_column = label_column;
    metadata.set_label_stats(&labels, &label_names);

    Ok(Dataset {
        metadata,
        source: FeatureSource::InMemory(data),
        labels,
        label_names,
    })
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

fn load_csv(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_path(path)
        .map_err(|e| DatasetError::Format(format!("CSV open failed: {}", e)))?;

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| DatasetError::Format(e.to_string()))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();

    let label_idx = match &opts.label_column {
        Some(name) => Some(
            headers
                .iter()
                .position(|h| h.eq_ignore_ascii_case(name))
                .ok_or_else(|| {
                    DatasetError::Format(format!("label column '{}' not found", name))
                })?,
        ),
        None => detect_label_column(&headers),
    };

    let feature_idx: Vec<usize> = (0..headers.len())
        .filter(|i| Some(*i) != label_idx)
        .collect();
    let n_cols = feature_idx.len();
    if n_cols == 0 {
        return Err(DatasetError::Format("CSV has no feature columns".into()));
    }

    // Streamed read: only the in-progress record lives outside `data`.
    let mut data: Vec<f32> = Vec::new();
    let mut label_strs: Vec<String> = Vec::new();
    for (row_no, record) in reader.records().enumerate() {
        if let Some(cap) = opts.max_rows {
            if row_no >= cap {
                break;
            }
        }
        let record = record.map_err(|e| DatasetError::Format(e.to_string()))?;
        for &ci in &feature_idx {
            let cell = record.get(ci).unwrap_or("").trim();
            let v = cell.parse::<f32>().map_err(|_| {
                DatasetError::Format(format!(
                    "row {}: non numeric value '{}' in column '{}'",
                    row_no + 1,
                    cell,
                    headers[ci]
                ))
            })?;
            data.push(v);
        }
        if let Some(li) = label_idx {
            label_strs.push(record.get(li).unwrap_or("").trim().to_string());
        }
    }

    finish_table_dataset(
        path,
        "csv",
        &headers,
        &feature_idx,
        label_idx,
        data,
        label_strs,
    )
}

/// Read a tabular Excel/ODS workbook (first sheet) into a dataset.
///
/// Shares the header -> feature/label resolution and metadata construction
/// with [`load_csv`] via [`finish_table_dataset`]; only the cell reading
/// differs.
fn load_excel(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    use crate::util::excel_cell_to_string as cell_to_string;
    use calamine::{open_workbook_auto, Reader};

    let mut workbook = open_workbook_auto(path)
        .map_err(|e| DatasetError::Format(format!("Excel open failed: {}", e)))?;
    let sheet_name = workbook
        .sheet_names()
        .first()
        .cloned()
        .ok_or_else(|| DatasetError::Format("workbook has no sheets".into()))?;
    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| DatasetError::Format(format!("sheet '{}': {}", sheet_name, e)))?;

    let mut rows_iter = range.rows();
    let headers: Vec<String> = rows_iter
        .next()
        .ok_or_else(|| DatasetError::Format("sheet is empty".into()))?
        .iter()
        .map(|c| cell_to_string(c).trim().to_string())
        .collect();

    let label_idx = match &opts.label_column {
        Some(name) => Some(
            headers
                .iter()
                .position(|h| h.eq_ignore_ascii_case(name))
                .ok_or_else(|| {
                    DatasetError::Format(format!("label column '{}' not found", name))
                })?,
        ),
        None => detect_label_column(&headers),
    };
    let feature_idx: Vec<usize> = (0..headers.len())
        .filter(|i| Some(*i) != label_idx)
        .collect();
    if feature_idx.is_empty() {
        return Err(DatasetError::Format("sheet has no feature columns".into()));
    }

    let mut data: Vec<f32> = Vec::new();
    let mut label_strs: Vec<String> = Vec::new();
    let mut row_no = 0usize;
    for row in rows_iter {
        let cells: Vec<String> = row.iter().map(cell_to_string).collect();
        // Skip fully blank rows (trailing rows are common in spreadsheets).
        if cells.iter().all(|c| c.is_empty()) {
            continue;
        }
        if let Some(cap) = opts.max_rows {
            if row_no >= cap {
                break;
            }
        }
        for &ci in &feature_idx {
            let cell = cells.get(ci).map(String::as_str).unwrap_or("").trim();
            let v = cell.parse::<f32>().map_err(|_| {
                DatasetError::Format(format!(
                    "row {}: non numeric value '{}' in column '{}'",
                    row_no + 1,
                    cell,
                    headers[ci]
                ))
            })?;
            data.push(v);
        }
        if let Some(li) = label_idx {
            label_strs.push(
                cells
                    .get(li)
                    .map(String::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string(),
            );
        }
        row_no += 1;
    }

    finish_table_dataset(
        path,
        "excel",
        &headers,
        &feature_idx,
        label_idx,
        data,
        label_strs,
    )
}

/// Resolve labels/metadata for a tabular import and pack it into a [`Dataset`].
///
/// Shared by [`load_csv`] and [`load_excel`]; `data` is row-major features
/// (feature columns only) and `label_strs` holds one raw label per row when a
/// label column was detected.
fn finish_table_dataset(
    path: &Path,
    format: &str,
    headers: &[String],
    feature_idx: &[usize],
    label_idx: Option<usize>,
    data: Vec<f32>,
    label_strs: Vec<String>,
) -> Result<Dataset> {
    let n_cols = feature_idx.len();
    let n_rows = if n_cols == 0 { 0 } else { data.len() / n_cols };

    let (labels, label_names, label_column) = if let Some(li) = label_idx {
        let mut names: Vec<String> = label_strs.clone();
        names.sort();
        names.dedup();
        let id_of: HashMap<&str, u32> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as u32))
            .collect();
        let labels = label_strs.iter().map(|s| id_of[s.as_str()]).collect();
        (labels, names, Some(headers[li].clone()))
    } else {
        unlabeled(n_rows)
    };

    let mut metadata = DatasetMetadata::new(&file_stem(path), format, n_rows, n_cols);
    metadata.source_path = path.to_string_lossy().to_string();
    metadata.column_names = feature_idx.iter().map(|&i| headers[i].clone()).collect();
    metadata.label_column = label_column;
    metadata.set_label_stats(&labels, &label_names);

    Ok(Dataset {
        metadata,
        source: FeatureSource::InMemory(data),
        labels,
        label_names,
    })
}

// ---------------------------------------------------------------------------
// IDX (MNIST-style benchmark files)
// ---------------------------------------------------------------------------

fn load_idx(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    let file = std::fs::File::open(path)?;
    // SAFETY: read-only mapping owned by the returned Dataset.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let (elem, dims, data_offset) = parse_idx_header(&map)?;
    let (mut n_rows, n_cols) = shape_to_2d(&dims)?;
    if let Some(cap) = opts.max_rows {
        n_rows = n_rows.min(cap);
    }

    let mut metadata = DatasetMetadata::new(&file_stem(path), "idx", n_rows, n_cols);
    metadata.source_path = path.to_string_lossy().to_string();
    metadata.memory_mapped = true;

    let (labels, label_names, label_column) = sibling_idx_labels(path, n_rows)?;
    metadata.label_column = label_column;
    metadata.set_label_stats(&labels, &label_names);

    Ok(Dataset {
        metadata,
        source: FeatureSource::Mmap {
            map,
            data_offset,
            elem,
        },
        labels,
        label_names,
    })
}

/// IDX header: 0x00 0x00 <dtype> <ndims> then ndims big-endian u32 dims.
fn parse_idx_header(bytes: &[u8]) -> Result<(ElemType, Vec<usize>, usize)> {
    if bytes.len() < 4 || bytes[0] != 0 || bytes[1] != 0 {
        return Err(DatasetError::Format("not an IDX file (bad magic)".into()));
    }
    let elem = match bytes[2] {
        0x08 => ElemType::U8,
        0x09 => ElemType::I8,
        0x0B => ElemType::I16,
        0x0C => ElemType::I32,
        0x0D => ElemType::F32,
        0x0E => ElemType::F64,
        t => {
            return Err(DatasetError::Unsupported(format!(
                "IDX dtype 0x{:02x} not supported",
                t
            )))
        }
    };
    // NOTE: IDX multi-byte payloads are big-endian; only u8/i8 are safe to
    // decode through the little-endian mmap path used by FeatureSource.
    if elem.size() > 1 {
        return Err(DatasetError::Unsupported(
            "only u8/i8 IDX payloads are supported (MNIST-style)".into(),
        ));
    }
    let ndims = bytes[3] as usize;
    let header_len = 4 + 4 * ndims;
    if bytes.len() < header_len {
        return Err(DatasetError::Format("truncated IDX header".into()));
    }
    let mut dims = Vec::with_capacity(ndims);
    for d in 0..ndims {
        let off = 4 + 4 * d;
        dims.push(u32::from_be_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]) as usize);
    }
    Ok((elem, dims, header_len))
}

/// MNIST convention: "train-images-idx3-ubyte" pairs "train-labels-idx1-ubyte".
fn sibling_idx_labels(path: &Path, n_rows: usize) -> Result<(Vec<u32>, Vec<String>, Option<String>)> {
    let fname = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if !fname.contains("images") {
        return Ok(unlabeled(n_rows));
    }
    let label_name = fname.replace("images", "labels").replace("idx3", "idx1");
    let cand: PathBuf = path.with_file_name(label_name);
    if !cand.is_file() {
        return Ok(unlabeled(n_rows));
    }
    let raw = std::fs::read(&cand)?;
    let (elem, dims, off) = parse_idx_header(&raw)?;
    let count: usize = dims.iter().product();
    if count < n_rows {
        return Err(DatasetError::Format("IDX label file shorter than rows".into()));
    }
    let values: Vec<f32> = (0..n_rows)
        .map(|i| elem.read_f32(&raw, off + i * elem.size()))
        .collect();
    let (labels, names) = labels_from_numeric(&values);
    Ok((labels, names, Some(file_stem(&cand))))
}

// ---------------------------------------------------------------------------
// Parquet (optional)
// ---------------------------------------------------------------------------

#[cfg(feature = "parquet-support")]
fn load_parquet(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::Field;

    let file = std::fs::File::open(path)?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| DatasetError::Format(format!("parquet open failed: {}", e)))?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let headers: Vec<String> = (0..schema.num_columns())
        .map(|i| schema.column(i).name().to_string())
        .collect();

    let label_idx = match &opts.label_column {
        Some(name) => headers.iter().position(|h| h.eq_ignore_ascii_case(name)),
        None => detect_label_column(&headers),
    };

    let mut data: Vec<f32> = Vec::new();
    let mut label_strs: Vec<String> = Vec::new();
    let mut n_cols = 0usize;
    let iter = reader
        .get_row_iter(None)
        .map_err(|e| DatasetError::Format(e.to_string()))?;
    for (row_no, row) in iter.enumerate() {
        if let Some(cap) = opts.max_rows {
            if row_no >= cap {
                break;
            }
        }
        let row = row.map_err(|e| DatasetError::Format(e.to_string()))?;
        let mut row_vals = Vec::new();
        for (ci, (_, field)) in row.get_column_iter().enumerate() {
            let as_f32 = match field {
                Field::Bool(b) => Some(*b as i32 as f32),
                Field::Byte(v) => Some(*v as f32),
                Field::Short(v) => Some(*v as f32),
                Field::Int(v) => Some(*v as f32),
                Field::Long(v) => Some(*v as f32),
                Field::UByte(v) => Some(*v as f32),
                Field::UShort(v) => Some(*v as f32),
                Field::UInt(v) => Some(*v as f32),
                Field::ULong(v) => Some(*v as f32),
                Field::Float(v) => Some(*v),
                Field::Double(v) => Some(*v as f32),
                _ => None,
            };
            if Some(ci) == label_idx {
                label_strs.push(match field {
                    Field::Str(s) => s.clone(),
                    other => other.to_string(),
                });
            } else {
                let v = as_f32.ok_or_else(|| {
                    DatasetError::Format(format!(
                        "row {}: non numeric parquet column '{}'",
                        row_no + 1,
                        headers[ci]
                    ))
                })?;
                row_vals.push(v);
            }
        }
        n_cols = row_vals.len();
        data.extend_from_slice(&row_vals);
    }
    if n_cols == 0 {
        return Err(DatasetError::Format("parquet has no feature columns".into()));
    }
    let n_rows = data.len() / n_cols;

    let (labels, label_names, label_column) = if let Some(li) = label_idx {
        let mut names = label_strs.clone();
        names.sort();
        names.dedup();
        let id_of: HashMap<&str, u32> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as u32))
            .collect();
        (
            label_strs.iter().map(|s| id_of[s.as_str()]).collect(),
            names,
            Some(headers[li].clone()),
        )
    } else {
        unlabeled(n_rows)
    };

    let mut metadata = DatasetMetadata::new(&file_stem(path), "parquet", n_rows, n_cols);
    metadata.source_path = path.to_string_lossy().to_string();
    metadata.column_names = headers
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != label_idx)
        .map(|(_, h)| h.clone())
        .collect();
    metadata.label_column = label_column;
    metadata.set_label_stats(&labels, &label_names);

    Ok(Dataset {
        metadata,
        source: FeatureSource::InMemory(data),
        labels,
        label_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal NPY v1 buffer with the given descr/shape header fields.
    fn npy_with_header(dict: &str, payload: &[u8]) -> Vec<u8> {
        let mut header = dict.to_string();
        header.push('\n');
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn descr_parsing_covers_all_supported_dtypes() {
        for (descr, elem) in [
            ("<f4", ElemType::F32),
            ("<f8", ElemType::F64),
            ("|i1", ElemType::I8),
            ("|u1", ElemType::U8),
            ("<i2", ElemType::I16),
            ("<u2", ElemType::U16),
            ("<i4", ElemType::I32),
            ("<u4", ElemType::U32),
            ("<i8", ElemType::I64),
            ("<u8", ElemType::U64),
            ("=f4", ElemType::F32),
            ("f4", ElemType::F32), // no byte-order prefix
        ] {
            assert_eq!(parse_descr(descr).unwrap(), elem, "descr {}", descr);
        }
    }

    #[test]
    fn descr_rejects_big_endian_strings_and_unknowns() {
        assert!(parse_descr(">f4").is_err());
        assert!(parse_descr("<U16").is_err());
        assert!(parse_descr("").is_err());
    }

    #[test]
    fn shape_extraction_handles_tuples() {
        let h = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }";
        assert_eq!(extract_shape(h).unwrap(), vec![3, 4]);
        let h1 = "{'descr': '<f4', 'fortran_order': False, 'shape': (7,), }";
        assert_eq!(extract_shape(h1).unwrap(), vec![7]);
        let h3 = "{'shape': (2, 3, 4)}";
        assert_eq!(extract_shape(h3).unwrap(), vec![2, 3, 4]);
        assert!(extract_shape("{'shape': (a,)}").is_err());
        assert!(extract_shape("{'no_shape': 1}").is_err());
    }

    #[test]
    fn shape_to_2d_folds_trailing_dims() {
        assert_eq!(shape_to_2d(&[10]).unwrap(), (10, 1));
        assert_eq!(shape_to_2d(&[10, 4]).unwrap(), (10, 4));
        assert_eq!(shape_to_2d(&[10, 2, 3]).unwrap(), (10, 6));
        assert!(shape_to_2d(&[]).is_err());
    }

    #[test]
    fn npy_header_version_2_uses_u32_length() {
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1), }\n";
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(2);
        out.push(0);
        out.extend_from_slice(&(dict.len() as u32).to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.extend_from_slice(&1.0f32.to_le_bytes());
        let h = parse_npy_header(&out).unwrap();
        assert_eq!(h.elem, ElemType::F32);
        assert_eq!(h.shape, vec![1, 1]);
        assert_eq!(h.data_offset, 12 + dict.len());
    }

    #[test]
    fn npy_header_rejects_bad_magic_and_version() {
        assert!(parse_npy_header(b"NOPE").is_err());
        let mut bad_ver = b"\x93NUMPY".to_vec();
        bad_ver.extend_from_slice(&[9, 0, 0, 0]);
        assert!(parse_npy_header(&bad_ver).is_err());
    }

    #[test]
    fn npy_header_detects_fortran_order() {
        let buf = npy_with_header(
            "{'descr': '<f4', 'fortran_order': True, 'shape': (1, 1), }",
            &1.0f32.to_le_bytes(),
        );
        assert!(parse_npy_header(&buf).unwrap().fortran_order);
        let buf2 = npy_with_header(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1), }",
            &1.0f32.to_le_bytes(),
        );
        assert!(!parse_npy_header(&buf2).unwrap().fortran_order);
    }

    #[test]
    fn decode_rejects_truncated_payload() {
        let buf = npy_with_header(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (4, 4), }",
            &[0u8; 4], // 4 bytes instead of 64
        );
        assert!(decode_npy_as_f32(&buf).is_err());
    }

    #[test]
    fn label_detection_prefers_strong_names_over_y() {
        let h = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // "y" alone is accepted...
        assert_eq!(detect_label_column(&h(&["a", "y", "b"])), Some(1));
        // ...but never beats an explicit "label"/"target"/"class" column.
        assert_eq!(detect_label_column(&h(&["x", "y", "label"])), Some(2));
        assert_eq!(detect_label_column(&h(&["y", "TARGET"])), Some(1));
        assert_eq!(detect_label_column(&h(&["Species", "y"])), Some(0));
        assert_eq!(detect_label_column(&h(&["a", "b"])), None);
    }

    #[test]
    fn labels_from_numeric_builds_sorted_vocabulary() {
        let (ids, names) = labels_from_numeric(&[5.0, -1.0, 5.0, 2.0]);
        assert_eq!(names, vec!["-1", "2", "5"]);
        assert_eq!(ids, vec![2, 0, 2, 1]);
    }

    #[test]
    fn idx_header_parsing_and_rejections() {
        // Valid u8, 2 dims.
        let mut buf = vec![0u8, 0, 0x08, 2];
        buf.extend_from_slice(&3u32.to_be_bytes());
        buf.extend_from_slice(&2u32.to_be_bytes());
        buf.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let (elem, dims, off) = parse_idx_header(&buf).unwrap();
        assert_eq!(elem, ElemType::U8);
        assert_eq!(dims, vec![3, 2]);
        assert_eq!(off, 12);

        // Bad magic.
        assert!(parse_idx_header(&[1, 2, 3, 4]).is_err());
        // Unsupported dtype code.
        assert!(parse_idx_header(&[0, 0, 0xFF, 1, 0, 0, 0, 1]).is_err());
        // Multi-byte dtype rejected (big-endian payload).
        assert!(parse_idx_header(&[0, 0, 0x0D, 1, 0, 0, 0, 1]).is_err());
        // Truncated dims.
        assert!(parse_idx_header(&[0, 0, 0x08, 2, 0, 0]).is_err());
    }
}
