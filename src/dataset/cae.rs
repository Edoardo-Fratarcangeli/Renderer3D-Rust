//! Import of real CFD / FEM result files.
//!
//! Supported formats:
//! - **VTK legacy** (`.vtk`) — ASCII and binary (big-endian per spec), for
//!   unstructured/structured grids, polydata, rectilinear grids and image
//!   data. Exported by ParaView, OpenFOAM (`foamToVTK`), CalculiX converters
//!   and most commercial solvers.
//! - **VTK XML** (`.vtu` / `.vtp`) — ascii, inline base64 and appended
//!   (raw or base64) data arrays, optionally zlib-compressed. Only the first
//!   `<Piece>` of the file is read.
//! - **CalculiX** (`.frd`) — ASCII result files written by `ccx` (nodal
//!   coordinates + nodal result blocks such as DISP, STRESS, NDTEMP).
//!
//! Every parser produces a [`CaeMesh`] (points + named per-point fields);
//! [`mesh_to_dataset`] then maps it onto the standard [`Dataset`] pipeline:
//! the feature columns are `x, y, z` plus every field component (vector
//! fields also get a `_mag` magnitude column), and the labels quantize one
//! chosen field into [`FIELD_BANDS`] contour bands, so the label legend reads
//! like a FEM color bar and doubles as a per-isovalue visibility filter.
//!
//! Large files: all three formats declare their point count up front, so
//! points are kept with a uniform index stride computed against the row
//! budget. Unlike a "first N rows" cut this keeps the whole domain covered,
//! which is what makes a subsampled 100M-point solution still readable.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use super::loader::LoadOptions;
use super::{metadata::DatasetMetadata, Dataset, DatasetError, ElemType, FeatureSource, Result};

/// Default point budget when the user did not cap rows. Solver outputs can
/// reach hundreds of millions of nodes; everything above this is strided
/// down (the renderer caps far lower anyway, this bounds RAM).
pub const DEFAULT_MAX_POINTS: usize = 2_000_000;

/// Number of contour bands the colored field is quantized into.
pub const FIELD_BANDS: usize = 10;

/// Hard cap on a single decoded/decompressed data array (bomb guard: the
/// sizes come from attacker-controlled headers).
const MAX_ARRAY_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GiB

/// Extensions routed to this module by the dataset loader.
pub fn is_cae_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("vtk" | "vtu" | "vtp" | "frd")
    )
}

// ---------------------------------------------------------------------------
// Common mesh model + Dataset mapping
// ---------------------------------------------------------------------------

/// One per-point result field (already subsampled to the kept points).
pub struct CaeField {
    pub name: String,
    pub ncomp: usize,
    /// Component names (e.g. `D1 D2 D3` from FRD). Empty = numeric suffixes.
    pub comp_names: Vec<String>,
    /// Kept-point-major values, `ncomp` per point.
    pub data: Vec<f32>,
}

/// Points + fields decoded from a result file, after stride subsampling.
pub struct CaeMesh {
    pub points: Vec<[f32; 3]>,
    pub fields: Vec<CaeField>,
}

/// Uniform keep-stride so at most `budget` of `n` items survive.
fn stride_for(n: usize, budget: usize) -> usize {
    n.div_ceil(budget.max(1)).max(1)
}

/// How many items survive a keep-stride of `stride`.
fn kept_count(n: usize, stride: usize) -> usize {
    n.div_ceil(stride.max(1))
}

/// `a * b` with overflow mapped to a format error (counts come from headers).
fn checked_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| DatasetError::Format("size overflow in file header".into()))
}

/// Compact human formatting for physical band bounds (adaptive precision).
fn fmt_val(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".to_string()
    } else if (1e-3..1e5).contains(&a) {
        let s = format!("{:.3}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        format!("{:.3e}", v)
    }
}

/// Quantize a per-point scalar into `bands` contour bands.
///
/// Returns per-point band ids plus the legend names (`"lo–hi unit"`), ordered
/// low → high so the label legend reads like a FEM color bar. Non-finite
/// values land in band 0. Shared with the synthetic CAE generators in
/// [`super::builtin`].
pub fn band_labels(values: &[f32], bands: usize, unit: &str) -> (Vec<u32>, Vec<String>) {
    let bands = bands.max(1);
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in values {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if !min.is_finite() {
        (min, max) = (0.0, 0.0);
    }
    let span = (max - min).max(f32::MIN_POSITIVE);

    let labels = values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                return 0;
            }
            let t = ((v - min) / span).clamp(0.0, 1.0);
            ((t * bands as f32) as usize).min(bands - 1) as u32
        })
        .collect();

    let suffix = if unit.is_empty() {
        String::new()
    } else {
        format!(" {}", unit)
    };
    let names = (0..bands)
        .map(|i| {
            let lo = min + span * (i as f32 / bands as f32);
            let hi = min + span * ((i + 1) as f32 / bands as f32);
            format!("{}–{}{}", fmt_val(lo), fmt_val(hi), suffix)
        })
        .collect();
    (labels, names)
}

/// Map a decoded mesh onto a [`Dataset`]: columns `x, y, z` + field
/// components (+ `_mag` for vectors), labels = contour bands of the field
/// selected by `field_sel` (a field or column name, case-insensitive),
/// falling back to `default_field` and then to the first field.
pub fn mesh_to_dataset(
    name: &str,
    format: &str,
    path: &Path,
    mesh: CaeMesh,
    field_sel: Option<&str>,
    default_field: Option<&str>,
) -> Result<Dataset> {
    let n = mesh.points.len();
    if n == 0 {
        return Err(DatasetError::Format("file contains no points".into()));
    }

    // Column layout: x, y, z, then per-field components (+ magnitude).
    let mut col_names: Vec<String> = vec!["x".into(), "y".into(), "z".into()];
    // Per field: the column a plain field-name selection should color by
    // (the scalar column, or the magnitude column for multi-component).
    let mut field_main_col: Vec<(String, usize)> = Vec::new();
    for f in &mesh.fields {
        if f.data.len() != checked_mul(n, f.ncomp.max(1))? {
            return Err(DatasetError::Format(format!(
                "field '{}' has {} values, expected {}x{}",
                f.name,
                f.data.len(),
                n,
                f.ncomp
            )));
        }
        if f.ncomp == 1 {
            field_main_col.push((f.name.clone(), col_names.len()));
            col_names.push(f.name.clone());
        } else {
            for c in 0..f.ncomp {
                let suffix = f
                    .comp_names
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| c.to_string());
                col_names.push(format!("{}_{}", f.name, suffix));
            }
            field_main_col.push((f.name.clone(), col_names.len()));
            col_names.push(format!("{}_mag", f.name));
        }
    }
    let n_cols = col_names.len();

    // Resolve which column the contour bands come from.
    let find = |sel: &str| -> Option<usize> {
        field_main_col
            .iter()
            .find(|(fname, _)| fname.eq_ignore_ascii_case(sel))
            .map(|(_, col)| *col)
            .or_else(|| {
                col_names
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case(sel))
            })
    };
    let band_col = match field_sel {
        Some(sel) => Some(find(sel).ok_or_else(|| {
            let avail: Vec<&str> = field_main_col.iter().map(|(f, _)| f.as_str()).collect();
            DatasetError::Format(format!(
                "field '{}' not found; available fields: {}",
                sel,
                if avail.is_empty() {
                    "(none)".to_string()
                } else {
                    avail.join(", ")
                }
            ))
        })?),
        None => default_field
            .and_then(find)
            .or_else(|| field_main_col.first().map(|(_, col)| *col)),
    };

    // Assemble the row-major feature matrix.
    let mut data = Vec::with_capacity(checked_mul(n, n_cols)?);
    for i in 0..n {
        data.extend_from_slice(&mesh.points[i]);
        for f in &mesh.fields {
            if f.ncomp == 1 {
                data.push(f.data[i]);
            } else {
                let mut sq = 0.0f32;
                for c in 0..f.ncomp {
                    let v = f.data[i * f.ncomp + c];
                    data.push(v);
                    sq += v * v;
                }
                data.push(sq.sqrt());
            }
        }
    }

    let (labels, label_names, label_column) = match band_col {
        Some(col) => {
            let vals: Vec<f32> = (0..n).map(|i| data[i * n_cols + col]).collect();
            let (labels, names) = band_labels(&vals, FIELD_BANDS, "");
            (labels, names, Some(col_names[col].clone()))
        }
        None => (vec![0; n], vec!["points".to_string()], None),
    };

    let mut metadata = DatasetMetadata::new(name, format, n, n_cols);
    metadata.source_path = path.to_string_lossy().to_string();
    metadata.column_names = col_names;
    metadata.label_column = label_column;
    metadata.set_label_stats(&labels, &label_names);

    Ok(Dataset {
        metadata,
        source: FeatureSource::InMemory(data),
        labels,
        label_names,
    })
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_string()
}

/// Decode one element at `off` honoring the file's byte order.
fn decode_elem(bytes: &[u8], off: usize, elem: ElemType, big_endian: bool) -> f32 {
    macro_rules! rd {
        ($t:ty, $n:expr) => {{
            let mut buf = [0u8; $n];
            buf.copy_from_slice(&bytes[off..off + $n]);
            if big_endian {
                <$t>::from_be_bytes(buf) as f32
            } else {
                <$t>::from_le_bytes(buf) as f32
            }
        }};
    }
    match elem {
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

// ---------------------------------------------------------------------------
// VTK legacy (.vtk)
// ---------------------------------------------------------------------------

/// Load a legacy VTK file (ASCII or big-endian binary).
pub fn load_vtk_legacy(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    let file = std::fs::File::open(path)?;
    // SAFETY: read-only mapping; all data is copied out before it drops.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let budget = opts.max_rows.unwrap_or(DEFAULT_MAX_POINTS);
    let (mesh, default_field) = parse_vtk_legacy(&map, budget)?;
    mesh_to_dataset(
        &file_stem(path),
        "vtk",
        path,
        mesh,
        opts.label_column.as_deref(),
        default_field.as_deref(),
    )
}

/// Byte cursor over a legacy VTK file: whitespace-token reader for keywords
/// and ASCII data, with direct byte access for binary payloads.
struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cur { b, p: 0 }
    }

    fn skip_ws(&mut self) {
        while self.p < self.b.len() && self.b[self.p].is_ascii_whitespace() {
            self.p += 1;
        }
    }

    fn token_opt(&mut self) -> Option<&'a str> {
        self.skip_ws();
        if self.p >= self.b.len() {
            return None;
        }
        let start = self.p;
        while self.p < self.b.len() && !self.b[self.p].is_ascii_whitespace() {
            self.p += 1;
        }
        std::str::from_utf8(&self.b[start..self.p]).ok()
    }

    fn token(&mut self) -> Result<&'a str> {
        self.token_opt()
            .ok_or_else(|| DatasetError::Format("unexpected end of VTK file".into()))
    }

    fn peek_token(&mut self) -> Option<&'a str> {
        let save = self.p;
        let t = self.token_opt();
        self.p = save;
        t
    }

    /// Rest of the current line (trimmed), cursor moved past the newline.
    fn line(&mut self) -> &'a str {
        let start = self.p;
        while self.p < self.b.len() && self.b[self.p] != b'\n' {
            self.p += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.p]).unwrap_or("");
        if self.p < self.b.len() {
            self.p += 1; // consume '\n'
        }
        s.trim()
    }

    /// Advance just past the next newline (binary payloads start there).
    fn skip_past_newline(&mut self) {
        while self.p < self.b.len() && self.b[self.p] != b'\n' {
            self.p += 1;
        }
        if self.p < self.b.len() {
            self.p += 1;
        }
    }

    fn usize_(&mut self) -> Result<usize> {
        let t = self.token()?;
        t.parse::<usize>()
            .map_err(|_| DatasetError::Format(format!("expected integer, got '{}'", t)))
    }

    fn f32_(&mut self) -> Result<f32> {
        let t = self.token()?;
        t.parse::<f32>()
            .map_err(|_| DatasetError::Format(format!("expected number, got '{}'", t)))
    }
}

/// Map a legacy VTK dtype token to an element type.
fn vtk_elem(tok: &str) -> Result<ElemType> {
    Ok(match tok.to_ascii_lowercase().as_str() {
        "float" => ElemType::F32,
        "double" => ElemType::F64,
        "int" | "vtktypeint32" => ElemType::I32,
        "unsigned_int" | "vtktypeuint32" => ElemType::U32,
        "long" | "vtktypeint64" => ElemType::I64,
        "unsigned_long" | "vtktypeuint64" => ElemType::U64,
        "short" => ElemType::I16,
        "unsigned_short" => ElemType::U16,
        "char" => ElemType::I8,
        "unsigned_char" => ElemType::U8,
        other => {
            return Err(DatasetError::Unsupported(format!(
                "VTK dtype '{}' not supported",
                other
            )))
        }
    })
}

/// Read `tuples * ncomp` values (ASCII tokens or big-endian binary), keeping
/// every `stride`-th tuple when `keep`, always consuming the full payload.
fn read_tuples(
    cur: &mut Cur,
    tuples: usize,
    ncomp: usize,
    elem: ElemType,
    ascii: bool,
    stride: usize,
    keep: bool,
) -> Result<Vec<f32>> {
    let stride = stride.max(1);
    let mut out = if keep {
        Vec::with_capacity(checked_mul(kept_count(tuples, stride), ncomp)?)
    } else {
        Vec::new()
    };
    if ascii {
        for t in 0..tuples {
            let k = keep && t % stride == 0;
            for _ in 0..ncomp {
                let tok = cur.token()?;
                if k {
                    out.push(tok.parse::<f32>().map_err(|_| {
                        DatasetError::Format(format!("non-numeric VTK value '{}'", tok))
                    })?);
                }
            }
        }
    } else {
        cur.skip_past_newline();
        let es = elem.size();
        let total = checked_mul(checked_mul(tuples, ncomp)?, es)?;
        if cur.p.checked_add(total).is_none_or(|end| end > cur.b.len()) {
            return Err(DatasetError::Format(
                "VTK binary payload shorter than declared".into(),
            ));
        }
        if keep {
            let mut t = 0;
            while t < tuples {
                let base = cur.p + t * ncomp * es;
                for c in 0..ncomp {
                    out.push(decode_elem(cur.b, base + c * es, elem, true));
                }
                t += stride;
            }
        }
        cur.p += total;
    }
    Ok(out)
}

/// Skip a legacy `METADATA` block (text lines until the first blank line).
fn skip_metadata(cur: &mut Cur) {
    loop {
        let save = cur.p;
        if save >= cur.b.len() {
            return;
        }
        let line = cur.line();
        if line.is_empty() {
            return;
        }
        // Defensive: a keyword line means the block already ended.
        let up = line.split_whitespace().next().unwrap_or("").to_uppercase();
        if matches!(
            up.as_str(),
            "POINT_DATA" | "CELL_DATA" | "SCALARS" | "VECTORS" | "FIELD" | "CELLS"
        ) {
            cur.p = save;
            return;
        }
    }
}

fn parse_vtk_legacy(bytes: &[u8], budget: usize) -> Result<(CaeMesh, Option<String>)> {
    let mut cur = Cur::new(bytes);
    let header = cur.line();
    if !header.contains("vtk DataFile") {
        return Err(DatasetError::Format(
            "not a legacy VTK file (missing '# vtk DataFile' header)".into(),
        ));
    }
    // Version decides the CELLS layout (5.1+ uses OFFSETS/CONNECTIVITY).
    let version: f32 = header
        .rsplit(' ')
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3.0);
    let _title = cur.line();
    let ascii = match cur.token()?.to_ascii_uppercase().as_str() {
        "ASCII" => true,
        "BINARY" => false,
        other => {
            return Err(DatasetError::Format(format!(
                "bad VTK format line '{}'",
                other
            )))
        }
    };

    let mut points: Option<Vec<[f32; 3]>> = None;
    let mut fields: Vec<CaeField> = Vec::new();
    let mut default_field: Option<String> = None;
    let mut n_points = 0usize;
    let mut stride = 1usize;
    let mut dims: Option<[usize; 3]> = None;
    let mut origin = [0.0f32; 3];
    let mut spacing = [1.0f32; 3];
    let mut rect: [Vec<f32>; 3] = Default::default();
    let mut cells_decl: Option<(usize, usize)> = None;
    // Active attribute section: (tuple count, is point data).
    let mut attr: Option<(usize, bool)> = None;

    // Set the point count (and stride) the first time it becomes known.
    let set_count = |n_points: &mut usize, stride: &mut usize, n: usize| -> Result<()> {
        if *n_points != 0 && *n_points != n {
            return Err(DatasetError::Format(format!(
                "inconsistent point counts in VTK file ({} vs {})",
                *n_points, n
            )));
        }
        *n_points = n;
        *stride = stride_for(n, budget);
        Ok(())
    };

    while let Some(kw) = cur.token_opt() {
        match kw.to_ascii_uppercase().as_str() {
            "DATASET" => {
                let _kind = cur.token()?;
            }
            "DIMENSIONS" => {
                let d = [cur.usize_()?, cur.usize_()?, cur.usize_()?];
                let n = checked_mul(checked_mul(d[0], d[1])?, d[2])?;
                set_count(&mut n_points, &mut stride, n)?;
                dims = Some(d);
            }
            "ORIGIN" => origin = [cur.f32_()?, cur.f32_()?, cur.f32_()?],
            "SPACING" | "ASPECT_RATIO" => spacing = [cur.f32_()?, cur.f32_()?, cur.f32_()?],
            "POINTS" => {
                let n = cur.usize_()?;
                let elem = vtk_elem(cur.token()?)?;
                set_count(&mut n_points, &mut stride, n)?;
                let vals = read_tuples(&mut cur, n, 3, elem, ascii, stride, true)?;
                points = Some(vals.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect());
            }
            "X_COORDINATES" | "Y_COORDINATES" | "Z_COORDINATES" => {
                let axis = match kw.as_bytes()[0].to_ascii_uppercase() {
                    b'X' => 0,
                    b'Y' => 1,
                    _ => 2,
                };
                let n = cur.usize_()?;
                let elem = vtk_elem(cur.token()?)?;
                rect[axis] = read_tuples(&mut cur, n, 1, elem, ascii, 1, true)?;
            }
            "CELLS" | "POLYGONS" | "LINES" | "VERTICES" | "TRIANGLE_STRIPS" => {
                let n = cur.usize_()?;
                let size = cur.usize_()?;
                cells_decl = Some((n, size));
                // VTK >= 5.1 splits cells into OFFSETS/CONNECTIVITY arrays;
                // before that the connectivity ints follow immediately.
                if version < 5.1 {
                    read_tuples(&mut cur, size, 1, ElemType::I32, ascii, 1, false)?;
                }
            }
            "OFFSETS" | "CONNECTIVITY" => {
                let elem = vtk_elem(cur.token()?)?;
                let (n_off, n_conn) = cells_decl.ok_or_else(|| {
                    DatasetError::Format("OFFSETS/CONNECTIVITY before CELLS".into())
                })?;
                let count = if kw.eq_ignore_ascii_case("OFFSETS") {
                    n_off
                } else {
                    n_conn
                };
                read_tuples(&mut cur, count, 1, elem, ascii, 1, false)?;
            }
            "CELL_TYPES" => {
                let n = cur.usize_()?;
                read_tuples(&mut cur, n, 1, ElemType::I32, ascii, 1, false)?;
            }
            "POINT_DATA" => {
                let n = cur.usize_()?;
                if n_points != 0 && n != n_points {
                    return Err(DatasetError::Format(format!(
                        "POINT_DATA count {} does not match {} points",
                        n, n_points
                    )));
                }
                set_count(&mut n_points, &mut stride, n)?;
                attr = Some((n, true));
            }
            "CELL_DATA" => {
                attr = Some((cur.usize_()?, false));
            }
            "SCALARS" => {
                let (count, keep) = attr
                    .ok_or_else(|| DatasetError::Format("SCALARS outside a data section".into()))?;
                let name = cur.token()?.to_string();
                let elem = vtk_elem(cur.token()?)?;
                // Optional component count, then the LOOKUP_TABLE reference.
                // The peek is deliberately strict (1..=9, digits only): in
                // binary files a missing count means the next bytes are raw
                // payload, which must not be misread as a token.
                let ncomp = match cur.peek_token() {
                    Some(t)
                        if t.len() == 1 && t.chars().all(|c| c.is_ascii_digit()) && t != "0" =>
                    {
                        cur.usize_()?
                    }
                    _ => 1,
                };
                if cur.peek_token().is_some_and(|t| t.eq_ignore_ascii_case("LOOKUP_TABLE")) {
                    let _ = cur.token()?;
                    let _table = cur.token()?;
                }
                let s = if keep { stride } else { 1 };
                let data = read_tuples(&mut cur, count, ncomp, elem, ascii, s, keep)?;
                if keep {
                    default_field.get_or_insert_with(|| name.clone());
                    fields.push(CaeField {
                        name,
                        ncomp,
                        comp_names: Vec::new(),
                        data,
                    });
                }
            }
            "VECTORS" | "NORMALS" => {
                let (count, keep) = attr
                    .ok_or_else(|| DatasetError::Format("VECTORS outside a data section".into()))?;
                let name = cur.token()?.to_string();
                let elem = vtk_elem(cur.token()?)?;
                let s = if keep { stride } else { 1 };
                let data = read_tuples(&mut cur, count, 3, elem, ascii, s, keep)?;
                if keep {
                    fields.push(CaeField {
                        name,
                        ncomp: 3,
                        comp_names: Vec::new(),
                        data,
                    });
                }
            }
            "TENSORS" => {
                let (count, keep) = attr
                    .ok_or_else(|| DatasetError::Format("TENSORS outside a data section".into()))?;
                let name = cur.token()?.to_string();
                let elem = vtk_elem(cur.token()?)?;
                let s = if keep { stride } else { 1 };
                let data = read_tuples(&mut cur, count, 9, elem, ascii, s, keep)?;
                if keep {
                    fields.push(CaeField {
                        name,
                        ncomp: 9,
                        comp_names: Vec::new(),
                        data,
                    });
                }
            }
            "FIELD" => {
                // FIELD blocks appear inside POINT_DATA/CELL_DATA sections and
                // standalone (global metadata like TimeValue); only arrays whose
                // tuple count matches the point count are kept.
                let _fname = cur.token()?;
                let m = cur.usize_()?;
                let (count, keep_section) = attr.unwrap_or((0, false));
                for _ in 0..m {
                    let name = cur.token()?.to_string();
                    let ncomp = cur.usize_()?;
                    let ntuples = cur.usize_()?;
                    let elem = vtk_elem(cur.token()?)?;
                    let keep_this = keep_section && ntuples == count && count == n_points;
                    let s = if keep_this { stride } else { 1 };
                    let data = read_tuples(&mut cur, ntuples, ncomp, elem, ascii, s, keep_this)?;
                    if keep_this {
                        default_field.get_or_insert_with(|| name.clone());
                        fields.push(CaeField {
                            name,
                            ncomp,
                            comp_names: Vec::new(),
                            data,
                        });
                    }
                }
            }
            "COLOR_SCALARS" => {
                let (count, _) = attr.ok_or_else(|| {
                    DatasetError::Format("COLOR_SCALARS outside a data section".into())
                })?;
                let _name = cur.token()?;
                let m = cur.usize_()?;
                let elem = if ascii { ElemType::F32 } else { ElemType::U8 };
                read_tuples(&mut cur, count, m, elem, ascii, 1, false)?;
            }
            "LOOKUP_TABLE" => {
                let _name = cur.token()?;
                let size = cur.usize_()?;
                let elem = if ascii { ElemType::F32 } else { ElemType::U8 };
                read_tuples(&mut cur, size, 4, elem, ascii, 1, false)?;
            }
            "METADATA" => skip_metadata(&mut cur),
            other => {
                return Err(DatasetError::Format(format!(
                    "unexpected VTK keyword '{}'",
                    other
                )))
            }
        }
    }

    // Grids without an explicit POINTS section synthesize their coordinates.
    let points = match points {
        Some(p) => p,
        None => {
            let d = dims.ok_or_else(|| {
                DatasetError::Format("VTK file has neither POINTS nor DIMENSIONS".into())
            })?;
            let use_rect = rect.iter().any(|r| !r.is_empty());
            if use_rect
                && (rect[0].len() != d[0] || rect[1].len() != d[1] || rect[2].len() != d[2])
            {
                return Err(DatasetError::Format(
                    "rectilinear coordinate arrays do not match DIMENSIONS".into(),
                ));
            }
            let mut pts = Vec::with_capacity(kept_count(n_points, stride));
            let mut idx = 0usize;
            for k in 0..d[2] {
                for j in 0..d[1] {
                    for i in 0..d[0] {
                        if idx.is_multiple_of(stride) {
                            pts.push(if use_rect {
                                [rect[0][i], rect[1][j], rect[2][k]]
                            } else {
                                [
                                    origin[0] + i as f32 * spacing[0],
                                    origin[1] + j as f32 * spacing[1],
                                    origin[2] + k as f32 * spacing[2],
                                ]
                            });
                        }
                        idx += 1;
                    }
                }
            }
            pts
        }
    };

    Ok((CaeMesh { points, fields }, default_field))
}

// ---------------------------------------------------------------------------
// VTK XML (.vtu / .vtp)
// ---------------------------------------------------------------------------

/// Load a VTK XML UnstructuredGrid / PolyData file (first `<Piece>` only).
pub fn load_vtk_xml(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    let file = std::fs::File::open(path)?;
    // SAFETY: read-only mapping; all data is copied out before it drops.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let budget = opts.max_rows.unwrap_or(DEFAULT_MAX_POINTS);
    let (mesh, default_field) = parse_vtk_xml(&map, budget)?;
    mesh_to_dataset(
        &file_stem(path),
        "vtu",
        path,
        mesh,
        opts.label_column.as_deref(),
        default_field.as_deref(),
    )
}

/// Attribute lookup inside the attribute region of an XML open tag.
fn xml_attr(attrs: &str, key: &str) -> Option<String> {
    let bytes = attrs.as_bytes();
    let mut from = 0;
    while let Some(rel) = attrs[from..].find(key) {
        let at = from + rel;
        let boundary_ok =
            at == 0 || bytes[at - 1].is_ascii_whitespace();
        let mut p = at + key.len();
        while p < attrs.len() && bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        if boundary_ok && p < attrs.len() && bytes[p] == b'=' {
            p += 1;
            while p < attrs.len() && bytes[p].is_ascii_whitespace() {
                p += 1;
            }
            if p < attrs.len() && (bytes[p] == b'"' || bytes[p] == b'\'') {
                let quote = bytes[p];
                let start = p + 1;
                if let Some(end) = attrs[start..].find(quote as char) {
                    return Some(attrs[start..start + end].to_string());
                }
            }
        }
        from = at + key.len();
    }
    None
}

/// Find the attribute region of the first `<tag ...>` open tag.
fn tag_attrs<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}", tag);
    let mut from = 0;
    while let Some(rel) = text[from..].find(&open) {
        let at = from + rel;
        let after = at + open.len();
        // Must be followed by whitespace or the tag close, not a longer name.
        let next = text.as_bytes().get(after).copied();
        if matches!(next, Some(b) if b.is_ascii_whitespace() || b == b'>' || b == b'/') {
            let end = text[after..].find('>')?;
            return Some(text[after..after + end].trim_end_matches('/'));
        }
        from = after;
    }
    None
}

/// Inner text of the first `<tag ...> ... </tag>` section.
fn xml_section<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{}", tag);
    let mut from = 0;
    while let Some(rel) = text[from..].find(&open) {
        let at = from + rel;
        let after = at + open.len();
        let next = text.as_bytes().get(after).copied();
        if matches!(next, Some(b) if b.is_ascii_whitespace() || b == b'>') {
            let body_start = at + text[at..].find('>')? + 1;
            let close = format!("</{}>", tag);
            let end = text[body_start..].find(&close)?;
            return Some(&text[body_start..body_start + end]);
        }
        from = after;
    }
    None
}

/// All `<DataArray ...>content</DataArray>` items in a section, as
/// (attribute region, content) pairs. Self-closing arrays have "" content.
fn data_arrays(sec: &str) -> Result<Vec<(&str, &str)>> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel) = sec[from..].find("<DataArray") {
        let at = from + rel;
        let tag_end = sec[at..]
            .find('>')
            .ok_or_else(|| DatasetError::Format("unterminated <DataArray> tag".into()))?;
        let attrs_region = &sec[at + "<DataArray".len()..at + tag_end];
        if attrs_region.trim_end().ends_with('/') {
            out.push((attrs_region.trim_end().trim_end_matches('/'), ""));
            from = at + tag_end + 1;
        } else {
            let body_start = at + tag_end + 1;
            let end = sec[body_start..]
                .find("</DataArray>")
                .ok_or_else(|| DatasetError::Format("missing </DataArray>".into()))?;
            out.push((attrs_region, &sec[body_start..body_start + end]));
            from = body_start + end + "</DataArray>".len();
        }
    }
    Ok(out)
}

/// Map a VTK XML `type` attribute to an element type.
fn xml_elem(t: &str) -> Result<ElemType> {
    Ok(match t {
        "Float32" => ElemType::F32,
        "Float64" => ElemType::F64,
        "Int8" => ElemType::I8,
        "UInt8" => ElemType::U8,
        "Int16" => ElemType::I16,
        "UInt16" => ElemType::U16,
        "Int32" => ElemType::I32,
        "UInt32" => ElemType::U32,
        "Int64" => ElemType::I64,
        "UInt64" => ElemType::U64,
        other => {
            return Err(DatasetError::Unsupported(format!(
                "VTK XML dtype '{}' not supported",
                other
            )))
        }
    })
}

/// Base64 value of one character, or None for padding / foreign bytes.
fn b64_val(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Decode base64 (skipping ASCII whitespace) until `max_out` bytes are
/// produced, a padding '=' is hit, or a non-base64 byte ends the stream.
fn b64_decode(input: &[u8], max_out: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(max_out.min(1 << 20));
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &c in input {
        if c.is_ascii_whitespace() {
            continue;
        }
        let Some(v) = b64_val(c) else { break };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            if out.len() >= max_out {
                break;
            }
        }
    }
    out
}

/// Characters one base64 "unit" of `nbytes` occupies (with padding).
fn b64_unit_len(nbytes: usize) -> usize {
    nbytes.div_ceil(3) * 4
}

/// Read one little/big-endian header word of `hs` bytes (4 or 8).
fn header_word(bytes: &[u8], off: usize, hs: usize, big_endian: bool) -> Result<u64> {
    if off + hs > bytes.len() {
        return Err(DatasetError::Format("truncated VTK XML data header".into()));
    }
    let mut v = [0u8; 8];
    v[..hs].copy_from_slice(&bytes[off..off + hs]);
    Ok(if hs == 8 {
        if big_endian {
            u64::from_be_bytes(bytes[off..off + 8].try_into().unwrap())
        } else {
            u64::from_le_bytes(v)
        }
    } else if big_endian {
        u32::from_be_bytes(bytes[off..off + 4].try_into().unwrap()) as u64
    } else {
        u32::from_le_bytes(v[..4].try_into().unwrap()) as u64
    })
}

/// Inflate one zlib block, capped at `expect` bytes.
fn inflate(data: &[u8], expect: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expect.min(64 * 1024 * 1024));
    flate2::read::ZlibDecoder::new(data)
        .take(expect as u64)
        .read_to_end(&mut out)
        .map_err(|e| DatasetError::Format(format!("zlib inflate failed: {}", e)))?;
    Ok(out)
}

/// Extract the raw bytes of one binary/appended VTK XML data array starting
/// at `stream` (base64 text or raw bytes), handling the size header and
/// optional zlib block compression.
fn payload_bytes<'a>(
    stream: &'a [u8],
    b64: bool,
    compressed: bool,
    hs: usize,
    big_endian: bool,
) -> Result<Cow<'a, [u8]>> {
    if !compressed {
        if b64 {
            let head = b64_decode(stream, hs);
            let nbytes = header_word(&head, 0, hs, big_endian)?;
            if nbytes > MAX_ARRAY_BYTES {
                return Err(DatasetError::Format("VTK XML array too large".into()));
            }
            let mut all = b64_decode(stream, hs + nbytes as usize);
            if all.len() < hs + nbytes as usize {
                return Err(DatasetError::Format("truncated base64 data array".into()));
            }
            all.drain(..hs);
            Ok(Cow::Owned(all))
        } else {
            let nbytes = header_word(stream, 0, hs, big_endian)? as usize;
            if nbytes as u64 > MAX_ARRAY_BYTES || hs + nbytes > stream.len() {
                return Err(DatasetError::Format("truncated appended data array".into()));
            }
            Ok(Cow::Borrowed(&stream[hs..hs + nbytes]))
        }
    } else {
        // Compressed header: [nblocks][blocksize][last partial size] then one
        // compressed size per block; the zlib blocks follow. In base64 files
        // the header and the data are two separate base64 units.
        let first = if b64 {
            b64_decode(&stream[..b64_unit_len(3 * hs).min(stream.len())], 3 * hs)
        } else {
            stream.get(..3 * hs).unwrap_or_default().to_vec()
        };
        let nblocks = header_word(&first, 0, hs, big_endian)? as usize;
        let blocksize = header_word(&first, hs, hs, big_endian)? as usize;
        let lastsize = header_word(&first, 2 * hs, hs, big_endian)? as usize;
        if nblocks == 0 {
            return Ok(Cow::Owned(Vec::new()));
        }
        let header_bytes = checked_mul(3 + nblocks, hs)?;
        let header = if b64 {
            b64_decode(
                &stream[..b64_unit_len(header_bytes).min(stream.len())],
                header_bytes,
            )
        } else {
            stream.get(..header_bytes).unwrap_or_default().to_vec()
        };
        if header.len() < header_bytes {
            return Err(DatasetError::Format("truncated compression header".into()));
        }
        let sizes: Vec<usize> = (0..nblocks)
            .map(|i| header_word(&header, (3 + i) * hs, hs, big_endian).map(|v| v as usize))
            .collect::<Result<_>>()?;
        let total_raw = checked_mul(nblocks - 1, blocksize)?
            + if lastsize == 0 { blocksize } else { lastsize };
        if total_raw as u64 > MAX_ARRAY_BYTES {
            return Err(DatasetError::Format("VTK XML array too large".into()));
        }

        let compressed_total: usize = sizes.iter().sum();
        let data_region: Cow<[u8]> = if b64 {
            Cow::Owned(b64_decode(
                &stream[b64_unit_len(header_bytes).min(stream.len())..],
                compressed_total,
            ))
        } else {
            let start = header_bytes;
            if start + compressed_total > stream.len() {
                return Err(DatasetError::Format("truncated compressed data".into()));
            }
            Cow::Borrowed(&stream[start..start + compressed_total])
        };
        if data_region.len() < compressed_total {
            return Err(DatasetError::Format("truncated compressed data".into()));
        }

        let mut out = Vec::with_capacity(total_raw.min(64 * 1024 * 1024));
        let mut off = 0usize;
        for (i, &csize) in sizes.iter().enumerate() {
            let expect = if i + 1 == nblocks && lastsize != 0 {
                lastsize
            } else {
                blocksize
            };
            out.extend_from_slice(&inflate(&data_region[off..off + csize], expect)?);
            off += csize;
        }
        Ok(Cow::Owned(out))
    }
}

/// Everything needed to decode any DataArray in one file.
struct XmlCtx<'a> {
    big_endian: bool,
    header_size: usize,
    compressed: bool,
    appended: Option<(&'a [u8], bool)>, // (blob, base64 encoded)
}

/// Decode one `<DataArray>`: returns (name, ncomp, kept tuple values).
fn read_data_array(
    attrs: &str,
    content: &str,
    tuples: usize,
    stride: usize,
    ctx: &XmlCtx,
) -> Result<(String, usize, Vec<f32>)> {
    let name = xml_attr(attrs, "Name").unwrap_or_default();
    let elem = xml_elem(&xml_attr(attrs, "type").unwrap_or_default())?;
    let ncomp: usize = xml_attr(attrs, "NumberOfComponents")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1);
    let format = xml_attr(attrs, "format").unwrap_or_else(|| "ascii".into());

    let stride = stride.max(1);
    let kept = kept_count(tuples, stride);
    let mut out = Vec::with_capacity(checked_mul(kept, ncomp)?);

    match format.as_str() {
        "ascii" => {
            let mut it = content.split_ascii_whitespace();
            for t in 0..tuples {
                let keep = t % stride == 0;
                for _ in 0..ncomp {
                    let tok = it.next().ok_or_else(|| {
                        DatasetError::Format(format!("array '{}' has too few values", name))
                    })?;
                    if keep {
                        out.push(tok.parse::<f32>().map_err(|_| {
                            DatasetError::Format(format!("non-numeric value '{}'", tok))
                        })?);
                    }
                }
            }
        }
        "binary" | "appended" => {
            let stream: &[u8];
            let b64: bool;
            if format == "binary" {
                stream = content.trim().as_bytes();
                b64 = true;
            } else {
                let (blob, blob_b64) = ctx.appended.ok_or_else(|| {
                    DatasetError::Format("appended array but no <AppendedData> block".into())
                })?;
                let offset: usize = xml_attr(attrs, "offset")
                    .and_then(|v| v.trim().parse().ok())
                    .ok_or_else(|| DatasetError::Format("appended array missing offset".into()))?;
                if offset >= blob.len() {
                    return Err(DatasetError::Format("appended offset out of range".into()));
                }
                stream = &blob[offset..];
                b64 = blob_b64;
            }
            let raw = payload_bytes(stream, b64, ctx.compressed, ctx.header_size, ctx.big_endian)?;
            let es = elem.size();
            let need = checked_mul(checked_mul(tuples, ncomp)?, es)?;
            if raw.len() < need {
                return Err(DatasetError::Format(format!(
                    "array '{}' payload shorter than declared ({} < {})",
                    name,
                    raw.len(),
                    need
                )));
            }
            let mut t = 0;
            while t < tuples {
                let base = t * ncomp * es;
                for c in 0..ncomp {
                    out.push(decode_elem(&raw, base + c * es, elem, ctx.big_endian));
                }
                t += stride;
            }
        }
        other => {
            return Err(DatasetError::Unsupported(format!(
                "VTK XML data format '{}'",
                other
            )))
        }
    }
    Ok((name, ncomp, out))
}

fn parse_vtk_xml(map: &[u8], budget: usize) -> Result<(CaeMesh, Option<String>)> {
    // Split off the appended blob: everything after the '_' that follows
    // <AppendedData ...>, up to </AppendedData>. The XML head stays text.
    let (xml, appended): (&str, Option<(&[u8], bool)>) =
        match find_bytes(map, b"<AppendedData") {
            Some(at) => {
                let head = std::str::from_utf8(&map[..at])
                    .map_err(|_| DatasetError::Format("VTK XML head is not UTF-8".into()))?;
                let tag_end = at + find_bytes(&map[at..], b">")
                    .ok_or_else(|| DatasetError::Format("unterminated <AppendedData>".into()))?;
                let attrs = std::str::from_utf8(&map[at + "<AppendedData".len()..tag_end])
                    .unwrap_or("");
                let b64 = xml_attr(attrs, "encoding")
                    .map(|e| e.eq_ignore_ascii_case("base64"))
                    .unwrap_or(false);
                let under = tag_end
                    + 1
                    + find_bytes(&map[tag_end + 1..], b"_").ok_or_else(|| {
                        DatasetError::Format("missing '_' in <AppendedData>".into())
                    })?;
                let end = rfind_bytes(map, b"</AppendedData>").unwrap_or(map.len());
                (head, Some((&map[under + 1..end], b64)))
            }
            None => (
                std::str::from_utf8(map)
                    .map_err(|_| DatasetError::Format("VTK XML file is not UTF-8".into()))?,
                None,
            ),
        };

    let vtkfile = tag_attrs(xml, "VTKFile")
        .ok_or_else(|| DatasetError::Format("missing <VTKFile> root element".into()))?;
    let big_endian = xml_attr(vtkfile, "byte_order")
        .map(|b| b.eq_ignore_ascii_case("BigEndian"))
        .unwrap_or(false);
    let header_size = match xml_attr(vtkfile, "header_type").as_deref() {
        Some("UInt64") => 8,
        _ => 4,
    };
    let compressed = match xml_attr(vtkfile, "compressor").as_deref() {
        None | Some("") => false,
        Some(c) if c.contains("ZLib") => true,
        Some(c) => {
            return Err(DatasetError::Unsupported(format!(
                "VTK XML compressor '{}' (only zlib is supported)",
                c
            )))
        }
    };
    let ctx = XmlCtx {
        big_endian,
        header_size,
        compressed,
        appended,
    };

    let piece = tag_attrs(xml, "Piece")
        .ok_or_else(|| DatasetError::Format("missing <Piece> element".into()))?;
    let n_points: usize = xml_attr(piece, "NumberOfPoints")
        .and_then(|v| v.trim().parse().ok())
        .ok_or_else(|| DatasetError::Format("missing NumberOfPoints".into()))?;
    if n_points == 0 {
        return Err(DatasetError::Format("file contains no points".into()));
    }
    let stride = stride_for(n_points, budget);

    let points_sec = xml_section(xml, "Points")
        .ok_or_else(|| DatasetError::Format("missing <Points> section".into()))?;
    let point_arrays = data_arrays(points_sec)?;
    let (pa, pc) = point_arrays
        .first()
        .ok_or_else(|| DatasetError::Format("missing points DataArray".into()))?;
    let (_, pcomp, pvals) = read_data_array(pa, pc, n_points, stride, &ctx)?;
    if pcomp != 3 {
        return Err(DatasetError::Format(format!(
            "points array has {} components, expected 3",
            pcomp
        )));
    }
    let points: Vec<[f32; 3]> = pvals.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

    let mut fields = Vec::new();
    let mut default_field = None;
    if let Some(pd) = xml_section(xml, "PointData") {
        // The PointData "Scalars" attribute names the writer's active field.
        default_field = tag_attrs(xml, "PointData").and_then(|a| xml_attr(a, "Scalars"));
        for (attrs, content) in data_arrays(pd)? {
            let (name, ncomp, data) = read_data_array(attrs, content, n_points, stride, &ctx)?;
            fields.push(CaeField {
                name,
                ncomp,
                comp_names: Vec::new(),
                data,
            });
        }
    }

    Ok((CaeMesh { points, fields }, default_field))
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn rfind_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).rposition(|w| w == needle)
}

// ---------------------------------------------------------------------------
// CalculiX FRD (.frd)
// ---------------------------------------------------------------------------

/// Load an ASCII CalculiX `.frd` result file.
pub fn load_frd(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    let file = std::fs::File::open(path)?;
    // SAFETY: read-only mapping; all data is copied out before it drops.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let budget = opts.max_rows.unwrap_or(DEFAULT_MAX_POINTS);
    let mesh = parse_frd(&map, budget)?;
    mesh_to_dataset(
        &file_stem(path),
        "frd",
        path,
        mesh,
        opts.label_column.as_deref(),
        None,
    )
}

/// Fixed-width FRD data line: 3-char code, 10-char node id, 12-char values.
/// Falls back to whitespace splitting for writers that pad differently.
fn frd_values(line: &str, values_from: usize, out: &mut Vec<f32>) {
    let area = if line.len() > values_from {
        &line[values_from..]
    } else {
        ""
    };
    let start = out.len();
    for chunk in area.as_bytes().chunks(12) {
        let s = std::str::from_utf8(chunk).unwrap_or("").trim();
        if s.is_empty() {
            break;
        }
        match s.parse::<f32>() {
            Ok(v) => out.push(v),
            Err(_) => {
                // Fixed-width slicing failed: retry as whitespace-separated.
                out.truncate(start);
                out.extend(
                    area.split_whitespace()
                        .filter_map(|t| t.parse::<f32>().ok()),
                );
                return;
            }
        }
    }
}

/// Node id from the fixed-width columns, whitespace fallback.
fn frd_node_id(line: &str) -> Option<i64> {
    line.get(3..13)
        .and_then(|s| s.trim().parse::<i64>().ok())
        .or_else(|| {
            line.split_whitespace()
                .nth(1)
                .and_then(|t| t.parse::<i64>().ok())
        })
}

fn parse_frd(map: &[u8], budget: usize) -> Result<CaeMesh> {
    let text = std::str::from_utf8(map).map_err(|_| {
        DatasetError::Unsupported("FRD file is not ASCII (binary FRD is not supported)".into())
    })?;
    let mut lines = text.lines();

    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut row_of_id: HashMap<i64, u32> = HashMap::new();
    // Insertion-ordered fields; a repeated name (next step) overwrites.
    let mut fields: Vec<CaeField> = Vec::new();
    let mut field_idx: HashMap<String, usize> = HashMap::new();

    while let Some(line) = lines.next() {
        let head = line.trim_start();
        if head.starts_with("9999") {
            break;
        }
        if head.starts_with("2C") {
            // Nodal coordinates block: "    2C <numnod> <format>".
            let toks: Vec<&str> = line.split_whitespace().collect();
            let numnod: usize = toks
                .get(1)
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| DatasetError::Format("bad FRD 2C node block header".into()))?;
            if toks
                .get(2)
                .and_then(|t| t.parse::<i64>().ok())
                .is_some_and(|f| f >= 2)
            {
                return Err(DatasetError::Unsupported(
                    "binary FRD node blocks are not supported (rerun ccx with ASCII output)"
                        .into(),
                ));
            }
            let stride = stride_for(numnod, budget);
            let mut seq = 0usize;
            for nl in lines.by_ref() {
                let nh = nl.trim_start();
                if nh.starts_with("-3") {
                    break;
                }
                if !nh.starts_with("-1") {
                    continue;
                }
                if seq.is_multiple_of(stride) {
                    let id = frd_node_id(nl)
                        .ok_or_else(|| DatasetError::Format("bad FRD node line".into()))?;
                    let mut coords = Vec::with_capacity(3);
                    frd_values(nl, 13, &mut coords);
                    if coords.len() < 3 {
                        return Err(DatasetError::Format("bad FRD node coordinates".into()));
                    }
                    row_of_id.insert(id, points.len() as u32);
                    points.push([coords[0], coords[1], coords[2]]);
                }
                seq += 1;
            }
        } else if head.starts_with("3C") {
            // Element block: irrelevant for the point cloud.
            for el in lines.by_ref() {
                if el.trim_start().starts_with("-3") {
                    break;
                }
            }
        } else if head.starts_with("100C") {
            // Nodal result block for one step.
            if points.is_empty() {
                return Err(DatasetError::Format(
                    "FRD result block before the node block".into(),
                ));
            }
            let mut name = String::new();
            let mut comp_names: Vec<String> = Vec::new();
            // Per-row values, filled as -1/-2 lines arrive.
            let mut rows: Vec<Vec<f32>> = vec![Vec::new(); points.len()];
            let mut pending: Option<(i64, Vec<f32>)> = None;
            let mut ncomp_seen = 0usize;

            let flush =
                |pending: &mut Option<(i64, Vec<f32>)>, rows: &mut Vec<Vec<f32>>, ncomp: &mut usize| {
                    if let Some((id, vals)) = pending.take() {
                        *ncomp = (*ncomp).max(vals.len());
                        if let Some(&row) = row_of_id.get(&id) {
                            rows[row as usize] = vals;
                        }
                    }
                };

            for rl in lines.by_ref() {
                let rh = rl.trim_start();
                if rh.starts_with("-3") {
                    break;
                }
                if rh.starts_with("-4") {
                    name = rl
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("field")
                        .to_string();
                } else if rh.starts_with("-5") {
                    if let Some(c) = rl.split_whitespace().nth(1) {
                        comp_names.push(c.to_string());
                    }
                } else if rh.starts_with("-1") {
                    flush(&mut pending, &mut rows, &mut ncomp_seen);
                    let id = frd_node_id(rl)
                        .ok_or_else(|| DatasetError::Format("bad FRD result line".into()))?;
                    let mut vals = Vec::new();
                    frd_values(rl, 13, &mut vals);
                    pending = Some((id, vals));
                } else if rh.starts_with("-2") {
                    if let Some((_, vals)) = pending.as_mut() {
                        frd_values(rl, 13, vals);
                    }
                }
            }
            flush(&mut pending, &mut rows, &mut ncomp_seen);

            if name.is_empty() || ncomp_seen == 0 {
                continue; // empty or unreadable block: skip it
            }
            let ncomp = ncomp_seen;
            let mut data = vec![0.0f32; points.len() * ncomp];
            for (row, vals) in rows.iter().enumerate() {
                for (c, &v) in vals.iter().take(ncomp).enumerate() {
                    data[row * ncomp + c] = v;
                }
            }
            comp_names.truncate(ncomp);
            let field = CaeField {
                name: name.clone(),
                ncomp,
                comp_names,
                data,
            };
            match field_idx.get(&name) {
                // Same result on a later step: keep the last one.
                Some(&i) => fields[i] = field,
                None => {
                    field_idx.insert(name, fields.len());
                    fields.push(field);
                }
            }
        }
    }

    if points.is_empty() {
        return Err(DatasetError::Format("FRD file has no node block".into()));
    }
    Ok(CaeMesh { points, fields })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn opts(max_rows: Option<usize>, field: Option<&str>) -> LoadOptions {
        LoadOptions {
            max_rows,
            label_column: field.map(|s| s.to_string()),
        }
    }

    fn write_temp(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::File::create(&path).unwrap().write_all(bytes).unwrap();
        path
    }

    // ── banding ──

    #[test]
    fn band_labels_quantizes_low_to_high() {
        let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let (labels, names) = band_labels(&vals, 10, "MPa");
        assert_eq!(names.len(), 10);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[99], 9);
        assert!(names[0].ends_with(" MPa"));
        // Constant field collapses into one band without dividing by zero.
        let (labels, names) = band_labels(&[5.0; 4], 10, "");
        assert!(labels.iter().all(|&l| l == 0));
        assert_eq!(names.len(), 10);
        // Non-finite values land in band 0.
        let (labels, _) = band_labels(&[f32::NAN, 0.0, 10.0], 10, "");
        assert_eq!(labels[0], 0);
    }

    // ── VTK legacy ──

    const VTK_ASCII: &str = "\
# vtk DataFile Version 3.0
test grid
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 float
0 0 0
1 0 0
0 1 0
0 0 1
CELLS 1 5
4 0 1 2 3
CELL_TYPES 1
10
POINT_DATA 4
SCALARS T float 1
LOOKUP_TABLE default
10 20 30 40
VECTORS U float
1 0 0
0 2 0
0 0 3
1 1 1
FIELD attributes 1
p 1 4 double
100 200 300 400
";

    #[test]
    fn vtk_legacy_ascii_maps_points_and_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "case.vtk", VTK_ASCII.as_bytes());
        let ds = load_vtk_legacy(&path, &opts(None, None)).unwrap();
        assert_eq!(ds.n_rows(), 4);
        assert_eq!(
            ds.metadata.column_names,
            vec!["x", "y", "z", "T", "U_0", "U_1", "U_2", "U_mag", "p"]
        );
        // Default banding field is the first SCALARS array.
        assert_eq!(ds.metadata.label_column.as_deref(), Some("T"));
        assert_eq!(ds.value(1, 0), 1.0); // x of point 1
        assert_eq!(ds.value(3, 3), 40.0); // T of point 3
        assert_eq!(ds.value(1, 7), 2.0); // |U| of point 1
        assert_eq!(ds.value(2, 8), 300.0); // p of point 2
        // 4 distinct T values over 10 bands: first band low, last band high.
        assert_eq!(ds.labels[0], 0);
        assert_eq!(ds.labels[3], 9);
    }

    #[test]
    fn vtk_legacy_field_selection_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "case.vtk", VTK_ASCII.as_bytes());
        let ds = load_vtk_legacy(&path, &opts(None, Some("p"))).unwrap();
        assert_eq!(ds.metadata.label_column.as_deref(), Some("p"));
        // Vector field name resolves to its magnitude column.
        let ds = load_vtk_legacy(&path, &opts(None, Some("u"))).unwrap();
        assert_eq!(ds.metadata.label_column.as_deref(), Some("U_mag"));
        // Unknown fields list what is available.
        let err = load_vtk_legacy(&path, &opts(None, Some("nope"))).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("T") && msg.contains("U") && msg.contains("p"), "{msg}");
    }

    #[test]
    fn vtk_legacy_stride_subsamples_uniformly() {
        let mut txt = String::from(
            "# vtk DataFile Version 3.0\nbig\nASCII\nDATASET POLYDATA\nPOINTS 100 float\n",
        );
        for i in 0..100 {
            txt.push_str(&format!("{} 0 0\n", i));
        }
        txt.push_str("POINT_DATA 100\nSCALARS v float 1\nLOOKUP_TABLE default\n");
        for i in 0..100 {
            txt.push_str(&format!("{}\n", i * 10));
        }
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "big.vtk", txt.as_bytes());
        let ds = load_vtk_legacy(&path, &opts(Some(10), None)).unwrap();
        assert_eq!(ds.n_rows(), 10);
        // Kept rows are strided over the whole domain, not the first 10.
        assert_eq!(ds.value(0, 0), 0.0);
        assert_eq!(ds.value(9, 0), 90.0);
        assert_eq!(ds.value(9, 3), 900.0); // field stays aligned with points
    }

    #[test]
    fn vtk_legacy_binary_big_endian() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"# vtk DataFile Version 3.0\nbin\nBINARY\nDATASET POLYDATA\n");
        buf.extend_from_slice(b"POINTS 2 float\n");
        for v in [0.0f32, 0.0, 0.0, 1.0, 2.0, 3.0] {
            buf.extend_from_slice(&v.to_be_bytes());
        }
        buf.extend_from_slice(b"\nPOINT_DATA 2\nSCALARS q double 1\nLOOKUP_TABLE default\n");
        for v in [1.5f64, -2.5] {
            buf.extend_from_slice(&v.to_be_bytes());
        }
        buf.extend_from_slice(b"\n");
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "bin.vtk", &buf);
        let ds = load_vtk_legacy(&path, &opts(None, None)).unwrap();
        assert_eq!(ds.n_rows(), 2);
        assert_eq!(ds.value(1, 0), 1.0);
        assert_eq!(ds.value(1, 2), 3.0);
        assert_eq!(ds.value(0, 3), 1.5);
        assert_eq!(ds.value(1, 3), -2.5);
    }

    #[test]
    fn vtk_legacy_structured_points_synthesizes_coords() {
        let txt = "\
# vtk DataFile Version 3.0
img
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS 2 2 1
ORIGIN 10 20 30
SPACING 1 2 3
POINT_DATA 4
SCALARS rho float 1
LOOKUP_TABLE default
1 2 3 4
";
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "img.vtk", txt.as_bytes());
        let ds = load_vtk_legacy(&path, &opts(None, None)).unwrap();
        assert_eq!(ds.n_rows(), 4);
        // x varies fastest: point 1 = origin + (1,0,0)*spacing.
        assert_eq!(ds.value(1, 0), 11.0);
        assert_eq!(ds.value(2, 1), 22.0);
        assert_eq!(ds.value(3, 3), 4.0);
    }

    #[test]
    fn vtk_legacy_rejects_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "bad.vtk", b"not a vtk file at all");
        assert!(load_vtk_legacy(&path, &opts(None, None)).is_err());
    }

    // ── VTK XML ──

    fn vtu_ascii(n: usize) -> String {
        let mut pts = String::new();
        let mut p = String::new();
        for i in 0..n {
            pts.push_str(&format!("{} 0 0 ", i));
            p.push_str(&format!("{} ", i * 100));
        }
        format!(
            r#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n}" NumberOfCells="0">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {pts}
        </DataArray>
      </Points>
      <PointData Scalars="p">
        <DataArray type="Float64" Name="p" format="ascii">
          {p}
        </DataArray>
      </PointData>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii"></DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"#
        )
    }

    #[test]
    fn vtu_ascii_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "flow.vtu", vtu_ascii(5).as_bytes());
        let ds = load_vtk_xml(&path, &opts(None, None)).unwrap();
        assert_eq!(ds.n_rows(), 5);
        assert_eq!(ds.metadata.column_names, vec!["x", "y", "z", "p"]);
        assert_eq!(ds.metadata.label_column.as_deref(), Some("p"));
        assert_eq!(ds.value(4, 0), 4.0);
        assert_eq!(ds.value(4, 3), 400.0);
    }

    #[test]
    fn vtu_ascii_stride() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "flow.vtu", vtu_ascii(100).as_bytes());
        let ds = load_vtk_xml(&path, &opts(Some(25), None)).unwrap();
        assert_eq!(ds.n_rows(), 25);
        assert_eq!(ds.value(24, 0), 96.0); // last kept point = index 96
        assert_eq!(ds.value(24, 3), 9600.0);
    }

    /// Standard base64 encoder (tests only).
    fn b64_encode(data: &[u8]) -> String {
        const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
            let v = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
            out.push(A[(v >> 18) as usize & 63] as char);
            out.push(A[(v >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 { A[(v >> 6) as usize & 63] as char } else { '=' });
            out.push(if chunk.len() > 2 { A[v as usize & 63] as char } else { '=' });
        }
        out
    }

    fn le_f32(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn vtu_inline_base64_uncompressed() {
        let pts = le_f32(&[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        let field = le_f32(&[7.0, 9.0]);
        let mut pblob = (pts.len() as u32).to_le_bytes().to_vec();
        pblob.extend_from_slice(&pts);
        let mut fblob = (field.len() as u32).to_le_bytes().to_vec();
        fblob.extend_from_slice(&field);
        let xml = format!(
            r#"<VTKFile type="UnstructuredGrid" byte_order="LittleEndian" header_type="UInt32">
  <UnstructuredGrid>
    <Piece NumberOfPoints="2" NumberOfCells="0">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="binary">{}</DataArray>
      </Points>
      <PointData>
        <DataArray type="Float32" Name="T" format="binary">{}</DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#,
            b64_encode(&pblob),
            b64_encode(&fblob)
        );
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "b64.vtu", xml.as_bytes());
        let ds = load_vtk_xml(&path, &opts(None, None)).unwrap();
        assert_eq!(ds.n_rows(), 2);
        assert_eq!(ds.value(1, 2), 3.0);
        assert_eq!(ds.value(1, 3), 9.0);
    }

    #[test]
    fn vtu_appended_raw_zlib_compressed() {
        use flate2::{write::ZlibEncoder, Compression};

        let compress = |raw: &[u8]| -> Vec<u8> {
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
            enc.write_all(raw).unwrap();
            enc.finish().unwrap()
        };
        // One compressed block per array: header [1][len][len][clen].
        let encode = |raw: &[u8]| -> Vec<u8> {
            let c = compress(raw);
            let mut out = Vec::new();
            for w in [1u32, raw.len() as u32, raw.len() as u32, c.len() as u32] {
                out.extend_from_slice(&w.to_le_bytes());
            }
            out.extend_from_slice(&c);
            out
        };
        let pts = encode(&le_f32(&[0.0, 0.0, 0.0, 4.0, 5.0, 6.0]));
        let field = encode(&le_f32(&[1.0, 2.0]));
        let off2 = pts.len();

        let mut file = format!(
            r#"<VTKFile type="UnstructuredGrid" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="2" NumberOfCells="0">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="appended" offset="0"/>
      </Points>
      <PointData>
        <DataArray type="Float32" Name="U" format="appended" offset="{off2}"/>
      </PointData>
    </Piece>
  </UnstructuredGrid>
  <AppendedData encoding="raw">
   _"#
        )
        .into_bytes();
        file.extend_from_slice(&pts);
        file.extend_from_slice(&field);
        file.extend_from_slice(b"\n  </AppendedData>\n</VTKFile>\n");

        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "raw.vtu", &file);
        let ds = load_vtk_xml(&path, &opts(None, None)).unwrap();
        assert_eq!(ds.n_rows(), 2);
        assert_eq!(ds.value(1, 0), 4.0);
        assert_eq!(ds.value(1, 2), 6.0);
        assert_eq!(ds.value(0, 3), 1.0);
        assert_eq!(ds.value(1, 3), 2.0);
    }

    // ── FRD ──

    const FRD: &str = "\
    1C  test
    1UUSER
    2C                           4                                     1
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2 1.00000E+00 0.00000E+00 0.00000E+00
 -1         3 0.00000E+00 1.00000E+00 0.00000E+00
 -1         4 0.00000E+00 0.00000E+00 1.00000E+00
 -3
    3C                           1                                     1
 -1         1    1    0    1
 -2         1         2         3         4
 -3
  100CL  101 1.00000000               4                     0    1
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -5  ALL         1    2    0    0    1ALL
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2 1.00000E+00 0.00000E+00 0.00000E+00
 -1         3 0.00000E+00 2.00000E+00 0.00000E+00
 -1         4 0.00000E+00 0.00000E+00 3.00000E+00
 -3
  100CL  101 1.00000000               4                     0    2
 -4  NDTEMP      1    1
 -5  T           1    1    0    0
 -1         1 2.00000E+01
 -1         2 4.00000E+01
 -1         3 6.00000E+01
 -1         4 8.00000E+01
 -3
 9999
";

    #[test]
    fn frd_parses_nodes_and_result_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "job.frd", FRD.as_bytes());
        let ds = load_frd(&path, &opts(None, None)).unwrap();
        assert_eq!(ds.n_rows(), 4);
        assert_eq!(
            ds.metadata.column_names,
            vec!["x", "y", "z", "DISP_D1", "DISP_D2", "DISP_D3", "DISP_mag", "NDTEMP"]
        );
        // First field drives the default banding.
        assert_eq!(ds.metadata.label_column.as_deref(), Some("DISP_mag"));
        assert_eq!(ds.value(3, 2), 1.0); // z of node 4
        assert_eq!(ds.value(2, 4), 2.0); // DISP D2 of node 3
        assert_eq!(ds.value(3, 6), 3.0); // |DISP| of node 4
        assert_eq!(ds.value(1, 7), 40.0); // NDTEMP of node 2
    }

    #[test]
    fn frd_field_selection_and_stride() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "job.frd", FRD.as_bytes());
        let ds = load_frd(&path, &opts(None, Some("NDTEMP"))).unwrap();
        assert_eq!(ds.metadata.label_column.as_deref(), Some("NDTEMP"));
        assert_eq!(ds.labels[0], 0);
        assert_eq!(ds.labels[3], 9);

        // Budget of 2 keeps nodes 1 and 3 (stride 2), results follow the map.
        let ds = load_frd(&path, &opts(Some(2), None)).unwrap();
        assert_eq!(ds.n_rows(), 2);
        assert_eq!(ds.value(1, 1), 1.0); // y of node 3
        assert_eq!(ds.value(1, 7), 60.0); // NDTEMP of node 3
    }

    #[test]
    fn frd_rejects_binary_and_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "bin.frd", b"    2C           4    2\n\x00\x01\x02");
        assert!(load_frd(&path, &opts(None, None)).is_err());
        let path = write_temp(&dir, "empty.frd", b"    1C  x\n 9999\n");
        assert!(load_frd(&path, &opts(None, None)).is_err());
    }
}
