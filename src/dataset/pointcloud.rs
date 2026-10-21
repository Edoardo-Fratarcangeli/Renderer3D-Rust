//! Import of glTF / GLB point clouds (primitives drawn with `mode = POINTS`).
//!
//! A glTF point cloud has no faces: feeding it to the solid-mesh importer
//! would stitch every three consecutive points into a bogus triangle. This
//! module instead maps the points onto the standard [`Dataset`] pipeline, so
//! they get the instanced point renderer, label/search/spatial filters, the
//! synced data table, picking and CSV export like any other point dataset.
//!
//! Feature columns: `x, y, z` (node transforms applied), then `r, g, b`
//! when the cloud carries `COLOR_0` (linear, as the spec defines it), then
//! `nx, ny, nz` for `NORMAL`, then every custom `_ATTRIBUTE` (scalar →
//! one column, vector → `name_0..name_n`), lowercased.
//!
//! Labels: a `_CLASSIFICATION` attribute (LAS/ASPRS codes, as written by
//! PDAL and Cesium exporters) becomes categorical labels; otherwise all
//! points share one label. `LoadOptions::label_column` overrides this with
//! any column, quantized into contour bands like the CFD/FEM importer.
//!
//! Large files: the point count is known from the accessors before any
//! decoding, so the row budget is applied as a uniform stride across the
//! whole cloud (spatially even, unlike a "first N points" cut).

use std::collections::BTreeMap;
use std::path::Path;

use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, SquareMatrix, Vector3, Vector4};
use gltf::accessor::DataType;
use gltf::mesh::{Mode, Semantic};

use super::cae::{band_labels, DEFAULT_MAX_POINTS, FIELD_BANDS};
use super::loader::LoadOptions;
use super::{metadata::DatasetMetadata, Dataset, DatasetError, FeatureSource, Result};

/// Extensions that can hold a point cloud.
pub fn is_point_cloud_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("gltf" | "glb")
    )
}

/// Primitive census of a glTF file, used to route it to the right importer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GltfContent {
    /// Primitives with a triangle topology (list, strip or fan).
    pub triangle_primitives: usize,
    /// Primitives drawn as points.
    pub point_primitives: usize,
    /// Total vertices across the point primitives.
    pub points: usize,
}

impl GltfContent {
    pub fn has_points(&self) -> bool {
        self.point_primitives > 0 && self.points > 0
    }

    pub fn has_triangles(&self) -> bool {
        self.triangle_primitives > 0
    }
}

/// Read only the glTF document (no buffer decoding for `.gltf`) and count
/// its primitives by topology.
pub fn inspect(path: &Path) -> Result<GltfContent> {
    let gltf = gltf::Gltf::open(path).map_err(|e| DatasetError::Format(describe_error(&e)))?;
    let mut content = GltfContent::default();
    for mesh in gltf.document.meshes() {
        for primitive in mesh.primitives() {
            match primitive.mode() {
                Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan => {
                    content.triangle_primitives += 1
                }
                Mode::Points => {
                    content.point_primitives += 1;
                    content.points += primitive
                        .get(&Semantic::Positions)
                        .map(|a| a.count())
                        .unwrap_or(0);
                }
                Mode::Lines | Mode::LineLoop | Mode::LineStrip => {}
            }
        }
    }
    Ok(content)
}

/// Human readable glTF error. Unsupported required extensions (Draco,
/// meshopt, quantization) are spelled out, since a bare "invalid glTF"
/// gives the user no hint that the file is compressed.
pub fn describe_error(err: &gltf::Error) -> String {
    if let gltf::Error::Validation(errors) = err {
        let required: Vec<String> = errors
            .iter()
            .map(|(path, _)| path.as_str().to_string())
            .filter(|p| p.starts_with("extensionsRequired"))
            .collect();
        if !required.is_empty() {
            return format!(
                "glTF requires unsupported extension(s): {} — re-export without \
                 compression (Draco / meshopt / quantization)",
                required.join(", ")
            );
        }
        let details: Vec<String> = errors
            .iter()
            .take(5)
            .map(|(path, e)| format!("{}: {}", path.as_str(), e))
            .collect();
        return format!("invalid glTF: {}", details.join("; "));
    }
    format!("glTF error: {}", err)
}

/// LAS 1.4 (ASPRS) standard point classes, indexed by code.
const LAS_CLASSES: [&str; 19] = [
    "never classified",
    "unclassified",
    "ground",
    "low vegetation",
    "medium vegetation",
    "high vegetation",
    "building",
    "low point (noise)",
    "model key-point",
    "water",
    "rail",
    "road surface",
    "overlap",
    "wire guard",
    "wire conductor",
    "transmission tower",
    "wire connector",
    "bridge deck",
    "high noise",
];

fn class_name(code: u32) -> String {
    LAS_CLASSES
        .get(code as usize)
        .map(|n| format!("{} · {}", code, n))
        .unwrap_or_else(|| format!("class {}", code))
}

/// A point primitive placed in the scene by its node's world transform.
struct Placed<'a> {
    primitive: gltf::Primitive<'a>,
    world: Matrix4<f32>,
}

fn collect_node<'a>(node: gltf::Node<'a>, parent: Matrix4<f32>, out: &mut Vec<Placed<'a>>) {
    let world = parent * Matrix4::from(node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives().filter(|p| p.mode() == Mode::Points) {
            out.push(Placed { primitive, world });
        }
    }
    for child in node.children() {
        collect_node(child, world, out);
    }
}

/// Point primitives with their world transforms: the default (or first)
/// scene graph when present, otherwise every mesh untransformed.
fn placed_point_primitives(doc: &gltf::Document) -> Vec<Placed<'_>> {
    let mut out = Vec::new();
    match doc.default_scene().or_else(|| doc.scenes().next()) {
        Some(scene) => {
            for node in scene.nodes() {
                collect_node(node, Matrix4::identity(), &mut out);
            }
        }
        None => {
            for mesh in doc.meshes() {
                for primitive in mesh.primitives().filter(|p| p.mode() == Mode::Points) {
                    out.push(Placed {
                        primitive,
                        world: Matrix4::identity(),
                    });
                }
            }
        }
    }
    out
}

/// Decode every element of a custom attribute accessor as `f32`
/// components (normalized integers mapped to [0, 1] / [-1, 1]). Sparse or
/// view-less accessors return `None` and the attribute is skipped.
fn read_accessor_f32(
    accessor: &gltf::Accessor,
    buffers: &[gltf::buffer::Data],
) -> Option<Vec<f32>> {
    if accessor.sparse().is_some() {
        return None;
    }
    let view = accessor.view()?;
    let data = buffers.get(view.buffer().index())?;
    let comp = accessor.data_type().size();
    let ncomp = accessor.dimensions().multiplicity();
    let stride = view.stride().unwrap_or(accessor.size());
    let start = view.offset() + accessor.offset();
    let count = accessor.count();
    if count > 0 {
        let last = start + (count - 1) * stride + comp * ncomp;
        if last > data.len() || last > view.offset() + view.length() {
            return None;
        }
    }
    let normalized = accessor.normalized();
    let mut out = Vec::with_capacity(count * ncomp);
    for i in 0..count {
        for c in 0..ncomp {
            let off = start + i * stride + c * comp;
            let b = &data[off..off + comp];
            let v = match accessor.data_type() {
                DataType::I8 => {
                    let v = b[0] as i8 as f32;
                    if normalized {
                        (v / 127.0).max(-1.0)
                    } else {
                        v
                    }
                }
                DataType::U8 => {
                    let v = b[0] as f32;
                    if normalized {
                        v / 255.0
                    } else {
                        v
                    }
                }
                DataType::I16 => {
                    let v = i16::from_le_bytes([b[0], b[1]]) as f32;
                    if normalized {
                        (v / 32767.0).max(-1.0)
                    } else {
                        v
                    }
                }
                DataType::U16 => {
                    let v = u16::from_le_bytes([b[0], b[1]]) as f32;
                    if normalized {
                        v / 65535.0
                    } else {
                        v
                    }
                }
                DataType::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                DataType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            };
            out.push(v);
        }
    }
    Some(out)
}

/// Load the point primitives of a glTF / GLB file as a [`Dataset`].
pub fn load_gltf(path: &Path, opts: &LoadOptions) -> Result<Dataset> {
    let gltf = gltf::Gltf::open(path).map_err(|e| DatasetError::Format(describe_error(&e)))?;
    let base = path.parent().unwrap_or_else(|| Path::new("./"));
    let buffers = gltf::import_buffers(&gltf.document, Some(base), gltf.blob.clone())
        .map_err(|e| DatasetError::Format(describe_error(&e)))?;
    let format = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("gltf")
        .to_ascii_lowercase();
    points_to_dataset(
        &gltf.document,
        &buffers,
        &file_stem(path),
        &format,
        path,
        opts,
    )
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("point cloud")
        .to_string()
}

/// Map already-decoded glTF point primitives onto a [`Dataset`].
fn points_to_dataset(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    name: &str,
    format: &str,
    path: &Path,
    opts: &LoadOptions,
) -> Result<Dataset> {
    let placed = placed_point_primitives(doc);
    let total: usize = placed
        .iter()
        .map(|p| {
            p.primitive
                .get(&Semantic::Positions)
                .map(|a| a.count())
                .unwrap_or(0)
        })
        .sum();
    if total == 0 {
        return Err(DatasetError::Unsupported(
            "glTF contains no point primitives (mode POINTS) — import it as a 3D model instead"
                .into(),
        ));
    }

    // Column layout: the union of the attributes over every primitive, so a
    // cloud split across primitives keeps one consistent table.
    let has_color = placed
        .iter()
        .any(|p| p.primitive.get(&Semantic::Colors(0)).is_some());
    let has_normals = placed
        .iter()
        .any(|p| p.primitive.get(&Semantic::Normals).is_some());
    let mut extras: BTreeMap<String, usize> = BTreeMap::new();
    for p in &placed {
        for (semantic, accessor) in p.primitive.attributes() {
            if let Semantic::Extras(attr) = semantic {
                let n = accessor.dimensions().multiplicity();
                let entry = extras.entry(attr.to_ascii_lowercase()).or_insert(n);
                *entry = (*entry).max(n);
            }
        }
    }

    let mut col_names: Vec<String> = vec!["x".into(), "y".into(), "z".into()];
    let color_col = has_color.then(|| {
        col_names.extend(["r", "g", "b"].map(String::from));
        3
    });
    let normal_col = has_normals.then(|| {
        let at = col_names.len();
        col_names.extend(["nx", "ny", "nz"].map(String::from));
        at
    });
    let mut extra_cols: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (attr, &n) in &extras {
        let at = col_names.len();
        if n == 1 {
            col_names.push(attr.clone());
        } else {
            col_names.extend((0..n).map(|c| format!("{}_{}", attr, c)));
        }
        extra_cols.insert(attr.clone(), (at, n));
    }
    let n_cols = col_names.len();

    let budget = opts.max_rows.unwrap_or(DEFAULT_MAX_POINTS).max(1);
    let stride = total.div_ceil(budget).max(1);
    let kept = total.div_ceil(stride);
    let mut data: Vec<f32> = Vec::with_capacity(
        kept.checked_mul(n_cols)
            .ok_or_else(|| DatasetError::Format("size overflow in point count".into()))?,
    );

    let mut global = 0usize;
    let mut rows = 0usize;
    for p in &placed {
        let reader = p
            .primitive
            .reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
        let Some(positions) = reader.read_positions() else {
            continue;
        };
        let positions: Vec<[f32; 3]> = positions.collect();
        let n = positions.len();
        let colors: Option<Vec<[f32; 3]>> =
            reader.read_colors(0).map(|c| c.into_rgb_f32().collect());
        let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|it| it.collect());
        let mut custom: Vec<(usize, usize, usize, Vec<f32>)> = Vec::new();
        for (semantic, accessor) in p.primitive.attributes() {
            if let Semantic::Extras(attr) = semantic {
                let (at, width) = extra_cols[&attr.to_ascii_lowercase()];
                let ncomp = accessor.dimensions().multiplicity();
                if accessor.count() != n {
                    continue;
                }
                if let Some(values) = read_accessor_f32(&accessor, buffers) {
                    custom.push((at, width, ncomp, values));
                }
            }
        }
        let normal_matrix = {
            let m = p.world;
            let linear = Matrix3::from_cols(m.x.truncate(), m.y.truncate(), m.z.truncate());
            linear.invert().map(|inv| inv.transpose()).unwrap_or(linear)
        };

        let mut row = vec![0.0f32; n_cols];
        for i in 0..n {
            let keep = global.is_multiple_of(stride);
            global += 1;
            if !keep {
                continue;
            }
            row.iter_mut().for_each(|v| *v = 0.0);
            let q = p.world * Vector4::new(positions[i][0], positions[i][1], positions[i][2], 1.0);
            row[..3].copy_from_slice(&[q.x, q.y, q.z]);
            if let Some(at) = color_col {
                let rgb = colors
                    .as_ref()
                    .and_then(|c| c.get(i))
                    .copied()
                    .unwrap_or([1.0; 3]);
                row[at..at + 3].copy_from_slice(&rgb);
            }
            if let (Some(at), Some(normals)) = (normal_col, normals.as_ref()) {
                if let Some(nrm) = normals.get(i) {
                    let v = normal_matrix * Vector3::new(nrm[0], nrm[1], nrm[2]);
                    let v = if v.magnitude2() > 0.0 {
                        v.normalize()
                    } else {
                        v
                    };
                    row[at..at + 3].copy_from_slice(&[v.x, v.y, v.z]);
                }
            }
            for (at, width, ncomp, values) in &custom {
                for c in 0..(*ncomp).min(*width) {
                    row[at + c] = values[i * ncomp + c];
                }
            }
            data.extend_from_slice(&row);
            rows += 1;
        }
    }
    if rows == 0 {
        return Err(DatasetError::Format(
            "glTF point primitives hold no positions".into(),
        ));
    }

    let classification_col = extra_cols.get("classification").map(|&(at, _)| at);
    let find = |sel: &str| col_names.iter().position(|c| c.eq_ignore_ascii_case(sel));
    let label_col = match opts.label_column.as_deref() {
        Some(sel) => Some(find(sel).ok_or_else(|| {
            DatasetError::Format(format!(
                "column '{}' not found; available columns: {}",
                sel,
                col_names.join(", ")
            ))
        })?),
        None => classification_col,
    };
    let (labels, label_names, label_column) = match label_col {
        Some(col) => {
            let values: Vec<f32> = (0..rows).map(|r| data[r * n_cols + col]).collect();
            let (labels, names) = if Some(col) == classification_col {
                class_labels(&values)
            } else {
                band_labels(&values, FIELD_BANDS, "")
            };
            (labels, names, Some(col_names[col].clone()))
        }
        None => (vec![0; rows], vec!["points".to_string()], None),
    };

    let mut metadata = DatasetMetadata::new(name, format, rows, n_cols);
    metadata.source_path = path.to_string_lossy().to_string();
    metadata.column_names = col_names;
    metadata.label_column = label_column;
    metadata.color_columns = color_col.map(|at| [at, at + 1, at + 2]);
    metadata.set_label_stats(&labels, &label_names);

    Ok(Dataset {
        metadata,
        source: FeatureSource::InMemory(data),
        labels,
        label_names,
    })
}

/// Categorical labels from LAS classification codes: one label per code
/// present, ordered by code, named after the ASPRS standard classes.
fn class_labels(values: &[f32]) -> (Vec<u32>, Vec<String>) {
    let codes: Vec<u32> = values
        .iter()
        .map(|&v| {
            if v.is_finite() && v >= 0.0 {
                v.round() as u32
            } else {
                0
            }
        })
        .collect();
    let mut present: Vec<u32> = codes.clone();
    present.sort_unstable();
    present.dedup();
    let labels = codes
        .iter()
        .map(|c| present.binary_search(c).unwrap_or(0) as u32)
        .collect();
    let names = present.iter().map(|&c| class_name(c)).collect();
    (labels, names)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GLB writer: JSON chunk + BIN chunk, both 4-byte aligned.
    fn glb(json: &str, bin: &[u8]) -> Vec<u8> {
        let mut json = json.as_bytes().to_vec();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let mut bin = bin.to_vec();
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin);
        out
    }

    /// Point cloud GLB: `n` points on the x axis, u8 normalized RGB colors,
    /// a u8 `_CLASSIFICATION` (alternating ground / building) and a f32
    /// `_INTENSITY`, placed by a node translated +10 on y.
    fn sample_glb(n: usize) -> Vec<u8> {
        let mut bin = Vec::new();
        for i in 0..n {
            for c in [i as f32, 0.0, 0.0] {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let color_off = bin.len();
        for i in 0..n {
            bin.extend_from_slice(&[255, (i % 256) as u8, 0, 0]); // padded to 4 bytes
        }
        let class_off = bin.len();
        for i in 0..n {
            bin.push(if i % 2 == 0 { 2 } else { 6 });
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let int_off = bin.len();
        for i in 0..n {
            bin.extend_from_slice(&(i as f32 * 0.5).to_le_bytes());
        }
        let json = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0, "translation": [0.0, 10.0, 0.0] }}],
  "buffers": [{{ "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": {pos_len} }},
    {{ "buffer": 0, "byteOffset": {color_off}, "byteLength": {color_len}, "byteStride": 4 }},
    {{ "buffer": 0, "byteOffset": {class_off}, "byteLength": {n} }},
    {{ "buffer": 0, "byteOffset": {int_off}, "byteLength": {int_len} }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": {n}, "type": "VEC3",
       "min": [0.0, 0.0, 0.0], "max": [{maxx}, 0.0, 0.0] }},
    {{ "bufferView": 1, "componentType": 5121, "normalized": true, "count": {n}, "type": "VEC3" }},
    {{ "bufferView": 2, "componentType": 5121, "count": {n}, "type": "SCALAR" }},
    {{ "bufferView": 3, "componentType": 5126, "count": {n}, "type": "SCALAR" }}
  ],
  "meshes": [{{ "primitives": [{{
    "mode": 0,
    "attributes": {{ "POSITION": 0, "COLOR_0": 1, "_CLASSIFICATION": 2, "_INTENSITY": 3 }}
  }}] }}]
}}"#,
            total = bin.len(),
            pos_len = n * 12,
            color_len = n * 4,
            int_len = n * 4,
            maxx = (n - 1) as f32,
        );
        glb(&json, &bin)
    }

    fn write_tmp(name: &str, bytes: &[u8]) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("r3d_pc_{}_{}", name, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{}.glb", name));
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    #[test]
    fn point_cloud_paths_are_recognised() {
        assert!(is_point_cloud_path(Path::new("scan.GLB")));
        assert!(is_point_cloud_path(Path::new("scan.gltf")));
        assert!(!is_point_cloud_path(Path::new("scan.vtk")));
        assert!(!is_point_cloud_path(Path::new("scan")));
    }

    #[test]
    fn inspect_counts_point_primitives() {
        let (dir, path) = write_tmp("inspect", &sample_glb(8));
        let content = inspect(&path).unwrap();
        assert!(content.has_points());
        assert!(!content.has_triangles());
        assert_eq!(content.points, 8);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn glb_points_become_dataset_rows_with_colors_and_classes() {
        let (dir, path) = write_tmp("load", &sample_glb(6));
        let ds = load_gltf(&path, &LoadOptions::default()).unwrap();
        assert_eq!(ds.n_rows(), 6);
        assert_eq!(
            ds.metadata.column_names,
            vec!["x", "y", "z", "r", "g", "b", "classification", "intensity"]
        );
        assert_eq!(ds.metadata.format, "glb");
        assert_eq!(ds.metadata.color_columns, Some([3, 4, 5]));
        // Node translation applied to every point.
        assert!((0..6).all(|r| (ds.value(r, 1) - 10.0).abs() < 1e-6));
        assert_eq!(ds.value(5, 0), 5.0);
        // Normalized u8 colors decoded to [0, 1].
        assert_eq!(ds.value(0, 3), 1.0);
        assert!((ds.value(3, 4) - 3.0 / 255.0).abs() < 1e-6);
        assert_eq!(ds.value(4, 7), 2.0);
        // Classification drives categorical labels.
        assert_eq!(ds.label_names, vec!["2 · ground", "6 · building"]);
        assert_eq!(ds.labels, vec![0, 1, 0, 1, 0, 1]);
        assert_eq!(ds.metadata.label_column.as_deref(), Some("classification"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn label_column_override_bands_any_column() {
        let (dir, path) = write_tmp("bands", &sample_glb(20));
        let opts = LoadOptions {
            max_rows: None,
            label_column: Some("Intensity".into()),
        };
        let ds = load_gltf(&path, &opts).unwrap();
        assert_eq!(ds.label_names.len(), FIELD_BANDS);
        assert_eq!(ds.metadata.label_column.as_deref(), Some("intensity"));

        let bad = LoadOptions {
            max_rows: None,
            label_column: Some("nope".into()),
        };
        let err = load_gltf(&path, &bad).unwrap_err().to_string();
        assert!(err.contains("nope") && err.contains("intensity"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn row_budget_strides_evenly_over_the_cloud() {
        let (dir, path) = write_tmp("stride", &sample_glb(100));
        let opts = LoadOptions {
            max_rows: Some(10),
            label_column: None,
        };
        let ds = load_gltf(&path, &opts).unwrap();
        assert_eq!(ds.n_rows(), 10);
        // Uniform stride keeps the far end of the cloud, not just its head.
        assert_eq!(ds.value(9, 0), 90.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn triangle_only_gltf_is_rejected_with_a_hint() {
        let mut bin = Vec::new();
        for c in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&c.to_le_bytes());
        }
        let json = r#"{
  "asset": { "version": "2.0" },
  "buffers": [{ "byteLength": 36 }],
  "bufferViews": [{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }],
  "accessors": [{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0.0,0.0,0.0], "max": [1.0,1.0,0.0] }],
  "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0 } }] }]
}"#;
        let (dir, path) = write_tmp("tri", &glb(json, &bin));
        let content = inspect(&path).unwrap();
        assert!(content.has_triangles() && !content.has_points());
        let err = load_gltf(&path, &LoadOptions::default()).unwrap_err();
        assert!(matches!(err, DatasetError::Unsupported(_)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn required_compression_extension_is_explained() {
        let json = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["KHR_draco_mesh_compression"],
  "extensionsRequired": ["KHR_draco_mesh_compression"]
}"#;
        let (dir, path) = write_tmp("draco", &glb(json, &[]));
        let err = inspect(&path).unwrap_err().to_string();
        assert!(err.contains("KHR_draco_mesh_compression"), "{err}");
        assert!(err.contains("re-export"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unknown_classes_get_a_generic_name() {
        let (labels, names) = class_labels(&[42.0, 1.0, 42.0]);
        assert_eq!(names, vec!["1 · unclassified", "class 42"]);
        assert_eq!(labels, vec![1, 0, 1]);
    }
}
