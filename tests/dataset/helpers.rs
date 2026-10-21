// Shared fixtures: hand-crafted NPY/NPZ/CSV/IDX files written to temp dirs.

use std::io::Write;
use std::path::{Path, PathBuf};

/// Serialize a little-endian f32 NPY (version 1.0) with the given shape.
pub fn npy_f32_bytes(shape: &[usize], data: &[f32]) -> Vec<u8> {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let shape_str = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );
    // Pad so that magic(6)+ver(2)+len(2)+header is a multiple of 64, ending in \n.
    let unpadded = 10 + header.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');

    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) {
    std::fs::write(path, npy_f32_bytes(shape, data)).unwrap();
}

/// Small labeled 2D dataset: 6 rows x 3 cols, labels [0,0,1,1,2,2].
pub fn sample_matrix() -> (Vec<f32>, Vec<f32>) {
    let features = vec![
        0.0, 0.1, 0.2, //
        0.1, 0.2, 0.3, //
        5.0, 5.1, 5.2, //
        5.1, 5.2, 5.3, //
        9.0, 9.1, 9.2, //
        9.1, 9.2, 9.3, //
    ];
    let labels = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    (features, labels)
}

/// Write features + sibling label file, return the features path.
pub fn write_sample_npy(dir: &Path) -> PathBuf {
    let (features, labels) = sample_matrix();
    let feat_path = dir.join("sample.npy");
    write_npy_f32(&feat_path, &[6, 3], &features);
    write_npy_f32(&dir.join("sample_labels.npy"), &[6], &labels);
    feat_path
}

/// Write an NPZ archive with X (features) and y (labels) entries.
pub fn write_sample_npz(dir: &Path) -> PathBuf {
    let (features, labels) = sample_matrix();
    let path = dir.join("sample.npz");
    let file = std::fs::File::create(&path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("X.npy", opts).unwrap();
    zip.write_all(&npy_f32_bytes(&[6, 3], &features)).unwrap();
    zip.start_file("y.npy", opts).unwrap();
    zip.write_all(&npy_f32_bytes(&[6], &labels)).unwrap();
    zip.finish().unwrap();
    path
}

pub fn write_sample_csv(dir: &Path) -> PathBuf {
    let path = dir.join("sample.csv");
    let mut text = String::from("a,b,c,label\n");
    let (features, labels) = sample_matrix();
    for r in 0..6 {
        text.push_str(&format!(
            "{},{},{},class_{}\n",
            features[r * 3],
            features[r * 3 + 1],
            features[r * 3 + 2],
            labels[r] as i32
        ));
    }
    std::fs::write(&path, text).unwrap();
    path
}

/// MNIST-style IDX pair: 4 "images" of 2x2 u8 pixels + label file.
pub fn write_sample_idx(dir: &Path) -> PathBuf {
    let images_path = dir.join("train-images-idx3.ubyte");
    let mut bytes = vec![0u8, 0, 0x08, 3];
    for dim in [4u32, 2, 2] {
        bytes.extend_from_slice(&dim.to_be_bytes());
    }
    bytes.extend_from_slice(&[
        10, 20, 30, 40, //
        50, 60, 70, 80, //
        90, 100, 110, 120, //
        130, 140, 150, 160,
    ]);
    std::fs::write(&images_path, bytes).unwrap();

    let labels_path = dir.join("train-labels-idx1.ubyte");
    let mut bytes = vec![0u8, 0, 0x08, 1];
    bytes.extend_from_slice(&4u32.to_be_bytes());
    bytes.extend_from_slice(&[7, 7, 3, 3]);
    std::fs::write(&labels_path, bytes).unwrap();

    images_path
}

/// Write a GLB point cloud (`mode = POINTS`) of `n` points on a helix with
/// u8-normalized RGB colors (`[i % 256, 0, 255 - i % 256]`, padded to 4
/// bytes) and a u8 `_CLASSIFICATION` cycling ground(2) / building(6) /
/// water(9). About 17 bytes per point, like a real scanned cloud.
pub fn write_point_cloud_glb(dir: &Path, name: &str, n: usize) -> PathBuf {
    let mut bin = Vec::with_capacity(n * 17);
    for i in 0..n {
        let t = i as f32 * 0.01;
        for c in [t.cos(), t.sin(), t * 0.1] {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    let color_off = bin.len();
    for i in 0..n {
        let v = (i % 256) as u8;
        bin.extend_from_slice(&[v, 0, 255 - v, 0]);
    }
    let class_off = bin.len();
    bin.extend((0..n).map(|i| [2u8, 6, 9][i % 3]));
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],
"nodes":[{{"mesh":0}}],"buffers":[{{"byteLength":{total}}}],
"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{pos_len}}},
{{"buffer":0,"byteOffset":{color_off},"byteLength":{color_len},"byteStride":4}},
{{"buffer":0,"byteOffset":{class_off},"byteLength":{n}}}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":{n},"type":"VEC3","min":[-1,-1,0],"max":[1,1,{maxz}]}},
{{"bufferView":1,"componentType":5121,"normalized":true,"count":{n},"type":"VEC3"}},
{{"bufferView":2,"componentType":5121,"count":{n},"type":"SCALAR"}}],
"meshes":[{{"primitives":[{{"mode":0,"attributes":{{"POSITION":0,"COLOR_0":1,"_CLASSIFICATION":2}}}}]}}]}}"#,
        total = bin.len(),
        pos_len = n * 12,
        color_len = n * 4,
        maxz = (n as f32 - 1.0) * 0.001,
    );
    let mut json = json.into_bytes();
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"glTF").unwrap();
    f.write_all(&2u32.to_le_bytes()).unwrap();
    f.write_all(&(total as u32).to_le_bytes()).unwrap();
    f.write_all(&(json.len() as u32).to_le_bytes()).unwrap();
    f.write_all(b"JSON").unwrap();
    f.write_all(&json).unwrap();
    f.write_all(&(bin.len() as u32).to_le_bytes()).unwrap();
    f.write_all(b"BIN\0").unwrap();
    f.write_all(&bin).unwrap();
    path
}
