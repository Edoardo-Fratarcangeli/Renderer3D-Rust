use rendering_3d::dataset::loader::{load, LoadOptions};

use crate::helpers;

#[test]
fn npy_loads_memory_mapped_with_sibling_labels() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_sample_npy(dir.path());

    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.n_rows(), 6);
    assert_eq!(ds.n_cols(), 3);
    assert!(ds.source.is_memory_mapped(), "NPY must be memory mapped");
    assert!(ds.metadata.memory_mapped);
    assert_eq!(ds.label_names, vec!["0", "1", "2"]);
    assert_eq!(ds.labels, vec![0, 0, 1, 1, 2, 2]);

    let mut row = Vec::new();
    ds.row(3, &mut row);
    assert_eq!(row, vec![5.1, 5.2, 5.3]);
    assert!((ds.value(5, 2) - 9.3).abs() < 1e-6);
    assert_eq!(ds.label_name(4), "2");
}

#[test]
fn npy_respects_max_rows_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_sample_npy(dir.path());
    let opts = LoadOptions {
        max_rows: Some(4),
        label_column: None,
    };
    let ds = load(&path, &opts).unwrap();
    assert_eq!(ds.n_rows(), 4);
    assert_eq!(ds.labels.len(), 4);
}

#[test]
fn npy_rejects_garbage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.npy");
    std::fs::write(&path, b"this is not numpy").unwrap();
    assert!(load(&path, &LoadOptions::default()).is_err());
}

#[test]
fn npz_loads_features_and_labels() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_sample_npz(dir.path());

    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.n_rows(), 6);
    assert_eq!(ds.n_cols(), 3);
    assert_eq!(ds.metadata.format, "npz");
    assert_eq!(ds.metadata.label_column.as_deref(), Some("y"));
    assert_eq!(ds.labels, vec![0, 0, 1, 1, 2, 2]);

    let mut row = Vec::new();
    ds.row(0, &mut row);
    assert!((row[1] - 0.1).abs() < 1e-6);
}

#[test]
fn csv_streams_rows_and_detects_label_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_sample_csv(dir.path());

    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.n_rows(), 6);
    assert_eq!(ds.n_cols(), 3);
    assert_eq!(ds.metadata.column_names, vec!["a", "b", "c"]);
    assert_eq!(ds.metadata.label_column.as_deref(), Some("label"));
    assert_eq!(
        ds.label_names,
        vec!["class_0", "class_1", "class_2"]
    );
    let stats = &ds.metadata.labels;
    assert_eq!(stats.len(), 3);
    assert!(stats.iter().all(|s| s.count == 2));
}

#[test]
fn excel_loads_features_and_detects_label_column() {
    use rust_xlsxwriter::Workbook;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.xlsx");

    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    for (col, header) in ["a", "b", "label"].iter().enumerate() {
        sheet.write_string(0, col as u16, *header).unwrap();
    }
    let rows: [(f64, f64, &str); 4] = [
        (1.0, 2.0, "x"),
        (3.0, 4.0, "y"),
        (5.0, 6.0, "x"),
        (7.0, 8.0, "y"),
    ];
    for (i, (a, b, label)) in rows.iter().enumerate() {
        let r = (i + 1) as u32;
        sheet.write_number(r, 0, *a).unwrap();
        sheet.write_number(r, 1, *b).unwrap();
        sheet.write_string(r, 2, *label).unwrap();
    }
    workbook.save(&path).unwrap();

    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.n_rows(), 4);
    assert_eq!(ds.n_cols(), 2);
    assert_eq!(ds.metadata.column_names, vec!["a", "b"]);
    assert_eq!(ds.metadata.format, "excel");
    assert_eq!(ds.metadata.label_column.as_deref(), Some("label"));
    assert_eq!(ds.label_names, vec!["x", "y"]);
    assert_eq!(ds.value(0, 0), 1.0);
    assert_eq!(ds.value(3, 1), 8.0);
}

#[test]
fn csv_reports_non_numeric_cells() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.csv");
    std::fs::write(&path, "a,b\n1.0,oops\n").unwrap();
    let err = load(&path, &LoadOptions::default()).unwrap_err();
    assert!(err.to_string().contains("non numeric"));
}

#[test]
fn idx_loads_mnist_style_pair_memory_mapped() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_sample_idx(dir.path());

    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.n_rows(), 4);
    assert_eq!(ds.n_cols(), 4); // 2x2 flattened
    assert!(ds.source.is_memory_mapped());
    assert_eq!(ds.label_names, vec!["3", "7"]);
    assert_eq!(ds.labels, vec![1, 1, 0, 0]);

    let mut row = Vec::new();
    ds.row(1, &mut row);
    assert_eq!(row, vec![50.0, 60.0, 70.0, 80.0]);
}

#[test]
fn unknown_extension_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.xyz");
    std::fs::write(&path, b"whatever").unwrap();
    let err = load(&path, &LoadOptions::default()).unwrap_err();
    assert!(err.to_string().contains("unknown dataset extension"));
}

#[test]
fn csv_label_named_y_does_not_shadow_label_column() {
    // Regression: a coordinate column named "y" must not be auto-picked as
    // the label when a strong name like "label" is present.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("xy.csv");
    std::fs::write(&path, "x,y,label\n1,2,a\n3,4,b\n").unwrap();
    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.metadata.label_column.as_deref(), Some("label"));
    assert_eq!(ds.metadata.column_names, vec!["x", "y"]);
    assert_eq!(ds.label_names, vec!["a", "b"]);
}

#[test]
fn csv_explicit_label_column_overrides_detection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("explicit.csv");
    std::fs::write(&path, "v,grp\n1.5,one\n2.5,two\n").unwrap();
    let opts = LoadOptions {
        max_rows: None,
        label_column: Some("grp".into()),
    };
    let ds = load(&path, &opts).unwrap();
    assert_eq!(ds.metadata.label_column.as_deref(), Some("grp"));
    assert_eq!(ds.label_names, vec!["one", "two"]);

    let missing = LoadOptions {
        max_rows: None,
        label_column: Some("nope".into()),
    };
    assert!(load(&path, &missing).is_err());
}

#[test]
fn csv_without_label_column_is_unlabeled() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nolabel.csv");
    std::fs::write(&path, "a,b\n1,2\n3,4\n").unwrap();
    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.label_names, vec!["unlabeled"]);
    assert_eq!(ds.labels, vec![0, 0]);
    assert_eq!(ds.metadata.label_column, None);
}

#[test]
fn csv_respects_max_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_sample_csv(dir.path());
    let opts = LoadOptions {
        max_rows: Some(2),
        label_column: None,
    };
    let ds = load(&path, &opts).unwrap();
    assert_eq!(ds.n_rows(), 2);
}

#[test]
fn npy_f64_payload_decodes_to_f32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("doubles.npy");
    // Hand-craft a v1 header with <f8 dtype.
    let mut header =
        String::from("{'descr': '<f8', 'fortran_order': False, 'shape': (2, 2), }");
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in [1.5f64, -2.0, 0.25, 1e6] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&path, bytes).unwrap();

    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert!(ds.source.is_memory_mapped());
    let mut row = Vec::new();
    ds.row(0, &mut row);
    assert_eq!(row, vec![1.5, -2.0]);
    assert_eq!(ds.value(1, 1), 1e6);
}

#[test]
fn npy_fortran_order_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fortran.npy");
    let mut header =
        String::from("{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2), }");
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0u8; 16]);
    std::fs::write(&path, bytes).unwrap();
    let err = load(&path, &LoadOptions::default()).unwrap_err();
    assert!(err.to_string().contains("fortran"));
}

#[test]
fn npz_without_label_entry_is_unlabeled() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nolabels.npz");
    let file = std::fs::File::create(&path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    use std::io::Write as _;
    zip.start_file("data.npy", opts).unwrap();
    zip.write_all(&helpers::npy_f32_bytes(&[3, 2], &[1., 2., 3., 4., 5., 6.]))
        .unwrap();
    zip.finish().unwrap();

    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.n_rows(), 3);
    assert_eq!(ds.label_names, vec!["unlabeled"]);
}

#[test]
fn npz_respects_max_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_sample_npz(dir.path());
    let opts = LoadOptions {
        max_rows: Some(3),
        label_column: None,
    };
    let ds = load(&path, &opts).unwrap();
    assert_eq!(ds.n_rows(), 3);
    assert_eq!(ds.labels.len(), 3);
}

#[test]
fn idx_without_sibling_labels_is_unlabeled() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.idx");
    let mut bytes = vec![0u8, 0, 0x08, 2];
    bytes.extend_from_slice(&2u32.to_be_bytes());
    bytes.extend_from_slice(&3u32.to_be_bytes());
    bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
    std::fs::write(&path, bytes).unwrap();

    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.n_rows(), 2);
    assert_eq!(ds.n_cols(), 3);
    assert_eq!(ds.label_names, vec!["unlabeled"]);
}

#[cfg(not(feature = "parquet-support"))]
#[test]
fn parquet_without_feature_gives_actionable_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.parquet");
    std::fs::write(&path, b"PAR1").unwrap();
    let err = load(&path, &LoadOptions::default()).unwrap_err();
    assert!(err.to_string().contains("parquet-support"));
}
