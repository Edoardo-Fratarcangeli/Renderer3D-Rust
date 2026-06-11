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
