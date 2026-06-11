use rendering_3d::dataset::metadata::DatasetMetadata;

#[test]
fn metadata_json_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let mut meta = DatasetMetadata::new("iris", "csv", 150, 4);
    meta.source_path = "/data/iris.csv".into();
    meta.label_column = Some("species".into());
    meta.set_label_stats(
        &[0, 0, 1, 2, 2, 2],
        &["setosa".into(), "versicolor".into(), "virginica".into()],
    );

    let path = dir.path().join("meta").join("iris.meta.json");
    meta.save_json(&path).unwrap();
    let reloaded = DatasetMetadata::load_json(&path).unwrap();
    assert_eq!(meta, reloaded);
    assert_eq!(reloaded.labels[2].name, "virginica");
    assert_eq!(reloaded.labels[2].count, 3);
}

#[test]
fn label_stats_count_per_label() {
    let mut meta = DatasetMetadata::new("t", "builtin", 5, 2);
    meta.set_label_stats(&[1, 1, 1, 0, 1], &["a".into(), "b".into()]);
    assert_eq!(meta.labels[0].count, 1);
    assert_eq!(meta.labels[1].count, 4);
}

#[test]
fn default_column_names_are_generated() {
    let meta = DatasetMetadata::new("t", "npy", 10, 3);
    assert_eq!(meta.column_names, vec!["f0", "f1", "f2"]);
}
