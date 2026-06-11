// End-to-end smoke test: import -> index cache -> projection cache ->
// filter -> point cloud -> export, exactly the path the UI drives.

use std::collections::HashSet;

use rendering_3d::dataset::builtin::{generate, BuiltinDataset};
use rendering_3d::dataset::export::export_csv;
use rendering_3d::dataset::index::{apply_filter, FilterSpec, SearchQuery};
use rendering_3d::dataset::loader::{load, LoadOptions};
use rendering_3d::dataset::preprocessor::ProjectionMethod;
use rendering_3d::ui::{load_dataset_pipeline, prepare_dataset};
use rendering_3d::visualization::point_cloud::{build_instances, PointCloudSettings};

use crate::helpers;

#[test]
fn full_pipeline_from_file_with_caches() {
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join(".r3d_cache");
    let path = helpers::write_sample_npy(dir.path());

    let loaded =
        load_dataset_pipeline(&path, None, ProjectionMethod::Pca, Some(&cache_dir)).unwrap();
    assert_eq!(loaded.dataset.n_rows(), 6);
    assert_eq!(loaded.projection.points.len(), 6);
    assert_eq!(loaded.index.n_rows, 6);
    assert!(!loaded.projection.from_cache);

    // Cache artifacts exist on disk.
    let cache_entries: Vec<_> = std::fs::read_dir(&cache_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert!(cache_entries.iter().any(|n| n.ends_with(".proj")));
    assert!(cache_entries.iter().any(|n| n.ends_with(".index.json")));
    assert!(cache_entries.iter().any(|n| n.ends_with(".meta.json")));

    // Second import hits the projection cache.
    let again =
        load_dataset_pipeline(&path, None, ProjectionMethod::Pca, Some(&cache_dir)).unwrap();
    assert!(again.projection.from_cache);
}

#[test]
fn builtin_blobs_filter_render_export() {
    let dir = tempfile::tempdir().unwrap();
    let ds = generate(
        BuiltinDataset::Blobs {
            clusters: 4,
            points_per_cluster: 50,
            dims: 8,
        },
        42,
    );
    let loaded = prepare_dataset(ds, ProjectionMethod::Pca, None).unwrap();
    assert_eq!(loaded.dataset.n_rows(), 200);

    // Filter to two clusters.
    let spec = FilterSpec {
        enabled_labels: HashSet::from([0, 2]),
        query: SearchQuery::All,
    };
    let rows = apply_filter(&loaded.dataset, &loaded.index, &spec);
    assert_eq!(rows.len(), 100);

    // Point cloud only contains the filtered rows, colored per label.
    let result = build_instances(
        &loaded.projection.points,
        &loaded.dataset.labels,
        &rows,
        &PointCloudSettings::default(),
    );
    assert_eq!(result.rendered_points, 100);
    assert!(!result.downsampled);

    // Export the subset and verify consistency on re-import.
    let out = dir.path().join("subset.csv");
    assert_eq!(export_csv(&loaded.dataset, &rows, &out).unwrap(), 100);
    let reloaded = load(&out, &LoadOptions::default()).unwrap();
    assert_eq!(reloaded.n_rows(), 100);
    assert_eq!(reloaded.label_names, vec!["cluster_0", "cluster_2"]);
}

#[test]
fn builtin_generators_are_deterministic() {
    for name in BuiltinDataset::ALL_NAMES {
        let kind = BuiltinDataset::default_of(name).unwrap();
        let a = generate(kind, 9);
        let b = generate(kind, 9);
        assert_eq!(a.labels, b.labels, "{} labels must be deterministic", name);
        let (mut ra, mut rb) = (Vec::new(), Vec::new());
        a.row(0, &mut ra);
        b.row(0, &mut rb);
        assert_eq!(ra, rb, "{} features must be deterministic", name);
        assert!(a.n_rows() > 0 && a.n_cols() > 0);
        assert_eq!(a.metadata.labels.len(), a.label_names.len());
    }
}
