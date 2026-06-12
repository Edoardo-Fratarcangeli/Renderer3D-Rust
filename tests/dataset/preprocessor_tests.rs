use rendering_3d::dataset::builtin::{generate, BuiltinDataset, Rng};
use rendering_3d::dataset::loader::{load, LoadOptions};
use rendering_3d::dataset::metadata::DatasetMetadata;
use rendering_3d::dataset::preprocessor::{
    cache_file_path, project, ProjectionMethod, VIEW_HALF_EXTENT,
};
use rendering_3d::dataset::{Dataset, FeatureSource};

use crate::helpers;

fn in_memory_dataset(data: Vec<f32>, n_rows: usize, n_cols: usize) -> Dataset {
    let metadata = DatasetMetadata::new("test", "builtin", n_rows, n_cols);
    Dataset {
        metadata,
        source: FeatureSource::InMemory(data),
        labels: vec![0; n_rows],
        label_names: vec!["unlabeled".into()],
    }
}

#[test]
fn pca_recovers_dominant_direction() {
    // Points spread along one axis embedded in 5D with small noise: the
    // first principal component must capture far more variance than the rest.
    let n = 500;
    let d = 5;
    let mut rng = Rng::new(7);
    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        let t = (i as f32 / n as f32 - 0.5) * 100.0;
        data.push(t); // dominant direction = axis 0
        for _ in 1..d {
            data.push(rng.next_gauss() * 0.1);
        }
    }
    let ds = in_memory_dataset(data, n, d);
    let proj = project(&ds, ProjectionMethod::Pca, None).unwrap();
    assert_eq!(proj.points.len(), n);

    let var = |axis: usize| -> f32 {
        let mean: f32 = proj.points.iter().map(|p| p[axis]).sum::<f32>() / n as f32;
        proj.points
            .iter()
            .map(|p| (p[axis] - mean).powi(2))
            .sum::<f32>()
            / n as f32
    };
    assert!(
        var(0) > var(1) * 50.0 && var(0) > var(2) * 50.0,
        "PC1 variance {} must dominate PC2 {} / PC3 {}",
        var(0),
        var(1),
        var(2)
    );
}

#[test]
fn projection_is_normalized_into_view_cube() {
    let ds = generate(
        BuiltinDataset::Blobs {
            clusters: 3,
            points_per_cluster: 100,
            dims: 6,
        },
        1,
    );
    let proj = project(&ds, ProjectionMethod::Pca, None).unwrap();
    for p in &proj.points {
        for a in 0..3 {
            assert!(
                p[a].abs() <= VIEW_HALF_EXTENT + 1e-3,
                "point {:?} escapes the view cube",
                p
            );
        }
    }
}

#[test]
fn low_dimensional_data_falls_back_to_direct() {
    let ds = generate(BuiltinDataset::Spirals { points_per_arm: 50 }, 3);
    assert_eq!(ds.n_cols(), 3);
    let proj = project(&ds, ProjectionMethod::Pca, None).unwrap();
    assert_eq!(proj.points.len(), ds.n_rows());
}

#[test]
fn projection_cache_roundtrip_for_file_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    let path = helpers::write_sample_npy(dir.path());
    let ds = load(&path, &LoadOptions::default()).unwrap();

    let first = project(&ds, ProjectionMethod::Pca, Some(&cache_dir)).unwrap();
    assert!(!first.from_cache, "first run must compute");
    let cache_file = cache_file_path(&cache_dir, &ds, ProjectionMethod::Pca);
    assert!(cache_file.is_file(), "cache file must be written");

    let second = project(&ds, ProjectionMethod::Pca, Some(&cache_dir)).unwrap();
    assert!(second.from_cache, "second run must hit the cache");
    assert_eq!(first.points, second.points);
}

#[test]
fn cache_key_distinguishes_methods() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_sample_npy(dir.path());
    let ds = load(&path, &LoadOptions::default()).unwrap();
    let a = cache_file_path(dir.path(), &ds, ProjectionMethod::Pca);
    let b = cache_file_path(dir.path(), &ds, ProjectionMethod::Direct);
    assert_ne!(a, b);
}
