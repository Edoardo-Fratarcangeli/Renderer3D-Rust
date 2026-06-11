// Lightweight benchmarks, ignored by default so CI stays fast.
// Run with: cargo test --release --test dataset -- --ignored --nocapture

use std::time::Instant;

use rendering_3d::dataset::builtin::{generate, BuiltinDataset};
use rendering_3d::dataset::index::DatasetIndex;
use rendering_3d::dataset::loader::{load, LoadOptions};
use rendering_3d::dataset::preprocessor::{project, ProjectionMethod};
use rendering_3d::visualization::point_cloud::{build_instances, PointCloudSettings};

use crate::helpers;

#[test]
#[ignore]
fn bench_pca_projection_50k_x_32() {
    let ds = generate(
        BuiltinDataset::Blobs {
            clusters: 10,
            points_per_cluster: 5_000,
            dims: 32,
        },
        1,
    );
    let t = Instant::now();
    let proj = project(&ds, ProjectionMethod::Pca, None).unwrap();
    println!(
        "PCA 50k x 32 -> 3D: {:?} ({} points)",
        t.elapsed(),
        proj.points.len()
    );
    assert_eq!(proj.points.len(), 50_000);
}

#[test]
#[ignore]
fn bench_mmap_npy_load_and_index_100k() {
    let dir = tempfile::tempdir().unwrap();
    let n = 100_000usize;
    let d = 16usize;
    let data: Vec<f32> = (0..n * d).map(|i| (i % 977) as f32).collect();
    let path = dir.path().join("big.npy");
    helpers::write_npy_f32(&path, &[n, d], &data);

    let t = Instant::now();
    let ds = load(&path, &LoadOptions::default()).unwrap();
    println!("mmap load {}x{}: {:?}", n, d, t.elapsed());
    assert!(ds.source.is_memory_mapped());

    let t = Instant::now();
    let idx = DatasetIndex::build(&ds.labels, ds.label_names.len());
    println!("index build: {:?}", t.elapsed());
    assert_eq!(idx.n_rows, n);
}

#[test]
#[ignore]
fn bench_instance_build_200k() {
    let n = 200_000usize;
    let points = vec![[1.0f32, 2.0, 3.0]; n];
    let labels: Vec<u32> = (0..n).map(|i| (i % 10) as u32).collect();
    let visible: Vec<u32> = (0..n as u32).collect();
    let t = Instant::now();
    let result = build_instances(&points, &labels, &visible, &PointCloudSettings::default());
    println!(
        "instance build {} points: {:?}",
        result.rendered_points,
        t.elapsed()
    );
}
