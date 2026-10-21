// glTF / GLB point clouds through the whole dataset stack: loader ->
// pipeline (direct projection, native colors) -> DatasetView (worker,
// color modes, filters, picking) -> point-cloud instances -> CSV export.

use std::collections::HashSet;

use rendering_3d::dataset::export::export_csv;
use rendering_3d::dataset::loader::{is_spatial_path, load, LoadOptions};
use rendering_3d::dataset::preprocessor::{ProjectionMethod, ProjectionSpec};
use rendering_3d::ui::{load_dataset_pipeline_opts, DatasetTab, DatasetView};
use rendering_3d::visualization::color_mapper::ColorMode;

use crate::helpers;

#[test]
fn glb_point_cloud_loads_through_the_generic_loader() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_point_cloud_glb(dir.path(), "scan.glb", 300);
    assert!(is_spatial_path(&path));

    let ds = load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(ds.n_rows(), 300);
    assert_eq!(ds.metadata.color_columns, Some([3, 4, 5]));
    assert_eq!(
        ds.label_names,
        vec!["2 · ground", "6 · building", "9 · water"]
    );
    assert!(ds.metadata.labels.iter().all(|s| s.count == 100));
}

#[test]
fn pipeline_keeps_geometry_and_native_colors() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_point_cloud_glb(dir.path(), "scan.glb", 512);
    let spec = ProjectionSpec::full(ProjectionMethod::Direct);
    let loaded = load_dataset_pipeline_opts(&path, &LoadOptions::default(), &spec, None).unwrap();

    let colors = loaded.colors.as_ref().expect("native colors");
    assert_eq!(colors.len(), 512);
    assert_eq!(colors[0], [0.0, 0.0, 1.0]);
    assert_eq!(colors[255], [1.0, 0.0, 0.0]);
    // Direct projection keeps the helix ordering along z.
    let z: Vec<f32> = loaded.projection.points.iter().map(|p| p[2]).collect();
    assert!(z.windows(2).all(|w| w[0] <= w[1]));
}

fn run_frame(ctx: &egui::Context, view: &mut DatasetView) {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(input, |ctx| {
        view.show(ctx);
    });
}

#[test]
fn handed_over_point_cloud_renders_in_native_colors_and_filters() {
    let dir = tempfile::tempdir().unwrap();
    let path = helpers::write_point_cloud_glb(dir.path(), "handover.glb", 90);
    let ctx = egui::Context::default();
    let mut view = DatasetView::new();

    view.import_point_cloud(path.clone());
    assert!(view.show_window);
    assert_eq!(view.tab, DatasetTab::Import);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while view.is_loading() {
        assert!(
            std::time::Instant::now() < deadline,
            "import did not finish"
        );
        run_frame(&ctx, &mut view);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let loaded = view.loaded.as_ref().expect("point cloud installed");
    assert_eq!(loaded.dataset.n_rows(), 90);
    assert_eq!(view.settings.color_mode, ColorMode::Native);
    let native = loaded.colors.clone().unwrap();

    // Instances carry the per-point colors of the file.
    let result = view.build_point_cloud();
    assert_eq!(result.rendered_points, 90);
    let mut drawn: Vec<[f32; 3]> = result
        .batches
        .iter()
        .flat_map(|b| {
            b.instances
                .iter()
                .map(|i| [i.color[0], i.color[1], i.color[2]])
        })
        .collect();
    let mut expected = native.clone();
    let key = |c: &[f32; 3]| (c[0] * 1e4) as i64 * 100_000 + (c[2] * 1e4) as i64;
    drawn.sort_by_key(key);
    expected.sort_by_key(key);
    assert_eq!(drawn, expected);

    // Every View-tab panel still renders with the native mode active.
    view.tab = DatasetTab::View;
    run_frame(&ctx, &mut view);

    // The classification labels filter the cloud like any dataset.
    view.enabled_labels = HashSet::from([1]);
    view.recompute_visible();
    assert_eq!(view.visible_rows.len(), 30);

    // Asking for bands switches away from the native colors.
    view.recolor_by_column(0, 4);
    assert_eq!(view.settings.color_mode, ColorMode::ByLabel);

    // Export keeps the color columns.
    let out = dir.path().join("subset.csv");
    let loaded = view.loaded.as_ref().unwrap();
    export_csv(&loaded.dataset, &view.visible_rows, &out).unwrap();
    let header = std::fs::read_to_string(&out).unwrap();
    assert!(header.lines().next().unwrap().contains("r,g,b"), "{header}");
}
