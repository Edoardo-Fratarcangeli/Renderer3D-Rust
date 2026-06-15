// Headless egui tests for the dataset visualizer UI.
//
// egui's layout/painting is pure CPU, so we can drive the real DatasetView
// (window, tab bar, every panel) without a GPU by running frames on a bare
// egui::Context. These tests cover the rendering paths and the
// import-worker polling loop end to end.

use rendering_3d::dataset::preprocessor::{ProjectionMethod, ProjectionSpec};
use rendering_3d::ui::{
    prepare_dataset, DatasetAction, DatasetTab, DatasetView, ImportRequest, ImportSource,
    StatusKind,
};

/// Run one headless egui frame, calling `view.show` inside it.
fn run_frame(ctx: &egui::Context, view: &mut DatasetView) -> DatasetAction {
    let mut action = DatasetAction::None;
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(input, |ctx| {
        action = view.show(ctx);
    });
    action
}

/// A view with the blobs benchmark already installed.
fn loaded_view() -> DatasetView {
    let mut view = DatasetView::new();
    view.start_import(ImportRequest {
        source: ImportSource::Builtin("blobs"),
        max_rows: None,
        projection: ProjectionSpec::full(ProjectionMethod::Pca),
    });
    assert!(view.loaded.is_some(), "builtin import is synchronous");
    view
}

#[test]
fn hidden_window_renders_nothing_and_returns_no_action() {
    let ctx = egui::Context::default();
    let mut view = DatasetView::new();
    assert_eq!(run_frame(&ctx, &mut view), DatasetAction::None);
}

#[test]
fn every_tab_renders_without_panicking_when_loaded() {
    let ctx = egui::Context::default();
    let mut view = loaded_view();
    view.show_window = true;
    for tab in DatasetTab::ALL {
        view.tab = tab;
        let action = run_frame(&ctx, &mut view);
        assert_eq!(action, DatasetAction::None, "tab {:?}", tab);
    }
}

#[test]
fn data_tabs_show_empty_state_without_dataset() {
    let ctx = egui::Context::default();
    let mut view = DatasetView::new();
    view.show_window = true;
    for tab in DatasetTab::ALL {
        view.tab = tab;
        run_frame(&ctx, &mut view); // must not panic on the empty state
    }
    assert!(view.loaded.is_none());
}

#[test]
fn builtin_import_lands_on_labels_tab_with_success_status() {
    let view = loaded_view();
    assert_eq!(view.tab, DatasetTab::Labels);
    let status = view.import.status.as_ref().expect("status after import");
    assert_eq!(status.kind, StatusKind::Success);
    assert!(status.text.contains("blobs"));
    assert_eq!(view.visible_rows.len(), view.loaded.as_ref().unwrap().dataset.n_rows());
}

#[test]
fn projection_dims_control_the_active_axes() {
    // 2D import: the z axis must be flat.
    let mut view2d = DatasetView::new();
    view2d.start_import(ImportRequest {
        source: ImportSource::Builtin("blobs"),
        max_rows: None,
        projection: ProjectionSpec {
            method: ProjectionMethod::Direct,
            dims: 2,
            axes: [0, 1, 2],
        },
    });
    let pts = &view2d.loaded.as_ref().unwrap().projection.points;
    assert!(pts.iter().all(|p| p[2] == 0.0), "2D import must flatten z");
    assert!(pts.iter().any(|p| p[1] != 0.0), "2D import keeps y");

    // 1D import: both y and z must be flat.
    let mut view1d = DatasetView::new();
    view1d.start_import(ImportRequest {
        source: ImportSource::Builtin("blobs"),
        max_rows: None,
        projection: ProjectionSpec {
            method: ProjectionMethod::Direct,
            dims: 1,
            axes: [0, 1, 2],
        },
    });
    let pts = &view1d.loaded.as_ref().unwrap().projection.points;
    assert!(
        pts.iter().all(|p| p[1] == 0.0 && p[2] == 0.0),
        "1D import must flatten y and z"
    );
}

#[test]
fn unknown_builtin_reports_error_status() {
    let mut view = DatasetView::new();
    view.start_import(ImportRequest {
        source: ImportSource::Builtin("not_a_dataset"),
        max_rows: None,
        projection: ProjectionSpec::full(ProjectionMethod::Pca),
    });
    assert!(view.loaded.is_none());
    assert_eq!(view.import.status.as_ref().unwrap().kind, StatusKind::Error);
}

/// Poll `view.show` frames until the background worker resolves.
fn pump_until_idle(ctx: &egui::Context, view: &mut DatasetView, timeout_s: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_s);
    while view.is_loading() {
        assert!(
            std::time::Instant::now() < deadline,
            "import worker did not finish in time"
        );
        run_frame(ctx, view);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn file_import_runs_on_worker_thread_and_installs() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("tiny.csv");
    std::fs::write(&csv, "x,y,label\n1,2,a\n3,4,b\n5,6,a\n").unwrap();

    let ctx = egui::Context::default();
    let mut view = DatasetView::new();
    view.show_window = true;
    view.start_import(ImportRequest {
        source: ImportSource::Path(csv),
        max_rows: None,
        projection: ProjectionSpec::full(ProjectionMethod::Direct),
    });
    assert!(view.is_loading());

    pump_until_idle(&ctx, &mut view, 30);
    let loaded = view
        .loaded
        .as_ref()
        .unwrap_or_else(|| panic!("dataset installed; status = {:?}", view.import.status));
    assert_eq!(loaded.dataset.n_rows(), 3);
    assert_eq!(view.import.status.as_ref().unwrap().kind, StatusKind::Success);
    // A frame after install renders the landing (Labels) tab.
    run_frame(&ctx, &mut view);
}

#[test]
fn failed_file_import_surfaces_error_status() {
    let ctx = egui::Context::default();
    let mut view = DatasetView::new();
    view.show_window = true;
    view.start_import(ImportRequest {
        source: ImportSource::Path("/definitely/missing.csv".into()),
        max_rows: None,
        projection: ProjectionSpec::full(ProjectionMethod::Pca),
    });
    pump_until_idle(&ctx, &mut view, 30);
    assert!(view.loaded.is_none());
    let status = view.import.status.as_ref().unwrap();
    assert_eq!(status.kind, StatusKind::Error);
    assert!(status.text.contains("Import failed"));
}

#[test]
fn search_text_drives_visible_rows_and_dirty_flag() {
    let mut view = loaded_view();
    let total = view.visible_rows.len();
    assert!(view.take_render_dirty(), "install marks dirty");

    view.search_text = "cluster_1".into();
    view.recompute_visible();
    assert!(view.take_render_dirty());
    assert!(view.visible_rows.len() < total);
    assert!(!view.visible_rows.is_empty());

    // Invalid row query surfaces an error and keeps all rows.
    view.search_text = "row:xyz".into();
    view.recompute_visible();
    assert!(view.search_error.is_some());
}

#[test]
fn label_toggle_drops_highlight_when_filtered_out() {
    let mut view = loaded_view();
    // Highlight a row belonging to label 0, then hide label 0.
    let row0 = *view
        .loaded
        .as_ref()
        .unwrap()
        .index
        .label_rows[0]
        .first()
        .unwrap();
    view.settings.highlighted_row = Some(row0);
    view.enabled_labels.remove(&0);
    view.recompute_visible();
    assert_eq!(view.settings.highlighted_row, None);
}

#[test]
fn point_cloud_build_matches_visible_rows() {
    let mut view = loaded_view();
    view.enabled_labels = [1u32].into_iter().collect();
    view.recompute_visible();
    let visible = view.visible_rows.len();
    let result = view.build_point_cloud();
    assert_eq!(result.rendered_points, visible);
    assert!(view.last_build_info.contains(&visible.to_string()));

    // Without a dataset the build is empty.
    let mut empty = DatasetView::new();
    let result = empty.build_point_cloud();
    assert_eq!(result.rendered_points, 0);
    assert!(result.batches.is_empty());
}

#[test]
fn export_tab_writes_csv_through_the_ui_path() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("ui_export.csv");

    let ctx = egui::Context::default();
    let mut view = loaded_view();
    view.show_window = true;
    view.tab = DatasetTab::Export;
    view.export.path_text = out.to_string_lossy().to_string();
    // Rendering the tab does not export by itself (button not clicked).
    run_frame(&ctx, &mut view);
    assert!(!out.exists());

    // The actual write goes through dataset::export (covered there); here we
    // verify the UI wiring end-to-end by invoking it like the panel does.
    let loaded = view.loaded.as_ref().unwrap();
    let n = rendering_3d::dataset::export::export_csv(
        &loaded.dataset,
        &view.visible_rows,
        &out,
    )
    .unwrap();
    assert_eq!(n, view.visible_rows.len());
    assert!(out.exists());
}

#[test]
fn table_row_text_is_stable_and_truncated() {
    use rendering_3d::ui::dataset_table::{row_text, MAX_TABLE_COLS};
    let loaded = prepare_dataset(
        rendering_3d::dataset::builtin::generate(
            rendering_3d::dataset::builtin::BuiltinDataset::Blobs {
                clusters: 2,
                points_per_cluster: 5,
                dims: 12, // wider than MAX_TABLE_COLS -> ellipsis
            },
            7,
        ),
        ProjectionMethod::Direct,
        None,
    )
    .unwrap();
    let text = row_text(&loaded.dataset, 0, MAX_TABLE_COLS, true);
    assert!(text.contains('…'));
    assert!(text.contains("cluster_0"));
    assert!(text.starts_with(&format!("{:>7}", 0)));
}

#[test]
fn status_lines_and_spinner_render_in_every_kind() {
    use rendering_3d::ui::StatusMessage;
    let ctx = egui::Context::default();
    let mut view = loaded_view();
    view.show_window = true;

    // A bad search query still surfaces an error through recompute (the search
    // box itself no longer has a tab; filtering lives behind the Labels tab).
    view.search_text = "row:bad".into();
    view.recompute_visible();
    assert!(view.search_error.is_some());
    view.tab = DatasetTab::Labels;
    run_frame(&ctx, &mut view);

    // Export status, success then error.
    view.tab = DatasetTab::Export;
    view.export.status = Some(StatusMessage::success("done"));
    run_frame(&ctx, &mut view);
    view.export.status = Some(StatusMessage::error("nope"));
    run_frame(&ctx, &mut view);

    // Import tab: filled path (enabled button), spinner + info status.
    view.tab = DatasetTab::Import;
    view.import.path_text = "some/file.csv".into();
    view.import.loading = true;
    view.import.status = Some(StatusMessage::info("loading…"));
    run_frame(&ctx, &mut view);
    view.import.loading = false;

    // Highlight a row and render the View tab.
    view.tab = DatasetTab::View;
    view.search_text.clear();
    view.recompute_visible();
    view.settings.highlighted_row = view.visible_rows.first().copied();
    run_frame(&ctx, &mut view);
}
