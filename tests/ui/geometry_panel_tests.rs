// Headless egui tests for the geometry import window.

use rendering_3d::ui::geometry_panel::GeometryView;
use rendering_3d::ui::StatusKind;

fn run_frame(ctx: &egui::Context, view: &mut GeometryView) -> Option<[f32; 3]> {
    let mut focus = None;
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(input, |ctx| {
        focus = view.show(ctx);
    });
    focus
}

#[test]
fn window_renders_empty_and_with_layers() {
    let ctx = egui::Context::default();
    let mut view = GeometryView::new();
    assert_eq!(run_frame(&ctx, &mut view), None); // hidden

    view.show_window = true;
    run_frame(&ctx, &mut view); // empty state

    view.paste_text = "cube 0 0 0 2\nsphere 1 1 1 0.5\npoint 2 2 2".into();
    view.import_pasted();
    assert_eq!(view.layers.len(), 1);
    run_frame(&ctx, &mut view); // layer list + status line
    assert_eq!(view.total_records(), 3);
}

#[test]
fn file_import_worker_installs_layer() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("geo.csv");
    std::fs::write(&csv, "shape,x,y,z\ncube,1,2,3\n").unwrap();

    let ctx = egui::Context::default();
    let mut view = GeometryView::new();
    view.show_window = true;
    view.path_text = csv.to_string_lossy().to_string();
    view.import_file();
    assert!(view.is_loading());

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while view.is_loading() {
        assert!(std::time::Instant::now() < deadline, "worker stuck");
        run_frame(&ctx, &mut view);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(view.layers.len(), 1);
    assert_eq!(view.status.as_ref().unwrap().kind, StatusKind::Success);

    // Failure path: missing file.
    view.path_text = "/missing/file.csv".into();
    view.import_file();
    while view.is_loading() {
        run_frame(&ctx, &mut view);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(view.layers.len(), 1);
    assert_eq!(view.status.as_ref().unwrap().kind, StatusKind::Error);
}

#[test]
fn dirty_flag_drives_buffer_rebuilds() {
    let mut view = GeometryView::new();
    view.paste_text = "cube 0 0 0".into();
    view.import_pasted();
    assert!(view.take_render_dirty());
    assert!(!view.take_render_dirty());
    view.layers[0].visible = false;
    // Visibility is toggled through the UI which marks dirty; emulate the
    // pipeline by rebuilding batches directly.
    assert!(view.build_geometry_batches().is_empty());
}
