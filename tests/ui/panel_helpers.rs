//! Headless egui coverage for the reusable dataset helper panels
//! (`search_panel`, `dataset_table`) that are no longer wired to a tab but
//! remain part of the public API.

use rendering_3d::dataset::builtin::{generate, BuiltinDataset};
use rendering_3d::dataset::preprocessor::ProjectionMethod;
use rendering_3d::ui::dataset_table::{row_text, show as table_show, MAX_TABLE_COLS};
use rendering_3d::ui::search_panel;
use rendering_3d::ui::{prepare_dataset, LoadedDataset};

fn ctx() -> egui::Context {
    egui::Context::default()
}

fn run<R>(ctx: &egui::Context, body: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000.0, 700.0),
        )),
        ..Default::default()
    };
    let mut out = None;
    let _ = ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            out = Some(body(ui));
        });
    });
    out.unwrap()
}

fn sample() -> LoadedDataset {
    prepare_dataset(
        generate(
            BuiltinDataset::Blobs {
                clusters: 2,
                points_per_cluster: 6,
                dims: 4,
            },
            3,
        ),
        ProjectionMethod::Direct,
        None,
    )
    .unwrap()
}

#[test]
fn search_panel_reports_typing_and_clear() {
    let ctx = ctx();
    let mut text = String::new();
    // Rendering an empty box reports no change and disables clear.
    let changed = run(&ctx, |ui| search_panel::show(ui, &mut text, &None));
    assert!(!changed);

    // With text and an error, the panel still renders (error banner path).
    text = "row:1".into();
    let err = Some("bad query".to_string());
    let _ = run(&ctx, |ui| search_panel::show(ui, &mut text, &err));
}

#[test]
fn dataset_table_renders_and_row_text_is_consistent() {
    let ctx = ctx();
    let loaded = sample();
    let visible: Vec<u32> = (0..loaded.dataset.n_rows() as u32).collect();
    let mut highlighted = None;

    // The table renders without panicking over the full visible set.
    let _ = run(&ctx, |ui| {
        table_show(
            ui,
            &loaded.dataset,
            &loaded.projection.points,
            &visible,
            &mut highlighted,
        )
    });

    // row_text is stable and contains the row index and its label.
    let n_cols = loaded.dataset.n_cols().min(MAX_TABLE_COLS);
    let text = row_text(&loaded.dataset, 0, n_cols, false);
    assert!(text.contains(loaded.dataset.label_name(0)));
    assert!(text.trim_start().starts_with('0'));
}

#[test]
fn dataset_table_truncates_wide_datasets() {
    // More feature columns than the cap → ellipsis marker.
    let loaded = prepare_dataset(
        generate(
            BuiltinDataset::Blobs {
                clusters: 2,
                points_per_cluster: 4,
                dims: MAX_TABLE_COLS + 5,
            },
            1,
        ),
        ProjectionMethod::Direct,
        None,
    )
    .unwrap();
    let text = row_text(&loaded.dataset, 0, MAX_TABLE_COLS, true);
    assert!(text.contains('…'));
}
