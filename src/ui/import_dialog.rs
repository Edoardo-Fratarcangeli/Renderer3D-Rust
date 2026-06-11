// Import panel: file path entry, row cap, projection method and one-click
// builtin benchmark datasets.

use super::{ImportRequest, ImportSource, ImportState};
use crate::dataset::builtin::BuiltinDataset;
use crate::dataset::preprocessor::ProjectionMethod;

pub fn show(ui: &mut egui::Ui, state: &mut ImportState) -> Option<ImportRequest> {
    let mut request = None;

    ui.horizontal(|ui| {
        ui.label("File:");
        ui.add(
            egui::TextEdit::singleline(&mut state.path_text)
                .hint_text("path/to/data.npy | .npz | .csv | .parquet | .idx"),
        );
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut state.limit_rows, "Limit rows");
        if state.limit_rows {
            ui.add(egui::DragValue::new(&mut state.max_rows).range(100..=10_000_000));
        }
        ui.checkbox(&mut state.use_pca, "PCA projection");
    });

    let method = if state.use_pca {
        ProjectionMethod::Pca
    } else {
        ProjectionMethod::Direct
    };

    ui.horizontal(|ui| {
        let can_load = !state.loading && !state.path_text.trim().is_empty();
        if ui
            .add_enabled(can_load, egui::Button::new("📂 Import file"))
            .clicked()
        {
            request = Some(ImportRequest {
                source: ImportSource::Path(state.path_text.trim().into()),
                max_rows: state.limit_rows.then_some(state.max_rows),
                method,
            });
        }
        if state.loading {
            ui.spinner();
        }
    });

    ui.separator();
    ui.label("Benchmark datasets:");
    ui.horizontal(|ui| {
        for name in BuiltinDataset::ALL_NAMES {
            if ui.add_enabled(!state.loading, egui::Button::new(name)).clicked() {
                request = Some(ImportRequest {
                    source: ImportSource::Builtin(name),
                    max_rows: None,
                    method,
                });
            }
        }
    });

    if !state.status.is_empty() {
        ui.separator();
        ui.label(&state.status);
    }
    request
}
