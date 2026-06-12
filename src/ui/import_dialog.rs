//! Import panel: file path entry, row cap, projection method and one-click
//! synthetic benchmark datasets.
//!
//! The panel is a pure function over [`ImportState`]: it never performs the
//! import itself, it only returns an [`ImportRequest`] for the caller
//! ([`super::DatasetView::start_import`]) to execute.

use super::{ImportRequest, ImportSource, ImportState};
use crate::dataset::builtin::BuiltinDataset;
use crate::dataset::preprocessor::ProjectionMethod;

/// Short, user-facing description for each builtin benchmark button.
pub fn builtin_description(name: &str) -> &'static str {
    match name {
        "blobs" => "5 gaussian clusters, 8D",
        "spirals" => "2 interleaved 3D spirals",
        "swiss_roll" => "classic manifold, 3D",
        _ => "synthetic dataset",
    }
}

/// Draw the import form; returns a request when the user confirms.
pub fn show(ui: &mut egui::Ui, state: &mut ImportState) -> Option<ImportRequest> {
    let mut request = None;

    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new("Import a dataset").heading());
        ui.label(
            egui::RichText::new("NPY · NPZ · CSV · Parquet · IDX — large files are memory mapped")
                .weak(),
        );
    });
    ui.add_space(8.0);

    egui::Grid::new("import_form_grid")
        .num_columns(2)
        .spacing([12.0, 8.0])
        .show(ui, |ui| {
            ui.label("File");
            ui.add(
                egui::TextEdit::singleline(&mut state.path_text)
                    .desired_width(f32::INFINITY)
                    .hint_text("path/to/data.npy"),
            );
            ui.end_row();

            ui.label("Rows");
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.limit_rows, "limit to");
                ui.add_enabled(
                    state.limit_rows,
                    egui::DragValue::new(&mut state.max_rows).range(100..=10_000_000),
                );
            });
            ui.end_row();

            ui.label("Projection");
            ui.horizontal(|ui| {
                ui.radio_value(&mut state.use_pca, true, "PCA (3 components)");
                ui.radio_value(&mut state.use_pca, false, "First 3 columns");
            });
            ui.end_row();
        });

    let method = if state.use_pca {
        ProjectionMethod::Pca
    } else {
        ProjectionMethod::Direct
    };

    ui.add_space(8.0);
    ui.vertical_centered(|ui| {
        let can_load = !state.loading && !state.path_text.trim().is_empty();
        ui.horizontal(|ui| {
            // Center the action button within the row.
            let button = egui::Button::new(
                egui::RichText::new("📂 Import file").size(14.0),
            )
            .min_size(egui::vec2(140.0, 28.0));
            if ui.add_enabled(can_load, button).clicked() {
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
    });

    ui.add_space(8.0);
    ui.separator();
    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new("…or try a benchmark dataset").weak());
    });
    ui.add_space(4.0);
    ui.columns(BuiltinDataset::ALL_NAMES.len(), |cols| {
        for (col, name) in cols.iter_mut().zip(BuiltinDataset::ALL_NAMES) {
            col.vertical_centered(|ui| {
                let button =
                    egui::Button::new(egui::RichText::new(name).strong())
                        .min_size(egui::vec2(100.0, 24.0));
                if ui.add_enabled(!state.loading, button).clicked() {
                    request = Some(ImportRequest {
                        source: ImportSource::Builtin(name),
                        max_rows: None,
                        method,
                    });
                }
                ui.label(egui::RichText::new(builtin_description(name)).weak().small());
            });
        }
    });

    if let Some(status) = &state.status {
        ui.add_space(8.0);
        ui.separator();
        ui.vertical_centered(|ui| status.show(ui));
    }
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_descriptions_exist_for_all_benchmarks() {
        for name in BuiltinDataset::ALL_NAMES {
            assert_ne!(builtin_description(name), "synthetic dataset");
        }
        assert_eq!(builtin_description("nope"), "synthetic dataset");
    }
}
