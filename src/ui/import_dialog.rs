//! Import panel: file path entry, row cap, projection method and one-click
//! synthetic benchmark datasets.
//!
//! The panel is a pure function over [`ImportState`]: it never performs the
//! import itself, it only returns an [`ImportRequest`] for the caller
//! ([`super::DatasetView::start_import`]) to execute.

use super::{ImportRequest, ImportSource, ImportState};
use crate::dataset::builtin::BuiltinDataset;

/// Short, user-facing description for each builtin benchmark button.
pub fn builtin_description(name: &str) -> String {
    match name {
        "blobs" => t!("dataset.builtin_blobs").to_string(),
        "spirals" => t!("dataset.builtin_spirals").to_string(),
        "swiss_roll" => t!("dataset.builtin_swiss_roll").to_string(),
        _ => t!("dataset.builtin_generic").to_string(),
    }
}

/// Mutually-exclusive PCA / Direct / Radial selector shared by the import form,
/// the View tab's reprojection panel and the Stream panel. Operates on the two
/// flags directly so any state that carries them can reuse it. Returns true if
/// the choice changed.
pub fn method_radio(ui: &mut egui::Ui, use_pca: &mut bool, use_radial: &mut bool) -> bool {
    // Map the two-bool state onto a single radio group: 0 PCA, 1 Direct, 2 Radial.
    let mut sel = if *use_radial {
        2
    } else if *use_pca {
        0
    } else {
        1
    };
    let before = sel;
    ui.radio_value(&mut sel, 0, t!("dataset.method_pca").to_string());
    ui.radio_value(&mut sel, 1, t!("dataset.method_direct").to_string());
    ui.radio_value(&mut sel, 2, t!("dataset.method_radial").to_string());
    if sel != before {
        *use_radial = sel == 2;
        *use_pca = sel == 0;
        true
    } else {
        false
    }
}

/// Draw the import form; returns a request when the user confirms.
pub fn show(ui: &mut egui::Ui, state: &mut ImportState) -> Option<ImportRequest> {
    let mut request = None;

    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new(t!("dataset.import_heading").to_string()).heading());
        ui.label(
            egui::RichText::new(t!("dataset.import_formats").to_string())
            .weak(),
        );
    });
    ui.add_space(8.0);

    egui::Grid::new("import_form_grid")
        .num_columns(2)
        .spacing([12.0, 8.0])
        .show(ui, |ui| {
            ui.label(t!("dataset.field_file").to_string());
            ui.add(
                egui::TextEdit::singleline(&mut state.path_text)
                    .desired_width(f32::INFINITY)
                    .hint_text("path/to/data.npy"),
            );
            ui.end_row();

            ui.label(t!("dataset.field_rows").to_string());
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.limit_rows, t!("dataset.limit_to").to_string());
                ui.add_enabled(
                    state.limit_rows,
                    egui::DragValue::new(&mut state.max_rows).range(100..=10_000_000),
                );
            });
            ui.end_row();

            ui.label(t!("dataset.field_projection").to_string());
            ui.horizontal(|ui| {
                method_radio(ui, &mut state.use_pca, &mut state.use_radial);
            });
            ui.end_row();

            ui.label(t!("dataset.dimensions").to_string());
            ui.horizontal(|ui| {
                ui.radio_value(&mut state.dims, 3, "3D");
                ui.radio_value(&mut state.dims, 2, "2D");
                ui.radio_value(&mut state.dims, 1, "1D");
            });
            ui.end_row();

            // For direct projection, choose which feature columns feed the
            // axes (by 0-based index; column names are not known until load —
            // refine them later from the View tab). PCA and radial use every
            // column, so the per-axis picker is hidden for them.
            if !state.use_pca && !state.use_radial {
                let dims = state.dims.clamp(1, 3) as usize;
                ui.label(t!("dataset.field_columns").to_string());
                ui.horizontal(|ui| {
                    for (a, axis) in ["X", "Y", "Z"].iter().enumerate().take(dims) {
                        ui.add(
                            egui::DragValue::new(&mut state.axes[a])
                                .range(0..=usize::MAX)
                                .prefix(format!("{}: ", axis)),
                        );
                    }
                });
                ui.end_row();
            }
        });

    let projection = state.projection();

    ui.add_space(8.0);
    ui.vertical_centered(|ui| {
        let can_load = !state.loading && !state.path_text.trim().is_empty();
        ui.horizontal(|ui| {
            // Center the action button within the row.
            let button = egui::Button::new(
                egui::RichText::new(t!("dataset.import_file").to_string()).size(14.0),
            )
            .min_size(egui::vec2(140.0, 28.0));
            if ui.add_enabled(can_load, button).clicked() {
                request = Some(ImportRequest {
                    source: ImportSource::Path(state.path_text.trim().into()),
                    max_rows: state.limit_rows.then_some(state.max_rows),
                    projection,
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
        ui.label(egui::RichText::new(t!("dataset.try_benchmark").to_string()).weak());
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
                        projection,
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
        let generic = t!("dataset.builtin_generic").to_string();
        for name in BuiltinDataset::ALL_NAMES {
            assert_ne!(builtin_description(name), generic);
        }
        assert_eq!(builtin_description("nope"), generic);
    }
}
