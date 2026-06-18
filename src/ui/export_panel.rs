//! Export panel: writes the rows matching the active filter to CSV.
//!
//! Like the other panels this is presentation-only: it returns the chosen
//! destination path and the caller performs the actual export.

use std::path::PathBuf;

use super::ExportState;

/// Draw the export form; returns the destination when the user confirms.
/// `n_visible` is the number of rows the export would contain.
pub fn show(ui: &mut egui::Ui, state: &mut ExportState, n_visible: usize) -> Option<PathBuf> {
    let mut out = None;

    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new(t!("dataset.export_heading").to_string()).heading());
        ui.label(
            egui::RichText::new(
                t!("dataset.export_hint", count = n_visible.to_string()).to_string(),
            )
            .weak(),
        );
    });
    ui.add_space(8.0);

    egui::Grid::new("export_form_grid")
        .num_columns(2)
        .spacing([12.0, 8.0])
        .show(ui, |ui| {
            ui.label(t!("dataset.field_path").to_string());
            ui.add(
                egui::TextEdit::singleline(&mut state.path_text)
                    .desired_width(f32::INFINITY)
                    .hint_text("export.csv"),
            );
            ui.end_row();
        });

    ui.add_space(8.0);
    ui.vertical_centered(|ui| {
        let button = egui::Button::new(egui::RichText::new(t!("dataset.export_button").to_string()).size(14.0))
            .min_size(egui::vec2(120.0, 28.0));
        let can_export = !state.path_text.trim().is_empty();
        if ui.add_enabled(can_export, button).clicked() {
            out = Some(PathBuf::from(state.path_text.trim()));
        }
    });

    if let Some(status) = &state.status {
        ui.add_space(8.0);
        ui.vertical_centered(|ui| status.show(ui));
    }
    out
}
