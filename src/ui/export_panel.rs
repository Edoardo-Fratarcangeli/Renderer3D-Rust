// Export panel: writes the rows matching the active filter to CSV.

use std::path::PathBuf;

use super::ExportState;

/// Returns the destination path when the user confirms the export.
pub fn show(ui: &mut egui::Ui, state: &mut ExportState) -> Option<PathBuf> {
    let mut out = None;
    ui.horizontal(|ui| {
        ui.label("Path:");
        ui.add(egui::TextEdit::singleline(&mut state.path_text).hint_text("export.csv"));
        if ui.button("💾 Export filtered rows").clicked() && !state.path_text.trim().is_empty() {
            out = Some(PathBuf::from(state.path_text.trim()));
        }
    });
    if !state.status.is_empty() {
        ui.label(&state.status);
    }
    out
}
