// Search box over the filtered selection. Syntax handled by
// dataset::index::SearchQuery (label substring, `row:N`, `c<i> <op> <v>`).

/// Returns true when the query text changed (caller re-filters).
pub fn show(ui: &mut egui::Ui, text: &mut String, error: &Option<String>) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("🔍");
        let resp = ui.add(
            egui::TextEdit::singleline(text)
                .hint_text("label substring | row:42 | c0 > 0.5"),
        );
        changed = resp.changed();
        if ui.button("✖").clicked() && !text.is_empty() {
            text.clear();
            changed = true;
        }
    });
    if let Some(err) = error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 120), err);
    }
    changed
}
