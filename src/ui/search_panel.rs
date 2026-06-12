//! Search box over the filtered selection.
//!
//! The query grammar is implemented (and documented) by
//! [`crate::dataset::index::SearchQuery`]: label substring, `row:N`, or a
//! numeric column predicate like `c0 > 0.5`.

/// Draw the search bar; returns `true` when the query text changed and the
/// caller should re-run the filter.
pub fn show(ui: &mut egui::Ui, text: &mut String, error: &Option<String>) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("🔍");
        let resp = ui.add(
            egui::TextEdit::singleline(text)
                .desired_width(ui.available_width() - 32.0)
                .hint_text("label substring | row:42 | c0 > 0.5"),
        );
        changed = resp.changed();
        if ui
            .add_enabled(!text.is_empty(), egui::Button::new("✖"))
            .on_hover_text("Clear search")
            .clicked()
        {
            text.clear();
            changed = true;
        }
    });
    if let Some(err) = error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 120), err);
    }
    changed
}
