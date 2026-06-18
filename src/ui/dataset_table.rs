//! Virtual-scrolling table over the filtered rows.
//!
//! Only the rows visible in the scroll viewport are decoded from the
//! (possibly memory-mapped) source, so the table stays O(viewport) even on
//! million-row datasets. Rows are striped for readability and span the full
//! width; clicking a row highlights its point and asks the host to focus
//! the camera on it.

use crate::dataset::Dataset;

/// Cap on feature columns shown; wide datasets (e.g. images) would explode
/// the table otherwise.
pub const MAX_TABLE_COLS: usize = 8;
const ROW_HEIGHT: f32 = 20.0;

/// Compose the monospace text of one table row (used by rendering & tests).
pub fn row_text(dataset: &Dataset, row: u32, n_cols: usize, truncated: bool) -> String {
    let mut text = format!("{:>7}", row);
    for c in 0..n_cols {
        text.push_str(&format!(" {:>9.3}", dataset.value(row as usize, c)));
    }
    if truncated {
        text.push_str(" …");
    }
    text.push_str(&format!("  {}", dataset.label_name(row as usize)));
    text
}

/// Draw the table; returns the projected position to focus when a row is
/// clicked (and toggles `highlighted`).
pub fn show(
    ui: &mut egui::Ui,
    dataset: &Dataset,
    points: &[[f32; 3]],
    visible_rows: &[u32],
    highlighted: &mut Option<u32>,
) -> Option<[f32; 3]> {
    let n_cols = dataset.n_cols().min(MAX_TABLE_COLS);
    let truncated = dataset.n_cols() > MAX_TABLE_COLS;
    let mut focus = None;

    // Header
    ui.horizontal(|ui| {
        let strong = |s: &str| egui::RichText::new(s.to_owned()).monospace().strong();
        ui.label(strong(&format!("{:>7}", "row")));
        for c in 0..n_cols {
            let name = dataset
                .metadata
                .column_names
                .get(c)
                .map(String::as_str)
                .unwrap_or("?");
            ui.label(strong(&format!("{:>9.9}", name)));
        }
        if truncated {
            ui.label(strong("…"));
        }
        ui.label(strong(&t!("dataset.table_label")));
    });
    ui.separator();

    let stripe = ui.visuals().faint_bg_color;
    egui::ScrollArea::vertical()
        .max_height(260.0)
        .auto_shrink([false, true])
        .show_rows(ui, ROW_HEIGHT, visible_rows.len(), |ui, range| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for vi in range {
                let row = visible_rows[vi];
                let is_sel = *highlighted == Some(row);
                let text = row_text(dataset, row, n_cols, truncated);

                // Zebra striping behind every other row.
                let fill = if vi % 2 == 0 {
                    stripe
                } else {
                    egui::Color32::TRANSPARENT
                };
                let resp = egui::Frame::none().fill(fill).show(ui, |ui| {
                    ui.add_sized(
                        [ui.available_width(), ROW_HEIGHT],
                        egui::SelectableLabel::new(
                            is_sel,
                            egui::RichText::new(text).monospace(),
                        ),
                    )
                });
                if resp.inner.clicked() {
                    if is_sel {
                        *highlighted = None;
                    } else {
                        *highlighted = Some(row);
                        focus = points.get(row as usize).copied();
                    }
                }
            }
        });

    ui.add_space(4.0);
    ui.vertical_centered(|ui| {
        ui.label(
            egui::RichText::new(
                t!(
                    "dataset.table_match_hint",
                    count = visible_rows.len().to_string()
                )
                .to_string(),
            )
            .weak(),
        );
    });
    focus
}
