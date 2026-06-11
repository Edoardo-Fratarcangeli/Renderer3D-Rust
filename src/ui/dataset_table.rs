// Virtual-scrolling table over the filtered rows. Only the rows visible in
// the scroll viewport are decoded from the (possibly memory-mapped) source.

use crate::dataset::Dataset;

/// Cap on feature columns shown; wide datasets (e.g. images) would explode
/// the table otherwise.
const MAX_TABLE_COLS: usize = 8;
const ROW_HEIGHT: f32 = 18.0;

/// Returns the projected position to focus when a row is clicked.
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
        ui.monospace(format!("{:>7}", "row"));
        for c in 0..n_cols {
            let name = dataset
                .metadata
                .column_names
                .get(c)
                .map(String::as_str)
                .unwrap_or("?");
            ui.monospace(format!("{:>9.9}", name));
        }
        if truncated {
            ui.monospace("…");
        }
        ui.monospace("label");
    });
    ui.separator();

    egui::ScrollArea::vertical()
        .max_height(220.0)
        .auto_shrink([false, true])
        .show_rows(ui, ROW_HEIGHT, visible_rows.len(), |ui, range| {
            for vi in range {
                let row = visible_rows[vi];
                let is_sel = *highlighted == Some(row);
                let mut text = format!("{:>7}", row);
                for c in 0..n_cols {
                    text.push_str(&format!(" {:>9.3}", dataset.value(row as usize, c)));
                }
                if truncated {
                    text.push_str(" …");
                }
                text.push_str(&format!("  {}", dataset.label_name(row as usize)));

                let resp = ui.selectable_label(is_sel, egui::RichText::new(text).monospace());
                if resp.clicked() {
                    if is_sel {
                        *highlighted = None;
                    } else {
                        *highlighted = Some(row);
                        focus = points.get(row as usize).copied();
                    }
                }
            }
        });

    ui.label(format!("{} rows match the current filter", visible_rows.len()));
    focus
}
