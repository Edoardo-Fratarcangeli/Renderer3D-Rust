//! Per-label visibility toggles and editable label colors.
//!
//! Pure egui panel over a `HashSet<u32>` of enabled label ids and the
//! per-label color table; the caller re-runs the filter / rebuilds the
//! instances when this returns `true`.

use std::collections::HashSet;

use crate::dataset::metadata::LabelStat;
use crate::visualization::color_mapper::color_for_label;

/// Draw the filter; returns `true` when the enabled set or a label color
/// changed. `colors` is the user-editable color table (see
/// [`crate::visualization::point_cloud::PointCloudSettings::label_colors`]);
/// it is padded with the default palette on demand.
pub fn show(
    ui: &mut egui::Ui,
    labels: &[LabelStat],
    enabled: &mut HashSet<u32>,
    colors: &mut Vec<[f32; 3]>,
) -> bool {
    let mut changed = false;
    // Make sure every label has an editable slot, seeded from the palette.
    if colors.len() < labels.len() {
        for id in colors.len()..labels.len() {
            colors.push(color_for_label(id as u32));
        }
    }

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(t!("dataset.visible_labels").to_string()).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button(t!("dataset.select_none").to_string()).clicked() {
                enabled.clear();
                changed = true;
            }
            if ui.small_button(t!("dataset.select_all").to_string()).clicked() {
                *enabled = (0..labels.len() as u32).collect();
                changed = true;
            }
        });
    });
    ui.add_space(4.0);

    // Two columns keep long label lists compact and centered.
    let half = labels.len().div_ceil(2);
    ui.columns(2, |cols| {
        for (id, stat) in labels.iter().enumerate() {
            let col = &mut cols[if id < half { 0 } else { 1 }];
            let idx = id;
            let id = id as u32;
            let mut on = enabled.contains(&id);
            col.horizontal(|ui| {
                // The swatch is a color picker: label colors are editable
                // exactly like at creation time.
                changed |= ui.color_edit_button_rgb(&mut colors[idx]).changed();
                if ui
                    .checkbox(&mut on, format!("{} ({})", stat.name, stat.count))
                    .changed()
                {
                    if on {
                        enabled.insert(id);
                    } else {
                        enabled.remove(&id);
                    }
                    changed = true;
                }
            });
        }
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::metadata::LabelStat;

    #[test]
    fn color_table_is_padded_with_the_default_palette() {
        let ctx = egui::Context::default();
        let labels: Vec<LabelStat> = (0..4)
            .map(|i| LabelStat {
                name: format!("l{i}"),
                count: 1,
            })
            .collect();
        let mut enabled: HashSet<u32> = (0..4).collect();
        // One edited color survives; the rest is seeded from the palette.
        let mut colors = vec![[0.1, 0.2, 0.3]];
        let _ = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                show(ui, &labels, &mut enabled, &mut colors);
            });
        });
        // The color picker roundtrips through 8-bit sRGB, so compare with a
        // tolerance instead of exact equality.
        let close = |a: [f32; 3], b: [f32; 3]| {
            a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.01)
        };
        assert_eq!(colors.len(), 4);
        assert!(close(colors[0], [0.1, 0.2, 0.3]), "edited color survives");
        for (i, c) in colors.iter().enumerate().skip(1) {
            assert!(close(*c, color_for_label(i as u32)));
        }
    }
}
