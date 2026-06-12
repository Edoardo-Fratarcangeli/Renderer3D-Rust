//! Per-label visibility toggles, colored like the 3D points.
//!
//! Pure egui panel over a `HashSet<u32>` of enabled label ids; the caller
//! re-runs the filter when this returns `true`.

use std::collections::HashSet;

use crate::dataset::metadata::LabelStat;
use crate::visualization::color_mapper::color_for_label;

/// Convert a normalized RGB triple to an egui color.
pub fn swatch_color(label: u32) -> egui::Color32 {
    let c = color_for_label(label);
    egui::Color32::from_rgb(
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
    )
}

/// Draw the filter; returns `true` when the enabled set changed.
pub fn show(ui: &mut egui::Ui, labels: &[LabelStat], enabled: &mut HashSet<u32>) -> bool {
    let mut changed = false;

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Visible labels").strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("None").clicked() {
                enabled.clear();
                changed = true;
            }
            if ui.small_button("All").clicked() {
                *enabled = (0..labels.len() as u32).collect();
                changed = true;
            }
        });
    });
    ui.add_space(4.0);

    // Two columns keep long label lists compact and centered.
    let half = (labels.len() + 1) / 2;
    ui.columns(2, |cols| {
        for (id, stat) in labels.iter().enumerate() {
            let col = &mut cols[if id < half { 0 } else { 1 }];
            let id = id as u32;
            let mut on = enabled.contains(&id);
            col.horizontal(|ui| {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 2.0, swatch_color(id));
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

    #[test]
    fn swatch_matches_point_cloud_palette() {
        for label in 0..16u32 {
            let c = color_for_label(label);
            let s = swatch_color(label);
            assert_eq!(s.r(), (c[0] * 255.0) as u8);
            assert_eq!(s.g(), (c[1] * 255.0) as u8);
            assert_eq!(s.b(), (c[2] * 255.0) as u8);
        }
    }
}
