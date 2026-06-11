// Per-label visibility toggles, colored like the 3D points.

use std::collections::HashSet;

use crate::dataset::metadata::LabelStat;
use crate::visualization::color_mapper::color_for_label;

/// Returns true when the enabled set changed.
pub fn show(ui: &mut egui::Ui, labels: &[LabelStat], enabled: &mut HashSet<u32>) -> bool {
    let mut changed = false;

    ui.horizontal(|ui| {
        if ui.button("All").clicked() {
            *enabled = (0..labels.len() as u32).collect();
            changed = true;
        }
        if ui.button("None").clicked() {
            enabled.clear();
            changed = true;
        }
    });

    for (id, stat) in labels.iter().enumerate() {
        let id = id as u32;
        let mut on = enabled.contains(&id);
        ui.horizontal(|ui| {
            let c = color_for_label(id);
            let color = egui::Color32::from_rgb(
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
            );
            let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, color);
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
    changed
}
