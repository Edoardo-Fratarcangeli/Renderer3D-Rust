// Hand-drawn label distribution bar chart (no plotting dependency).

use std::collections::HashSet;

use crate::dataset::metadata::LabelStat;
use crate::visualization::color_mapper::color_for_label;

pub fn show(ui: &mut egui::Ui, labels: &[LabelStat], enabled: &HashSet<u32>) {
    let max_count = labels.iter().map(|l| l.count).max().unwrap_or(1).max(1);
    let bar_h = 14.0;
    let gap = 4.0;
    let label_w = 110.0;
    let avail_w = (ui.available_width() - label_w - 50.0).max(40.0);

    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(
            ui.available_width(),
            labels.len() as f32 * (bar_h + gap),
        ),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);

    for (id, stat) in labels.iter().enumerate() {
        let y = rect.top() + id as f32 * (bar_h + gap);
        let c = color_for_label(id as u32);
        let on = enabled.contains(&(id as u32));
        let color = if on {
            egui::Color32::from_rgb(
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
            )
        } else {
            egui::Color32::from_gray(80)
        };
        painter.text(
            egui::pos2(rect.left(), y + bar_h * 0.5),
            egui::Align2::LEFT_CENTER,
            &stat.name,
            egui::FontId::proportional(12.0),
            ui.visuals().text_color(),
        );
        let w = avail_w * stat.count as f32 / max_count as f32;
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(rect.left() + label_w, y),
                egui::vec2(w.max(1.0), bar_h),
            ),
            2.0,
            color,
        );
        painter.text(
            egui::pos2(rect.left() + label_w + w + 6.0, y + bar_h * 0.5),
            egui::Align2::LEFT_CENTER,
            format!("{}", stat.count),
            egui::FontId::proportional(11.0),
            ui.visuals().text_color(),
        );
    }
}
