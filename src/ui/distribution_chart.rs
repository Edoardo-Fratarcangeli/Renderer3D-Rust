//! Hand-drawn label distribution bar chart (no plotting dependency).
//!
//! One horizontal bar per label, colored like the 3D points; disabled
//! labels are greyed out. Hovering a bar shows count and percentage.

use std::collections::HashSet;

use crate::dataset::metadata::LabelStat;
use super::label_filter::swatch_color;

/// Percentage of `count` over the total of all labels (0 when empty).
pub fn percentage(count: usize, labels: &[LabelStat]) -> f32 {
    let total: usize = labels.iter().map(|l| l.count).sum();
    if total == 0 {
        0.0
    } else {
        count as f32 * 100.0 / total as f32
    }
}

/// Draw the chart.
pub fn show(ui: &mut egui::Ui, labels: &[LabelStat], enabled: &HashSet<u32>) {
    let max_count = labels.iter().map(|l| l.count).max().unwrap_or(1).max(1);
    let bar_h = 16.0;
    let label_w = 110.0;

    for (id, stat) in labels.iter().enumerate() {
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), bar_h),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        let on = enabled.contains(&(id as u32));
        let color = if on {
            swatch_color(id as u32)
        } else {
            egui::Color32::from_gray(80)
        };

        painter.text(
            egui::pos2(rect.left(), rect.center().y),
            egui::Align2::LEFT_CENTER,
            &stat.name,
            egui::FontId::proportional(12.0),
            ui.visuals().text_color(),
        );
        let avail_w = (rect.width() - label_w - 50.0).max(40.0);
        let w = avail_w * stat.count as f32 / max_count as f32;
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(rect.left() + label_w, rect.top() + 2.0),
                egui::vec2(w.max(1.0), bar_h - 4.0),
            ),
            3.0,
            color,
        );
        painter.text(
            egui::pos2(rect.left() + label_w + w + 6.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            format!("{}", stat.count),
            egui::FontId::proportional(11.0),
            ui.visuals().text_color(),
        );
        resp.on_hover_text(format!(
            "{}: {} rows ({:.1}%)",
            stat.name,
            stat.count,
            percentage(stat.count, labels)
        ));
        ui.add_space(3.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(counts: &[usize]) -> Vec<LabelStat> {
        counts
            .iter()
            .enumerate()
            .map(|(i, &count)| LabelStat {
                name: format!("l{}", i),
                count,
            })
            .collect()
    }

    #[test]
    fn percentage_sums_to_hundred() {
        let labels = stats(&[25, 25, 50]);
        let total: f32 = labels.iter().map(|l| percentage(l.count, &labels)).sum();
        assert!((total - 100.0).abs() < 1e-3);
        assert!((percentage(50, &labels) - 50.0).abs() < 1e-3);
    }

    #[test]
    fn percentage_of_empty_set_is_zero() {
        assert_eq!(percentage(0, &[]), 0.0);
        assert_eq!(percentage(5, &stats(&[0, 0])), 0.0);
    }
}
