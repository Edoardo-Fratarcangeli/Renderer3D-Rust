//! Shared chrome for the top-level tool windows (Objects, Import, Data,
//! Stream, Network…).
//!
//! Every tool opened from the toolbar gets the same look and behavior from
//! one place: spawn centered on screen (draggable afterwards), a close box
//! bound to the tool's `show_window` flag, and a consistent size policy.
//! Panels keep owning their content; this module only removes the
//! per-window copies of the window setup and the tab-bar boilerplate.

/// Size policy of a tool window.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SizePolicy {
    /// The window keeps this exact footprint; long content scrolls inside.
    Fixed([f32; 2]),
    /// The window starts at this size and the user may resize it.
    Resizable([f32; 2]),
}

/// Builder for a consistently styled tool window.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolWindow {
    title: String,
    size: SizePolicy,
}

impl ToolWindow {
    /// A resizable window spawning at `size` (the common case).
    pub fn new(title: impl Into<String>, size: [f32; 2]) -> Self {
        Self {
            title: title.into(),
            size: SizePolicy::Resizable(size),
        }
    }

    /// A fixed-footprint window: it never grows with its content.
    pub fn fixed(title: impl Into<String>, size: [f32; 2]) -> Self {
        Self {
            title: title.into(),
            size: SizePolicy::Fixed(size),
        }
    }

    /// Draw the window when `show_window` is set; clear the flag when the
    /// user closes it. Returns the closure's result while the window is
    /// visible.
    pub fn show<R>(
        self,
        ctx: &egui::Context,
        show_window: &mut bool,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> Option<R> {
        if !*show_window {
            return None;
        }
        let mut open = true;
        let mut window = egui::Window::new(self.title)
            .open(&mut open)
            // Spawn centered on screen; remains draggable afterwards.
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.screen_rect().center());
        window = match self.size {
            SizePolicy::Fixed(size) => window.fixed_size(size),
            SizePolicy::Resizable(size) => window.default_size(size).resizable(true),
        };
        let result = window.show(ctx, add_contents).and_then(|r| r.inner);
        if !open {
            *show_window = false;
        }
        result
    }
}

/// One entry of a [`tab_bar`].
pub struct TabItem<T> {
    pub value: T,
    /// Label shown on the selectable tab (icon + localized title).
    pub label: egui::RichText,
    /// Grayed out (not selectable) when false.
    pub enabled: bool,
}

impl<T> TabItem<T> {
    /// An always-enabled tab with a plain-text label.
    pub fn new(value: T, label: impl Into<String>) -> Self {
        Self {
            value,
            label: egui::RichText::new(label.into()),
            enabled: true,
        }
    }

    /// Disable the tab unless `enabled` holds.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Replace the label with a styled one (color, size…).
    pub fn styled(mut self, label: egui::RichText) -> Self {
        self.label = label;
        self
    }
}

/// Centered tab bar used by every multi-tab tool window. Updates `current`
/// on click and returns true when the selection changed.
pub fn tab_bar<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    current: &mut T,
    items: Vec<TabItem<T>>,
) -> bool {
    let mut changed = false;
    ui.vertical_centered(|ui| {
        ui.horizontal(|ui| {
            for item in items {
                let selected = *current == item.value;
                if ui
                    .add_enabled(
                        item.enabled,
                        egui::SelectableLabel::new(selected, item.label.size(14.0)),
                    )
                    .clicked()
                    && !selected
                {
                    *current = item.value;
                    changed = true;
                }
            }
        });
    });
    ui.separator();
    changed
}

/// Centered section heading with an optional weak hint line under it —
/// the pattern every panel hand-rolled before.
pub fn section_heading(ui: &mut egui::Ui, heading: impl Into<String>, hint: Option<String>) {
    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new(heading.into()).heading());
        if let Some(hint) = hint {
            ui.label(egui::RichText::new(hint).weak());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab_bar_switches_only_between_enabled_tabs() {
        let ctx = egui::Context::default();
        let mut current = 0usize;
        let _ = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let changed = tab_bar(
                    ui,
                    &mut current,
                    vec![
                        TabItem::new(0usize, "a"),
                        TabItem::new(1usize, "b").enabled(false),
                    ],
                );
                // No interaction in a headless frame: selection unchanged.
                assert!(!changed);
                assert_eq!(current, 0);
            });
        });
    }

    #[test]
    fn hidden_tool_window_draws_nothing() {
        let ctx = egui::Context::default();
        let _ = ctx.run(Default::default(), |ctx| {
            let mut show = false;
            let ran = ToolWindow::new("t", [100.0, 100.0]).show(ctx, &mut show, |_| 42);
            assert_eq!(ran, None);

            let mut show = true;
            let ran = ToolWindow::fixed("t2", [100.0, 100.0]).show(ctx, &mut show, |_| 42);
            assert_eq!(ran, Some(42));
            assert!(show, "window stays open until the user closes it");
        });
    }
}
