//! Unified entity list (the expandable panel at the bottom of the main
//! window).
//!
//! Every entity in the scene — hand-built objects, imported geometry layers,
//! datasets, neural networks — is rendered by the same [`row`] widget with
//! the same affordances: select, open properties, toggle visibility/label,
//! focus the camera, delete. Hosts describe a row with [`EntityRow`] and act
//! on the returned [`RowAction`]; no per-type list code.

/// Colors shared by all row icons.
mod colors {
    use egui::Color32;

    pub const DANGER: Color32 = Color32::from_rgb(255, 100, 100);
    pub const VISIBLE: Color32 = Color32::from_rgb(100, 255, 100);
    pub const HIDDEN: Color32 = Color32::from_rgb(255, 100, 100);
    pub const LABEL_ON: Color32 = Color32::from_rgb(0, 200, 255);
    pub const LABEL_OFF: Color32 = Color32::from_rgb(150, 150, 150);
    pub const EDITING: Color32 = Color32::from_rgb(0, 150, 255);
}

/// Description of one list row. Build it from any entity type.
pub struct EntityRow<'a> {
    /// Leading emoji tag (📊, 🧠, 🧊…); `None` for plain scene objects.
    pub icon: Option<&'a str>,
    /// Entity name, drawn strong.
    pub title: &'a str,
    /// Weak detail text after the name, e.g. "(1234 rows)".
    pub subtitle: Option<String>,
    /// Highlight the title as selected (only meaningful when `selectable`).
    pub selected: bool,
    /// Whether clicking the title selects the entity.
    pub selectable: bool,
    /// Visibility toggle state; `None` hides the toggle.
    pub visible: Option<bool>,
    /// Viewport-label toggle state; `None` hides the toggle.
    pub show_label: Option<bool>,
    /// Show the edit (properties) button; `editing` tints it while active.
    pub can_edit: bool,
    pub editing: bool,
    /// Show the camera-focus button.
    pub can_focus: bool,
    /// Show the delete button.
    pub can_delete: bool,
    /// Offer the CAD transform entries (duplicate / mirror / flip) in the
    /// right-click context menu (scene objects only).
    pub cad_actions: bool,
}

impl<'a> EntityRow<'a> {
    /// A minimal row: icon + name, delete/focus/visibility, no selection.
    pub fn entry(icon: &'a str, title: &'a str) -> Self {
        Self {
            icon: Some(icon),
            title,
            subtitle: None,
            selected: false,
            selectable: false,
            visible: None,
            show_label: None,
            can_edit: false,
            editing: false,
            can_focus: true,
            can_delete: true,
            cad_actions: false,
        }
    }

    pub fn subtitle(mut self, s: impl Into<String>) -> Self {
        self.subtitle = Some(s.into());
        self
    }

    pub fn visible(mut self, v: bool) -> Self {
        self.visible = Some(v);
        self
    }
}

/// World axis used by the mirror context-menu entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorAxis {
    X,
    Y,
    Z,
}

/// What the user asked to do with the row this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowAction {
    None,
    /// Title clicked; `exclusive` is false when Ctrl/Shift/Cmd was held.
    Select { exclusive: bool },
    /// Title double-clicked (open properties).
    Open,
    ToggleVisibility,
    ToggleLabel,
    Edit,
    Delete,
    Focus,
    /// Context menu: clone the entity in place.
    Duplicate,
    /// Context menu: add a duplicate mirrored across the world plane
    /// orthogonal to this axis.
    Mirror(MirrorAxis),
    /// Context menu: flip the entity 180° around the vertical (Z) axis.
    Flip,
}

/// Draw one row; returns the (single) action requested this frame.
pub fn row(ui: &mut egui::Ui, spec: &EntityRow) -> RowAction {
    let mut action = RowAction::None;
    ui.horizontal(|ui| {
        // --- Title (selectable or plain) ---
        let title = match spec.icon {
            Some(icon) => format!("{icon} {}", spec.title),
            None => spec.title.to_string(),
        };
        let text = egui::RichText::new(title).strong().size(14.0);
        let response = if spec.selectable {
            let response = ui.selectable_label(spec.selected, text);
            if response.clicked() {
                let m = ui.ctx().input(|i| i.modifiers);
                action = RowAction::Select {
                    exclusive: !(m.ctrl || m.shift || m.mac_cmd),
                };
            }
            if response.double_clicked() {
                action = RowAction::Open;
            }
            response
        } else {
            // Sense clicks so the right-click context menu opens on plain
            // (non-selectable) rows too.
            ui.add(egui::Label::new(text).sense(egui::Sense::click()))
        };
        // Right-click context menu: the classic entity actions, gated by the
        // same capability flags as the trailing icon buttons.
        response.context_menu(|ui| {
            if spec.can_edit && ui.button(t!("list.edit").to_string()).clicked() {
                action = RowAction::Edit;
                ui.close_menu();
            }
            if spec.can_focus && ui.button(t!("menu.focus").to_string()).clicked() {
                action = RowAction::Focus;
                ui.close_menu();
            }
            if spec.cad_actions {
                ui.separator();
                if ui.button(t!("menu.duplicate").to_string()).clicked() {
                    action = RowAction::Duplicate;
                    ui.close_menu();
                }
                ui.menu_button(t!("menu.mirror").to_string(), |ui| {
                    for (axis, label) in [
                        (MirrorAxis::X, "X"),
                        (MirrorAxis::Y, "Y"),
                        (MirrorAxis::Z, "Z"),
                    ] {
                        if ui.button(label).clicked() {
                            action = RowAction::Mirror(axis);
                            ui.close_menu();
                        }
                    }
                });
                if ui.button(t!("menu.flip").to_string()).clicked() {
                    action = RowAction::Flip;
                    ui.close_menu();
                }
            }
            if spec.can_delete {
                ui.separator();
                if ui.button(t!("list.delete").to_string()).clicked() {
                    action = RowAction::Delete;
                    ui.close_menu();
                }
            }
        });
        if let Some(subtitle) = &spec.subtitle {
            ui.label(egui::RichText::new(subtitle).weak());
        }

        // --- Trailing icon buttons (right-aligned, uniform order) ---
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if spec.can_edit {
                let color = if spec.editing {
                    colors::EDITING
                } else {
                    ui.visuals().text_color()
                };
                if icon_button(ui, "✏", color, &t!("list.edit")) {
                    action = RowAction::Edit;
                }
            }
            if spec.can_delete && icon_button(ui, "🗑", colors::DANGER, &t!("list.delete")) {
                action = RowAction::Delete;
            }
            if spec.can_focus
                && icon_button(ui, "🎯", ui.visuals().text_color(), &t!("list.focus"))
            {
                action = RowAction::Focus;
            }
            if let Some(show_label) = spec.show_label {
                let color = if show_label {
                    colors::LABEL_ON
                } else {
                    colors::LABEL_OFF
                };
                if icon_button(ui, "🏷", color, &t!("list.toggle_label")) {
                    action = RowAction::ToggleLabel;
                }
            }
            if let Some(visible) = spec.visible {
                let (icon, color) = if visible {
                    ("👁", colors::VISIBLE)
                } else {
                    ("🕶", colors::HIDDEN)
                };
                if icon_button(ui, icon, color, &t!("list.visibility")) {
                    action = RowAction::ToggleVisibility;
                }
            }
        });
    });
    action
}

/// Flat, borderless icon button used by every row (same look everywhere).
fn icon_button(ui: &mut egui::Ui, icon: &str, color: egui::Color32, hover: &str) -> bool {
    ui.add(
        egui::Button::new(egui::RichText::new(icon).color(color).size(16.0))
            .frame(false),
    )
    .on_hover_text(hover)
    .clicked()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_covers_the_common_entry_shape() {
        let row = EntityRow::entry("📊", "mnist").subtitle("(10 rows)").visible(true);
        assert_eq!(row.icon, Some("📊"));
        assert_eq!(row.subtitle.as_deref(), Some("(10 rows)"));
        assert_eq!(row.visible, Some(true));
        assert!(!row.selectable);
        assert!(row.can_delete && row.can_focus && !row.can_edit);
    }

    #[test]
    fn headless_row_reports_no_action() {
        let ctx = egui::Context::default();
        let _ = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let spec = EntityRow::entry("🧊", "layer").visible(false);
                assert_eq!(row(ui, &spec), RowAction::None);
            });
        });
    }
}
