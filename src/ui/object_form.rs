//! Shared property form for scene objects.
//!
//! Creation ("Add object") and editing ("Properties") are the same form over
//! a [`SceneObject`] draft: name, type, transform, color, label and the
//! type-specific geometry properties. The host owns the draft and applies it
//! on [`ObjectFormAction::Confirm`]; the form itself never touches the scene,
//! so it can be driven headless by the tests in `tests/ui`.

use crate::scene::{GeometryType, SceneObject};

/// Whether the form is creating a new object or editing an existing one.
/// Only the window title and the confirm-button label differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectFormMode {
    Create,
    Edit,
}

impl ObjectFormMode {
    fn window_title(&self) -> String {
        match self {
            ObjectFormMode::Create => t!("add.window_title").to_string(),
            ObjectFormMode::Edit => t!("props.window_title").to_string(),
        }
    }

    fn confirm_label(&self) -> String {
        match self {
            ObjectFormMode::Create => t!("common.add").to_string(),
            ObjectFormMode::Edit => t!("common.confirm").to_string(),
        }
    }
}

/// What the user asked the host to do with the draft this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectFormAction {
    /// Keep showing the form.
    None,
    /// Apply the draft (add it to the scene / overwrite the object).
    Confirm,
    /// Discard the draft and close the form.
    Cancel,
}

/// Draw the object form as a floating window over `draft`.
///
/// `focus_name` requests keyboard focus on the name field once (the flag is
/// consumed). Closing the window with its ✕ button yields
/// [`ObjectFormAction::Cancel`], exactly like the Cancel button or Escape.
pub fn window(
    ctx: &egui::Context,
    draft: &mut SceneObject,
    focus_name: &mut bool,
    mode: ObjectFormMode,
) -> ObjectFormAction {
    let mut action = ObjectFormAction::None;
    let mut open = true;
    egui::Window::new(mode.window_title())
        .id(egui::Id::new(("object_form", mode == ObjectFormMode::Create)))
        .open(&mut open)
        .resizable(true)
        .show(ctx, |ui| {
            action = show(ui, draft, focus_name, mode);
        });
    if !open {
        action = ObjectFormAction::Cancel;
    }
    action
}

/// Draw the form body into an arbitrary `Ui` (used by [`window`] and by the
/// headless tests). Returns the requested action.
pub fn show(
    ui: &mut egui::Ui,
    draft: &mut SceneObject,
    focus_name: &mut bool,
    mode: ObjectFormMode,
) -> ObjectFormAction {
    let mut action = ObjectFormAction::None;

    // --- Name ---
    ui.horizontal(|ui| {
        ui.label(t!("common.name").to_string());
        let res = ui.text_edit_singleline(&mut draft.label);
        if *focus_name {
            res.request_focus();
            *focus_name = false;
        }
    });
    ui.separator();

    // --- Type ---
    ui.horizontal(|ui| {
        ui.label(t!("common.type").to_string());
        egui::ComboBox::from_id_source(("object_form_type", draft.id))
            .selected_text(type_label(draft.geometry_type))
            .show_ui(ui, |ui| {
                for gt in [GeometryType::Cube, GeometryType::Sphere, GeometryType::Plane] {
                    ui.selectable_value(&mut draft.geometry_type, gt, type_label(gt));
                }
            });
    });

    // --- Transform ---
    ui.heading(t!("add.transform").to_string());
    ui.horizontal(|ui| {
        ui.label(t!("common.position").to_string());
        ui.add(egui::DragValue::new(&mut draft.instance.position.x).speed(0.1));
        ui.add(egui::DragValue::new(&mut draft.instance.position.y).speed(0.1));
        ui.add(egui::DragValue::new(&mut draft.instance.position.z).speed(0.1));
    });
    ui.horizontal(|ui| {
        ui.label(t!("common.rotation").to_string());
        let mut changed = false;
        for axis in 0..3 {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut draft.rotation_euler[axis])
                        .speed(1.0)
                        .suffix("°"),
                )
                .changed();
        }
        if changed {
            draft.update_rotation();
        }
    });
    ui.horizontal(|ui| {
        ui.label(t!("common.scale").to_string());
        ui.add(
            egui::DragValue::new(&mut draft.instance.scale.x)
                .speed(0.01)
                .max_decimals(2),
        );
        ui.add(
            egui::DragValue::new(&mut draft.instance.scale.y)
                .speed(0.01)
                .max_decimals(2),
        );
        ui.add(
            egui::DragValue::new(&mut draft.instance.scale.z)
                .speed(0.01)
                .max_decimals(2),
        );
    });

    // --- Appearance ---
    ui.separator();
    ui.horizontal(|ui| {
        ui.label(t!("common.color").to_string());
        ui.color_edit_button_rgb(&mut draft.color);
    });
    ui.checkbox(&mut draft.show_label, t!("add.show_label").to_string());

    // --- Type-specific geometry properties ---
    ui.separator();
    ui.heading(t!("add.geometry_props").to_string());
    geometry_props(ui, draft);

    // --- Confirm / Cancel ---
    ui.separator();
    ui.horizontal(|ui| {
        if ui
            .add_sized([80.0, 24.0], egui::Button::new(t!("common.cancel").to_string()))
            .clicked()
            || ui.input(|i| i.key_pressed(egui::Key::Escape))
        {
            action = ObjectFormAction::Cancel;
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add_sized([80.0, 24.0], egui::Button::new(mode.confirm_label()))
                .clicked()
                || ui.input(|i| i.key_pressed(egui::Key::Enter))
            {
                action = ObjectFormAction::Confirm;
            }
        });
    });

    action
}

/// Localized display name of a geometry type.
pub fn type_label(gt: GeometryType) -> String {
    match gt {
        GeometryType::Cube => t!("shape.cube").to_string(),
        GeometryType::Sphere => t!("shape.sphere").to_string(),
        GeometryType::Plane => t!("shape.plane").to_string(),
        other => format!("{:?}", other),
    }
}

/// The per-type property rows (side / radius / surface), kept in sync with
/// the uniform scale exactly as before.
fn geometry_props(ui: &mut egui::Ui, draft: &mut SceneObject) {
    match draft.geometry_type {
        GeometryType::Cube => {
            ui.horizontal(|ui| {
                ui.label(t!("add.side_length").to_string());
                if ui
                    .add(egui::DragValue::new(&mut draft.cube_side).speed(0.1))
                    .changed()
                {
                    let s = draft.cube_side;
                    draft.instance.scale = cgmath::Vector3::new(s, s, s);
                }
            });
        }
        GeometryType::Sphere => {
            ui.horizontal(|ui| {
                ui.label(t!("add.radius").to_string());
                if ui
                    .add(egui::DragValue::new(&mut draft.sphere_radius).speed(0.1))
                    .changed()
                {
                    // Mesh is radius 0.5, so scale = 2 * radius.
                    let s = draft.sphere_radius * 2.0;
                    draft.instance.scale = cgmath::Vector3::new(s, s, s);
                }
            });
        }
        GeometryType::Plane => {
            ui.horizontal(|ui| {
                ui.label(t!("add.surface_area").to_string());
                if ui
                    .add(
                        egui::DragValue::new(&mut draft.plane_surface)
                            .speed(0.1)
                            .range(0.1..=100.0),
                    )
                    .changed()
                {
                    let s = draft.plane_surface.sqrt();
                    draft.instance.scale = cgmath::Vector3::new(s, 1.0, s);
                }
            });
            ui.checkbox(&mut draft.show_normal, t!("add.show_normal").to_string());
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_have_distinct_titles_and_confirm_labels() {
        assert_ne!(
            ObjectFormMode::Create.window_title(),
            ObjectFormMode::Edit.window_title()
        );
        assert_ne!(
            ObjectFormMode::Create.confirm_label(),
            ObjectFormMode::Edit.confirm_label()
        );
    }

    #[test]
    fn type_labels_are_localized_and_distinct() {
        let labels = [
            type_label(GeometryType::Cube),
            type_label(GeometryType::Sphere),
            type_label(GeometryType::Plane),
        ];
        for (i, a) in labels.iter().enumerate() {
            assert!(!a.is_empty());
            for b in &labels[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
