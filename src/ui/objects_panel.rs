//! Objects window: everything the user builds by hand, in one tool.
//!
//! Four tabs over the shared tool-window chrome:
//! - **Primitive** — cube / sphere / plane via the shared object form
//!   (the same form used to edit objects later from the list).
//! - **Array** — CAD-style parametric repetition of the primitive draft:
//!   linear, circular, or along the curve sketched in the Sketch tab.
//! - **Sketch** — 2D profiles (polygons, circles, arcs, custom point lists)
//!   turned into surfaces, or open polylines.
//! - **Composer** — weld closed surfaces into 3D solids by shared edges.
//!
//! The panel never touches the scene: the host drains
//! [`ObjectsView::take_created`] (primitives and arrays) and
//! [`SketchView::take_requests`] (surfaces/solids) each frame.

use crate::pattern::{self, Placement};
use crate::scene::SceneObject;

use super::object_form::{self, ObjectFormAction, ObjectFormMode};
use super::sketch_panel::{PlaneChoice, SketchView};
use super::tool_window::{self, TabItem, ToolWindow};
use super::StatusMessage;

/// Tabs of the Objects window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectsTab {
    /// Primitive creation (cube / sphere / plane).
    Primitive,
    /// Parametric repetition of the primitive draft (CAD-style arrays).
    Array,
    /// 2D profile → surface / polyline.
    Sketch,
    /// Compose closed surfaces into a solid.
    Composer,
}

impl ObjectsTab {
    /// All tabs in display order.
    pub const ALL: [ObjectsTab; 4] = [
        ObjectsTab::Primitive,
        ObjectsTab::Array,
        ObjectsTab::Sketch,
        ObjectsTab::Composer,
    ];

    /// Localized tab title.
    pub fn title(&self) -> String {
        match self {
            ObjectsTab::Primitive => t!("objects.tab_primitive").to_string(),
            ObjectsTab::Array => t!("objects.tab_array").to_string(),
            ObjectsTab::Sketch => t!("objects.tab_sketch").to_string(),
            ObjectsTab::Composer => t!("objects.tab_composer").to_string(),
        }
    }
}

/// Which parametric pattern the Array tab builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayKind {
    Linear,
    Circular,
    /// Along the curve currently configured in the Sketch tab.
    Path,
}

/// State of the Array tab.
pub struct ArrayState {
    pub kind: ArrayKind,
    pub count: usize,
    /// Linear: step between instances.
    pub step: [f32; 3],
    /// Circular: plane, center, radius and swept angles.
    pub plane: PlaneChoice,
    pub center: [f32; 3],
    pub radius: f32,
    pub start_deg: f32,
    pub sweep_deg: f32,
    /// Orient instances along the pattern direction (circular/path).
    pub orient: bool,
    pub status: Option<StatusMessage>,
}

impl Default for ArrayState {
    fn default() -> Self {
        Self {
            kind: ArrayKind::Linear,
            count: 5,
            step: [1.5, 0.0, 0.0],
            plane: PlaneChoice::Xy,
            center: [0.0; 3],
            radius: 3.0,
            start_deg: 0.0,
            sweep_deg: 360.0,
            orient: true,
            status: None,
        }
    }
}

/// In-plane axes (u, v) of a sketch plane, used by the circular pattern.
fn plane_axes(plane: PlaneChoice) -> ([f32; 3], [f32; 3]) {
    match plane {
        PlaneChoice::Xy => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        PlaneChoice::Xz => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        PlaneChoice::Yz => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    }
}

/// Root state of the Objects window.
pub struct ObjectsView {
    /// Whether the window is visible.
    pub show_window: bool,
    /// Currently selected tab.
    pub tab: ObjectsTab,
    /// Primitive draft edited by the form. `None` until the host seeds it
    /// with a fresh id (see [`ObjectsView::needs_draft`]).
    pub draft: Option<SceneObject>,
    /// One-shot request to focus the name field.
    pub focus_name: bool,
    /// Sketch + composer state (shared with the host's build requests).
    pub sketch: SketchView,
    /// Parametric array (repetition) state.
    pub array: ArrayState,

    created: Vec<SceneObject>,
}

impl Default for ObjectsView {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectsView {
    pub fn new() -> Self {
        Self {
            show_window: false,
            tab: ObjectsTab::Primitive,
            draft: None,
            focus_name: false,
            sketch: SketchView::new(),
            array: ArrayState::default(),
            created: Vec::new(),
        }
    }

    /// Open the window on the Primitive tab with a fresh draft (used by the
    /// toolbar button and the `n` shortcut).
    pub fn open_primitive(&mut self, draft: SceneObject) {
        self.show_window = true;
        self.tab = ObjectsTab::Primitive;
        self.draft = Some(draft);
        self.focus_name = true;
    }

    /// True when a tab that consumes the primitive draft (Primitive itself,
    /// or Array which uses it as template) is visible without one; the host
    /// answers by seeding a draft with the next free object id.
    pub fn needs_draft(&self) -> bool {
        self.show_window
            && matches!(self.tab, ObjectsTab::Primitive | ObjectsTab::Array)
            && self.draft.is_none()
    }

    /// Drain the objects created this frame (single primitive or array).
    pub fn take_created(&mut self) -> Vec<SceneObject> {
        std::mem::take(&mut self.created)
    }

    /// Draw the window.
    pub fn show(&mut self, ctx: &egui::Context) {
        let mut show_window = self.show_window;
        ToolWindow::new(t!("objects.window_title"), [460.0, 540.0]).show(
            ctx,
            &mut show_window,
            |ui| self.contents(ui),
        );
        self.show_window = show_window;
    }

    fn contents(&mut self, ui: &mut egui::Ui) {
        let items = ObjectsTab::ALL
            .iter()
            .map(|&tab| TabItem::new(tab, tab.title()))
            .collect();
        tool_window::tab_bar(ui, &mut self.tab, items);

        match self.tab {
            ObjectsTab::Primitive => {
                if let Some(mut draft) = self.draft.take() {
                    match object_form::show(ui, &mut draft, &mut self.focus_name, ObjectFormMode::Create)
                    {
                        // Confirmed: hand the draft to the host; a fresh one
                        // is seeded on the next frame (needs_draft).
                        ObjectFormAction::Confirm => self.created.push(draft),
                        // Cancel just discards the edits, same effect.
                        ObjectFormAction::Cancel => {}
                        ObjectFormAction::None => self.draft = Some(draft),
                    }
                }
            }
            ObjectsTab::Array => self.array_section(ui),
            ObjectsTab::Sketch => self.sketch.sketch_section(ui),
            ObjectsTab::Composer => self.sketch.composer_section(ui),
        }
    }

    /// The Array tab: repeat the primitive draft along a parametric pattern.
    fn array_section(&mut self, ui: &mut egui::Ui) {
        // Template summary: the draft configured in the Primitive tab.
        let template_name = self
            .draft
            .as_ref()
            .map(|d| object_form::type_label(d.geometry_type))
            .unwrap_or_default();
        tool_window::section_heading(
            ui,
            t!("objects.array_heading"),
            Some(t!("objects.array_hint", template = template_name).to_string()),
        );

        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.array.kind,
                ArrayKind::Linear,
                t!("objects.array_linear").to_string(),
            );
            ui.selectable_value(
                &mut self.array.kind,
                ArrayKind::Circular,
                t!("objects.array_circular").to_string(),
            );
            ui.selectable_value(
                &mut self.array.kind,
                ArrayKind::Path,
                t!("objects.array_path").to_string(),
            );
        });
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label(t!("objects.array_count").to_string());
            ui.add(egui::Slider::new(&mut self.array.count, 2..=200));
        });
        match self.array.kind {
            ArrayKind::Linear => {
                ui.horizontal(|ui| {
                    ui.label(t!("objects.array_step").to_string());
                    for axis in 0..3 {
                        ui.add(egui::DragValue::new(&mut self.array.step[axis]).speed(0.1));
                    }
                });
            }
            ArrayKind::Circular => {
                ui.horizontal(|ui| {
                    ui.label(t!("sketch.plane").to_string());
                    ui.selectable_value(&mut self.array.plane, PlaneChoice::Xy, "XY");
                    ui.selectable_value(&mut self.array.plane, PlaneChoice::Xz, "XZ");
                    ui.selectable_value(&mut self.array.plane, PlaneChoice::Yz, "YZ");
                });
                ui.horizontal(|ui| {
                    ui.label(t!("objects.array_center").to_string());
                    for axis in 0..3 {
                        ui.add(egui::DragValue::new(&mut self.array.center[axis]).speed(0.1));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(t!("sketch.radius").to_string());
                    ui.add(egui::Slider::new(&mut self.array.radius, 0.1..=20.0));
                });
                ui.horizontal(|ui| {
                    ui.label(t!("sketch.arc_angles").to_string());
                    ui.add(
                        egui::DragValue::new(&mut self.array.start_deg)
                            .speed(1.0)
                            .suffix("°"),
                    );
                    ui.label("+");
                    ui.add(
                        egui::DragValue::new(&mut self.array.sweep_deg)
                            .speed(1.0)
                            .suffix("°"),
                    );
                });
                ui.checkbox(&mut self.array.orient, t!("objects.array_orient").to_string());
            }
            ArrayKind::Path => {
                // The curve comes from the Sketch tab's current form.
                ui.label(egui::RichText::new(t!("objects.array_path_hint").to_string()).weak());
                ui.checkbox(&mut self.array.orient, t!("objects.array_orient").to_string());
            }
        }

        ui.add_space(6.0);
        ui.vertical_centered(|ui| {
            let can = self.draft.is_some();
            if ui
                .add_enabled(
                    can,
                    egui::Button::new(
                        t!("objects.array_create", count = self.array.count).to_string(),
                    ),
                )
                .clicked()
            {
                self.create_array();
            }
        });

        if let Some(status) = &self.array.status {
            ui.add_space(4.0);
            ui.vertical_centered(|ui| status.show(ui));
        }
    }

    /// Build the placements for the current pattern and queue the clones.
    fn create_array(&mut self) {
        let Some(template) = self.draft.clone() else {
            return;
        };
        let placements: Result<Vec<Placement>, String> = match self.array.kind {
            ArrayKind::Linear => {
                let base = [
                    template.instance.position.x,
                    template.instance.position.y,
                    template.instance.position.z,
                ];
                Ok(pattern::linear(base, self.array.step, self.array.count))
            }
            ArrayKind::Circular => {
                let (u, v) = plane_axes(self.array.plane);
                Ok(pattern::circular(
                    pattern::Ring {
                        center: self.array.center,
                        u,
                        v,
                        radius: self.array.radius,
                        start_deg: self.array.start_deg,
                        sweep_deg: self.array.sweep_deg,
                    },
                    self.array.count,
                    self.array.orient,
                ))
            }
            ArrayKind::Path => self
                .sketch
                .current_world_polyline()
                .and_then(|(points, closed)| {
                    pattern::along_path(&points, closed, self.array.count, self.array.orient)
                        .ok_or_else(|| t!("objects.array_path_invalid").to_string())
                }),
        };
        match placements {
            Ok(placements) => {
                let n = placements.len();
                for (i, placement) in placements.into_iter().enumerate() {
                    let mut obj = template.clone();
                    obj.label = format!("{} {}", template.label, i + 1);
                    obj.instance.position = placement.position.into();
                    if let Some(euler) = placement.rotation_euler {
                        obj.rotation_euler = euler;
                        obj.update_rotation();
                    }
                    self.created.push(obj);
                }
                self.array.status = Some(StatusMessage::success(
                    t!("objects.array_created", count = n).to_string(),
                ));
            }
            Err(msg) => {
                self.array.status = Some(StatusMessage::error(msg));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::GeometryType;

    fn draft(id: usize) -> SceneObject {
        SceneObject::new(id, format!("Object {id}"), [0.0; 3], GeometryType::Cube)
    }

    #[test]
    fn tab_titles_are_localized_and_distinct() {
        for (i, a) in ObjectsTab::ALL.iter().enumerate() {
            assert!(!a.title().is_empty());
            for b in &ObjectsTab::ALL[i + 1..] {
                assert_ne!(a.title(), b.title());
            }
        }
    }

    #[test]
    fn primitive_tab_asks_for_a_draft_only_when_visible() {
        let mut view = ObjectsView::new();
        assert!(!view.needs_draft(), "hidden window needs nothing");
        view.open_primitive(draft(1));
        assert!(!view.needs_draft(), "open_primitive seeds the draft");
        assert_eq!(view.tab, ObjectsTab::Primitive);
        assert!(view.focus_name);

        view.draft = None;
        assert!(view.needs_draft());
        view.tab = ObjectsTab::Sketch;
        assert!(!view.needs_draft(), "other tabs don't need a draft");
    }

    #[test]
    fn created_primitive_is_drained_once() {
        let mut view = ObjectsView::new();
        view.created.push(draft(7));
        assert_eq!(view.take_created().len(), 1);
        assert!(view.take_created().is_empty());
    }

    #[test]
    fn linear_array_clones_the_template_with_numbered_labels() {
        let mut view = ObjectsView::new();
        view.draft = Some(draft(3));
        view.array.kind = ArrayKind::Linear;
        view.array.count = 4;
        view.array.step = [2.0, 0.0, 0.0];
        view.create_array();
        let objs = view.take_created();
        assert_eq!(objs.len(), 4);
        assert_eq!(objs[0].label, "Object 3 1");
        assert_eq!(objs[3].label, "Object 3 4");
        assert!((objs[3].instance.position.x - 6.0).abs() < 1e-5);
        assert!(matches!(
            view.array.status.as_ref().unwrap().kind,
            crate::ui::StatusKind::Success
        ));
        // The template draft itself is untouched (still available).
        assert!(view.draft.is_some());
    }

    #[test]
    fn circular_array_orients_instances_on_the_ring() {
        let mut view = ObjectsView::new();
        view.draft = Some(draft(1));
        view.array.kind = ArrayKind::Circular;
        view.array.count = 8;
        view.array.radius = 2.0;
        view.array.orient = true;
        view.create_array();
        let objs = view.take_created();
        assert_eq!(objs.len(), 8);
        // All on the radius-2 ring around the default center (origin).
        for o in &objs {
            let p = o.instance.position;
            assert!(((p.x * p.x + p.y * p.y + p.z * p.z).sqrt() - 2.0).abs() < 1e-4);
        }
        // Oriented: at least one instance carries a non-trivial rotation.
        assert!(objs.iter().any(|o| o.rotation_euler[2].abs() > 1.0));
    }

    #[test]
    fn path_array_follows_the_sketch_curve() {
        use crate::ui::sketch_panel::ShapeMode;
        let mut view = ObjectsView::new();
        view.draft = Some(draft(1));
        // A custom open polyline in the Sketch tab: 2 units along X.
        view.sketch.shape = ShapeMode::Custom;
        view.sketch.closed = false;
        view.sketch.custom_text = "0,0\n2,0".into();
        view.array.kind = ArrayKind::Path;
        view.array.count = 3;
        view.array.orient = false;
        view.create_array();
        let objs = view.take_created();
        assert_eq!(objs.len(), 3);
        assert!((objs[1].instance.position.x - 1.0).abs() < 1e-4, "midpoint");

        // An invalid sketch surfaces an error instead of objects.
        view.sketch.custom_text = "not points".into();
        view.create_array();
        assert!(view.take_created().is_empty());
        assert!(matches!(
            view.array.status.as_ref().unwrap().kind,
            crate::ui::StatusKind::Error
        ));
    }
}
