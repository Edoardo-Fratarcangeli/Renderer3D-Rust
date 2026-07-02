//! Objects window: everything the user builds by hand, in one tool.
//!
//! Three tabs over the shared tool-window chrome:
//! - **Primitive** — cube / sphere / plane via the shared object form
//!   (the same form used to edit objects later from the list).
//! - **Sketch** — 2D profiles (polygons, circles, custom point lists) turned
//!   into surfaces, or open polylines.
//! - **Composer** — weld closed surfaces into 3D solids by shared edges.
//!
//! The panel never touches the scene: the host drains
//! [`ObjectsView::take_created`] (primitives) and
//! [`SketchView::take_requests`] (surfaces/solids) each frame.

use crate::scene::SceneObject;

use super::object_form::{self, ObjectFormAction, ObjectFormMode};
use super::sketch_panel::SketchView;
use super::tool_window::{self, TabItem, ToolWindow};

/// Tabs of the Objects window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectsTab {
    /// Primitive creation (cube / sphere / plane).
    Primitive,
    /// 2D profile → surface / polyline.
    Sketch,
    /// Compose closed surfaces into a solid.
    Composer,
}

impl ObjectsTab {
    /// All tabs in display order.
    pub const ALL: [ObjectsTab; 3] = [
        ObjectsTab::Primitive,
        ObjectsTab::Sketch,
        ObjectsTab::Composer,
    ];

    /// Localized tab title.
    pub fn title(&self) -> String {
        match self {
            ObjectsTab::Primitive => t!("objects.tab_primitive").to_string(),
            ObjectsTab::Sketch => t!("objects.tab_sketch").to_string(),
            ObjectsTab::Composer => t!("objects.tab_composer").to_string(),
        }
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

    created: Option<SceneObject>,
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
            created: None,
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

    /// True when the Primitive tab is visible but has no draft yet; the host
    /// answers by seeding one with the next free object id.
    pub fn needs_draft(&self) -> bool {
        self.show_window && self.tab == ObjectsTab::Primitive && self.draft.is_none()
    }

    /// Drain the primitive confirmed this frame, if any.
    pub fn take_created(&mut self) -> Option<SceneObject> {
        self.created.take()
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
                        ObjectFormAction::Confirm => self.created = Some(draft),
                        // Cancel just discards the edits, same effect.
                        ObjectFormAction::Cancel => {}
                        ObjectFormAction::None => self.draft = Some(draft),
                    }
                }
            }
            ObjectsTab::Sketch => self.sketch.sketch_section(ui),
            ObjectsTab::Composer => self.sketch.composer_section(ui),
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
        view.created = Some(draft(7));
        assert_eq!(view.take_created().map(|o| o.id), Some(7));
        assert!(view.take_created().is_none());
    }
}
