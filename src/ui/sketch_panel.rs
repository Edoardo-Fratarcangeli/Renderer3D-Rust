//! Sketch & solid composer window.
//!
//! Lets the user draw a 2D profile (regular polygon, circle, or a custom point
//! list — closed surface or open polyline) on a chosen plane, turn it into a
//! surface, and then compose several closed surfaces into a 3D solid by welding
//! their shared edges.
//!
//! Like the other panels, this is plain state + a `show` method: all geometry
//! lives in [`crate::sketch`] / [`crate::brep`], so the form can be driven
//! headless by the tests below. The host (`State`) drains
//! [`SketchView::take_requests`] each frame and performs the GPU upload.

use crate::sketch::{Plane, Profile, Sketch, Vec2, DEFAULT_TOLERANCE};

use super::StatusMessage;

/// Which world plane the sketch is drawn on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneChoice {
    Xy,
    Xz,
    Yz,
}

impl PlaneChoice {
    fn plane(self, origin: [f32; 3]) -> Plane {
        let mut p = match self {
            PlaneChoice::Xy => Plane::xy(),
            PlaneChoice::Xz => Plane::xz(),
            PlaneChoice::Yz => Plane::yz(),
        };
        p.origin = origin;
        p
    }
}

/// Which kind of profile the form produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeMode {
    /// Regular straight-edged N-gon.
    Polygon,
    /// Curved circle (built from arcs).
    Circle,
    /// Custom point list (closed surface or open polyline).
    Custom,
}

/// A surface already created and kept as a candidate face for the composer.
#[derive(Debug, Clone, PartialEq)]
pub struct CreatedSurface {
    pub name: String,
    /// World-space boundary loop (closed profiles only).
    pub loop3: Vec<[f32; 3]>,
    pub selected: bool,
}

/// What the sketch window asks the host to build this frame.
#[derive(Debug, Clone, PartialEq)]
pub enum SketchRequest {
    /// Add a surface (filled face or ribbon) from this sketch.
    Surface { sketch: Sketch, label: String },
    /// Compose a solid from these world-space face loops.
    Solid {
        loops: Vec<Vec<[f32; 3]>>,
        label: String,
    },
}

/// Root state of the sketch / composer window.
pub struct SketchView {
    pub show_window: bool,
    pub plane: PlaneChoice,
    pub shape: ShapeMode,
    pub n_sides: usize,
    pub radius: f32,
    /// Closure flag for the custom point list.
    pub closed: bool,
    /// Plane origin offset, so successive faces can be stacked into a solid.
    pub offset: [f32; 3],
    /// Custom points, one `x,y` per line.
    pub custom_text: String,
    pub status: Option<StatusMessage>,
    /// Surfaces available to the composer.
    pub surfaces: Vec<CreatedSurface>,

    pending: Vec<SketchRequest>,
    next_id: usize,
}

impl Default for SketchView {
    fn default() -> Self {
        Self::new()
    }
}

impl SketchView {
    pub fn new() -> Self {
        Self {
            show_window: false,
            plane: PlaneChoice::Xy,
            shape: ShapeMode::Polygon,
            n_sides: 5,
            radius: 1.0,
            closed: true,
            offset: [0.0, 0.0, 0.0],
            custom_text: String::new(),
            status: None,
            surfaces: Vec::new(),
            pending: Vec::new(),
            next_id: 0,
        }
    }

    /// Drain the build requests queued this frame (host performs the upload).
    pub fn take_requests(&mut self) -> Vec<SketchRequest> {
        std::mem::take(&mut self.pending)
    }

    /// Build the profile (and its closed flag) from the current form.
    fn build_profile(&self) -> Result<(Profile, bool), String> {
        match self.shape {
            ShapeMode::Polygon => Profile::regular_polygon(self.n_sides, self.radius)
                .map(|p| (p, true))
                .map_err(|e| e.to_string()),
            ShapeMode::Circle => Profile::circle(self.radius)
                .map(|p| (p, true))
                .map_err(|e| e.to_string()),
            ShapeMode::Custom => {
                let pts = parse_points(&self.custom_text)?;
                Profile::from_points(&pts, self.closed)
                    .map(|p| (p, self.closed))
                    .map_err(|e| e.to_string())
            }
        }
    }

    /// Validate the form, queue a surface build, and (for closed profiles)
    /// remember the surface as a composer face.
    pub fn create_surface(&mut self) {
        let (profile, closed) = match self.build_profile() {
            Ok(v) => v,
            Err(msg) => {
                self.status = Some(StatusMessage::error(
                    t!("sketch.profile_invalid", msg = msg).to_string(),
                ));
                return;
            }
        };
        let sketch = Sketch::new(self.plane.plane(self.offset), profile);
        // Surface the mesh validity early so the user gets immediate feedback
        // (the host will rebuild it, but this catches degenerate input here).
        match sketch.world_polyline(DEFAULT_TOLERANCE) {
            Ok(loop3) => {
                self.next_id += 1;
                let name = t!("sketch.surface_name", id = self.next_id).to_string();
                if closed {
                    self.surfaces.push(CreatedSurface {
                        name: name.clone(),
                        loop3,
                        selected: true,
                    });
                }
                self.pending.push(SketchRequest::Surface {
                    sketch,
                    label: name,
                });
                self.status = Some(StatusMessage::info(t!("sketch.surface_queued").to_string()));
            }
            Err(e) => {
                self.status = Some(StatusMessage::error(
                    t!("sketch.profile_invalid", msg = e.to_string()).to_string(),
                ));
            }
        }
    }

    /// Number of composer faces currently selected.
    pub fn selected_count(&self) -> usize {
        self.surfaces.iter().filter(|s| s.selected).count()
    }

    /// Queue a solid built from the selected surfaces' loops.
    pub fn compose_selected(&mut self) {
        let loops: Vec<Vec<[f32; 3]>> = self
            .surfaces
            .iter()
            .filter(|s| s.selected)
            .map(|s| s.loop3.clone())
            .collect();
        if loops.len() < 2 {
            self.status = Some(StatusMessage::error(
                t!("sketch.need_two_faces").to_string(),
            ));
            return;
        }
        self.next_id += 1;
        let label = t!("sketch.solid_name", id = self.next_id).to_string();
        self.pending.push(SketchRequest::Solid { loops, label });
        self.status = Some(StatusMessage::info(t!("sketch.solid_queued").to_string()));
    }

    /// Poll nothing; draw the window. Returns nothing (status is internal).
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.show_window {
            return;
        }
        let mut open = true;
        let screen_center = ctx.screen_rect().center();
        egui::Window::new(t!("sketch.window_title").to_string())
            .open(&mut open)
            .default_size([440.0, 520.0])
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(screen_center)
            .resizable(true)
            .show(ctx, |ui| self.contents(ui));
        if !open {
            self.show_window = false;
        }
    }

    fn contents(&mut self, ui: &mut egui::Ui) {
        // --- Plane ---
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new(t!("sketch.plane_heading").to_string()).heading());
        });
        ui.horizontal(|ui| {
            ui.label(t!("sketch.plane").to_string());
            ui.selectable_value(&mut self.plane, PlaneChoice::Xy, "XY");
            ui.selectable_value(&mut self.plane, PlaneChoice::Xz, "XZ");
            ui.selectable_value(&mut self.plane, PlaneChoice::Yz, "YZ");
        });
        ui.horizontal(|ui| {
            ui.label(t!("sketch.offset").to_string());
            for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*axis);
                ui.add(egui::DragValue::new(&mut self.offset[i]).speed(0.1));
            }
        });

        ui.add_space(6.0);
        ui.separator();

        // --- Shape ---
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new(t!("sketch.shape_heading").to_string()).heading());
        });
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.shape,
                ShapeMode::Polygon,
                t!("sketch.shape_polygon").to_string(),
            );
            ui.selectable_value(
                &mut self.shape,
                ShapeMode::Circle,
                t!("sketch.shape_circle").to_string(),
            );
            ui.selectable_value(
                &mut self.shape,
                ShapeMode::Custom,
                t!("sketch.shape_custom").to_string(),
            );
        });

        match self.shape {
            ShapeMode::Polygon => {
                ui.horizontal(|ui| {
                    ui.label(t!("sketch.sides").to_string());
                    ui.add(egui::Slider::new(&mut self.n_sides, 3..=64));
                });
                ui.horizontal(|ui| {
                    ui.label(t!("sketch.radius").to_string());
                    ui.add(egui::Slider::new(&mut self.radius, 0.1..=10.0));
                });
            }
            ShapeMode::Circle => {
                ui.horizontal(|ui| {
                    ui.label(t!("sketch.radius").to_string());
                    ui.add(egui::Slider::new(&mut self.radius, 0.1..=10.0));
                });
            }
            ShapeMode::Custom => {
                ui.checkbox(&mut self.closed, t!("sketch.closed").to_string());
                ui.label(egui::RichText::new(t!("sketch.custom_hint").to_string()).weak());
                ui.add(
                    egui::TextEdit::multiline(&mut self.custom_text)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY)
                        .hint_text("0,0\n2,0\n2,2\n0,2"),
                );
            }
        }

        ui.add_space(4.0);
        ui.vertical_centered(|ui| {
            if ui.button(t!("sketch.create_surface").to_string()).clicked() {
                self.create_surface();
            }
        });

        if let Some(status) = &self.status {
            ui.add_space(4.0);
            ui.vertical_centered(|ui| status.show(ui));
        }

        // --- Composer ---
        ui.add_space(8.0);
        ui.separator();
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new(t!("sketch.composer_heading").to_string()).heading());
            ui.label(egui::RichText::new(t!("sketch.composer_hint").to_string()).weak());
        });

        if self.surfaces.is_empty() {
            ui.add_space(8.0);
            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new(t!("sketch.no_surfaces").to_string()).weak());
            });
        } else {
            let mut remove_idx = None;
            egui::ScrollArea::vertical()
                .max_height(160.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for (i, s) in self.surfaces.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut s.selected, "");
                            ui.label(egui::RichText::new(&s.name).strong());
                            ui.label(
                                egui::RichText::new(format!("({} pts)", s.loop3.len())).weak(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("🗑").clicked() {
                                        remove_idx = Some(i);
                                    }
                                },
                            );
                        });
                    }
                });
            if let Some(i) = remove_idx {
                self.surfaces.remove(i);
            }
            ui.add_space(4.0);
            ui.vertical_centered(|ui| {
                let can = self.selected_count() >= 2;
                if ui
                    .add_enabled(
                        can,
                        egui::Button::new(t!("sketch.compose_solid").to_string()),
                    )
                    .clicked()
                {
                    self.compose_selected();
                }
            });
        }
    }
}

/// Parse a custom point list: one `x,y` (or `x y`) per non-empty line.
fn parse_points(text: &str) -> Result<Vec<Vec2>, String> {
    let mut pts = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line
            .split([',', ' ', '\t'])
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() != 2 {
            return Err(t!("sketch.bad_point_line", line = (i + 1)).to_string());
        }
        let x = parts[0]
            .parse::<f32>()
            .map_err(|_| t!("sketch.bad_point_line", line = (i + 1)).to_string())?;
        let y = parts[1]
            .parse::<f32>()
            .map_err(|_| t!("sketch.bad_point_line", line = (i + 1)).to_string())?;
        pts.push([x, y]);
    }
    if pts.len() < 2 {
        return Err(t!("sketch.too_few_points").to_string());
    }
    Ok(pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_surface_is_queued_and_stored_as_a_face() {
        let mut view = SketchView::new();
        view.shape = ShapeMode::Polygon;
        view.n_sides = 6;
        view.create_surface();
        // One closed face stored for the composer + one surface request queued.
        assert_eq!(view.surfaces.len(), 1);
        assert_eq!(view.surfaces[0].loop3.len(), 6);
        let reqs = view.take_requests();
        assert_eq!(reqs.len(), 1);
        assert!(matches!(reqs[0], SketchRequest::Surface { .. }));
        // Draining clears the queue.
        assert!(view.take_requests().is_empty());
    }

    #[test]
    fn open_custom_polyline_is_not_stored_as_a_face() {
        let mut view = SketchView::new();
        view.shape = ShapeMode::Custom;
        view.closed = false;
        view.custom_text = "0,0\n1,0\n1,1".into();
        view.create_surface();
        // Open polyline still produces a (ribbon) surface request…
        let reqs = view.take_requests();
        assert_eq!(reqs.len(), 1);
        // …but is not a closed face, so it can't seed a solid.
        assert!(view.surfaces.is_empty());
    }

    #[test]
    fn invalid_custom_input_sets_an_error_and_queues_nothing() {
        let mut view = SketchView::new();
        view.shape = ShapeMode::Custom;
        view.custom_text = "not a point".into();
        view.create_surface();
        assert!(view.take_requests().is_empty());
        assert!(matches!(
            view.status.as_ref().unwrap().kind,
            super::super::StatusKind::Error
        ));
    }

    #[test]
    fn composing_needs_at_least_two_selected_faces() {
        let mut view = SketchView::new();
        view.shape = ShapeMode::Polygon;
        view.create_surface(); // one face
        view.take_requests();
        view.compose_selected();
        // Only one face selected → error, nothing queued.
        assert!(view.take_requests().is_empty());
        assert!(matches!(
            view.status.as_ref().unwrap().kind,
            super::super::StatusKind::Error
        ));

        // Add a second face, then composing succeeds.
        view.create_surface();
        view.take_requests();
        view.compose_selected();
        let reqs = view.take_requests();
        assert_eq!(reqs.len(), 1);
        match &reqs[0] {
            SketchRequest::Solid { loops, .. } => assert_eq!(loops.len(), 2),
            _ => panic!("expected a solid request"),
        }
    }

    #[test]
    fn parse_points_accepts_commas_and_spaces() {
        assert_eq!(
            parse_points("0,0\n1 2\n3,4").unwrap(),
            vec![[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]
        );
        assert!(parse_points("0,0").is_err(), "need at least two points");
        assert!(parse_points("1,2,3").is_err(), "two coords per line");
    }
}
