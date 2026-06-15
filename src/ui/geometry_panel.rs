//! Geometry import window: paste geometry text, import files from many
//! formats, and manage the resulting layers (visibility, focus, removal).
//!
//! Mirrors the dataset window's architecture: state lives in
//! [`GeometryView`], file parsing runs on a worker thread, and the host
//! (`State`) drains a dirty flag to rebuild GPU instance buffers only when
//! the visible layer set changes.

use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use crate::geometry::{build_batches, loader, GeometryLayer};
use crate::model::InstanceRaw;
use crate::scene::GeometryType;
use crate::visualization::color_mapper::color_for_label;

use super::StatusMessage;

/// Root state of the geometry import window.
pub struct GeometryView {
    /// Whether the window is visible.
    pub show_window: bool,
    /// Multiline paste area content.
    pub paste_text: String,
    /// File path field content.
    pub path_text: String,
    /// Last import outcome.
    pub status: Option<StatusMessage>,
    /// True while a file import worker is running.
    pub loading: bool,
    /// Imported layers, drawn in order.
    pub layers: Vec<GeometryLayer>,

    /// File path field for 3D solid model import (STL/OBJ/glTF).
    pub mesh_path_text: String,
    /// True while the host is loading a 3D model on a worker thread.
    pub mesh_loading: bool,
    /// A 3D-model path the user asked to import; the host
    /// ([`crate::state::State`]) drains it, loads the mesh on a worker thread
    /// and adds it to the scene as a new object.
    mesh_request: Option<std::path::PathBuf>,

    /// Counter used to name pasted layers and pick default colors.
    next_layer_id: usize,
    render_dirty: bool,
    worker: Option<Receiver<Result<GeometryLayer, String>>>,
}

impl Default for GeometryView {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometryView {
    pub fn new() -> Self {
        Self {
            show_window: false,
            paste_text: String::new(),
            path_text: String::new(),
            status: None,
            loading: false,
            layers: Vec::new(),
            mesh_path_text: String::new(),
            mesh_loading: false,
            mesh_request: None,
            next_layer_id: 0,
            render_dirty: false,
            worker: None,
        }
    }

    /// True once per change to the visible layer set (host rebuilds buffers).
    pub fn take_render_dirty(&mut self) -> bool {
        std::mem::take(&mut self.render_dirty)
    }

    /// True while the import worker is running.
    pub fn is_loading(&self) -> bool {
        self.loading
    }

    /// Total records across all layers (visible or not).
    pub fn total_records(&self) -> usize {
        self.layers.iter().map(|l| l.len()).sum()
    }

    /// Default color for the next layer, cycling the label palette so each
    /// imported layer is visually distinct out of the box.
    pub fn next_default_color(&self) -> [f32; 3] {
        color_for_label(self.next_layer_id as u32)
    }

    /// Add a layer and report success.
    pub fn add_layer(&mut self, layer: GeometryLayer) {
        self.status = Some(StatusMessage::success(format!(
            "Added layer '{}' with {} geometries",
            layer.name,
            layer.len()
        )));
        self.layers.push(layer);
        self.next_layer_id += 1;
        self.render_dirty = true;
    }

    /// Parse the paste area into a new layer.
    pub fn import_pasted(&mut self) {
        let name = format!("pasted_{}", self.next_layer_id + 1);
        match loader::layer_from_string(&self.paste_text, name, self.next_default_color()) {
            Ok(layer) => {
                self.add_layer(layer);
                self.paste_text.clear();
            }
            Err(e) => self.status = Some(StatusMessage::error(format!("Parse failed: {}", e))),
        }
    }

    /// Start a background import of the file in `path_text`.
    pub fn import_file(&mut self) {
        let path = PathBuf::from(self.path_text.trim());
        self.loading = true;
        self.status = Some(StatusMessage::info(format!("Loading {} ...", path.display())));
        let default_color = self.next_default_color();
        let (tx, rx) = std::sync::mpsc::channel();
        self.worker = Some(rx);
        std::thread::spawn(move || {
            let result =
                loader::layer_from_path(&path, default_color).map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
    }

    /// Instanced batches for all visible layers.
    pub fn build_geometry_batches(&self) -> Vec<(GeometryType, Vec<InstanceRaw>)> {
        build_batches(&self.layers)
    }

    /// Take a pending 3D-model import request, if the user clicked "Import 3D
    /// model" this frame. The host loads the file and adds it to the scene.
    pub fn take_mesh_request(&mut self) -> Option<std::path::PathBuf> {
        self.mesh_request.take()
    }

    fn poll_worker(&mut self) {
        let Some(rx) = &self.worker else { return };
        match rx.try_recv() {
            Ok(Ok(layer)) => {
                self.worker = None;
                self.loading = false;
                self.add_layer(layer);
            }
            Ok(Err(msg)) => {
                self.worker = None;
                self.loading = false;
                self.status = Some(StatusMessage::error(format!("Import failed: {}", msg)));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.worker = None;
                self.loading = false;
                self.status = Some(StatusMessage::error("Import worker died unexpectedly"));
            }
        }
    }

    /// Poll the worker, draw the window; returns a camera focus target when
    /// the user clicks a layer's focus button.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<[f32; 3]> {
        self.poll_worker();
        if !self.show_window {
            return None;
        }

        let mut focus = None;
        let mut open = true;
        let screen_center = ctx.screen_rect().center();
        egui::Window::new("🧊 Solids Import")
            .open(&mut open)
            .default_size([460.0, 480.0])
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(screen_center)
            .resizable(true)
            .show(ctx, |ui| {
                focus = self.contents(ui);
            });
        if !open {
            self.show_window = false;
        }
        focus
    }

    fn contents(&mut self, ui: &mut egui::Ui) -> Option<[f32; 3]> {
        let mut focus = None;

        // --- 3D solid model import (STL / OBJ / glTF) ---
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new("Import a 3D model").heading());
            ui.label(egui::RichText::new("STL · OBJ · glTF / GLB · STEP").weak());
        });
        ui.horizontal(|ui| {
            ui.label("Model");
            ui.add(
                egui::TextEdit::singleline(&mut self.mesh_path_text)
                    .desired_width(f32::INFINITY)
                    .hint_text("path/to/model.stl"),
            );
        });
        ui.vertical_centered(|ui| {
            ui.horizontal(|ui| {
                let can_load = !self.mesh_loading && !self.mesh_path_text.trim().is_empty();
                if ui
                    .add_enabled(can_load, egui::Button::new("📥 Import 3D model"))
                    .clicked()
                {
                    self.mesh_request =
                        Some(std::path::PathBuf::from(self.mesh_path_text.trim()));
                    self.mesh_loading = true;
                }
                if self.mesh_loading {
                    ui.spinner();
                }
            });
        });

        ui.add_space(8.0);
        ui.separator();

        // --- Paste area ---
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new("Paste geometry data").heading());
            ui.label(
                egui::RichText::new(
                    "DSL (`cube 0 0 0 2 #ff8800`), XYZ points or JSON — auto-detected",
                )
                .weak(),
            );
        });
        ui.add_space(4.0);
        ui.add(
            egui::TextEdit::multiline(&mut self.paste_text)
                .desired_rows(4)
                .desired_width(f32::INFINITY)
                .hint_text("cube 0 0 0 2 #ff8800 base\nsphere 0 0 2 0.5 color=0,1,0"),
        );
        ui.vertical_centered(|ui| {
            let can_parse = !self.paste_text.trim().is_empty();
            if ui
                .add_enabled(can_parse, egui::Button::new("➕ Add layer from text"))
                .clicked()
            {
                self.import_pasted();
            }
        });

        ui.add_space(8.0);
        ui.separator();

        // --- File import ---
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new("Import a file").heading());
            // Tabular data (CSV/Excel) lives in the Dataset window now; Solids
            // covers geometry descriptions and (soon) 3D mesh formats.
            ui.label(egui::RichText::new("JSON · XYZ · TXT (DSL)").weak());
        });
        ui.horizontal(|ui| {
            ui.label("File");
            ui.add(
                egui::TextEdit::singleline(&mut self.path_text)
                    .desired_width(f32::INFINITY)
                    .hint_text("path/to/geometries.json"),
            );
        });
        ui.vertical_centered(|ui| {
            ui.horizontal(|ui| {
                let can_load = !self.loading && !self.path_text.trim().is_empty();
                if ui
                    .add_enabled(can_load, egui::Button::new("📂 Import file"))
                    .clicked()
                {
                    self.import_file();
                }
                if self.loading {
                    ui.spinner();
                }
            });
        });

        if let Some(status) = &self.status {
            ui.add_space(4.0);
            ui.vertical_centered(|ui| status.show(ui));
        }

        // --- Layer list ---
        if !self.layers.is_empty() {
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Layers").heading());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!("{} geometries total", self.total_records()))
                            .weak(),
                    );
                });
            });

            let mut remove_idx = None;
            let mut changed = false;
            egui::ScrollArea::vertical()
                .max_height(180.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for (i, layer) in self.layers.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let eye = if layer.visible { "👁" } else { "🕶" };
                            if ui
                                .button(eye)
                                .on_hover_text("Toggle layer visibility")
                                .clicked()
                            {
                                layer.visible = !layer.visible;
                                changed = true;
                            }
                            ui.label(egui::RichText::new(&layer.name).strong());
                            ui.label(egui::RichText::new(format!("({})", layer.len())).weak());
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .button("🗑")
                                        .on_hover_text("Remove layer")
                                        .clicked()
                                    {
                                        remove_idx = Some(i);
                                    }
                                    if ui
                                        .button("🎯")
                                        .on_hover_text("Focus camera on layer")
                                        .clicked()
                                    {
                                        focus = layer.centroid();
                                    }
                                },
                            );
                        });
                    }
                });
            if let Some(i) = remove_idx {
                self.layers.remove(i);
                changed = true;
            }
            if changed {
                self.render_dirty = true;
            }
        } else {
            ui.add_space(12.0);
            ui.vertical_centered(|ui| {
                ui.label(egui::RichText::new("No layers yet — paste data or import a file.").weak());
            });
        }
        focus
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::GeometryRecord;

    fn layer(n: usize) -> GeometryLayer {
        GeometryLayer::new(
            format!("l{}", n),
            (0..n)
                .map(|i| GeometryRecord::new(GeometryType::Cube, [i as f32, 0.0, 0.0]))
                .collect(),
        )
    }

    #[test]
    fn add_layer_sets_status_and_dirty() {
        let mut view = GeometryView::new();
        assert!(!view.take_render_dirty());
        view.add_layer(layer(3));
        assert!(view.take_render_dirty());
        assert_eq!(view.total_records(), 3);
        assert!(matches!(
            view.status.as_ref().unwrap().kind,
            super::super::StatusKind::Success
        ));
    }

    #[test]
    fn layer_default_colors_cycle_the_palette() {
        let mut view = GeometryView::new();
        let c0 = view.next_default_color();
        view.add_layer(layer(1));
        let c1 = view.next_default_color();
        assert_ne!(c0, c1);
    }

    #[test]
    fn pasted_text_becomes_a_layer_and_clears_the_box() {
        let mut view = GeometryView::new();
        view.paste_text = "cube 0 0 0\nsphere 1 1 1 0.5".into();
        view.import_pasted();
        assert_eq!(view.layers.len(), 1);
        assert_eq!(view.layers[0].len(), 2);
        assert!(view.paste_text.is_empty());

        // Bad text -> error status, no layer.
        view.paste_text = "triangle 0 0 0".into();
        view.import_pasted();
        assert_eq!(view.layers.len(), 1);
        assert!(matches!(
            view.status.as_ref().unwrap().kind,
            super::super::StatusKind::Error
        ));
    }

    #[test]
    fn batches_follow_layer_visibility() {
        let mut view = GeometryView::new();
        view.add_layer(layer(4));
        assert_eq!(view.build_geometry_batches()[0].1.len(), 4);
        view.layers[0].visible = false;
        assert!(view.build_geometry_batches().is_empty());
    }
}
