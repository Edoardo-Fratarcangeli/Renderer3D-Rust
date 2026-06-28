//! LLM / SLM parameter visualizer window.
//!
//! Mirrors the architecture of [`super::geometry_panel`]:
//!  - [`LlmView`] owns all state and is embedded in `State`.
//!  - `State` calls [`LlmView::show`] every frame; the returned
//!    [`LlmAction`] is applied after the egui closure.
//!  - [`LlmView::build_render_data`] produces the [`LlmRenderData`] that
//!    `State` uploads to GPU and draws via the existing triangle / line
//!    pipelines.
//!
//! ### Shader convention
//! Node sphere instances use `InstanceRaw::color.a` to encode glow state:
//!  - `alpha = 1.0` → inactive node (normal diffuse shading).
//!  - `alpha ∈ (1.5, 2.5)` → selected object (existing golden glow path).
//!  - `alpha ∈ [3.0, 4.0]` → LLM active node; triggers the cyan→white
//!    emissive path added to the fragment shader.
//!    `alpha - 3.0` is the glow intensity ∈ [0, 1].

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use cgmath::Matrix4;

use crate::llm::activation::ActivationState;
use crate::llm::loader;
use crate::llm::network::{LayerKind, NetworkGraph, NODE_BASE_SCALE, NODE_MAX_SCALE};
use crate::llm::tokenizer::Tokenizer;
use crate::model::{InstanceRaw, Vertex};

use super::StatusMessage;

// ─── Public action type ───────────────────────────────────────────────────────

/// What the LLM panel asks the host (`State`) to do this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LlmAction {
    None,
    /// Move the camera target to the graph centroid.
    FocusGraph([f32; 3]),
}

// ─── Render data ─────────────────────────────────────────────────────────────

/// GPU-ready data produced by [`LlmView::build_render_data`].
pub struct LlmRenderData {
    /// One `InstanceRaw` per visible node (sphere instances).
    pub node_instances: Vec<InstanceRaw>,
    /// Interleaved line-list vertex pairs – one pair per edge.
    pub edge_vertices: Vec<Vertex>,
}

// ─── LlmView ─────────────────────────────────────────────────────────────────

pub struct LlmView {
    /// Whether the import/control window is visible.
    pub show_window: bool,
    /// The loaded network graph (none until a model file is imported).
    pub graph: Option<NetworkGraph>,
    /// Tokenizer derived from the embedded vocabulary, if any.
    pub tokenizer: Option<Tokenizer>,
    /// Current activation animation state.
    pub activation: Option<ActivationState>,

    /// Text in the prompt input field.
    pub prompt_text: String,
    /// Text in the file-path input field.
    pub path_text: String,
    /// Last import / animation status message.
    pub status: Option<StatusMessage>,
    /// Whether the 3D visualization is drawn.
    pub visible: bool,
    /// Uniform multiplier applied on top of the weight-driven node radius.
    pub node_scale_mult: f32,

    render_dirty: bool,
    animation_active: bool,
    worker: Option<mpsc::Receiver<Result<(NetworkGraph, Option<Tokenizer>), String>>>,
}

impl Default for LlmView {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmView {
    pub fn new() -> Self {
        Self {
            show_window: false,
            graph: None,
            tokenizer: None,
            activation: None,
            prompt_text: String::new(),
            path_text: String::new(),
            status: None,
            visible: true,
            node_scale_mult: 1.0,
            render_dirty: false,
            animation_active: false,
            worker: None,
        }
    }

    // ── Dirty-flag ──────────────────────────────────────────────────────────

    /// True each frame while an animation is running; true exactly once when a
    /// rebuild is required for any other reason (load, clear, visibility change).
    pub fn take_render_dirty(&mut self) -> bool {
        if self.animation_active {
            if let Some(anim) = &self.activation {
                if anim.is_finished(Instant::now()) {
                    self.animation_active = false;
                    self.render_dirty = true; // One final quiet frame
                }
            } else {
                self.animation_active = false;
            }
            return true;
        }
        std::mem::take(&mut self.render_dirty)
    }

    pub fn mark_dirty(&mut self) {
        self.render_dirty = true;
    }

    pub fn is_loading(&self) -> bool {
        self.worker.is_some()
    }

    // ── Loading ─────────────────────────────────────────────────────────────

    /// Spawn a worker thread that parses `path` off the UI thread.
    pub fn load_file(&mut self, path: PathBuf) {
        self.status = Some(StatusMessage::info(format!(
            "Loading {}…",
            path.display()
        )));
        let (tx, rx) = mpsc::channel();
        self.worker = Some(rx);
        std::thread::spawn(move || {
            let result = std::fs::read(&path)
                .map_err(|e| e.to_string())
                .and_then(|data| {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    match ext.as_str() {
                        "gguf" => loader::from_gguf(&data).map_err(|e| e.to_string()),
                        "json" => loader::from_json(&data).map_err(|e| e.to_string()),
                        other  => Err(format!(
                            "Unsupported model format '.{other}'. Use .json or .gguf"
                        )),
                    }
                });
            let _ = tx.send(result);
        });
    }

    /// Poll the import worker and install the result.
    pub fn poll_worker(&mut self) {
        let Some(rx) = &self.worker else { return };
        match rx.try_recv() {
            Ok(Ok((graph, tokenizer))) => {
                self.worker = None;
                let n_layers = graph.layers.len();
                let n_nodes  = graph.node_count();
                let n_edges  = graph.edges.len();
                self.status = Some(StatusMessage::success(format!(
                    "Loaded '{}' — {n_layers} layers · {n_nodes} nodes · {n_edges} edges",
                    graph.name
                )));
                self.graph     = Some(graph);
                self.tokenizer = tokenizer;
                self.activation = None;
                self.animation_active = false;
                self.visible = true;
                self.render_dirty = true;
            }
            Ok(Err(msg)) => {
                self.worker = None;
                self.status = Some(StatusMessage::error(msg));
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.worker = None;
                self.status = Some(StatusMessage::error(
                    "Import worker thread died unexpectedly".to_owned(),
                ));
            }
        }
    }

    // ── Prompt injection ────────────────────────────────────────────────────

    /// Tokenize `self.prompt_text` and start the activation animation.
    pub fn inject_prompt(&mut self) {
        let Some(graph) = &self.graph else { return };

        let tokens: Vec<u32> = if let Some(tok) = &self.tokenizer {
            tok.encode(&self.prompt_text)
        } else {
            // No vocabulary — hash each whitespace-split word.
            self.prompt_text
                .split_whitespace()
                .map(|w| {
                    let mut h: u32 = 5381;
                    for b in w.bytes() {
                        h = h.wrapping_mul(33).wrapping_add(b as u32);
                    }
                    h % 256
                })
                .collect()
        };

        if tokens.is_empty() {
            self.status = Some(StatusMessage::error("Prompt is empty".to_owned()));
            return;
        }

        let word_count = self.prompt_text.split_whitespace().count();
        let preview: Vec<&str> = self.prompt_text.split_whitespace().take(6).collect();
        let preview_str = if word_count > 6 {
            format!("{}…", preview.join(" "))
        } else {
            preview.join(" ")
        };

        self.status = Some(StatusMessage::info(format!(
            "Injecting {word_count} token(s): '{preview_str}'"
        )));
        self.activation = Some(ActivationState::simulate(graph, &tokens));
        self.animation_active = true;
        self.render_dirty = true;
    }

    /// Remove the loaded model and reset all state.
    pub fn clear(&mut self) {
        self.graph = None;
        self.tokenizer = None;
        self.activation = None;
        self.animation_active = false;
        self.visible = false;
        self.render_dirty = true;
        self.status = None;
    }

    pub fn set_visible(&mut self, visible: bool) {
        if self.visible != visible {
            self.visible = visible;
            self.render_dirty = true;
        }
    }

    /// World-space centroid of the network, used to focus the camera.
    pub fn centroid(&self) -> Option<[f32; 3]> {
        self.graph.as_ref()?.centroid()
    }

    // ── GPU data ────────────────────────────────────────────────────────────

    /// Build all renderable data for the current frame.
    ///
    /// Called each frame that `take_render_dirty()` returns `true`.
    pub fn build_render_data(&self) -> LlmRenderData {
        if !self.visible {
            return LlmRenderData { node_instances: vec![], edge_vertices: vec![] };
        }
        let Some(graph) = &self.graph else {
            return LlmRenderData { node_instances: vec![], edge_vertices: vec![] };
        };

        let now = Instant::now();

        // ── Node instances ─────────────────────────────────────────────────
        let mut node_instances = Vec::new();
        for (li, layer) in graph.layers.iter().enumerate() {
            let base_col = layer.kind.base_color();
            for (ni, node) in layer.nodes.iter().enumerate() {
                let glow = self
                    .activation
                    .as_ref()
                    .map(|a| a.glow_at(li, ni, now))
                    .unwrap_or(0.0);

                let scale =
                    (NODE_BASE_SCALE + node.weight_magnitude * (NODE_MAX_SCALE - NODE_BASE_SCALE))
                        * self.node_scale_mult
                        * (1.0 + glow * 0.5); // swell on activation

                let [r, g, b] = if glow > 0.01 {
                    let cyan  = [0.0f32, 0.65, 1.0];
                    let white = [1.0f32, 0.95, 0.85];
                    lerp3(lerp3(base_col, cyan, glow), white, glow * glow)
                } else {
                    // Dim inactive nodes so active ones pop visually.
                    [base_col[0] * 0.45, base_col[1] * 0.45, base_col[2] * 0.45]
                };

                // alpha ≥ 3.0 triggers the LLM emissive path in the shader.
                let alpha = if glow > 0.01 { 3.0 + glow } else { 1.0 };

                let p = node.position;
                let model = Matrix4::from_translation(cgmath::Vector3::new(p[0], p[1], p[2]))
                    * Matrix4::from_scale(scale);

                node_instances.push(InstanceRaw {
                    model: model.into(),
                    color: [r, g, b, alpha],
                });
            }
        }

        // ── Edge vertices (line-list pairs) ────────────────────────────────
        let mut edge_vertices = Vec::new();
        for edge in &graph.edges {
            let Some(fl) = graph.layers.get(edge.from_layer) else { continue };
            let Some(tl) = graph.layers.get(edge.to_layer)   else { continue };
            let Some(fn_) = fl.nodes.get(edge.from_node)     else { continue };
            let Some(tn)  = tl.nodes.get(edge.to_node)       else { continue };

            let gf = self.activation.as_ref().map(|a| a.glow_at(edge.from_layer, edge.from_node, now)).unwrap_or(0.0);
            let gt = self.activation.as_ref().map(|a| a.glow_at(edge.to_layer, edge.to_node, now)).unwrap_or(0.0);
            let edge_glow = (gf + gt) * 0.5;

            let dim  = [0.10f32, 0.14, 0.26];
            let lit  = [0.00f32, 0.65, 1.00];
            let col  = lerp3(dim, lit, edge_glow);

            let normal = [0.0f32, 0.0, 1.0];
            edge_vertices.push(Vertex { position: fn_.position, color: col, normal });
            edge_vertices.push(Vertex { position: tn.position,  color: col, normal });
        }

        LlmRenderData { node_instances, edge_vertices }
    }

    // ── egui UI ─────────────────────────────────────────────────────────────

    /// Poll the worker and draw the LLM window. Returns a host action.
    pub fn show(&mut self, ctx: &egui::Context) -> LlmAction {
        self.poll_worker();

        if !self.show_window {
            return LlmAction::None;
        }

        // Keep repainting while animation runs.
        if self.animation_active {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }

        let mut action = LlmAction::None;
        let mut open   = true;
        let center     = ctx.screen_rect().center();

        egui::Window::new(t!("llm.window_title").to_string())
            .open(&mut open)
            .fixed_size([480.0, 540.0])
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(center)
            .show(ctx, |ui| {
                action = self.window_body(ui);
            });

        if !open {
            self.show_window = false;
        }
        action
    }

    fn window_body(&mut self, ui: &mut egui::Ui) -> LlmAction {
        let mut action = LlmAction::None;
        let mut do_clear = false;

        // ── Import section ─────────────────────────────────────────────────
        ui.heading(t!("llm.heading_import").to_string());
        ui.label(t!("llm.import_hint").to_string());
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.path_text)
                    .hint_text("model.gguf  or  model.json")
                    .desired_width(310.0),
            );
            let loading = self.is_loading();
            if ui
                .add_enabled(!loading, egui::Button::new(t!("llm.btn_load").to_string()))
                .clicked()
            {
                let p = PathBuf::from(&self.path_text);
                if p.exists() {
                    self.load_file(p);
                } else {
                    self.status = Some(StatusMessage::error(format!(
                        "{}: {}",
                        t!("llm.err_not_found"),
                        self.path_text
                    )));
                }
            }
            if loading {
                ui.spinner();
            }
        });

        if let Some(s) = &self.status {
            s.show(ui);
        }

        ui.separator();

        // ── Model info ─────────────────────────────────────────────────────
        if let Some(graph) = &self.graph {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(&graph.name)
                        .strong()
                        .size(15.0),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(t!("llm.btn_clear").to_string()).clicked() {
                        do_clear = true;
                    }
                    if let Some(c) = self.centroid() {
                        if ui.button(t!("llm.btn_focus").to_string()).clicked() {
                            action = LlmAction::FocusGraph(c);
                        }
                    }
                });
            });

            let n_nodes: usize = graph.layers.iter().map(|l| l.nodes.len()).sum();
            ui.label(
                egui::RichText::new(format!(
                    "{} {} · {} {} · {} {}",
                    graph.layers.len(), t!("llm.layers"),
                    n_nodes, t!("llm.nodes"),
                    graph.edges.len(), t!("llm.edges"),
                ))
                .weak(),
            );

            egui::ScrollArea::vertical()
                .id_source("llm_layer_scroll")
                .max_height(130.0)
                .show(ui, |ui| {
                    for layer in &graph.layers {
                        ui.horizontal(|ui| {
                            let icon = match layer.kind {
                                LayerKind::Embedding   => "📥",
                                LayerKind::Attention   => "👁",
                                LayerKind::FeedForward => "⚡",
                                LayerKind::LayerNorm   => "📐",
                                LayerKind::Output      => "📤",
                            };
                            ui.label(icon);
                            ui.label(&layer.name);
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} {}",
                                            layer.nodes.len(),
                                            t!("llm.nodes")
                                        ))
                                        .weak(),
                                    );
                                },
                            );
                        });
                    }
                });

            ui.separator();
        }

        // ── Prompt section ─────────────────────────────────────────────────
        ui.heading(t!("llm.heading_prompt").to_string());

        let has_model = self.graph.is_some();
        ui.add_enabled_ui(has_model, |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut self.prompt_text)
                    .hint_text(t!("llm.prompt_hint").to_string())
                    .desired_rows(3)
                    .desired_width(f32::INFINITY),
            );

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button(t!("llm.btn_inject").to_string()).clicked() {
                    self.inject_prompt();
                }
                if ui.button(t!("llm.btn_clear_anim").to_string()).clicked() {
                    self.activation = None;
                    self.animation_active = false;
                    self.render_dirty = true;
                }
                if self.animation_active {
                    ui.spinner();
                    ui.colored_label(
                        egui::Color32::from_rgb(0, 200, 255),
                        t!("llm.propagating").to_string(),
                    );
                } else if self.activation.is_some() {
                    ui.colored_label(
                        egui::Color32::from_rgb(120, 220, 120),
                        t!("llm.activated").to_string(),
                    );
                }
            });

            ui.add_space(6.0);
            ui.add(
                egui::Slider::new(&mut self.node_scale_mult, 0.4..=3.0)
                    .text(t!("llm.node_scale").to_string()),
            );
        });

        if !has_model {
            ui.colored_label(
                egui::Color32::DARK_GRAY,
                t!("llm.no_model_hint").to_string(),
            );
        }

        if do_clear {
            self.clear();
        }
        action
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}
