//! LLM / SLM parameter visualizer window.
//!
//! ### Architecture overview
//! Mirrors [`super::geometry_panel`]:
//!  - [`LlmView`] owns all state; embedded in `State`.
//!  - `State` calls [`LlmView::show`] each frame; the returned [`LlmAction`]
//!    is applied after the egui closure.
//!  - [`LlmView::build_render_data`] produces [`LlmRenderData`] that `State`
//!    uploads and draws via the triangle / line pipelines.
//!
//! ### Tabs
//!  - **Model** — import .gguf/.json, layer inspector, prompt injection.
//!  - **Ollama** — browse locally-installed Ollama models, load graph,
//!                 run streaming inference with live token display.
//!
//! ### Shader convention
//! `InstanceRaw::color.a` encodes glow state:
//!  - `alpha = 1.0`        → inactive (diffuse only).
//!  - `alpha ∈ (1.5, 2.5)` → selected object (golden glow, existing path).
//!  - `alpha ∈ [3.0, 4.0]` → LLM active node; intensity = alpha − 3.0.
//!    The emissive color is read from `instance_color.rgb` (color-agnostic).

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use cgmath::Matrix4;

use crate::llm::activation::{ActivationMode, ActivationState};
use crate::llm::loader;
use crate::llm::network::{LayerKind, NetworkGraph, NODE_BASE_SCALE, NODE_MAX_SCALE};
use crate::llm::ollama::{self, OllamaEvent, OllamaModel};
use crate::llm::tokenizer::Tokenizer;
use crate::model::{InstanceRaw, Vertex};

use super::StatusMessage;

// ─── Constants ────────────────────────────────────────────────────────────────

const CYAN:   [f32; 3] = [0.00, 0.65, 1.00];
const WHITE:  [f32; 3] = [1.00, 0.95, 0.85];
const ORANGE: [f32; 3] = [1.00, 0.55, 0.05];
const YELLOW: [f32; 3] = [1.00, 0.90, 0.30];

// ─── Public action type ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LlmAction {
    None,
    FocusGraph([f32; 3]),
}

// ─── Render data ─────────────────────────────────────────────────────────────

pub struct LlmRenderData {
    pub node_instances: Vec<InstanceRaw>,
    pub edge_vertices:  Vec<Vertex>,
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum LlmTab {
    #[default]
    Model,
    Ollama,
}

// ─── LlmView ─────────────────────────────────────────────────────────────────

pub struct LlmView {
    pub show_window: bool,
    pub graph:       Option<NetworkGraph>,
    pub tokenizer:   Option<Tokenizer>,
    pub activation:  Option<ActivationState>,

    pub prompt_text:     String,
    pub path_text:       String,
    pub status:          Option<StatusMessage>,
    pub visible:         bool,
    pub node_scale_mult: f32,
    pub wave_speed: f32,

    render_dirty:     bool,
    animation_active: bool,

    // File import worker
    worker: Option<mpsc::Receiver<Result<(NetworkGraph, Option<Tokenizer>), String>>>,

    // ── Ollama ──────────────────────────────────────────────────────────────
    active_tab:    LlmTab,
    ollama_models: Vec<OllamaModel>,
    /// Which row is selected in the Ollama model list.
    ollama_sel:    Option<usize>,
    ollama_status: Option<StatusMessage>,
    /// Worker loading the Ollama model list.
    ollama_list_rx: Option<mpsc::Receiver<Result<Vec<OllamaModel>, String>>>,
    /// Worker loading an Ollama model graph via /api/show.
    ollama_graph_rx: Option<mpsc::Receiver<Result<(NetworkGraph, Option<Tokenizer>), String>>>,

    // ── Inference (Ollama streaming) ─────────────────────────────────────────
    inference_text:   String,
    inference_active: bool,
    inference_rx:     Option<mpsc::Receiver<OllamaEvent>>,
    /// Which Ollama model is currently wired for inference.
    inference_model:  String,
}

impl Default for LlmView {
    fn default() -> Self { Self::new() }
}

impl LlmView {
    pub fn new() -> Self {
        Self {
            show_window: false,
            graph:       None,
            tokenizer:   None,
            activation:  None,
            prompt_text: String::new(),
            path_text:   String::new(),
            status:      None,
            visible:     true,
            node_scale_mult: 1.0,
            wave_speed: 1.0,
            render_dirty:     false,
            animation_active: false,
            worker:           None,
            active_tab:       LlmTab::default(),
            ollama_models:    Vec::new(),
            ollama_sel:       None,
            ollama_status:    None,
            ollama_list_rx:   None,
            ollama_graph_rx:  None,
            inference_text:   String::new(),
            inference_active: false,
            inference_rx:     None,
            inference_model:  String::new(),
        }
    }

    // ── Dirty flag ──────────────────────────────────────────────────────────

    pub fn take_render_dirty(&mut self) -> bool {
        if self.animation_active {
            if let Some(anim) = &self.activation {
                if anim.is_finished(Instant::now()) {
                    self.animation_active = false;
                    self.render_dirty = true;
                }
            } else {
                self.animation_active = false;
            }
            return true;
        }
        std::mem::take(&mut self.render_dirty)
    }

    pub fn mark_dirty(&mut self) { self.render_dirty = true; }

    pub fn is_loading(&self) -> bool {
        self.worker.is_some() || self.ollama_graph_rx.is_some()
    }

    // ── File import ─────────────────────────────────────────────────────────

    pub fn load_file(&mut self, path: PathBuf) {
        self.status = Some(StatusMessage::info(format!("Loading {}…", path.display())));
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

    fn poll_worker(&mut self) {
        let Some(rx) = &self.worker else { return };
        match rx.try_recv() {
            Ok(Ok((graph, tokenizer))) => {
                self.worker = None;
                self.install_graph(graph, tokenizer, |g| {
                    format!("Loaded '{}' — {} layers · {} nodes · {} edges",
                        g.name, g.layers.len(), g.node_count(), g.edges.len())
                });
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

    // ── Ollama model list ───────────────────────────────────────────────────

    fn fetch_ollama_list(&mut self) {
        self.ollama_status = Some(StatusMessage::info("Connecting to Ollama…".to_owned()));
        let (tx, rx) = mpsc::channel();
        self.ollama_list_rx = Some(rx);
        std::thread::spawn(move || {
            let res = ollama::list_models().map_err(|e| e.to_string());
            let _ = tx.send(res);
        });
    }

    fn poll_ollama_list(&mut self) {
        let Some(rx) = &self.ollama_list_rx else { return };
        match rx.try_recv() {
            Ok(Ok(models)) => {
                self.ollama_list_rx = None;
                let n = models.len();
                self.ollama_models = models;
                self.ollama_sel = None;
                self.ollama_status = Some(StatusMessage::success(
                    format!("Found {n} Ollama model(s)")
                ));
            }
            Ok(Err(msg)) => {
                self.ollama_list_rx = None;
                self.ollama_status = Some(StatusMessage::error(msg));
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.ollama_list_rx = None;
                self.ollama_status = Some(StatusMessage::error(
                    "Ollama list worker died".to_owned(),
                ));
            }
        }
    }

    // ── Load Ollama model graph ─────────────────────────────────────────────

    fn load_ollama_model(&mut self, model_name: &str) {
        let name = model_name.to_owned();
        self.ollama_status = Some(StatusMessage::info(format!("Loading graph for {name}…")));
        let (tx, rx) = mpsc::channel();
        self.ollama_graph_rx = Some(rx);
        std::thread::spawn(move || {
            let res = ollama::load_model_graph(&name).map_err(|e| e.to_string());
            let _ = tx.send(res);
        });
    }

    fn poll_ollama_graph(&mut self) {
        let Some(rx) = &self.ollama_graph_rx else { return };
        match rx.try_recv() {
            Ok(Ok((graph, tokenizer))) => {
                self.ollama_graph_rx = None;
                // Set inference model name from graph name (strip " (Ollama)" suffix).
                let model_name = graph.name
                    .trim_end_matches(" (Ollama)")
                    .to_owned();
                self.inference_model = model_name;
                self.install_graph(graph, tokenizer, |g| {
                    format!("Loaded '{}' from Ollama — {} layers · {} nodes",
                        g.name, g.layers.len(), g.node_count())
                });
                self.ollama_status = self.status.clone();
            }
            Ok(Err(msg)) => {
                self.ollama_graph_rx = None;
                self.ollama_status = Some(StatusMessage::error(msg));
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.ollama_graph_rx = None;
                self.ollama_status = Some(StatusMessage::error(
                    "Ollama graph worker died".to_owned(),
                ));
            }
        }
    }

    // ── Inference streaming ─────────────────────────────────────────────────

    fn start_inference(&mut self) {
        let Some(graph) = &self.graph else { return };
        if self.inference_model.is_empty() {
            self.ollama_status = Some(StatusMessage::error(
                "No Ollama model selected. Load one from the Ollama tab first.".to_owned(),
            ));
            return;
        }
        if self.prompt_text.trim().is_empty() {
            self.ollama_status = Some(StatusMessage::error(
                "Prompt is empty".to_owned(),
            ));
            return;
        }

        self.inference_text.clear();
        self.inference_active = true;

        // Start a visual forward wave tied to the prompt.
        let tokens: Vec<u32> = self
            .prompt_text
            .split_whitespace()
            .map(|w| {
                let mut h: u32 = 5381;
                for b in w.bytes() { h = h.wrapping_mul(33).wrapping_add(b as u32); }
                h % 256
            })
            .collect();
        self.activation = Some(ActivationState::simulate(graph, &tokens).with_speed(self.wave_speed));
        self.animation_active = true;
        self.render_dirty = true;

        // Spawn the Ollama inference thread.
        self.inference_rx = Some(ollama::run_inference(
            &self.inference_model,
            &self.prompt_text,
        ));
        self.ollama_status = Some(StatusMessage::info(format!(
            "Running inference on '{}'…",
            self.inference_model
        )));
    }

    fn poll_inference(&mut self) {
        let Some(rx) = &self.inference_rx else { return };
        loop {
            match rx.try_recv() {
                Ok(OllamaEvent::Token(tok)) => {
                    self.inference_text.push_str(&tok);
                }
                Ok(OllamaEvent::Done) => {
                    self.inference_rx = None;
                    self.inference_active = false;
                    self.ollama_status = Some(StatusMessage::success("Inference complete".to_owned()));
                    break;
                }
                Ok(OllamaEvent::Error(e)) => {
                    self.inference_rx = None;
                    self.inference_active = false;
                    self.ollama_status = Some(StatusMessage::error(e));
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.inference_rx = None;
                    self.inference_active = false;
                    break;
                }
            }
        }
    }

    // ── Prompt injection (manual / file-loaded model) ───────────────────────

    pub fn inject_prompt(&mut self) {
        let Some(graph) = &self.graph else { return };

        let tokens: Vec<u32> = if let Some(tok) = &self.tokenizer {
            tok.encode(&self.prompt_text)
        } else {
            self.prompt_text
                .split_whitespace()
                .map(|w| {
                    let mut h: u32 = 5381;
                    for b in w.bytes() { h = h.wrapping_mul(33).wrapping_add(b as u32); }
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
        self.activation = Some(ActivationState::simulate(graph, &tokens).with_speed(self.wave_speed));
        self.animation_active = true;
        self.render_dirty = true;
    }

    // ── Training simulation ─────────────────────────────────────────────────

    fn start_training_sim(&mut self) {
        let Some(graph) = &self.graph else { return };
        self.activation = Some(ActivationState::simulate_training(graph).with_speed(self.wave_speed));
        self.animation_active = true;
        self.render_dirty = true;
        self.status = Some(StatusMessage::info(
            "Training simulation: forward (cyan) + backward gradient wave (orange)".to_owned(),
        ));
    }

    // ── Graph install helper ────────────────────────────────────────────────

    fn install_graph(
        &mut self,
        graph: NetworkGraph,
        tokenizer: Option<Tokenizer>,
        msg: impl FnOnce(&NetworkGraph) -> String,
    ) {
        let s = msg(&graph);
        self.status = Some(StatusMessage::success(s));
        self.graph = Some(graph);
        self.tokenizer = tokenizer;
        self.activation = None;
        self.animation_active = false;
        self.visible = true;
        self.render_dirty = true;
    }

    // ── Clear ───────────────────────────────────────────────────────────────

    pub fn clear(&mut self) {
        self.graph = None;
        self.tokenizer = None;
        self.activation = None;
        self.animation_active = false;
        self.inference_active = false;
        self.inference_rx = None;
        self.inference_text.clear();
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

    pub fn centroid(&self) -> Option<[f32; 3]> {
        self.graph.as_ref()?.centroid()
    }

    // ── GPU render data ─────────────────────────────────────────────────────

    pub fn build_render_data(&self) -> LlmRenderData {
        if !self.visible {
            return LlmRenderData { node_instances: vec![], edge_vertices: vec![] };
        }
        let Some(graph) = &self.graph else {
            return LlmRenderData { node_instances: vec![], edge_vertices: vec![] };
        };

        let now = Instant::now();

        let mut node_instances = Vec::new();
        for (li, layer) in graph.layers.iter().enumerate() {
            let base_col = layer.kind.base_color();
            for (ni, node) in layer.nodes.iter().enumerate() {
                let fwd  = self.activation.as_ref().map(|a| a.glow_at(li, ni, now)).unwrap_or(0.0);
                let back = self.activation.as_ref().map(|a| a.back_glow_at(li, ni, now)).unwrap_or(0.0);
                let total_glow = (fwd + back).min(1.0);

                let scale = (NODE_BASE_SCALE
                    + node.weight_magnitude * (NODE_MAX_SCALE - NODE_BASE_SCALE))
                    * self.node_scale_mult
                    * (1.0 + total_glow * 0.5);

                let [r, g, b] = if total_glow > 0.01 {
                    // Blend forward (cyan→white) and backward (orange→yellow) by strength.
                    let fwd_col  = lerp3(lerp3(base_col, CYAN, fwd), WHITE, fwd * fwd);
                    let back_col = lerp3(lerp3(base_col, ORANGE, back), YELLOW, back * back);
                    let w = fwd + back + 1e-8;
                    lerp3(back_col, fwd_col, fwd / w)
                } else {
                    [base_col[0] * 0.45, base_col[1] * 0.45, base_col[2] * 0.45]
                };

                let alpha = if total_glow > 0.01 { 3.0 + total_glow } else { 1.0 };

                let p = node.position;
                let model = Matrix4::from_translation(cgmath::Vector3::new(p[0], p[1], p[2]))
                    * Matrix4::from_scale(scale);

                node_instances.push(InstanceRaw {
                    model: model.into(),
                    color: [r, g, b, alpha],
                });
            }
        }

        let mut edge_vertices = Vec::new();
        for edge in &graph.edges {
            let Some(fl)  = graph.layers.get(edge.from_layer)  else { continue };
            let Some(tl)  = graph.layers.get(edge.to_layer)    else { continue };
            let Some(fn_) = fl.nodes.get(edge.from_node)       else { continue };
            let Some(tn)  = tl.nodes.get(edge.to_node)         else { continue };

            let gf = self.activation.as_ref()
                .map(|a| a.glow_at(edge.from_layer, edge.from_node, now))
                .unwrap_or(0.0);
            let gt = self.activation.as_ref()
                .map(|a| a.glow_at(edge.to_layer, edge.to_node, now))
                .unwrap_or(0.0);
            let bf = self.activation.as_ref()
                .map(|a| a.back_glow_at(edge.from_layer, edge.from_node, now))
                .unwrap_or(0.0);
            let bt = self.activation.as_ref()
                .map(|a| a.back_glow_at(edge.to_layer, edge.to_node, now))
                .unwrap_or(0.0);

            let dim  = [0.10f32, 0.14, 0.26];
            let fwd_glow  = (gf + gt) * 0.5;
            let back_glow = (bf + bt) * 0.5;
            let col = if back_glow > fwd_glow {
                lerp3(dim, ORANGE, back_glow)
            } else {
                lerp3(dim, CYAN, fwd_glow)
            };

            let normal = [0.0f32, 0.0, 1.0];
            edge_vertices.push(Vertex { position: fn_.position, color: col, normal });
            edge_vertices.push(Vertex { position: tn.position,  color: col, normal });
        }

        LlmRenderData { node_instances, edge_vertices }
    }

    // ── egui UI ─────────────────────────────────────────────────────────────

    pub fn show(&mut self, ctx: &egui::Context) -> LlmAction {
        self.poll_worker();
        self.poll_ollama_list();
        self.poll_ollama_graph();
        self.poll_inference();

        if !self.show_window {
            return LlmAction::None;
        }

        if self.animation_active || self.inference_active {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }

        let mut action = LlmAction::None;
        let mut open   = true;
        let center     = ctx.screen_rect().center();

        egui::Window::new(t!("llm.window_title").to_string())
            .open(&mut open)
            .fixed_size([520.0, 600.0])
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(center)
            .show(ctx, |ui| {
                action = self.window_body(ui);
            });

        if !open { self.show_window = false; }
        action
    }

    fn window_body(&mut self, ui: &mut egui::Ui) -> LlmAction {
        // ── Tab bar ────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.active_tab, LlmTab::Model,  t!("llm.tab_model").to_string());
            ui.selectable_value(&mut self.active_tab, LlmTab::Ollama, t!("llm.tab_ollama").to_string());
        });
        ui.separator();

        match self.active_tab {
            LlmTab::Model  => self.tab_model(ui),
            LlmTab::Ollama => { self.tab_ollama(ui); LlmAction::None }
        }
    }

    // ── Model tab ─────────────────────────────────────────────────────────────

    fn tab_model(&mut self, ui: &mut egui::Ui) -> LlmAction {
        let mut action   = LlmAction::None;
        let mut do_clear = false;

        // Import section
        ui.heading(t!("llm.heading_import").to_string());
        ui.label(t!("llm.import_hint").to_string());
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.path_text)
                    .hint_text("model.gguf  or  model.json")
                    .desired_width(310.0),
            );
            let loading = self.worker.is_some();
            if ui.add_enabled(!loading, egui::Button::new(t!("llm.btn_load").to_string())).clicked() {
                let p = PathBuf::from(&self.path_text);
                if p.exists() {
                    self.load_file(p);
                } else {
                    self.status = Some(StatusMessage::error(format!(
                        "{}: {}", t!("llm.err_not_found"), self.path_text
                    )));
                }
            }
            if loading { ui.spinner(); }
        });

        if let Some(s) = &self.status {
            s.show(ui);
        }
        ui.separator();

        // Model info
        if let Some(graph) = &self.graph {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&graph.name).strong().size(15.0));
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

            // Collect layer data to avoid borrow conflicts with self.activation inside egui closure.
            let layer_summaries: Vec<(String, LayerKind, usize, (f32, f32, f32))> = graph
                .layers
                .iter()
                .map(|l| (l.name.clone(), l.kind, l.nodes.len(), layer_stats(l)))
                .collect();

            egui::ScrollArea::vertical()
                .id_source("llm_layer_scroll")
                .max_height(120.0)
                .show(ui, |ui| {
                    for (li, (name, kind, node_count, (wmin, wmax, wmean))) in layer_summaries.iter().enumerate() {
                        // Compute mean_glow for this layer once per row (reused for bar and label).
                        let mean_glow: f32 = if self.animation_active {
                            let now = Instant::now();
                            (0..*node_count)
                                .map(|ni| self.activation.as_ref().map_or(0.0, |a| a.glow_at(li, ni, now)))
                                .sum::<f32>()
                                / (*node_count).max(1) as f32
                        } else {
                            0.0
                        };

                        let row_resp = ui.horizontal(|ui| {
                            let icon = match kind {
                                LayerKind::Embedding   => "📥",
                                LayerKind::Attention   => "👁",
                                LayerKind::FeedForward => "⚡",
                                LayerKind::LayerNorm   => "📐",
                                LayerKind::Output      => "📤",
                            };
                            ui.label(icon);
                            ui.label(name.as_str());
                            ui.add(
                                egui::ProgressBar::new(*wmean)
                                    .desired_width(50.0),
                            );
                            if self.animation_active && mean_glow > 0.005 {
                                ui.colored_label(
                                    egui::Color32::from_rgb(0, 180, 230),
                                    format!("{:.0}%", mean_glow * 100.0),
                                );
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{node_count} {}", t!("llm.nodes")
                                    ))
                                    .weak(),
                                );
                            });
                        }).response;
                        row_resp.on_hover_ui(|ui| {
                            ui.label(format!("Nodes: {node_count}"));
                            ui.label(format!("Weight  min {wmin:.3} · mean {wmean:.3} · max {wmax:.3}"));
                            if let Some(anim) = &self.activation {
                                let now = Instant::now();
                                let mean_glow: f32 = (0..*node_count)
                                    .map(|ni| anim.glow_at(li, ni, now))
                                    .sum::<f32>()
                                    / (*node_count).max(1) as f32;
                                ui.label(format!("Activation: {:.1} %", mean_glow * 100.0));
                            }
                        });
                    }
                });

            ui.separator();
        }

        // Prompt injection
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
                if ui.button(t!("llm.btn_train_step").to_string()).clicked() {
                    self.start_training_sim();
                }
                if ui.button(t!("llm.btn_clear_anim").to_string()).clicked() {
                    self.activation = None;
                    self.animation_active = false;
                    self.render_dirty = true;
                }
            });
            ui.horizontal(|ui| {
                if self.animation_active {
                    ui.spinner();
                    let label = match self.activation.as_ref().map(|a| a.mode) {
                        Some(ActivationMode::Training) => t!("llm.training").to_string(),
                        _                              => t!("llm.propagating").to_string(),
                    };
                    ui.colored_label(egui::Color32::from_rgb(0, 200, 255), label);
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
            ui.add(
                egui::Slider::new(&mut self.wave_speed, 0.25..=4.0)
                    .text(t!("llm.wave_speed").to_string()),
            );
        });

        if !has_model {
            ui.colored_label(egui::Color32::DARK_GRAY, t!("llm.no_model_hint").to_string());
        }

        if do_clear { self.clear(); }
        action
    }

    // ── Ollama tab ────────────────────────────────────────────────────────────

    fn tab_ollama(&mut self, ui: &mut egui::Ui) {
        ui.heading(t!("llm.ollama_heading").to_string());
        ui.label(t!("llm.ollama_hint").to_string());
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            let refreshing = self.ollama_list_rx.is_some();
            if ui.add_enabled(!refreshing, egui::Button::new(t!("llm.btn_refresh").to_string())).clicked() {
                self.fetch_ollama_list();
            }
            if refreshing { ui.spinner(); }

            // Load selected model graph
            let can_load = self.ollama_sel.is_some() && self.ollama_graph_rx.is_none();
            if ui.add_enabled(can_load, egui::Button::new(t!("llm.btn_load_model").to_string())).clicked() {
                if let Some(idx) = self.ollama_sel {
                    if let Some(m) = self.ollama_models.get(idx) {
                        let name = m.name.clone();
                        self.load_ollama_model(&name);
                    }
                }
            }
            if self.ollama_graph_rx.is_some() { ui.spinner(); }
        });

        if let Some(s) = &self.ollama_status.clone() {
            s.show(ui);
        }

        ui.add_space(4.0);

        // Model list
        if !self.ollama_models.is_empty() {
            egui::ScrollArea::vertical()
                .id_source("ollama_model_list")
                .max_height(200.0)
                .show(ui, |ui| {
                    // Clone to avoid borrow conflict with self.ollama_sel
                    let models: Vec<_> = self.ollama_models
                        .iter()
                        .map(|m| (m.name.clone(), m.size_bytes, m.family.clone(), m.parameter_size.clone()))
                        .collect();
                    for (idx, (name, size_bytes, family, param_sz)) in models.iter().enumerate() {
                        let selected = self.ollama_sel == Some(idx);
                        let label = format!(
                            "{name}  [{param_sz}] ({:.1} GB)",
                            *size_bytes as f64 / 1_000_000_000.0
                        );
                        let resp = ui.selectable_label(selected, &label);
                        if resp.clicked() {
                            self.ollama_sel = Some(idx);
                        }
                        if !family.is_empty() {
                            resp.on_hover_text(format!("Family: {family}"));
                        }
                    }
                });
        } else {
            ui.colored_label(egui::Color32::DARK_GRAY, t!("llm.ollama_empty").to_string());
        }

        ui.separator();

        // Inference section
        ui.heading(t!("llm.inference_heading").to_string());

        if !self.inference_model.is_empty() {
            ui.label(
                egui::RichText::new(format!("Model: {}", self.inference_model))
                    .weak()
                    .italics(),
            );
        }

        let has_graph = self.graph.is_some();
        let has_infer = !self.inference_model.is_empty() && has_graph;
        ui.add_enabled_ui(has_infer, |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut self.prompt_text)
                    .hint_text(t!("llm.prompt_hint").to_string())
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                let running = self.inference_active;
                if ui.add_enabled(!running, egui::Button::new(t!("llm.btn_run_inference").to_string())).clicked() {
                    self.start_inference();
                }
                if running {
                    ui.spinner();
                    ui.colored_label(
                        egui::Color32::from_rgb(0, 200, 255),
                        t!("llm.inference_active").to_string(),
                    );
                }
                if ui.button(t!("llm.btn_clear_anim").to_string()).clicked() {
                    self.inference_text.clear();
                    self.inference_active = false;
                    self.inference_rx = None;
                    self.activation = None;
                    self.animation_active = false;
                    self.render_dirty = true;
                }
            });
        });

        if !has_graph {
            ui.colored_label(
                egui::Color32::DARK_GRAY,
                t!("llm.inference_load_hint").to_string(),
            );
        }

        // Token output
        if !self.inference_text.is_empty() || self.inference_active {
            ui.add_space(4.0);
            ui.label(t!("llm.inference_output").to_string());
            egui::ScrollArea::vertical()
                .id_source("infer_output")
                .max_height(160.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.label(&self.inference_text);
                });
        }
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

/// Returns (min, max, mean) of weight_magnitude across all nodes in a layer.
fn layer_stats(layer: &crate::llm::network::Layer) -> (f32, f32, f32) {
    if layer.nodes.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min = f32::MAX;
    let mut max = 0.0f32;
    let mut sum = 0.0f32;
    for node in &layer.nodes {
        let w = node.weight_magnitude;
        if w < min { min = w; }
        if w > max { max = w; }
        sum += w;
    }
    (min, max, sum / layer.nodes.len() as f32)
}
