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
//!    run streaming inference with live token display.
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

use cgmath::{InnerSpace, Matrix4};

use crate::llm::activation::{ActivationMode, ActivationState};
use crate::llm::exporter::AnimExporter;
use crate::llm::db_import;
use crate::llm::graph_import;
use crate::llm::loader;
use crate::llm::network::{
    LayerKind, NetworkGraph, NodeStyle, NODE_BASE_SCALE, NODE_MAX_SCALE, NODE_SPACING,
};
use crate::llm::ollama::{self, OllamaEvent, OllamaModel};
use crate::llm::tokenizer::Tokenizer;
use crate::model::{InstanceRaw, Vertex};
use crate::scene::GeometryType;

use super::StatusMessage;

// ─── Constants ────────────────────────────────────────────────────────────────

const CYAN:   [f32; 3] = [0.00, 0.65, 1.00];
const WHITE:  [f32; 3] = [1.00, 0.95, 0.85];
const ORANGE: [f32; 3] = [1.00, 0.55, 0.05];
const YELLOW: [f32; 3] = [1.00, 0.90, 0.30];
#[allow(dead_code)]
const PURPLE: [f32; 3] = [0.55, 0.20, 0.70];

// ─── Public action type ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LlmAction {
    None,
    FocusGraph([f32; 3]),
}

/// Channel receiving a loaded network graph (+ optional tokenizer) from a
/// background worker (file import and Ollama /api/show share it).
type GraphWorkerRx = mpsc::Receiver<Result<(NetworkGraph, Option<Tokenizer>), String>>;

/// Per-layer digest for the inspector list: (name, kind, node count, color).
type LayerSummary = (String, LayerKind, usize, (f32, f32, f32));

// ─── Render data ─────────────────────────────────────────────────────────────

pub struct LlmRenderData {
    /// Node instances grouped by primitive mesh. Layers without a per-layer
    /// shape override (see [`crate::llm::network::Layer::shape`]) land in the
    /// batch of the global render style's geometry.
    pub node_batches: Vec<(GeometryType, Vec<InstanceRaw>)>,
    pub edge_vertices:  Vec<Vertex>,
    /// Solid edge beams (stretched cubes), used instead of line-list edges
    /// when [`LlmView::edge_width`] > 0 so links have configurable thickness.
    pub edge_solid_instances: Vec<InstanceRaw>,
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

    /// How node primitives are drawn (spheres / voxels / points).
    pub render_style: NodeStyle,
    /// When true, all node base colors are overridden by [`Self::accent_color`].
    pub accent_enabled: bool,
    /// User accent color (linear RGB) used when [`Self::accent_enabled`].
    pub accent_color: [f32; 3],
    /// Grid cell size used to snap node positions in [`NodeStyle::Voxels`].
    pub voxel_grid: f32,
    /// Multiplier on node positions: spreads or compacts the whole graph
    /// without touching the stored layout ("scala distanze").
    pub distance_scale: f32,
    /// Edge thickness in world units; 0 draws classic 1-px lines, > 0 draws
    /// solid instanced beams ("dimensione link").
    pub edge_width: f32,
    /// Override the automatic edge coloring with [`Self::edge_color`].
    pub edge_color_enabled: bool,
    /// User edge color (linear RGB) used when [`Self::edge_color_enabled`].
    pub edge_color: [f32; 3],
    /// Filter text of the per-node editing table (data graphs only).
    pub node_table_filter: String,

    render_dirty:     bool,
    animation_active: bool,

    // File import worker
    worker: Option<GraphWorkerRx>,

    // ── Ollama ──────────────────────────────────────────────────────────────
    active_tab:    LlmTab,
    ollama_models: Vec<OllamaModel>,
    /// Which row is selected in the Ollama model list.
    ollama_sel:    Option<usize>,
    ollama_status: Option<StatusMessage>,
    /// Worker loading the Ollama model list.
    ollama_list_rx: Option<mpsc::Receiver<Result<Vec<OllamaModel>, String>>>,
    /// Worker loading an Ollama model graph via /api/show.
    ollama_graph_rx: Option<GraphWorkerRx>,

    // ── Inference (Ollama streaming) ─────────────────────────────────────────
    inference_text:   String,
    inference_active: bool,
    inference_rx:     Option<mpsc::Receiver<OllamaEvent>>,
    /// Which Ollama model is currently wired for inference.
    inference_model:  String,
    /// Rolling buffer of pseudo-token IDs for per-token mini-waves.
    inference_token_buf: Vec<u32>,
    /// Tokens received since the last mini-wave was triggered.
    tokens_since_wave: u32,
    /// Per-step loss values for the live training loss curve.
    loss_history: Vec<f32>,
    /// Count of training steps taken since model loaded.
    training_step: u32,
    /// When true, skip every N-1 transformer blocks in 3D render.
    compact_view: bool,
    /// How many blocks to skip per stride window in compact mode.
    compact_stride: usize,
    /// Animation frame exporter.
    pub exporter: AnimExporter,
    /// File path for the exported animation JSON.
    export_path: String,
    /// Status message from the last export operation.
    export_status: Option<StatusMessage>,
    /// Node selected by ray-pick in the 3D viewport: (layer_idx, node_idx).
    pub selected_node: Option<(usize, usize)>,
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
            render_style:     NodeStyle::default(),
            accent_enabled:   false,
            accent_color:     [0.20, 0.65, 1.00],
            voxel_grid:       NODE_SPACING,
            distance_scale:   1.0,
            edge_width:       0.0,
            edge_color_enabled: false,
            edge_color:       [0.25, 0.55, 0.95],
            node_table_filter: String::new(),
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
            inference_token_buf: Vec::new(),
            tokens_since_wave: 0,
            loss_history: Vec::new(),
            training_step: 0,
            compact_view: false,
            compact_stride: 4,
            exporter: AnimExporter::new(),
            export_path: String::new(),
            export_status: None,
            selected_node: None,
        }
    }

    // ── Dirty flag ──────────────────────────────────────────────────────────

    pub fn take_render_dirty(&mut self) -> bool {
        if self.animation_active {
            if let Some(anim) = &self.activation {
                if anim.is_finished(Instant::now()) {
                    self.animation_active = false;
                    self.render_dirty = true;
                    // Auto-stop exporter when animation ends.
                    if self.exporter.active {
                        self.exporter.stop();
                    }
                }
            } else {
                self.animation_active = false;
            }
            return true;
        }
        std::mem::take(&mut self.render_dirty)
    }

    pub fn mark_dirty(&mut self) { self.render_dirty = true; }

    fn is_layer_visible(&self, layer_idx: usize, kind: LayerKind) -> bool {
        if !self.compact_view || self.compact_stride <= 1 {
            return true;
        }
        // Always show Embedding and Output.
        if matches!(kind, LayerKind::Embedding | LayerKind::Output) {
            return true;
        }
        // Show one "set" of layers per stride window.
        // Each transformer block = 3+ layers (Attn, LN, FFN/Experts). Show block 0, stride, 2*stride, …
        layer_idx % (self.compact_stride * 3) < 3
    }

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
                    let stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("graph")
                        .to_string();
                    match ext.as_str() {
                        "gguf" => loader::from_gguf(&data).map_err(|e| e.to_string()),
                        // `.json` is ambiguous: try the neural-network model
                        // description first, then the JSON graph shapes
                        // (node-link, Cytoscape, JSON-LD, RDF/JSON).
                        "json" => loader::from_json(&data).map_err(|e| e.to_string()).or_else(
                            |model_err| {
                                graph_import::from_json_auto(&data, &stem)
                                    .map(|g| (g, None))
                                    .map_err(|graph_err| {
                                        format!(
                                            "not a model JSON ({model_err}) nor a graph JSON \
                                             ({graph_err})"
                                        )
                                    })
                            },
                        ),
                        ext if graph_import::is_graph_extension(ext) => {
                            graph_import::load_graph(&data, ext, &stem).map(|g| (g, None))
                        }
                        ext if db_import::is_db_extension(ext) => {
                            db_import::load_db(&data, ext, &stem).map(|g| (g, None))
                        }
                        other => Err(format!(
                            "Unsupported format '.{other}'. Models: .json/.gguf — graphs: \
                             .json/.cyjs/.graphml/.gexf/.gdf/.dot/.gv/.tgf/.net/.sif/.jsonld/.rj \
                             — databases: .sql/.sqlite/.sqlite3/.db/.db3"
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
                    // Convert token text fragment to a pseudo-ID via djb2 hash.
                    let id: u32 = tok.bytes().fold(5381u32, |h, b| h.wrapping_mul(33).wrapping_add(b as u32)) % 512;
                    self.inference_token_buf.push(id);
                    self.tokens_since_wave += 1;
                    const WAVE_EVERY: u32 = 5;
                    if self.tokens_since_wave >= WAVE_EVERY {
                        if let Some(graph) = &self.graph {
                            let recent: Vec<u32> = self.inference_token_buf
                                .iter().rev().take(WAVE_EVERY as usize).copied().collect();
                            let speed = self.wave_speed;
                            self.activation = Some(
                                ActivationState::simulate_fast(graph, &recent)
                                    .with_speed(speed)
                            );
                            self.animation_active = true;
                            self.render_dirty = true;
                        }
                        self.tokens_since_wave = 0;
                    }
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
        self.training_step += 1;
        let t = self.training_step as f32;
        // Exponential decay + small stochastic noise.
        let noise = ((t * std::f32::consts::E * 0.37).sin().powi(2)) * 0.12;
        let loss = 3.5 * (-0.07 * t).exp() + noise + 0.05;
        self.loss_history.push(loss);
        if self.loss_history.len() > 60 {
            self.loss_history.remove(0);
        }
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
        self.inference_token_buf.clear();
        self.tokens_since_wave = 0;
        self.loss_history.clear();
        self.training_step = 0;
        self.exporter.stop();
        self.export_status = None;
        self.visible = false;
        self.render_dirty = true;
        self.status = None;
        self.selected_node = None;
    }

    pub fn set_visible(&mut self, visible: bool) {
        if self.visible != visible {
            self.visible = visible;
            self.render_dirty = true;
        }
    }

    pub fn centroid(&self) -> Option<[f32; 3]> {
        let c = self.graph.as_ref()?.centroid()?;
        let ds = self.distance_scale.max(0.01);
        Some([c[0] * ds, c[1] * ds, c[2] * ds])
    }

    /// Primitive mesh the renderer should instance for node bodies.
    pub fn node_geometry(&self) -> GeometryType {
        match self.render_style {
            NodeStyle::Voxels => GeometryType::Cube,
            // Spheres and Points both use the sphere mesh; Points just scales
            // it down (see [`NodeStyle::scale_mult`]).
            NodeStyle::Spheres | NodeStyle::Points => GeometryType::Sphere,
        }
    }

    /// Install a pre-built architecture graph (catalog preset). Mirrors the
    /// file-import path but is synchronous since presets are tiny.
    pub fn load_preset(&mut self, graph: NetworkGraph) {
        self.install_graph(graph, None, |g| {
            format!("Loaded '{}' — {} layers · {} nodes · {} edges",
                g.name, g.layers.len(), g.node_count(), g.edges.len())
        });
    }

    /// Update render style and accent color; marks the scene dirty only when a
    /// value actually changed so it is cheap to call every frame.
    pub fn apply_style(
        &mut self,
        style: NodeStyle,
        accent_enabled: bool,
        accent_color: [f32; 3],
    ) {
        let changed = self.render_style != style
            || self.accent_enabled != accent_enabled
            || self.accent_color != accent_color;
        if changed {
            self.render_style = style;
            self.accent_enabled = accent_enabled;
            self.accent_color = accent_color;
            self.render_dirty = true;
        }
    }

    // ── Ray-pick against node spheres ──────────────────────────────────────

    /// Test a world-space ray against all visible LLM node spheres.
    /// Updates `selected_node` and returns `true` if any node was hit.
    pub fn pick_node(
        &mut self,
        ray_origin: cgmath::Vector3<f32>,
        ray_dir: cgmath::Vector3<f32>,
    ) -> bool {
        let Some(graph) = &self.graph else {
            self.selected_node = None;
            return false;
        };
        if !self.visible {
            self.selected_node = None;
            return false;
        }

        let mut best: Option<((usize, usize), f32)> = None;
        let ds = self.distance_scale.max(0.01);

        for (li, layer) in graph.layers.iter().enumerate() {
            if !self.is_layer_visible(li, layer.kind) {
                continue;
            }
            for (ni, node) in layer.nodes.iter().enumerate() {
                let c = cgmath::Vector3::new(
                    node.position[0] * ds,
                    node.position[1] * ds,
                    node.position[2] * ds,
                );
                let scale = (NODE_BASE_SCALE
                    + node.weight_magnitude * (NODE_MAX_SCALE - NODE_BASE_SCALE))
                    * self.node_scale_mult;
                // The primitive sphere has unit diameter (radius 0.5); inflate
                // the pick radius so small nodes are still easy to click.
                let r = scale * 0.5 * 2.0;

                let oc = ray_origin - c;
                let b = 2.0 * ray_dir.dot(oc);
                let c_term = oc.dot(oc) - r * r;
                let disc = b * b - 4.0 * c_term;
                if disc >= 0.0 {
                    let t = (-b - disc.sqrt()) * 0.5;
                    if t > 0.0 && best.is_none_or(|(_, bt)| t < bt) {
                        best = Some(((li, ni), t));
                    }
                }
            }
        }

        let new_sel = best.map(|(sel, _)| sel);
        if new_sel != self.selected_node {
            self.selected_node = new_sel;
            // The selection highlight lives in the instance colors.
            self.render_dirty = true;
        }
        new_sel.is_some()
    }

    // ── GPU render data ─────────────────────────────────────────────────────

    pub fn build_render_data(&mut self) -> LlmRenderData {
        let empty = || LlmRenderData {
            node_batches: vec![],
            edge_vertices: vec![],
            edge_solid_instances: vec![],
        };
        if !self.visible {
            return empty();
        }
        let Some(graph) = &self.graph else {
            return empty();
        };

        let now = Instant::now();
        let ds = self.distance_scale.max(0.01);

        // Selection highlight: the picked node glows golden (selected shader
        // path) and its direct neighbors brighten, so the neighborhood being
        // explored is readable in the 3D view.
        let neighbor_set: std::collections::HashSet<(usize, usize)> = self
            .selected_node
            .map(|(sl, sn)| {
                graph
                    .neighbors(sl, sn)
                    .into_iter()
                    .map(|(l, n, _, _)| (l, n))
                    .collect()
            })
            .unwrap_or_default();

        let style = self.render_style;
        let style_mult = style.scale_mult();
        let default_geom = self.node_geometry();
        let mut node_batches: Vec<(GeometryType, Vec<InstanceRaw>)> = Vec::new();
        for (li, layer) in graph.layers.iter().enumerate() {
            if !self.is_layer_visible(li, layer.kind) { continue; }
            // Per-layer shape override (database tables) wins over the style.
            let geom = layer.shape.unwrap_or(default_geom);
            let batch = match node_batches.iter().position(|(g, _)| *g == geom) {
                Some(idx) => idx,
                None => {
                    node_batches.push((geom, Vec::new()));
                    node_batches.len() - 1
                }
            };
            // Accent color overrides the per-kind palette when enabled.
            let layer_col = if self.accent_enabled {
                self.accent_color
            } else {
                layer.kind.base_color()
            };
            for (ni, node) in layer.nodes.iter().enumerate() {
                // Per-node file/table colors win unless the accent override is on.
                let base_col = match (self.accent_enabled, node.color) {
                    (false, Some(c)) => c,
                    _ => layer_col,
                };
                let fwd  = self.activation.as_ref().map(|a| a.glow_at(li, ni, now)).unwrap_or(0.0);
                let back = self.activation.as_ref().map(|a| a.back_glow_at(li, ni, now)).unwrap_or(0.0);
                let total_glow = (fwd + back).min(1.0);

                let is_selected = self.selected_node == Some((li, ni));
                let is_neighbor = neighbor_set.contains(&(li, ni));

                let scale = (NODE_BASE_SCALE
                    + node.weight_magnitude * (NODE_MAX_SCALE - NODE_BASE_SCALE))
                    * self.node_scale_mult
                    * style_mult
                    * (1.0 + total_glow * 0.5)
                    * if is_selected { 1.3 } else { 1.0 };

                let [r, g, b] = if total_glow > 0.01 {
                    // Blend forward (cyan→white) and backward (orange→yellow) by strength.
                    let fwd_col  = lerp3(lerp3(base_col, CYAN, fwd), WHITE, fwd * fwd);
                    let back_col = lerp3(lerp3(base_col, ORANGE, back), YELLOW, back * back);
                    let w = fwd + back + 1e-8;
                    lerp3(back_col, fwd_col, fwd / w)
                } else {
                    // Neighbors of the selected node stay near full brightness.
                    let dim = if is_selected || is_neighbor { 0.85 } else { 0.45 };
                    [base_col[0] * dim, base_col[1] * dim, base_col[2] * dim]
                };

                let alpha = if total_glow > 0.01 {
                    3.0 + total_glow
                } else if is_selected {
                    2.0 // golden selected-object shader path
                } else {
                    1.0
                };

                // In voxel mode snap node centers onto a regular grid so the
                // network reads as stacked blocks instead of free spheres.
                let scaled = [
                    node.position[0] * ds,
                    node.position[1] * ds,
                    node.position[2] * ds,
                ];
                let p = if style == NodeStyle::Voxels && layer.shape.is_none() {
                    snap_to_grid(scaled, self.voxel_grid)
                } else {
                    scaled
                };
                let model = Matrix4::from_translation(cgmath::Vector3::new(p[0], p[1], p[2]))
                    * Matrix4::from_scale(scale);

                node_batches[batch].1.push(InstanceRaw {
                    model: model.into(),
                    color: [r, g, b, alpha],
                });
            }
        }

        // Capture animation frame for export if recording.
        if self.exporter.active {
            let layer_fwd: Vec<Vec<f32>> = graph.layers.iter().enumerate()
                .map(|(li, layer)| {
                    layer.nodes.iter().enumerate()
                        .map(|(ni, _)| self.activation.as_ref().map_or(0.0, |a| a.glow_at(li, ni, now)))
                        .collect()
                })
                .collect();
            let layer_bwd: Vec<Vec<f32>> = graph.layers.iter().enumerate()
                .map(|(li, layer)| {
                    layer.nodes.iter().enumerate()
                        .map(|(ni, _)| self.activation.as_ref().map_or(0.0, |a| a.back_glow_at(li, ni, now)))
                        .collect()
                })
                .collect();
            self.exporter.capture(&layer_fwd, &layer_bwd);
        }

        let mut edge_vertices = Vec::new();
        let mut edge_solid_instances = Vec::new();
        for edge in &graph.edges {
            let Some(fl)  = graph.layers.get(edge.from_layer)  else { continue };
            let Some(tl)  = graph.layers.get(edge.to_layer)    else { continue };
            if !self.is_layer_visible(edge.from_layer, fl.kind) { continue; }
            if !self.is_layer_visible(edge.to_layer,   tl.kind) { continue; }
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

            let fwd_glow  = (gf + gt) * 0.5;
            let back_glow = (bf + bt) * 0.5;
            // Passive dim base is darker for low-importance edges, brighter
            // for high; the user edge color replaces the automatic ramp.
            let (base_dim, base_lit) = if self.edge_color_enabled {
                let c = self.edge_color;
                (
                    lerp3([c[0] * 0.25, c[1] * 0.25, c[2] * 0.25], c, edge.importance),
                    c,
                )
            } else {
                (
                    lerp3([0.03, 0.04, 0.09], [0.10, 0.14, 0.26], edge.importance),
                    lerp3(CYAN, WHITE, edge.importance * 0.5),
                )
            };
            let col = if back_glow > fwd_glow {
                lerp3(lerp3(base_dim, ORANGE, edge.importance), ORANGE, back_glow)
            } else {
                lerp3(base_dim, base_lit, fwd_glow.max(edge.importance * 0.25))
            };

            let a = [fn_.position[0] * ds, fn_.position[1] * ds, fn_.position[2] * ds];
            let b = [tn.position[0] * ds, tn.position[1] * ds, tn.position[2] * ds];
            if self.edge_width > 0.0 {
                if let Some(instance) = beam_instance(a, b, self.edge_width, col) {
                    edge_solid_instances.push(instance);
                }
            } else {
                let normal = [0.0f32, 0.0, 1.0];
                edge_vertices.push(Vertex { position: a, color: col, normal });
                edge_vertices.push(Vertex { position: b, color: col, normal });
            }
        }

        LlmRenderData { node_batches, edge_vertices, edge_solid_instances }
    }

    // ── Editable nodes table (data graphs) ──────────────────────────────────

    /// Per-node editing table for imported graph files: label, degree, and
    /// live-editable size and color per node. This is what makes every part
    /// of a graph import configurable after load, like the dataset pipeline.
    fn nodes_table_ui(&mut self, ui: &mut egui::Ui) {
        const MAX_TABLE_ROWS: usize = 500;
        if !self.graph.as_ref().is_some_and(|g| g.data_graph) {
            return;
        }
        ui.add_space(6.0);
        let heading = t!("llm.nodes_table").to_string();
        let filter_hint = t!("llm.node_filter").to_string();
        let mut dirty = false;
        let mut select: Option<(usize, usize)> = None;
        egui::CollapsingHeader::new(heading)
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("🔍");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.node_table_filter)
                            .desired_width(180.0)
                            .hint_text(filter_hint),
                    );
                });
                let filter = self.node_table_filter.trim().to_ascii_lowercase();
                let selected = self.selected_node;
                let Some(graph) = self.graph.as_mut() else { return };

                // Degree from the real edge list. Graph files keep every node
                // in layer 0; database imports use one layer per table.
                let mut degree: std::collections::HashMap<(usize, usize), u32> =
                    std::collections::HashMap::new();
                for e in &graph.edges {
                    *degree.entry((e.from_layer, e.from_node)).or_default() += 1;
                    *degree.entry((e.to_layer, e.to_node)).or_default() += 1;
                }
                let multi_layer = graph.layers.len() > 1;

                let mut shown = 0usize;
                egui::ScrollArea::vertical()
                    .max_height(230.0)
                    .show(ui, |ui| {
                        egui::Grid::new("graph_nodes_grid")
                            .num_columns(4)
                            .striped(true)
                            .spacing([14.0, 4.0])
                            .show(ui, |ui| {
                                for h in [
                                    t!("llm.node_label"),
                                    t!("llm.node_degree"),
                                    t!("llm.node_size"),
                                    t!("llm.node_color"),
                                ] {
                                    ui.label(egui::RichText::new(h.to_string()).weak().small());
                                }
                                ui.end_row();
                                'outer: for (li, layer) in
                                    graph.layers.iter_mut().enumerate()
                                {
                                    let layer_col = layer.kind.base_color();
                                    let layer_name = layer.name.clone();
                                    for (ni, node) in layer.nodes.iter_mut().enumerate() {
                                        let mut label = node
                                            .label
                                            .clone()
                                            .unwrap_or_else(|| ni.to_string());
                                        if !filter.is_empty()
                                            && !label.to_ascii_lowercase().contains(&filter)
                                            && !layer_name.to_ascii_lowercase().contains(&filter)
                                        {
                                            continue;
                                        }
                                        if shown >= MAX_TABLE_ROWS {
                                            break 'outer;
                                        }
                                        shown += 1;
                                        if multi_layer {
                                            label = format!("{layer_name} · {label}");
                                        }
                                        // Clicking a row selects the node in the
                                        // 3D view and opens it in the inspector.
                                        if ui
                                            .selectable_label(selected == Some((li, ni)), label)
                                            .clicked()
                                        {
                                            select = Some((li, ni));
                                        }
                                        ui.label(
                                            degree
                                                .get(&(li, ni))
                                                .copied()
                                                .unwrap_or(0)
                                                .to_string(),
                                        );
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut node.weight_magnitude)
                                                    .range(0.05..=1.0)
                                                    .speed(0.01),
                                            )
                                            .changed()
                                        {
                                            dirty = true;
                                        }
                                        let mut c = node.color.unwrap_or(layer_col);
                                        if ui.color_edit_button_rgb(&mut c).changed() {
                                            node.color = Some(c);
                                            dirty = true;
                                        }
                                        ui.end_row();
                                    }
                                }
                            });
                    });
                if shown >= MAX_TABLE_ROWS {
                    ui.label(
                        egui::RichText::new(
                            t!("llm.nodes_truncated", max = MAX_TABLE_ROWS.to_string())
                                .to_string(),
                        )
                        .weak()
                        .small(),
                    );
                }
            });
        if let Some(sel) = select {
            self.selected_node = Some(sel);
            dirty = true;
        }
        if dirty {
            self.render_dirty = true;
        }
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

        let mut show_window = self.show_window;
        let action = super::tool_window::ToolWindow::fixed(t!("llm.window_title"), [520.0, 600.0])
            .show(ctx, &mut show_window, |ui| self.window_body(ui))
            .unwrap_or(LlmAction::None);
        self.show_window = show_window;
        action
    }

    fn window_body(&mut self, ui: &mut egui::Ui) -> LlmAction {
        // ── Tab bar ────────────────────────────────────────────────────────
        super::tool_window::tab_bar(
            ui,
            &mut self.active_tab,
            vec![
                super::tool_window::TabItem::new(LlmTab::Model, t!("llm.tab_model")),
                super::tool_window::TabItem::new(LlmTab::Ollama, t!("llm.tab_ollama")),
            ],
        );

        let action = match self.active_tab {
            LlmTab::Model  => self.tab_model(ui),
            LlmTab::Ollama => { self.tab_ollama(ui); LlmAction::None }
        };

        let focus = self.show_node_inspector(ui);
        if matches!(action, LlmAction::None) {
            if let Some(p) = focus {
                return LlmAction::FocusGraph(p);
            }
        }
        action
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
                    .hint_text("model.gguf · graph.graphml · dump.sql · data.sqlite")
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
                if let Some(gb) = graph.estimated_vram_gb {
                    ui.label(egui::RichText::new(format!("~{gb:.1} GB FP16")).weak().small());
                }
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
            let layer_summaries: Vec<LayerSummary> = graph
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
                            ui.label(kind.icon());
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
            if ui
                .add(
                    egui::Slider::new(&mut self.node_scale_mult, 0.4..=3.0)
                        .text(t!("llm.node_scale").to_string()),
                )
                .changed()
            {
                self.mark_dirty();
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.distance_scale, 0.2..=5.0)
                        .logarithmic(true)
                        .text(t!("llm.distance_scale").to_string()),
                )
                .changed()
            {
                self.mark_dirty();
            }
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut self.edge_width, 0.0..=0.20)
                            .text(t!("llm.edge_width").to_string()),
                    )
                    .changed()
                {
                    self.mark_dirty();
                }
            });
            ui.horizontal(|ui| {
                if ui
                    .checkbox(&mut self.edge_color_enabled, t!("llm.edge_color").to_string())
                    .changed()
                {
                    self.mark_dirty();
                }
                ui.add_enabled_ui(self.edge_color_enabled, |ui| {
                    if ui.color_edit_button_rgb(&mut self.edge_color).changed() {
                        self.mark_dirty();
                    }
                });
            });
            ui.add(
                egui::Slider::new(&mut self.wave_speed, 0.25..=4.0)
                    .text(t!("llm.wave_speed").to_string()),
            );
            ui.horizontal(|ui| {
                if ui.checkbox(&mut self.compact_view, t!("llm.compact_view").to_string()).changed() {
                    self.mark_dirty();
                }
                if self.compact_view
                    && ui
                        .add(
                            egui::Slider::new(&mut self.compact_stride, 2..=8usize)
                                .text(t!("llm.compact_stride").to_string()),
                        )
                        .changed()
                {
                    self.mark_dirty();
                }
            });
        });

        self.nodes_table_ui(ui);

        // Loss curve (shown below sliders when training steps have been taken)
        if !self.loss_history.is_empty() {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(t!("llm.loss_curve").to_string()).small().weak());
            let desired = egui::vec2(ui.available_width(), 64.0);
            let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 3.0, egui::Color32::from_gray(22));
            let n = self.loss_history.len();
            let max_loss = self.loss_history.iter().cloned().fold(0.0f32, f32::max).max(0.01);
            let to_px = |i: usize, l: f32| -> egui::Pos2 {
                let x = rect.left() + (i as f32 / (n - 1).max(1) as f32) * rect.width();
                let y = rect.bottom() - (l / max_loss) * rect.height() * 0.88;
                egui::pos2(x, y)
            };
            for i in 0..n.saturating_sub(1) {
                painter.line_segment(
                    [to_px(i, self.loss_history[i]), to_px(i + 1, self.loss_history[i + 1])],
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 170, 40)),
                );
            }
            if let Some(&last_loss) = self.loss_history.last() {
                painter.text(
                    rect.right_top() + egui::vec2(-4.0, 3.0),
                    egui::Align2::RIGHT_TOP,
                    format!("loss {last_loss:.3}  step {}", self.training_step),
                    egui::FontId::proportional(10.0),
                    egui::Color32::from_rgb(255, 210, 100),
                );
            }
        }

        // Export animation UI
        if has_model {
            ui.separator();
            ui.heading(t!("llm.export_heading").to_string());
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.export_path)
                        .hint_text("animation.json")
                        .desired_width(240.0),
                );
                let recording = self.exporter.active;
                let btn_label = if recording {
                    format!("⏹ Stop ({} frames)", self.exporter.frame_count())
                } else {
                    t!("llm.btn_record").to_string()
                };
                if ui.button(&btn_label).clicked() {
                    if recording {
                        self.exporter.stop();
                        // Auto-export on stop.
                        if !self.export_path.is_empty() {
                            match self.exporter.export_json(&self.export_path) {
                                Ok(n) => self.export_status = Some(StatusMessage::success(
                                    format!("Exported {n} frames → {}", self.export_path)
                                )),
                                Err(e) => self.export_status = Some(StatusMessage::error(e.to_string())),
                            }
                        }
                    } else {
                        self.exporter.start();
                        self.export_status = Some(StatusMessage::info("Recording…".to_owned()));
                    }
                }
            });
            if let Some(s) = &self.export_status.clone() {
                s.show(ui);
            }
        }

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

        // ── Token output ────────────────────────────────────────────────────
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
    // ── Node inspector ───────────────────────────────────────────────────────

    /// Selected-node panel. For neural networks it shows layer/weight/glow;
    /// for imported data graphs it is a full explorer: the node's internal
    /// data (every file attribute), live size/color editing, and a clickable
    /// neighbor list to navigate the graph (select or 🎯 focus the camera).
    /// Returns a world-space point to focus when the user asks for it.
    fn show_node_inspector(&mut self, ui: &mut egui::Ui) -> Option<[f32; 3]> {
        const MAX_NEIGHBOR_ROWS: usize = 200;
        let (li, ni) = self.selected_node?;
        let ds = self.distance_scale.max(0.01);

        // Immutable snapshot first: the editing widgets below need &mut graph.
        let (
            data_graph,
            layer_kind,
            layer_name,
            n_in_layer,
            node_label,
            node_pos,
            node_attrs,
            neighbors,
        ) = {
            let graph = self.graph.as_ref()?;
            let layer = graph.layers.get(li)?;
            let node = layer.nodes.get(ni)?;
            // Deduped neighbor snapshot: (layer, node, outgoing, label, position).
            let mut seen = std::collections::HashSet::new();
            let neighbors: Vec<(usize, usize, bool, String, [f32; 3])> = graph
                .neighbors(li, ni)
                .into_iter()
                .filter(|&(l, n, _, _)| seen.insert((l, n)))
                .filter_map(|(l, n, _, out)| {
                    let nb = graph.layers.get(l)?.nodes.get(n)?;
                    let label = nb
                        .label
                        .clone()
                        .unwrap_or_else(|| format!("{}·{}", l, n));
                    Some((l, n, out, label, nb.position))
                })
                .collect();
            (
                graph.data_graph,
                layer.kind,
                layer.name.clone(),
                layer.nodes.len(),
                node.label.clone(),
                node.position,
                node.attrs.clone(),
                neighbors,
            )
        };

        let now = Instant::now();
        let fwd  = self.activation.as_ref().map(|a| a.glow_at(li, ni, now)).unwrap_or(0.0);
        let back = self.activation.as_ref().map(|a| a.back_glow_at(li, ni, now)).unwrap_or(0.0);

        ui.separator();
        ui.horizontal(|ui| {
            ui.strong("Node Inspector");
            let title = match &node_label {
                Some(l) => l.clone(),
                None => format!("Layer {} · Node {}/{}", li, ni, n_in_layer),
            };
            ui.label(egui::RichText::new(title).weak());
        });

        if !data_graph {
            ui.horizontal(|ui| {
                let kind_label = format!("{:?}", layer_kind);
                ui.label(egui::RichText::new(kind_label).color(egui::Color32::from_rgb(
                    (layer_kind.base_color()[0] * 255.0) as u8,
                    (layer_kind.base_color()[1] * 255.0) as u8,
                    (layer_kind.base_color()[2] * 255.0) as u8,
                )));
                ui.label(egui::RichText::new(&layer_name).italics().weak());
            });
        }

        let mut focus: Option<[f32; 3]> = None;
        let mut select: Option<(usize, usize)> = None;
        let mut dirty = false;

        // Live size/color editing on the selected node (data graphs); NN
        // nodes keep the read-only weight bar.
        if data_graph {
            if let Some(node) = self
                .graph
                .as_mut()
                .and_then(|g| g.layers.get_mut(li))
                .and_then(|l| l.nodes.get_mut(ni))
            {
                ui.horizontal(|ui| {
                    ui.label(t!("llm.node_size").to_string());
                    if ui
                        .add(
                            egui::DragValue::new(&mut node.weight_magnitude)
                                .range(0.05..=1.0)
                                .speed(0.01),
                        )
                        .changed()
                    {
                        dirty = true;
                    }
                    ui.label(t!("llm.node_color").to_string());
                    let mut c = node.color.unwrap_or_else(|| layer_kind.base_color());
                    if ui.color_edit_button_rgb(&mut c).changed() {
                        node.color = Some(c);
                        dirty = true;
                    }
                    if ui
                        .button("🎯")
                        .on_hover_text(t!("menu.focus").to_string())
                        .clicked()
                    {
                        focus = Some([node_pos[0] * ds, node_pos[1] * ds, node_pos[2] * ds]);
                    }
                });
            }
        } else {
            ui.horizontal(|ui| {
                let weight = self
                    .graph
                    .as_ref()
                    .and_then(|g| g.layers.get(li))
                    .and_then(|l| l.nodes.get(ni))
                    .map(|n| n.weight_magnitude)
                    .unwrap_or(0.0);
                ui.label(format!("Weight: {:.4}", weight));
                ui.add(egui::ProgressBar::new(weight).desired_width(60.0));
            });
        }

        ui.label(
            egui::RichText::new(format!(
                "Pos: ({:.2}, {:.2}, {:.2})",
                node_pos[0], node_pos[1], node_pos[2]
            ))
            .weak()
            .small(),
        );

        // The node's internal data, exactly as the file carried it.
        if !node_attrs.is_empty() {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(t!("llm.node_attrs").to_string()).strong().small());
            egui::ScrollArea::vertical()
                .id_source("node_attrs_scroll")
                .max_height(120.0)
                .show(ui, |ui| {
                    egui::Grid::new("node_attrs_grid")
                        .num_columns(2)
                        .striped(true)
                        .spacing([12.0, 2.0])
                        .show(ui, |ui| {
                            for (key, value) in &node_attrs {
                                ui.label(egui::RichText::new(key).weak());
                                ui.label(value);
                                ui.end_row();
                            }
                        });
                });
        }

        // Neighbor navigation: click a neighbor to select it, 🎯 to fly there.
        if data_graph {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(
                    t!("llm.node_neighbors", count = neighbors.len().to_string()).to_string(),
                )
                .strong()
                .small(),
            );
            egui::ScrollArea::vertical()
                .id_source("node_neighbors_scroll")
                .max_height(150.0)
                .show(ui, |ui| {
                    for (l, n, out, label, pos) in neighbors.iter().take(MAX_NEIGHBOR_ROWS) {
                        ui.horizontal(|ui| {
                            let arrow = if *out { "→" } else { "←" };
                            if ui
                                .selectable_label(false, format!("{arrow} {label}"))
                                .clicked()
                            {
                                select = Some((*l, *n));
                            }
                            if ui
                                .small_button("🎯")
                                .on_hover_text(t!("menu.focus").to_string())
                                .clicked()
                            {
                                focus = Some([pos[0] * ds, pos[1] * ds, pos[2] * ds]);
                            }
                        });
                    }
                    if neighbors.len() > MAX_NEIGHBOR_ROWS {
                        ui.label(
                            egui::RichText::new(
                                t!("llm.nodes_truncated", max = MAX_NEIGHBOR_ROWS.to_string())
                                    .to_string(),
                            )
                            .weak()
                            .small(),
                        );
                    }
                });
        }

        if fwd > 0.01 || back > 0.01 {
            ui.horizontal(|ui| {
                if fwd > 0.01 {
                    ui.colored_label(
                        egui::Color32::from_rgb(0, 180, 230),
                        format!("Fwd: {:.1}%", fwd * 100.0),
                    );
                }
                if back > 0.01 {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 140, 30),
                        format!("Bwd: {:.1}%", back * 100.0),
                    );
                }
            });
        }

        if let Some(sel) = select {
            self.selected_node = Some(sel);
            dirty = true;
        }
        if dirty {
            self.render_dirty = true;
        }
        focus
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Snap a world-space position onto a regular grid of cell size `g`.
#[inline]
/// Instance matrix for a solid edge beam: a unit cube stretched from `a` to
/// `b` with square cross-section `width`. None for degenerate (zero-length)
/// edges such as self-loops on coincident nodes.
fn beam_instance(a: [f32; 3], b: [f32; 3], width: f32, color: [f32; 3]) -> Option<InstanceRaw> {
    use cgmath::Vector3;
    let d = Vector3::new(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    let len = d.magnitude();
    if len < 1e-6 {
        return None;
    }
    let dir = d / len;
    // Any axis not parallel to the edge yields a stable cross-section basis.
    let up = if dir.x.abs() < 0.9 {
        Vector3::unit_x()
    } else {
        Vector3::unit_y()
    };
    let u = dir.cross(up).normalize();
    let v = dir.cross(u);
    let mid = Vector3::new((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5);
    let model = Matrix4::from_cols(
        (dir * len).extend(0.0),
        (u * width).extend(0.0),
        (v * width).extend(0.0),
        mid.extend(1.0),
    );
    Some(InstanceRaw {
        model: model.into(),
        color: [color[0], color[1], color[2], 1.0],
    })
}

fn snap_to_grid(p: [f32; 3], g: f32) -> [f32; 3] {
    if g <= 1e-4 {
        return p;
    }
    [
        (p[0] / g).round() * g,
        (p[1] / g).round() * g,
        (p[2] / g).round() * g,
    ]
}

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
