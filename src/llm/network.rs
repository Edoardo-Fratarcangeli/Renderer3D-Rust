//! Core data structures for the neural-network visualization graph.
//!
//! A [`NetworkGraph`] is the in-memory representation of a loaded model.
//! `layout()` assigns every node a 3D world-space position:
//!  - X axis: layer depth (embedding → … → output)
//!  - Y-Z plane: neuron grid within each layer
//!
//! Node sphere radius scales with [`Node::weight_magnitude`] ∈ [0, 1].
//! Active nodes encode glow intensity in the instance alpha channel
//! (≥ 3.0 triggers the LLM shader path).

pub const MAX_NODES_PER_LAYER: usize = 64;
pub const LAYER_SPACING: f32 = 3.5;
pub const NODE_SPACING: f32 = 0.45;
pub const NODE_BASE_SCALE: f32 = 0.10;
pub const NODE_MAX_SCALE: f32 = 0.32;

/// Type of a layer in the visualized network.
///
/// The first six variants describe transformer/LLM building blocks; the rest
/// generalize the visualizer to the wider ML/DL zoo (CNNs, RNNs, graph and
/// point-cloud networks, autoencoders, …) so any architecture in
/// [`crate::llm::catalog`] can be represented with a meaningful color/icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    // ── Transformer / LLM ──
    Embedding,
    Attention,
    FeedForward,
    LayerNorm,
    Output,
    /// MoE expert FFN — only a subset activate per token.
    Expert,

    // ── Generic ML / DL ──
    /// Raw input layer (image tensor, feature vector, point set, …).
    Input,
    /// Fully-connected / dense (MLP) layer.
    Dense,
    /// Convolutional layer (2D or 3D feature maps).
    Convolution,
    /// Pooling / subsampling layer.
    Pooling,
    /// Recurrent cell block (RNN / LSTM / GRU).
    Recurrent,
    /// Graph message-passing layer (GCN / GAT / GraphSAGE).
    Graph,
    /// Point-cloud set-abstraction / grouping layer (PointNet family).
    PointSet,
    /// Residual / skip-connection junction.
    Residual,
    /// Latent / bottleneck code (autoencoder, VAE).
    Latent,
    /// Up-sampling / transposed-convolution / decoder layer.
    Upsample,
}

impl LayerKind {
    /// Dim base color for inactive nodes by layer type.
    pub fn base_color(self) -> [f32; 3] {
        match self {
            LayerKind::Embedding   => [0.25, 0.35, 0.75],
            LayerKind::Attention   => [0.15, 0.55, 0.45],
            LayerKind::FeedForward => [0.45, 0.25, 0.65],
            LayerKind::LayerNorm   => [0.55, 0.55, 0.35],
            LayerKind::Output      => [0.75, 0.35, 0.25],
            LayerKind::Expert      => [0.55, 0.20, 0.70],
            LayerKind::Input       => [0.30, 0.55, 0.75],
            LayerKind::Dense       => [0.40, 0.45, 0.70],
            LayerKind::Convolution => [0.20, 0.60, 0.65],
            LayerKind::Pooling     => [0.35, 0.50, 0.40],
            LayerKind::Recurrent   => [0.65, 0.40, 0.55],
            LayerKind::Graph       => [0.30, 0.65, 0.35],
            LayerKind::PointSet    => [0.70, 0.55, 0.25],
            LayerKind::Residual    => [0.60, 0.60, 0.60],
            LayerKind::Latent      => [0.70, 0.25, 0.50],
            LayerKind::Upsample    => [0.45, 0.55, 0.70],
        }
    }

    /// Short emoji icon shown in layer lists and inspectors.
    pub fn icon(self) -> &'static str {
        match self {
            LayerKind::Embedding   => "📥",
            LayerKind::Attention   => "👁",
            LayerKind::FeedForward => "⚡",
            LayerKind::LayerNorm   => "📐",
            LayerKind::Output      => "📤",
            LayerKind::Expert      => "🔷",
            LayerKind::Input       => "🟢",
            LayerKind::Dense       => "▦",
            LayerKind::Convolution => "🟦",
            LayerKind::Pooling     => "🔽",
            LayerKind::Recurrent   => "🔁",
            LayerKind::Graph       => "🕸",
            LayerKind::PointSet    => "✶",
            LayerKind::Residual    => "➕",
            LayerKind::Latent      => "🎯",
            LayerKind::Upsample    => "🔼",
        }
    }
}

/// How node primitives are drawn in the 3D viewport.
///
/// The geometry mesh is chosen by the renderer ([`crate::state`]) from
/// [`NodeStyle::geometry`]; `Voxels` additionally snaps node positions to a
/// regular grid so a network reads as a stack of voxel blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeStyle {
    /// Smooth spheres (default).
    #[default]
    Spheres,
    /// Axis-aligned cubes on a voxel grid.
    Voxels,
    /// Small dot-like spheres (point-cloud look).
    Points,
}

impl NodeStyle {
    pub const ALL: [NodeStyle; 3] = [NodeStyle::Spheres, NodeStyle::Voxels, NodeStyle::Points];

    /// Per-node scale multiplier applied on top of the user node-scale slider.
    pub fn scale_mult(self) -> f32 {
        match self {
            NodeStyle::Spheres => 1.0,
            NodeStyle::Voxels  => 1.0,
            NodeStyle::Points  => 0.45,
        }
    }
}

/// Configuration for Mixture-of-Experts models.
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub n_experts: usize,
    pub experts_per_token: usize,
}

#[derive(Clone)]
pub struct Node {
    pub position: [f32; 3],
    /// Normalized importance weight ∈ [0, 1]; drives sphere radius.
    pub weight_magnitude: f32,
    /// Explicit per-node color (from graph files or table edits); None means
    /// the layer color (or the accent override) applies.
    pub color: Option<[f32; 3]>,
    /// Node label from graph files (id or label attribute); None for
    /// neural-network nodes, which are addressed by layer + index.
    pub label: Option<String>,
    /// The node's internal data: every attribute the source file carried
    /// that is not already mapped (label / position / size / color), as
    /// key–value pairs shown by the node inspector.
    pub attrs: Vec<(String, String)>,
}

impl Node {
    /// Unpositioned, unlabeled node with the given normalized weight.
    pub fn new(weight_magnitude: f32) -> Self {
        Node {
            position: [0.0; 3],
            weight_magnitude,
            color: None,
            label: None,
            attrs: Vec::new(),
        }
    }
}

pub struct Layer {
    pub name: String,
    pub kind: LayerKind,
    pub nodes: Vec<Node>,
    /// Per-layer primitive override for node bodies (database import gives
    /// every table its own shape). None follows the global render style.
    pub shape: Option<crate::scene::GeometryType>,
}

/// Sparse directed edge between two nodes across adjacent layers.
pub struct Edge {
    pub from_layer: usize,
    pub from_node:  usize,
    pub to_layer:   usize,
    pub to_node:    usize,
    /// Normalized connection importance ∈ [0, 1]; drives passive edge brightness.
    pub importance: f32,
}

pub struct NetworkGraph {
    pub name: String,
    pub layers: Vec<Layer>,
    pub edges: Vec<Edge>,
    /// Estimated FP16 VRAM in GB from architecture metadata; None if not known.
    pub estimated_vram_gb: Option<f64>,
    /// MoE configuration if the model uses Mixture of Experts.
    pub moe_config: Option<MoeConfig>,
    /// True when this graph came from a graph data file (GraphML, GEXF, DOT,
    /// …) rather than a neural-network description: positions and edges are
    /// real data — [`Self::layout`] must not overwrite them — and the UI
    /// offers the per-node editing table.
    pub data_graph: bool,
    /// Import caveat surfaced by the UI (e.g. "sampled 200000 of 1.2M rows"
    /// for big databases, or a schema-only fallback). None when the source
    /// was imported in full.
    pub note: Option<String>,
}

impl NetworkGraph {
    /// Compute 3D positions for all nodes and build a representative edge set.
    /// Call once after loading; mutates positions in place. No-op for data
    /// graphs, whose positions and edges come from the file / graph layout.
    pub fn layout(&mut self) {
        if self.data_graph {
            return;
        }
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let x = layer_idx as f32 * LAYER_SPACING;
            let n = layer.nodes.len();
            let cols = ((n as f32).sqrt().ceil() as usize).max(1);
            let rows = n.div_ceil(cols);
            let y_off = (cols.saturating_sub(1) as f32) * NODE_SPACING * 0.5;
            let z_off = (rows.saturating_sub(1) as f32) * NODE_SPACING * 0.5;
            for (i, node) in layer.nodes.iter_mut().enumerate() {
                let col = i % cols;
                let row = i / cols;
                node.position = [
                    x,
                    col as f32 * NODE_SPACING - y_off,
                    row as f32 * NODE_SPACING - z_off,
                ];
            }
        }
        self.build_sample_edges();
    }

    fn build_sample_edges(&mut self) {
        self.edges.clear();
        let num_layers = self.layers.len();
        for li in 0..num_layers.saturating_sub(1) {
            let n_from = self.layers[li].nodes.len();
            let n_to   = self.layers[li + 1].nodes.len();
            // Keep at most 40 edges per layer pair to avoid visual clutter.
            let stride = (n_from / 40).max(1);
            for fi in (0..n_from).step_by(stride) {
                let ti = (fi * n_to / n_from.max(1)).min(n_to.saturating_sub(1));
                let w_from = self.layers[li].nodes[fi].weight_magnitude;
                let w_to   = self.layers[li + 1].nodes[ti].weight_magnitude;
                // Geometric mean of endpoint weights.
                let importance = (w_from * w_to).sqrt();
                self.edges.push(Edge {
                    from_layer: li,
                    from_node:  fi,
                    to_layer:   li + 1,
                    to_node:    ti,
                    importance,
                });
            }
        }
        // Normalize importance to [0, 1].
        let max_imp = self.edges.iter().map(|e| e.importance).fold(0.0f32, f32::max).max(1e-8);
        for edge in &mut self.edges {
            edge.importance /= max_imp;
        }
    }

    /// Every node connected to `(layer, node)` by an edge, as
    /// `(layer, node, importance, outgoing)` — the navigation base for the
    /// node inspector's neighbor list.
    pub fn neighbors(&self, layer: usize, node: usize) -> Vec<(usize, usize, f32, bool)> {
        let mut out = Vec::new();
        for e in &self.edges {
            if e.from_layer == layer && e.from_node == node {
                out.push((e.to_layer, e.to_node, e.importance, true));
            } else if e.to_layer == layer && e.to_node == node {
                out.push((e.from_layer, e.from_node, e.importance, false));
            }
        }
        out
    }

    /// World-space centroid of all nodes, used to auto-focus the camera.
    pub fn centroid(&self) -> Option<[f32; 3]> {
        let mut sum = [0.0f32; 3];
        let mut count = 0usize;
        for layer in &self.layers {
            for node in &layer.nodes {
                sum[0] += node.position[0];
                sum[1] += node.position[1];
                sum[2] += node.position[2];
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        let n = count as f32;
        Some([sum[0] / n, sum[1] / n, sum[2] / n])
    }

    pub fn node_count(&self) -> usize {
        self.layers.iter().map(|l| l.nodes.len()).sum()
    }
}
