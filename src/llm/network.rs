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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Embedding,
    Attention,
    FeedForward,
    LayerNorm,
    Output,
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
        }
    }
}

#[derive(Clone)]
pub struct Node {
    pub position: [f32; 3],
    /// Normalized importance weight ∈ [0, 1]; drives sphere radius.
    pub weight_magnitude: f32,
}

pub struct Layer {
    pub name: String,
    pub kind: LayerKind,
    pub nodes: Vec<Node>,
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
}

impl NetworkGraph {
    /// Compute 3D positions for all nodes and build a representative edge set.
    /// Call once after loading; mutates positions in place.
    pub fn layout(&mut self) {
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
