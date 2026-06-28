//! Activation simulation and wave-propagation animation.
//!
//! `ActivationState::simulate` computes a per-node activation value from the
//! injected token IDs using a deterministic pseudo-random function so the
//! visualization is reproducible without running an actual forward pass.
//!
//! On each render frame `glow_at(layer, node, now)` returns a value in [0, 1]
//! that drives the shader's emissive effect.  The animation is a Gaussian wave
//! front that propagates from the embedding layer to the output layer over
//! `WAVE_DURATION_SECS`.
//!
//! For [`ActivationMode::Training`] an additional backward gradient wave
//! (orange) starts at 60 % of the forward duration and propagates in reverse
//! (output → embedding), simulating the backward pass.

use std::time::Instant;

use crate::llm::network::{LayerKind, NetworkGraph};

/// Total duration of the forward propagation wave in seconds.
const WAVE_DURATION_SECS: f32 = 4.0;
/// Width (σ) of the Gaussian wave front (fraction of total depth).
const WAVE_SIGMA: f32 = 0.12;
/// Fraction of the forward duration that the decay tail lasts after the wave.
const DECAY_TAIL: f32 = 0.35;
/// Backward wave starts after this fraction of the forward duration.
const BACK_WAVE_START: f32 = 0.60;

// ─── Mode ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationMode {
    /// Single forward pass (inference or manual prompt injection).
    Inference,
    /// Forward pass (cyan) followed by a backward gradient wave (orange).
    Training,
}

// ─── ActivationState ──────────────────────────────────────────────────────────

pub struct ActivationState {
    /// Per-node activation strength indexed [layer_idx][node_idx].
    values: Vec<Vec<f32>>,
    pub started_at: Instant,
    /// Duration of the forward wave alone.
    duration: f32,
    /// Total animation lifetime (forward + backward tail for Training mode).
    total_duration: f32,
    pub token_count: usize,
    pub mode: ActivationMode,
}

impl ActivationState {
    /// Create a forward-only inference animation.
    pub fn simulate(graph: &NetworkGraph, tokens: &[u32]) -> Self {
        Self::build(graph, tokens, ActivationMode::Inference)
    }

    /// Create a training animation (forward cyan + backward orange gradient).
    pub fn simulate_training(graph: &NetworkGraph) -> Self {
        // Use a fixed pseudo-random token set so every training step looks
        // slightly different without requiring real gradients.
        let tokens: Vec<u32> = (0u32..8)
            .map(|i| i.wrapping_mul(7919).wrapping_add(13) % 1024)
            .collect();
        Self::build(graph, &tokens, ActivationMode::Training)
    }

    /// Scale the animation duration. `mult > 1.0` = faster, `< 1.0` = slower.
    pub fn with_speed(mut self, mult: f32) -> Self {
        let factor = 1.0 / mult.max(0.1);
        self.duration *= factor;
        self.total_duration *= factor;
        self
    }

    fn build(graph: &NetworkGraph, tokens: &[u32], mode: ActivationMode) -> Self {
        let n_tokens = tokens.len().max(1);
        let values = graph
            .layers
            .iter()
            .enumerate()
            .map(|(li, layer)| {
                let kind_boost = match layer.kind {
                    LayerKind::Embedding   => 1.30,
                    LayerKind::Attention   => 1.20,
                    LayerKind::FeedForward => 1.00,
                    LayerKind::LayerNorm   => 0.75,
                    LayerKind::Output      => 1.10,
                };
                layer
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(ni, node)| {
                        let influence: f32 = tokens
                            .iter()
                            .enumerate()
                            .map(|(ti, &tok)| {
                                let angle = tok as f32 * 2.399_785
                                    + li as f32 * 1.732_051
                                    + ni as f32 * 0.618_034
                                    + ti as f32 * 0.381_966;
                                (angle.sin() * 0.5 + 0.5) * node.weight_magnitude
                            })
                            .sum::<f32>()
                            / n_tokens as f32;
                        (influence * kind_boost).clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        let duration = WAVE_DURATION_SECS * (1.0 + graph.layers.len() as f32 * 0.04);

        // Training: forward (duration) + gap before back wave + backward run + tail.
        let total_duration = match mode {
            ActivationMode::Inference => duration * (1.0 + DECAY_TAIL),
            ActivationMode::Training  => duration * (BACK_WAVE_START + 1.0 + DECAY_TAIL),
        };

        Self { values, started_at: Instant::now(), duration, total_duration, token_count: tokens.len(), mode }
    }

    // ── Forward glow ───────────────────────────────────────────────────────────

    /// Cyan→white glow intensity ∈ [0, 1] for the given node (forward pass).
    pub fn glow_at(&self, layer_idx: usize, node_idx: usize, now: Instant) -> f32 {
        let elapsed = now.duration_since(self.started_at).as_secs_f32();
        if elapsed > self.duration * (1.0 + DECAY_TAIL) {
            return 0.0;
        }
        let base = self.node_value(layer_idx, node_idx);
        let num_layers = self.values.len().max(1);
        let layer_frac = layer_idx as f32 / num_layers.saturating_sub(1).max(1) as f32;
        let wave_pos = (elapsed / self.duration).min(1.0 + DECAY_TAIL);
        wave_glow(base, wave_pos, layer_frac)
    }

    // ── Backward glow (Training only) ─────────────────────────────────────────

    /// Orange→yellow glow intensity ∈ [0, 1] for the backward gradient wave.
    /// Always returns 0 in [`ActivationMode::Inference`].
    pub fn back_glow_at(&self, layer_idx: usize, node_idx: usize, now: Instant) -> f32 {
        if self.mode != ActivationMode::Training {
            return 0.0;
        }
        let elapsed = now.duration_since(self.started_at).as_secs_f32();
        let back_elapsed = (elapsed - self.duration * BACK_WAVE_START).max(0.0);
        if back_elapsed <= 0.0 {
            return 0.0;
        }
        let base = self.node_value(layer_idx, node_idx);
        let num_layers = self.values.len().max(1);
        // Reverse: output layer → embedding; output starts at reversed_frac = 0.
        let reversed_frac =
            1.0 - layer_idx as f32 / num_layers.saturating_sub(1).max(1) as f32;
        let wave_pos = (back_elapsed / self.duration).min(1.0 + DECAY_TAIL);
        wave_glow(base, wave_pos, reversed_frac)
    }

    /// True once both waves (and their decay tails) have fully elapsed.
    pub fn is_finished(&self, now: Instant) -> bool {
        now.duration_since(self.started_at).as_secs_f32() > self.total_duration
    }

    fn node_value(&self, layer: usize, node: usize) -> f32 {
        self.values
            .get(layer)
            .and_then(|l| l.get(node))
            .copied()
            .unwrap_or(0.0)
    }
}

// ─── Shared math ──────────────────────────────────────────────────────────────

/// Gaussian wave glow for a node at `layer_frac` ∈ [0,1] given the current
/// wave front at `wave_pos` ∈ [0, 1 + DECAY_TAIL].
#[inline]
fn wave_glow(base: f32, wave_pos: f32, layer_frac: f32) -> f32 {
    let wave_intensity = gaussian(wave_pos - layer_frac, WAVE_SIGMA);
    let passed = (wave_pos - layer_frac).max(0.0);
    let residual = base * 0.25 * (1.0 - (passed / DECAY_TAIL).clamp(0.0, 1.0));
    (base * wave_intensity + residual).clamp(0.0, 1.0)
}

#[inline]
fn gaussian(x: f32, sigma: f32) -> f32 {
    (-0.5 * (x / sigma).powi(2)).exp()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::network::{Layer, LayerKind, NetworkGraph, Node};

    fn tiny_graph() -> NetworkGraph {
        let make = |kind, n: usize| Layer {
            name: format!("{kind:?}"),
            kind,
            nodes: (0..n)
                .map(|_| Node { position: [0.0; 3], weight_magnitude: 0.5 })
                .collect(),
        };
        let mut g = NetworkGraph {
            name: "test".into(),
            layers: vec![
                make(LayerKind::Embedding, 4),
                make(LayerKind::Attention, 4),
                make(LayerKind::Output, 4),
            ],
            edges: vec![],
            estimated_vram_gb: None,
        };
        g.layout();
        g
    }

    #[test]
    fn forward_glow_at_t0() {
        let g = tiny_graph();
        let s = ActivationState::simulate(&g, &[1, 2, 3]);
        let g0 = s.glow_at(0, 0, s.started_at);
        let g2 = s.glow_at(2, 0, s.started_at);
        assert!(g0 > 0.0, "embedding should glow at t=0");
        assert!(g2 < g0,  "output glow < embedding glow at t=0");
    }

    #[test]
    fn glow_zero_after_animation() {
        let g = tiny_graph();
        let s = ActivationState::simulate(&g, &[42]);
        let far = s.started_at + std::time::Duration::from_secs_f32(s.total_duration + 1.0);
        assert_eq!(s.glow_at(0, 0, far), 0.0);
        assert!(s.is_finished(far));
    }

    #[test]
    fn back_glow_zero_in_inference_mode() {
        let g = tiny_graph();
        let s = ActivationState::simulate(&g, &[1]);
        // Mid animation — backward glow must stay zero in Inference mode.
        let mid = s.started_at + std::time::Duration::from_secs_f32(s.duration * 1.0);
        assert_eq!(s.back_glow_at(2, 0, mid), 0.0);
    }

    #[test]
    fn back_glow_nonzero_in_training_mode() {
        let g = tiny_graph();
        let s = ActivationState::simulate_training(&g);
        // After the back wave starts, the output layer (idx 2, reversed_frac=0)
        // should have nonzero back_glow.
        let after_back_start = s.started_at
            + std::time::Duration::from_secs_f32(s.duration * (BACK_WAVE_START + 0.05));
        let bg = s.back_glow_at(2, 0, after_back_start);
        assert!(bg > 0.0, "output layer should glow in backward pass");
    }

    #[test]
    fn training_total_duration_longer_than_inference() {
        let g = tiny_graph();
        let inf = ActivationState::simulate(&g, &[1]);
        let trn = ActivationState::simulate_training(&g);
        assert!(trn.total_duration > inf.total_duration);
    }
}
