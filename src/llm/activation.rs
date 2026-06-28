//! Activation simulation and wave-propagation animation.
//!
//! `ActivationState::simulate` computes a per-node activation value from the
//! injected token IDs using a deterministic pseudo-random function so the
//! visualization is reproducible without running an actual forward pass.
//!
//! On each render frame `glow_at(layer, node, now)` returns a value in [0, 1]
//! that drives the shader's cyan→white emissive effect. The animation is a
//! Gaussian wave front that propagates from the embedding layer to the output
//! layer over `WAVE_DURATION_SECS`, followed by a decay tail.

use std::time::Instant;

use crate::llm::network::{LayerKind, NetworkGraph};

/// Total duration of the propagation wave in seconds.
const WAVE_DURATION_SECS: f32 = 4.0;
/// Width (σ) of the Gaussian wave front (fraction of total depth).
const WAVE_SIGMA: f32 = 0.12;
/// Fraction of total duration that the decay tail lasts after the wave exits.
const DECAY_TAIL: f32 = 0.35;

pub struct ActivationState {
    /// Activation strength per node, indexed [layer_idx][node_idx].
    values: Vec<Vec<f32>>,
    started_at: Instant,
    duration: f32,
    pub token_count: usize,
}

impl ActivationState {
    pub fn simulate(graph: &NetworkGraph, tokens: &[u32]) -> Self {
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
                        // Deterministic pseudo-random activation derived from token
                        // positions so each unique prompt produces a distinct pattern.
                        let token_influence: f32 = tokens
                            .iter()
                            .enumerate()
                            .map(|(ti, &tok)| {
                                let angle =
                                    tok as f32 * 2.399_785 // golden-ratio multiplier
                                    + li as f32 * 1.732_051
                                    + ni as f32 * 0.618_034
                                    + ti as f32 * 0.381_966;
                                (angle.sin() * 0.5 + 0.5) * node.weight_magnitude
                            })
                            .sum::<f32>()
                            / n_tokens as f32;

                        (token_influence * kind_boost).clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        // Slightly stretch the animation for deeper models.
        let duration = WAVE_DURATION_SECS * (1.0 + graph.layers.len() as f32 * 0.04);

        Self {
            values,
            started_at: Instant::now(),
            duration,
            token_count: tokens.len(),
        }
    }

    /// Current glow intensity ∈ [0, 1] for the given node.
    pub fn glow_at(&self, layer_idx: usize, node_idx: usize, now: Instant) -> f32 {
        let elapsed = now.duration_since(self.started_at).as_secs_f32();
        if elapsed > self.duration * (1.0 + DECAY_TAIL) {
            return 0.0;
        }

        let base = self
            .values
            .get(layer_idx)
            .and_then(|l| l.get(node_idx))
            .copied()
            .unwrap_or(0.0);

        let num_layers = self.values.len().max(1);
        let layer_frac = layer_idx as f32 / num_layers.saturating_sub(1).max(1) as f32;

        // Wave-front position normalised to [0, 1+DECAY_TAIL].
        let wave_pos = (elapsed / self.duration).min(1.0 + DECAY_TAIL);

        // Gaussian centred on the current wave position.
        let wave_intensity = gaussian(wave_pos - layer_frac, WAVE_SIGMA);

        // Residual glow that lingers after the wave front passes.
        let passed = (wave_pos - layer_frac).max(0.0);
        let residual = base * 0.25 * (1.0 - (passed / DECAY_TAIL).clamp(0.0, 1.0));

        (base * wave_intensity + residual).clamp(0.0, 1.0)
    }

    /// True once the wave front and its decay tail have fully elapsed.
    pub fn is_finished(&self, now: Instant) -> bool {
        now.duration_since(self.started_at).as_secs_f32()
            > self.duration * (1.0 + DECAY_TAIL)
    }
}

#[inline]
fn gaussian(x: f32, sigma: f32) -> f32 {
    (-0.5 * (x / sigma).powi(2)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::network::{Layer, LayerKind, NetworkGraph, Node};

    fn tiny_graph() -> NetworkGraph {
        let make_layer = |kind, n: usize| Layer {
            name: format!("{kind:?}"),
            kind,
            nodes: (0..n)
                .map(|_| Node { position: [0.0; 3], weight_magnitude: 0.5 })
                .collect(),
        };
        let mut g = NetworkGraph {
            name: "test".into(),
            layers: vec![
                make_layer(LayerKind::Embedding, 4),
                make_layer(LayerKind::Attention, 4),
                make_layer(LayerKind::Output, 4),
            ],
            edges: vec![],
        };
        g.layout();
        g
    }

    #[test]
    fn glow_is_zero_before_start() {
        let graph = tiny_graph();
        let state = ActivationState::simulate(&graph, &[1, 2, 3]);
        // Immediately after creation the wave hasn't propagated yet; first layer
        // should have nonzero glow (Gaussian centred at 0), last layer near zero.
        let g0 = state.glow_at(0, 0, state.started_at);
        let g2 = state.glow_at(2, 0, state.started_at);
        assert!(g0 > 0.0, "embedding should glow at t=0");
        assert!(g2 < g0, "output glow should be less than embedding at t=0");
    }

    #[test]
    fn glow_is_zero_after_animation_finishes() {
        let graph = tiny_graph();
        let state = ActivationState::simulate(&graph, &[42]);
        let far_future = state.started_at
            + std::time::Duration::from_secs_f32(state.duration * 2.0);
        assert_eq!(state.glow_at(0, 0, far_future), 0.0);
        assert!(state.is_finished(far_future));
    }
}
