//! Generic LLM architecture registry.
//!
//! Detects the model family from arch strings / model names and produces a
//! GQA-aware [`Vec<Layer>`] for the 3D visualization.
//!
//! Key type hierarchy:
//!  [`ArchFamily`] (enum) → [`ArchSpec`] (parsed params) → [`build_layers`] → [`Vec<Layer>`]

use std::collections::HashMap;

use crate::llm::network::{Layer, LayerKind, Node, MAX_NODES_PER_LAYER};

// ─── Family detection ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchFamily {
    Llama, Mistral, Gemma, Phi, Qwen, Falcon,
    Gpt2, Bloom, Mpt, StableLm, Unknown,
}

impl ArchFamily {
    /// Identify the family from the GGUF `general.architecture` string and/or
    /// the human-readable model name.  Case-insensitive, order matters.
    pub fn detect(arch_str: &str, model_name: &str) -> Self {
        let a = arch_str.to_ascii_lowercase();
        let n = model_name.to_ascii_lowercase();
        macro_rules! hits { ($($s:literal),+) => { $(a.contains($s) || n.contains($s))||+ } }
        if hits!("llama", "vicuna", "alpaca", "codellama", "wizard") { return Self::Llama; }
        if hits!("mistral", "mixtral", "zephyr")                     { return Self::Mistral; }
        if hits!("gemma")                                             { return Self::Gemma; }
        if a.starts_with("phi") || n.starts_with("phi")             { return Self::Phi; }
        if hits!("qwen")                                              { return Self::Qwen; }
        if hits!("falcon")                                            { return Self::Falcon; }
        if a.contains("gpt2") || a.contains("gpt-2")                { return Self::Gpt2; }
        if hits!("bloom")                                             { return Self::Bloom; }
        if a.starts_with("mpt") || n.starts_with("mpt")             { return Self::Mpt; }
        if hits!("stablelm")                                          { return Self::StableLm; }
        Self::Unknown
    }
}

// ─── ArchSpec ─────────────────────────────────────────────────────────────────

/// Parsed architectural parameters for one model instance.
#[derive(Debug, Clone)]
pub struct ArchSpec {
    pub family: ArchFamily,
    pub n_layers: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    /// For GQA (Grouped-Query Attention): `n_kv_heads < n_heads`.
    /// Equals `n_heads` for standard Multi-Head Attention.
    pub n_kv_heads: usize,
    pub ffn_size: usize,
}

impl ArchSpec {
    /// Build from a flat numeric metadata map whose keys follow the convention
    /// `"{arch_str}.{param}"` — the same layout used by both GGUF KV pairs and
    /// Ollama's `model_info` JSON object.
    ///
    /// ```text
    /// "llama.block_count"              -> n_layers
    /// "llama.embedding_length"         -> hidden_size
    /// "llama.attention.head_count"     -> n_heads
    /// "llama.attention.head_count_kv"  -> n_kv_heads
    /// "llama.feed_forward_length"      -> ffn_size
    /// ```
    pub fn from_metadata(
        family: ArchFamily,
        arch_str: &str,
        meta: &HashMap<String, u64>,
    ) -> Self {
        let get = |suffix: &str| -> usize {
            meta.get(&format!("{arch_str}.{suffix}"))
                .copied()
                .unwrap_or(0) as usize
        };
        let n_layers    = get("block_count").max(1);
        let hidden_size = get("embedding_length").max(64);
        let n_heads     = get("attention.head_count").max(1);
        let kv_raw      = get("attention.head_count_kv");
        let n_kv_heads  = if kv_raw == 0 { n_heads } else { kv_raw };
        let ffn_raw     = get("feed_forward_length");
        let ffn_size    = if ffn_raw == 0 {
            hidden_size * default_ffn_ratio(family)
        } else {
            ffn_raw
        };
        Self { family, n_layers, hidden_size, n_heads, n_kv_heads, ffn_size }
    }

    /// Canonical default sizes for each well-known family (7 B-class models).
    pub fn default_for(family: ArchFamily) -> Self {
        let (n_layers, hidden, heads, kv, ffn_r) = match family {
            ArchFamily::Llama    => (32, 4096, 32, 32, 4),
            ArchFamily::Mistral  => (32, 4096, 32,  8, 4),
            ArchFamily::Gemma    => (28, 3072, 16, 16, 3),
            ArchFamily::Phi      => (24, 2048, 32, 32, 4),
            ArchFamily::Qwen     => (32, 4096, 32,  8, 3),
            ArchFamily::Falcon   => (32, 4544, 71,  1, 4),
            ArchFamily::Gpt2     => (12,  768, 12, 12, 4),
            ArchFamily::Bloom    => (24, 1024, 16, 16, 4),
            ArchFamily::Mpt      => (32, 4096, 32, 32, 4),
            ArchFamily::StableLm => (32, 4096, 32,  8, 4),
            ArchFamily::Unknown  => (12,  768, 12, 12, 4),
        };
        Self {
            family,
            n_layers,
            hidden_size: hidden,
            n_heads: heads,
            n_kv_heads: kv,
            ffn_size: hidden * ffn_r,
        }
    }
}

fn default_ffn_ratio(family: ArchFamily) -> usize {
    match family {
        ArchFamily::Gemma | ArchFamily::Qwen => 3,
        _ => 4,
    }
}

// ─── VRAM estimate ────────────────────────────────────────────────────────────

/// Rough FP16 VRAM estimate in GB.
///
/// Formula: embedding + n_layers × (attn QKV+O + FFN gate/up/down + norms) × 2 bytes.
/// Vocabulary approximated to 32 000 tokens if unknown.
pub fn estimate_vram_gb(spec: &ArchSpec) -> f64 {
    let h   = spec.hidden_size as u64;
    let n   = spec.n_layers    as u64;
    let ff  = spec.ffn_size    as u64;
    let kv_dim = (h / spec.n_heads.max(1) as u64) * spec.n_kv_heads as u64;
    let vocab: u64 = 32_000;
    let embedding  = vocab * h * 2;                          // token embed + lm head
    let attn       = (h * h + 2 * h * kv_dim + h * h) * 2; // Q,K,V,O proj
    let ffn_block  = (h * ff + ff * h + h * ff) * 2;        // gate, up, down
    let ln         = h * 4;                                  // 2 norms × 2 bytes
    let per_block  = attn + ffn_block + ln;
    (embedding + n * per_block) as f64 / 1_073_741_824.0    // bytes → GB
}

// ─── Layer builder ────────────────────────────────────────────────────────────

/// Build a visualization [`Vec<Layer>`] from an [`ArchSpec`].
///
/// Each transformer block produces three layers:
///  1. Attention (GQA-aware node sizing)
///  2. Layer-norm (thin representation)
///  3. Feed-forward
///
/// All layers are capped at [`MAX_NODES_PER_LAYER`] nodes.
///
/// Returns `(layers, estimated_fp16_vram_gb)`.
pub fn build_layers(spec: &ArchSpec) -> (Vec<Layer>, f64) {
    let mut layers = Vec::with_capacity(2 + spec.n_layers * 3);

    layers.push(Layer {
        name: "Embedding".into(),
        kind: LayerKind::Embedding,
        nodes: uniform_nodes(spec.hidden_size.min(MAX_NODES_PER_LAYER), 0.60),
    });

    for i in 0..spec.n_layers {
        layers.push(Layer {
            name: format!("Block {i} · Attn"),
            kind: LayerKind::Attention,
            nodes: attn_nodes(spec),
        });
        let ln_n = (spec.hidden_size / 32).clamp(4, MAX_NODES_PER_LAYER);
        layers.push(Layer {
            name: format!("Block {i} · LN"),
            kind: LayerKind::LayerNorm,
            nodes: uniform_nodes(ln_n, 0.40),
        });
        layers.push(Layer {
            name: format!("Block {i} · FFN"),
            kind: LayerKind::FeedForward,
            nodes: uniform_nodes(spec.ffn_size.min(MAX_NODES_PER_LAYER), 0.50),
        });
    }

    layers.push(Layer {
        name: "Output".into(),
        kind: LayerKind::Output,
        nodes: uniform_nodes(spec.hidden_size.min(MAX_NODES_PER_LAYER), 0.55),
    });

    (layers, estimate_vram_gb(spec))
}

/// GQA-aware attention node list.
///
/// When `n_kv_heads < n_heads`, every `group_size`-th node is a KV head
/// (weight_magnitude = 0.55, rendered smaller) while the rest are Q heads (0.80).
fn attn_nodes(spec: &ArchSpec) -> Vec<Node> {
    let total = spec.n_heads.min(MAX_NODES_PER_LAYER);
    if spec.n_kv_heads == 0 || spec.n_kv_heads >= spec.n_heads {
        return uniform_nodes(total, 0.70);
    }
    let group = (spec.n_heads / spec.n_kv_heads).max(1);
    (0..total)
        .map(|i| Node {
            position: [0.0; 3],
            weight_magnitude: if i % group == 0 { 0.55 } else { 0.80 },
        })
        .collect()
}

fn uniform_nodes(n: usize, w: f32) -> Vec<Node> {
    (0..n.min(MAX_NODES_PER_LAYER))
        .map(|_| Node { position: [0.0; 3], weight_magnitude: w })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_llama_from_arch_str() {
        assert_eq!(ArchFamily::detect("llama", ""), ArchFamily::Llama);
        assert_eq!(ArchFamily::detect("mistral", ""), ArchFamily::Mistral);
        assert_eq!(ArchFamily::detect("gemma", ""), ArchFamily::Gemma);
    }

    #[test]
    fn detects_from_model_name() {
        assert_eq!(ArchFamily::detect("", "mistral-7b-v0.1"), ArchFamily::Mistral);
        assert_eq!(ArchFamily::detect("", "phi-2"), ArchFamily::Phi);
        assert_eq!(ArchFamily::detect("", "codellama-13b"), ArchFamily::Llama);
    }

    #[test]
    fn gqa_nodes_have_mixed_weights() {
        let spec = ArchSpec {
            family: ArchFamily::Mistral,
            n_layers: 1,
            hidden_size: 4096,
            n_heads: 32,
            n_kv_heads: 8,
            ffn_size: 14336,
        };
        let nodes = attn_nodes(&spec);
        assert_eq!(nodes.len(), 32);
        // Every 4th node is a KV head → smaller weight
        assert!((nodes[0].weight_magnitude - 0.55).abs() < 1e-6);
        assert!((nodes[1].weight_magnitude - 0.80).abs() < 1e-6);
        assert!((nodes[4].weight_magnitude - 0.55).abs() < 1e-6);
    }

    #[test]
    fn default_for_produces_valid_spec() {
        let spec = ArchSpec::default_for(ArchFamily::Llama);
        assert!(spec.n_layers > 0);
        assert!(spec.ffn_size > spec.hidden_size);
    }

    #[test]
    fn build_layers_produces_correct_count() {
        let spec = ArchSpec::default_for(ArchFamily::Gpt2);
        // 1 embedding + n_layers * 3 (attn+ln+ffn) + 1 output
        let (layers, _vram) = build_layers(&spec);
        assert_eq!(layers.len(), 2 + spec.n_layers * 3);
    }
}
