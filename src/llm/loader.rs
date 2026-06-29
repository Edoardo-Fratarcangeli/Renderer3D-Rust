//! Load a [`NetworkGraph`] from:
//!  - **JSON** – a lightweight model-description format (see `JsonModel`).
//!  - **GGUF** – the binary format used by llama.cpp and compatible runtimes.
//!    Only the metadata header and tensor info sections are read; large weight
//!    tensors are never fully loaded into memory.
//!
//! Both loaders return a `(NetworkGraph, Option<Tokenizer>)` pair.  The
//! tokenizer is populated when the source embeds a vocabulary.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::llm::arch::{ArchFamily, ArchSpec};
use crate::llm::network::{Layer, LayerKind, NetworkGraph, Node, MAX_NODES_PER_LAYER};
use crate::llm::tokenizer::Tokenizer;

// ─── JSON format ─────────────────────────────────────────────────────────────

/// Top-level JSON model descriptor.
///
/// ```json
/// {
///   "name": "TinyGPT",
///   "vocabulary": ["hello", "world", …],
///   "layers": [
///     { "name": "embedding", "kind": "Embedding", "node_weights": [0.8, 0.3, …] },
///     { "name": "block_0.attn", "kind": "Attention", "num_nodes": 32 }
///   ]
/// }
/// ```
#[derive(Deserialize)]
pub struct JsonModel {
    pub name: String,
    #[serde(default)]
    pub vocabulary: Vec<String>,
    pub layers: Vec<JsonLayer>,
}

#[derive(Deserialize)]
pub struct JsonLayer {
    pub name: String,
    pub kind: JsonLayerKind,
    /// Per-node mean absolute weight (drives sphere size). If absent, `num_nodes`
    /// uniform nodes are created with weight 0.5.
    #[serde(default)]
    pub node_weights: Vec<f32>,
    #[serde(default)]
    pub num_nodes: Option<usize>,
}

#[derive(Deserialize)]
pub enum JsonLayerKind {
    Embedding,
    Attention,
    FeedForward,
    LayerNorm,
    Output,
}

/// Parse a JSON model description into a [`NetworkGraph`].
pub fn from_json(data: &[u8]) -> Result<(NetworkGraph, Option<Tokenizer>)> {
    let model: JsonModel =
        serde_json::from_slice(data).context("Invalid JSON model file")?;

    let tokenizer = if model.vocabulary.is_empty() {
        None
    } else {
        Some(Tokenizer::from_vocab(model.vocabulary))
    };

    let layers = model
        .layers
        .into_iter()
        .map(|jl| {
            let kind = match jl.kind {
                JsonLayerKind::Embedding   => LayerKind::Embedding,
                JsonLayerKind::Attention   => LayerKind::Attention,
                JsonLayerKind::FeedForward => LayerKind::FeedForward,
                JsonLayerKind::LayerNorm   => LayerKind::LayerNorm,
                JsonLayerKind::Output      => LayerKind::Output,
            };
            let raw = if !jl.node_weights.is_empty() {
                jl.node_weights
            } else {
                vec![0.5; jl.num_nodes.unwrap_or(16)]
            };
            let nodes = capped_nodes(&raw);
            Layer { name: jl.name, kind, nodes }
        })
        .collect();

    let mut graph = NetworkGraph { name: model.name, layers, edges: vec![], estimated_vram_gb: None, moe_config: None };
    graph.layout();
    Ok((graph, tokenizer))
}

// ─── GGUF binary format ───────────────────────────────────────────────────────

/// Parse the metadata header of a GGUF file.
///
/// Large weight tensors are never loaded; we only read key-value metadata
/// (model name, architecture, layer counts) and tensor info (shapes).
/// Node weights are synthesised from the tensor shape statistics.
pub fn from_gguf(data: &[u8]) -> Result<(NetworkGraph, Option<Tokenizer>)> {
    let mut r = GgufReader::new(data);

    let magic = r.read_u32().context("GGUF: reading magic")?;
    if magic != 0x4655_4747 {
        bail!("Not a GGUF file (magic mismatch; expected 0x46554747, got 0x{magic:08X})");
    }
    let version = r.read_u32().context("GGUF: reading version")?;
    if !(1..=3).contains(&version) {
        bail!("Unsupported GGUF version {version} (expected 1-3)");
    }

    let n_tensors = r.read_u64().context("GGUF: n_tensors")? as usize;
    let n_kv      = r.read_u64().context("GGUF: n_kv")?      as usize;

    // ── Key-value metadata ────────────────────────────────────────────
    let mut kv: HashMap<String, GgufValue> = HashMap::with_capacity(n_kv);
    for _ in 0..n_kv {
        let key   = r.read_string().context("GGUF: kv key")?;
        let vtype = r.read_u32().context("GGUF: kv vtype")?;
        let val   = r.read_value(vtype).with_context(|| format!("GGUF: kv value for '{key}'"))?;
        kv.insert(key, val);
    }

    // ── Tensor info (read shapes, skip data offsets) ──────────────────
    let mut tensor_shapes: HashMap<String, Vec<u64>> = HashMap::new();
    for _ in 0..n_tensors {
        let name   = r.read_string().context("GGUF: tensor name")?;
        let n_dims = r.read_u32().context("GGUF: n_dims")? as usize;
        let dims: Vec<u64> = (0..n_dims)
            .map(|_| r.read_u64())
            .collect::<Result<_>>()
            .context("GGUF: tensor dims")?;
        let _dtype  = r.read_u32().context("GGUF: tensor dtype")?;
        let _offset = r.read_u64().context("GGUF: tensor offset")?;
        tensor_shapes.insert(name, dims);
    }

    // ── Derive architecture metadata ──────────────────────────────────
    let arch = kv
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .unwrap_or("llama")
        .to_owned();

    let name = kv
        .get("general.name")
        .and_then(GgufValue::as_str)
        .unwrap_or(&arch)
        .to_owned();

    // ── Vocabulary ────────────────────────────────────────────────────
    let vocab: Vec<String> = kv
        .get("tokenizer.ggml.tokens")
        .and_then(GgufValue::as_string_array)
        .unwrap_or_default();

    let tokenizer = if vocab.is_empty() {
        None
    } else {
        Some(Tokenizer::from_vocab(vocab))
    };

    // ── Build architecture spec from KV numeric metadata ──────────────
    let numeric_meta: HashMap<String, u64> = kv
        .iter()
        .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n)))
        .collect();

    let family = ArchFamily::detect(&arch, &name);
    let spec   = ArchSpec::from_metadata(family, &arch, &numeric_meta);
    let (layers, vram_gb, moe_config) = crate::llm::arch::build_layers(&spec);

    let mut graph = NetworkGraph { name, layers, edges: vec![], estimated_vram_gb: Some(vram_gb), moe_config };
    graph.layout();
    Ok((graph, tokenizer))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Downsample weights to ≤ MAX_NODES_PER_LAYER and normalise to [0, 1].
fn capped_nodes(weights: &[f32]) -> Vec<Node> {
    let n = weights.len().min(MAX_NODES_PER_LAYER);
    let stride = (weights.len() / n).max(1);
    let sampled: Vec<f32> = weights.iter().step_by(stride).take(n).map(|&w| w.abs()).collect();
    let max = sampled.iter().cloned().fold(0.0f32, f32::max).max(1e-8);
    sampled
        .into_iter()
        .map(|w| Node { position: [0.0; 3], weight_magnitude: w / max })
        .collect()
}

// ─── GGUF binary reader ───────────────────────────────────────────────────────

struct GgufReader<'a> {
    data: &'a [u8],
    pos:  usize,
}

impl<'a> GgufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            bail!("Unexpected end of GGUF data at offset {} (needed {n} bytes)", self.pos);
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let b   = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    fn read_value(&mut self, vtype: u32) -> Result<GgufValue> {
        match vtype {
            0  => Ok(GgufValue::U8(self.read_u8()?)),
            1  => Ok(GgufValue::I8(self.read_u8()? as i8)),
            2  => Ok(GgufValue::U16(self.read_u16()?)),
            3  => Ok(GgufValue::I16(self.read_u16()? as i16)),
            4  => Ok(GgufValue::U32(self.read_u32()?)),
            5  => Ok(GgufValue::I32(self.read_u32()? as i32)),
            6  => Ok(GgufValue::F32(self.read_f32()?)),
            7  => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8  => Ok(GgufValue::Str(self.read_string()?)),
            9  => {
                let item_type = self.read_u32()?;
                // Cap array length to avoid OOM on corrupt/truncated files.
                let count = (self.read_u64()? as usize).min(131_072);
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.read_value(item_type)?);
                }
                Ok(GgufValue::Array(items))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_u64()? as i64)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            _  => bail!("Unknown GGUF value type {vtype}"),
        }
    }
}

enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16),
    U32(u32), I32(i32), F32(f32), Bool(bool),
    Str(String), Array(Vec<GgufValue>),
    U64(u64), I64(i64), F64(f64),
}

impl GgufValue {
    fn as_str(&self) -> Option<&str> {
        if let GgufValue::Str(s) = self { Some(s) } else { None }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v)  => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    fn as_string_array(&self) -> Option<Vec<String>> {
        if let GgufValue::Array(items) = self {
            Some(
                items
                    .iter()
                    .filter_map(|v| {
                        if let GgufValue::Str(s) = v { Some(s.clone()) } else { None }
                    })
                    .collect(),
            )
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_minimal_model_parses() {
        let json = br#"{
          "name": "TinyModel",
          "layers": [
            { "name": "emb", "kind": "Embedding", "num_nodes": 8 },
            { "name": "out", "kind": "Output",    "node_weights": [0.1, 0.9, 0.5] }
          ]
        }"#;
        let (graph, tok) = from_json(json).unwrap();
        assert_eq!(graph.name, "TinyModel");
        assert_eq!(graph.layers.len(), 2);
        assert_eq!(graph.layers[0].nodes.len(), 8);
        assert_eq!(graph.layers[1].nodes.len(), 3);
        assert!(tok.is_none());
    }

    #[test]
    fn json_with_vocab_builds_tokenizer() {
        let json = br#"{
          "name": "M",
          "vocabulary": ["hello", "world"],
          "layers": [{ "name": "e", "kind": "Embedding", "num_nodes": 2 }]
        }"#;
        let (_, tok) = from_json(json).unwrap();
        assert!(tok.is_some());
        assert_eq!(tok.unwrap().vocab_size(), 2);
    }

    #[test]
    fn gguf_wrong_magic_is_rejected() {
        let bad = [0u8; 32];
        assert!(from_gguf(&bad).is_err());
    }
}
