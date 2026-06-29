//! Animation frame capture and JSON export.
//!
//! [`AnimExporter`] collects per-node glow snapshots while an animation runs
//! and serialises them to a compact JSON format suitable for offline replay or
//! analysis.

use std::time::Instant;
use anyhow::{Context, Result};

/// One captured snapshot of all node glows.
#[derive(serde::Serialize, Clone)]
pub struct FrameCapture {
    /// Seconds elapsed since animation start.
    pub t: f32,
    /// Flat array of forward glow ∈ [0,1], length = total_nodes.
    pub fwd: Vec<f32>,
    /// Flat array of backward glow (training only), same length.
    pub bwd: Vec<f32>,
    /// Cumulative node count per layer (layer boundary indices).
    pub layer_ends: Vec<usize>,
}

pub struct AnimExporter {
    pub active: bool,
    frames: Vec<FrameCapture>,
    started_at: Option<Instant>,
    last_capture_t: f32,
    /// Minimum seconds between frame captures.
    pub frame_interval: f32,
}

impl Default for AnimExporter {
    fn default() -> Self { Self::new() }
}

impl AnimExporter {
    pub fn new() -> Self {
        Self {
            active: false,
            frames: Vec::new(),
            started_at: None,
            last_capture_t: -1.0,
            frame_interval: 1.0 / 24.0,
        }
    }

    pub fn start(&mut self) {
        self.frames.clear();
        self.started_at = Some(Instant::now());
        self.last_capture_t = -1.0;
        self.active = true;
    }

    pub fn stop(&mut self) {
        self.active = false;
    }

    /// Record one frame. Call every time `build_render_data` runs while active.
    pub fn capture(
        &mut self,
        layer_fwd: &[Vec<f32>],
        layer_bwd: &[Vec<f32>],
    ) {
        if !self.active { return; }
        let elapsed = self.started_at
            .map(|s| s.elapsed().as_secs_f32())
            .unwrap_or(0.0);
        if elapsed - self.last_capture_t < self.frame_interval { return; }
        self.last_capture_t = elapsed;

        let mut fwd_flat   = Vec::new();
        let mut bwd_flat   = Vec::new();
        let mut layer_ends = Vec::new();
        for (fwd_layer, bwd_layer) in layer_fwd.iter().zip(layer_bwd.iter()) {
            fwd_flat.extend_from_slice(fwd_layer);
            bwd_flat.extend_from_slice(bwd_layer);
            layer_ends.push(fwd_flat.len());
        }
        self.frames.push(FrameCapture { t: elapsed, fwd: fwd_flat, bwd: bwd_flat, layer_ends });
    }

    /// Write all captured frames to a JSON file. Returns frame count.
    pub fn export_json(&self, path: &str) -> Result<usize> {
        let n = self.frames.len();
        if n == 0 { anyhow::bail!("No frames captured"); }
        let json = serde_json::json!({
            "version": 1,
            "fps": (1.0 / self.frame_interval) as u32,
            "frame_count": n,
            "frames": self.frames,
        });
        let file = std::fs::File::create(path)
            .with_context(|| format!("Cannot write to {path}"))?;
        serde_json::to_writer(file, &json)
            .context("JSON serialisation failed")?;
        Ok(n)
    }

    pub fn frame_count(&self) -> usize { self.frames.len() }
}
