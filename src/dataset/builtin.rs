//! Synthetic benchmark datasets generated in-process, so the visualizer can
//! be exercised (and smoke-tested) without any files on disk.
//!
//! Deterministic: the same seed always produces the same dataset.

use super::{metadata::DatasetMetadata, Dataset, FeatureSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinDataset {
    /// Isotropic gaussian blobs, one cluster per label.
    Blobs {
        clusters: usize,
        points_per_cluster: usize,
        dims: usize,
    },
    /// Two interleaved 3D spirals (binary classification benchmark).
    Spirals { points_per_arm: usize },
    /// Swiss-roll manifold, labeled by angle quartile.
    SwissRoll { points: usize },
}

impl BuiltinDataset {
    pub const ALL_NAMES: [&'static str; 3] = ["blobs", "spirals", "swiss_roll"];

    pub fn name(&self) -> &'static str {
        match self {
            BuiltinDataset::Blobs { .. } => "blobs",
            BuiltinDataset::Spirals { .. } => "spirals",
            BuiltinDataset::SwissRoll { .. } => "swiss_roll",
        }
    }

    pub fn default_of(name: &str) -> Option<Self> {
        match name {
            "blobs" => Some(BuiltinDataset::Blobs {
                clusters: 5,
                points_per_cluster: 400,
                dims: 8,
            }),
            "spirals" => Some(BuiltinDataset::Spirals {
                points_per_arm: 800,
            }),
            "swiss_roll" => Some(BuiltinDataset::SwissRoll { points: 1500 }),
            _ => None,
        }
    }
}

/// Tiny deterministic PRNG (xorshift64*), enough for synthetic data.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Approximate standard normal (sum of uniforms).
    pub fn next_gauss(&mut self) -> f32 {
        let mut acc = 0.0f32;
        for _ in 0..6 {
            acc += self.next_f32();
        }
        (acc - 3.0) * (12.0f32 / 6.0).sqrt() * 0.5
    }
}

pub fn generate(kind: BuiltinDataset, seed: u64) -> Dataset {
    match kind {
        BuiltinDataset::Blobs {
            clusters,
            points_per_cluster,
            dims,
        } => blobs(clusters, points_per_cluster, dims, seed),
        BuiltinDataset::Spirals { points_per_arm } => spirals(points_per_arm, seed),
        BuiltinDataset::SwissRoll { points } => swiss_roll(points, seed),
    }
}

fn finish(
    name: &str,
    dims: usize,
    data: Vec<f32>,
    labels: Vec<u32>,
    label_names: Vec<String>,
) -> Dataset {
    let n_rows = data.len() / dims;
    let mut metadata = DatasetMetadata::new(name, "builtin", n_rows, dims);
    metadata.label_column = Some("label".to_string());
    metadata.set_label_stats(&labels, &label_names);
    Dataset {
        metadata,
        source: FeatureSource::InMemory(data),
        labels,
        label_names,
    }
}

fn blobs(clusters: usize, per_cluster: usize, dims: usize, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(clusters * per_cluster * dims);
    let mut labels = Vec::with_capacity(clusters * per_cluster);
    // Cluster centers spread on a hypersphere of radius 10.
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| {
            let mut c: Vec<f32> = (0..dims).map(|_| rng.next_gauss()).collect();
            let norm = c.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            c.iter_mut().for_each(|v| *v = *v / norm * 10.0);
            c
        })
        .collect();
    for (ci, center) in centers.iter().enumerate() {
        for _ in 0..per_cluster {
            for d in 0..dims {
                data.push(center[d] + rng.next_gauss());
            }
            labels.push(ci as u32);
        }
    }
    let names = (0..clusters).map(|i| format!("cluster_{}", i)).collect();
    finish("blobs", dims, data, labels, names)
}

fn spirals(per_arm: usize, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(per_arm * 2 * 3);
    let mut labels = Vec::with_capacity(per_arm * 2);
    for arm in 0..2u32 {
        let phase = arm as f32 * std::f32::consts::PI;
        for i in 0..per_arm {
            let t = i as f32 / per_arm as f32 * 4.0 * std::f32::consts::PI;
            let r = 0.5 + t * 0.4;
            data.push(r * (t + phase).cos() + rng.next_gauss() * 0.1);
            data.push(r * (t + phase).sin() + rng.next_gauss() * 0.1);
            data.push(t * 0.3 + rng.next_gauss() * 0.1);
            labels.push(arm);
        }
    }
    let names = vec!["arm_0".to_string(), "arm_1".to_string()];
    finish("spirals", 3, data, labels, names)
}

fn swiss_roll(points: usize, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(points * 3);
    let mut labels = Vec::with_capacity(points);
    for _ in 0..points {
        let t = 1.5 * std::f32::consts::PI * (1.0 + 2.0 * rng.next_f32());
        let y = 21.0 * rng.next_f32();
        data.push(t * t.cos());
        data.push(y);
        data.push(t * t.sin());
        // Quartile of the unrolled coordinate as the label.
        let q = ((t - 1.5 * std::f32::consts::PI) / (3.0 * std::f32::consts::PI) * 4.0)
            .floor()
            .clamp(0.0, 3.0) as u32;
        labels.push(q);
    }
    let names = (0..4).map(|i| format!("quartile_{}", i)).collect();
    finish("swiss_roll", 3, data, labels, names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        for _ in 0..100 {
            let v = a.next_f32();
            assert_eq!(v, b.next_f32());
            assert!((0.0..1.0).contains(&v));
        }
        // Zero seed must not lock the generator at zero.
        let mut z = Rng::new(0);
        assert_ne!(z.next_u64(), 0);
    }

    #[test]
    fn gauss_is_roughly_centered() {
        let mut rng = Rng::new(99);
        let n = 5000;
        let mean: f32 = (0..n).map(|_| rng.next_gauss()).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.1, "gauss mean {} too far from 0", mean);
    }

    #[test]
    fn names_round_trip_through_default_of() {
        for name in BuiltinDataset::ALL_NAMES {
            let kind = BuiltinDataset::default_of(name).unwrap();
            assert_eq!(kind.name(), name);
        }
        assert!(BuiltinDataset::default_of("missing").is_none());
    }
}
