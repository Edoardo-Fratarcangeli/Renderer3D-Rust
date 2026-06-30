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

    // ── CAE / simulation fields (FEM-style scalar fields over a geometry) ──
    /// Heated solid block: temperature field decaying from a hot face.
    ThermalSolid { points: usize },
    /// Cantilever beam: bending-stress field (von Mises-like magnitude).
    StressBeam { points: usize },
    /// Channel flow past a cylinder: velocity-magnitude field with a wake.
    FluidCylinder { points: usize },
    /// Vibrating plate: a mode shape whose displacement amplitude is the field
    /// (the out-of-plane displacement is baked into the z coordinate).
    ModalPlate { points: usize },
}

impl BuiltinDataset {
    /// Synthetic ML benchmark generators.
    pub const ALL_NAMES: [&'static str; 3] = ["blobs", "spirals", "swiss_roll"];

    /// CAE / simulation field generators (temperature, stress, fluid, modal).
    /// Each renders as a FEM-style banded contour cloud over a real geometry.
    pub const CAE_NAMES: [&'static str; 4] =
        ["cae_thermal", "cae_stress", "cae_flow", "cae_modal"];

    pub fn name(&self) -> &'static str {
        match self {
            BuiltinDataset::Blobs { .. } => "blobs",
            BuiltinDataset::Spirals { .. } => "spirals",
            BuiltinDataset::SwissRoll { .. } => "swiss_roll",
            BuiltinDataset::ThermalSolid { .. } => "cae_thermal",
            BuiltinDataset::StressBeam { .. } => "cae_stress",
            BuiltinDataset::FluidCylinder { .. } => "cae_flow",
            BuiltinDataset::ModalPlate { .. } => "cae_modal",
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
            "cae_thermal" => Some(BuiltinDataset::ThermalSolid { points: 6000 }),
            "cae_stress" => Some(BuiltinDataset::StressBeam { points: 6000 }),
            "cae_flow" => Some(BuiltinDataset::FluidCylinder { points: 6000 }),
            "cae_modal" => Some(BuiltinDataset::ModalPlate { points: 6000 }),
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
        BuiltinDataset::ThermalSolid { points } => cae_thermal(points, seed),
        BuiltinDataset::StressBeam { points } => cae_stress(points, seed),
        BuiltinDataset::FluidCylinder { points } => cae_flow(points, seed),
        BuiltinDataset::ModalPlate { points } => cae_modal(points, seed),
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
            for &c in center.iter() {
                data.push(c + rng.next_gauss());
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

// ─── CAE / simulation fields ────────────────────────────────────────────────────

/// Number of contour bands a continuous field is quantized into (FEM legend).
const FIELD_BANDS: usize = 10;

/// Turn sampled 3D points + a per-point scalar field into a [`Dataset`] where
/// the field is quantized into [`FIELD_BANDS`] contour bands stored as labels.
///
/// This is the single place CAE generators share: color-by-label then renders
/// the field as a banded FEM-style contour plot, and the band legend doubles as
/// a per-isovalue visibility filter — all via the existing dataset pipeline.
fn field_dataset(name: &str, points: &[[f32; 3]], field: &[f32], unit: &str) -> Dataset {
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &f in field {
        min = min.min(f);
        max = max.max(f);
    }
    let span = (max - min).max(1e-9);

    let mut data = Vec::with_capacity(points.len() * 3);
    let mut labels = Vec::with_capacity(points.len());
    for (p, &f) in points.iter().zip(field) {
        data.extend_from_slice(p);
        let t = ((f - min) / span).clamp(0.0, 1.0);
        let band = ((t * FIELD_BANDS as f32) as usize).min(FIELD_BANDS - 1);
        labels.push(band as u32);
    }

    // Band legend reads like a contour scale: low → high physical values.
    let names = (0..FIELD_BANDS)
        .map(|i| {
            let lo = min + span * (i as f32 / FIELD_BANDS as f32);
            let hi = min + span * ((i + 1) as f32 / FIELD_BANDS as f32);
            format!("{lo:.2}–{hi:.2} {unit}")
        })
        .collect();
    finish(name, 3, data, labels, names)
}

/// Heated solid block — temperature decays exponentially from a hot face at
/// x = 0, like a steady-state conduction field.
fn cae_thermal(points: usize, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    let mut pts = Vec::with_capacity(points);
    let mut field = Vec::with_capacity(points);
    for _ in 0..points {
        let x = rng.next_f32() * 4.0;
        let y = rng.next_f32() * 2.0;
        let z = rng.next_f32() * 2.0;
        // Distance from the hot face/edge at (0, 1, 1).
        let d = ((x).powi(2) + (y - 1.0).powi(2) + (z - 1.0).powi(2)).sqrt();
        let temp = 20.0 + 280.0 * (-0.7 * d).exp(); // °C, ambient 20.
        pts.push([x, y, z]);
        field.push(temp);
    }
    field_dataset("cae_thermal", &pts, &field, "°C")
}

/// Cantilever beam fixed at x = 0, loaded at the tip — bending stress grows with
/// the moment arm (L − x) and the distance from the neutral axis (y).
fn cae_stress(points: usize, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    let length = 8.0;
    let mut pts = Vec::with_capacity(points);
    let mut field = Vec::with_capacity(points);
    for _ in 0..points {
        let x = rng.next_f32() * length;
        let y = (rng.next_f32() - 0.5) * 1.0;
        let z = (rng.next_f32() - 0.5) * 1.0;
        // Bending stress ∝ moment(L-x) × fibre distance |y|, plus a shear term.
        let bending = (length - x) * y.abs() * 6.0;
        let shear = (1.0 - (2.0 * y).abs()).max(0.0) * 4.0;
        let sigma = (bending + shear).max(0.0); // MPa.
        pts.push([x, y, z]);
        field.push(sigma);
    }
    field_dataset("cae_stress", &pts, &field, "MPa")
}

/// 2D channel flow past a circular cylinder (potential flow) — the field is the
/// velocity magnitude, low in the wake and accelerated around the flanks.
fn cae_flow(points: usize, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    let a = 0.6; // cylinder radius.
    let (cx, cz) = (2.0f32, 0.0f32);
    let u_inf = 1.0f32;
    let mut pts = Vec::with_capacity(points);
    let mut field = Vec::with_capacity(points);
    while pts.len() < points {
        let x = rng.next_f32() * 8.0;
        let z = (rng.next_f32() - 0.5) * 4.0;
        let y = (rng.next_f32() - 0.5) * 0.6; // thin channel.
        let (dx, dz) = (x - cx, z - cz);
        let r = (dx * dx + dz * dz).sqrt();
        if r < a {
            continue; // inside the cylinder.
        }
        let theta = dz.atan2(dx);
        // Potential-flow velocity components around a cylinder.
        let ur = u_inf * (1.0 - (a * a) / (r * r)) * theta.cos();
        let ut = -u_inf * (1.0 + (a * a) / (r * r)) * theta.sin();
        let mut speed = (ur * ur + ut * ut).sqrt();
        // Crude viscous wake: damp speed just downstream of the cylinder.
        if dx > 0.0 && dz.abs() < a * 1.5 {
            speed *= 0.25 + 0.75 * (dx / 6.0).min(1.0);
        }
        pts.push([x, y, z]);
        field.push(speed);
    }
    field_dataset("cae_flow", &pts, &field, "m/s")
}

/// Vibrating simply-supported plate — a (2,1) mode shape. The out-of-plane
/// displacement is baked into z so the mode is visible in 3D, and the field is
/// the displacement amplitude.
fn cae_modal(points: usize, seed: u64) -> Dataset {
    let mut rng = Rng::new(seed);
    let (lx, ly) = (4.0f32, 3.0f32);
    let (m, n) = (2.0f32, 1.0f32);
    let mut pts = Vec::with_capacity(points);
    let mut field = Vec::with_capacity(points);
    for _ in 0..points {
        let x = rng.next_f32() * lx;
        let y = rng.next_f32() * ly;
        let w = (m * std::f32::consts::PI * x / lx).sin()
            * (n * std::f32::consts::PI * y / ly).sin();
        pts.push([x, y, w]); // z = displacement → mode shape visible.
        field.push(w.abs());
    }
    field_dataset("cae_modal", &pts, &field, "mm")
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
        for name in BuiltinDataset::ALL_NAMES.iter().chain(BuiltinDataset::CAE_NAMES.iter()) {
            let kind = BuiltinDataset::default_of(name).unwrap();
            assert_eq!(kind.name(), *name);
        }
        assert!(BuiltinDataset::default_of("missing").is_none());
    }

    #[test]
    fn cae_fields_generate_3d_banded_clouds() {
        for name in BuiltinDataset::CAE_NAMES {
            let kind = BuiltinDataset::default_of(name).unwrap();
            let ds = generate(kind, 7);
            assert!(ds.n_rows() > 100, "{name} produced too few points");
            assert_eq!(ds.n_cols(), 3, "{name} must be a 3D field cloud");
            // Field banding must yield more than one contour band (a real range).
            let distinct: std::collections::HashSet<u32> = ds.labels.iter().copied().collect();
            assert!(distinct.len() > 1, "{name} field collapsed to one band");
            assert!(distinct.len() <= FIELD_BANDS, "{name} exceeded band count");
            // Coordinates must be finite (potential-flow guards against r→0 etc.).
            if let super::FeatureSource::InMemory(data) = &ds.source {
                assert!(data.iter().all(|v| v.is_finite()), "{name} has non-finite coords");
            }
        }
    }
}
