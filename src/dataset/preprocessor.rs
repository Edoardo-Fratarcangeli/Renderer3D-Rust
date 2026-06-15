//! 3D projection of the feature matrix (PCA), with a persistent disk cache
//! so large datasets are only projected once.
//!
//! The pipeline streams rows through [`Dataset::row`]: the full feature
//! matrix is never duplicated in RAM, only the `(n_rows, 3)` result is
//! materialized. Cache keys ([`cache_key`]) include file size and mtime, so
//! source edits invalidate stale entries automatically.

use std::path::{Path, PathBuf};

use super::{fnv1a64, Dataset, DatasetError, Result};

/// Maximum rows used to estimate mean/covariance. Projection itself always
/// covers every row; this only bounds the O(n * d^2) estimation pass.
const MAX_ESTIMATION_ROWS: usize = 5000;
/// Covariance is O(d^2) memory; wider datasets must be reduced upstream.
const MAX_DIMS: usize = 4096;
const POWER_ITERATIONS: usize = 60;

/// Half-extent of the cube the projected cloud is normalized into.
pub const VIEW_HALF_EXTENT: f32 = 5.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionMethod {
    /// Use the first three feature columns directly (no analysis).
    Direct,
    /// Principal component analysis, top three components.
    Pca,
}

impl ProjectionMethod {
    pub fn tag(&self) -> &'static str {
        match self {
            ProjectionMethod::Direct => "direct",
            ProjectionMethod::Pca => "pca",
        }
    }
}

/// A fully specified projection: which method, how many spatial dimensions to
/// map (1, 2 or 3), and — for [`ProjectionMethod::Direct`] — which feature
/// column feeds each output axis. This is what makes the dataset import
/// configurable (project to a 1D line, a 2D plane, or the full 3D space, over
/// a chosen subset of columns / principal components).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionSpec {
    pub method: ProjectionMethod,
    /// Number of spatial dimensions actually used (clamped to 1..=3). Unused
    /// axes are left at 0, so a 2D projection lies on the z = 0 plane and a 1D
    /// projection on the x axis.
    pub dims: u8,
    /// For `Direct`: the source feature-column index mapped to X, Y, Z. Only
    /// the first `dims` entries matter. Ignored for `Pca` (axes are the top
    /// principal components).
    pub axes: [usize; 3],
}

impl Default for ProjectionSpec {
    fn default() -> Self {
        Self::full(ProjectionMethod::Pca)
    }
}

impl ProjectionSpec {
    /// Full 3D projection over the first three columns / components.
    pub fn full(method: ProjectionMethod) -> Self {
        Self {
            method,
            dims: 3,
            axes: [0, 1, 2],
        }
    }

    /// Effective dimension count, clamped to the supported 1..=3 range.
    pub fn dims(&self) -> usize {
        self.dims.clamp(1, 3) as usize
    }

    /// Stable, unique cache/identity tag, e.g. `"pca-2"` or
    /// `"direct-3-0_4_7"`.
    pub fn tag(&self) -> String {
        match self.method {
            ProjectionMethod::Pca => format!("pca-{}", self.dims()),
            ProjectionMethod::Direct => format!(
                "direct-{}-{}_{}_{}",
                self.dims(),
                self.axes[0],
                self.axes[1],
                self.axes[2]
            ),
        }
    }
}

/// Projected 3D coordinates, normalized to fit the view cube.
#[derive(Debug, Clone, PartialEq)]
pub struct Projection {
    pub points: Vec<[f32; 3]>,
    pub method: String,
    /// True when this projection was read back from the disk cache.
    pub from_cache: bool,
}

/// Compute (or load from cache) the full 3D projection of a dataset.
///
/// Convenience wrapper over [`project_spec`] using a full 3D spec.
/// `cache_dir = None` disables caching entirely.
pub fn project(
    dataset: &Dataset,
    method: ProjectionMethod,
    cache_dir: Option<&Path>,
) -> Result<Projection> {
    project_spec(dataset, &ProjectionSpec::full(method), cache_dir)
}

/// Compute (or load from cache) the projection described by `spec`.
///
/// The result is always a `Vec<[f32; 3]>`; for 1D/2D projections the unused
/// axes are held at 0. `cache_dir = None` disables caching entirely.
pub fn project_spec(
    dataset: &Dataset,
    spec: &ProjectionSpec,
    cache_dir: Option<&Path>,
) -> Result<Projection> {
    let cache_path = cache_dir.map(|dir| cache_file_path_spec(dir, dataset, spec));
    if let Some(path) = &cache_path {
        if let Ok(points) = load_projection_cache(path, dataset.n_rows()) {
            return Ok(Projection {
                points,
                method: spec.tag(),
                from_cache: true,
            });
        }
    }

    let dims = spec.dims();
    let mut points = match spec.method {
        ProjectionMethod::Direct => project_direct_axes(dataset, dims, spec.axes),
        ProjectionMethod::Pca => project_pca_n(dataset, dims)?,
    };
    normalize_points(&mut points);

    if let Some(path) = &cache_path {
        // Cache write failures are non-fatal: the projection is still valid.
        let _ = save_projection_cache(path, &points);
    }
    Ok(Projection {
        points,
        method: spec.tag(),
        from_cache: false,
    })
}

/// Map `dims` chosen feature columns onto the X/Y/Z axes; unused axes stay 0.
fn project_direct_axes(dataset: &Dataset, dims: usize, axes: [usize; 3]) -> Vec<[f32; 3]> {
    let dims = dims.min(3);
    let mut buf = Vec::new();
    (0..dataset.n_rows())
        .map(|i| {
            dataset.row(i, &mut buf);
            let mut p = [0.0f32; 3];
            for (a, slot) in p.iter_mut().enumerate().take(dims) {
                *slot = buf.get(axes[a]).copied().unwrap_or(0.0);
            }
            p
        })
        .collect()
}

fn project_pca_n(dataset: &Dataset, n_components: usize) -> Result<Vec<[f32; 3]>> {
    let d = dataset.n_cols();
    let n = dataset.n_rows();
    let n_components = n_components.clamp(1, 3);
    if n == 0 {
        return Ok(Vec::new());
    }
    if d <= n_components {
        // Not enough columns to reduce: use the columns directly.
        return Ok(project_direct_axes(dataset, n_components, [0, 1, 2]));
    }
    if d > MAX_DIMS {
        return Err(DatasetError::Unsupported(format!(
            "PCA limited to {} dimensions (dataset has {})",
            MAX_DIMS, d
        )));
    }

    // Subsample rows evenly for the estimation pass.
    let est_n = n.min(MAX_ESTIMATION_ROWS);
    let stride = (n / est_n).max(1);
    let est_rows: Vec<usize> = (0..n).step_by(stride).take(est_n).collect();

    // Pass 1: mean.
    let mut mean = vec![0.0f64; d];
    let mut buf = Vec::new();
    for &i in &est_rows {
        dataset.row(i, &mut buf);
        for c in 0..d {
            mean[c] += buf[c] as f64;
        }
    }
    let inv = 1.0 / est_rows.len() as f64;
    for m in &mut mean {
        *m *= inv;
    }

    // Pass 2: covariance (upper triangle, symmetrized).
    let mut cov = vec![0.0f64; d * d];
    let mut centered = vec![0.0f64; d];
    for &i in &est_rows {
        dataset.row(i, &mut buf);
        for c in 0..d {
            centered[c] = buf[c] as f64 - mean[c];
        }
        for r in 0..d {
            let cr = centered[r];
            if cr == 0.0 {
                continue;
            }
            let row_base = r * d;
            for c in r..d {
                cov[row_base + c] += cr * centered[c];
            }
        }
    }
    for r in 0..d {
        for c in r..d {
            let v = cov[r * d + c] * inv;
            cov[r * d + c] = v;
            cov[c * d + r] = v;
        }
    }

    // Top-`n_components` eigenvectors via power iteration with deflation.
    let mut components: Vec<Vec<f64>> = Vec::with_capacity(n_components);
    for k in 0..n_components {
        let mut v: Vec<f64> = (0..d)
            .map(|i| {
                // Deterministic pseudo-random start (xorshift on index).
                let mut x = (i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15) ^ (k as u64 + 7);
                x ^= x >> 33;
                x = x.wrapping_mul(0xFF51AFD7ED558CCD);
                ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            })
            .collect();
        normalize_vec(&mut v);
        for _ in 0..POWER_ITERATIONS {
            // w = C v
            let mut w = vec![0.0f64; d];
            for r in 0..d {
                let mut acc = 0.0;
                let base = r * d;
                for c in 0..d {
                    acc += cov[base + c] * v[c];
                }
                w[r] = acc;
            }
            // Deflate against previously found components.
            for comp in &components {
                let dot: f64 = w.iter().zip(comp).map(|(a, b)| a * b).sum();
                for (wi, ci) in w.iter_mut().zip(comp) {
                    *wi -= dot * ci;
                }
            }
            if !normalize_vec(&mut w) {
                break; // Degenerate direction (e.g. constant data).
            }
            v = w;
        }
        components.push(v);
    }

    // Pass 3: project every row.
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        dataset.row(i, &mut buf);
        let mut p = [0.0f32; 3];
        for (k, comp) in components.iter().enumerate() {
            let mut acc = 0.0f64;
            for c in 0..d {
                acc += (buf[c] as f64 - mean[c]) * comp[c];
            }
            p[k] = acc as f32;
        }
        points.push(p);
    }
    Ok(points)
}

fn normalize_vec(v: &mut [f64]) -> bool {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    true
}

/// Center the cloud and scale it uniformly into the view cube.
pub fn normalize_points(points: &mut [[f32; 3]]) {
    if points.is_empty() {
        return;
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in points.iter() {
        for a in 0..3 {
            min[a] = min[a].min(p[a]);
            max[a] = max[a].max(p[a]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let extent = (0..3).map(|a| max[a] - min[a]).fold(0.0f32, f32::max);
    let scale = if extent > 1e-12 {
        2.0 * VIEW_HALF_EXTENT / extent
    } else {
        1.0
    };
    for p in points.iter_mut() {
        for a in 0..3 {
            p[a] = (p[a] - center[a]) * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Disk cache
// ---------------------------------------------------------------------------
//
// Layout: <cache_dir>/<key>.proj — little-endian f32 triples, prefixed with
// a u64 row count. The key hashes source path, file size/mtime, shape and
// projection method, so edits to the source invalidate the cache naturally.

pub fn cache_key(dataset: &Dataset, method: ProjectionMethod) -> u64 {
    cache_key_spec(dataset, &ProjectionSpec::full(method))
}

/// Cache key for a fully specified projection (see [`cache_key`]).
pub fn cache_key_spec(dataset: &Dataset, spec: &ProjectionSpec) -> u64 {
    let meta = &dataset.metadata;
    let (size, mtime) = std::fs::metadata(&meta.source_path)
        .map(|m| {
            (
                m.len(),
                m.modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            )
        })
        .unwrap_or((0, 0));
    fnv1a64(&[
        meta.source_path.as_bytes(),
        meta.name.as_bytes(),
        &size.to_le_bytes(),
        &mtime.to_le_bytes(),
        &(meta.n_rows as u64).to_le_bytes(),
        &(meta.n_cols as u64).to_le_bytes(),
        spec.tag().as_bytes(),
    ])
}

pub fn cache_file_path(cache_dir: &Path, dataset: &Dataset, method: ProjectionMethod) -> PathBuf {
    cache_file_path_spec(cache_dir, dataset, &ProjectionSpec::full(method))
}

pub fn cache_file_path_spec(cache_dir: &Path, dataset: &Dataset, spec: &ProjectionSpec) -> PathBuf {
    cache_dir.join(format!("{:016x}.proj", cache_key_spec(dataset, spec)))
}

pub fn save_projection_cache(path: &Path, points: &[[f32; 3]]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut bytes = Vec::with_capacity(8 + points.len() * 12);
    bytes.extend_from_slice(&(points.len() as u64).to_le_bytes());
    for p in points {
        for v in p {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn load_projection_cache(path: &Path, expected_rows: usize) -> Result<Vec<[f32; 3]>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        return Err(DatasetError::Format("projection cache too small".into()));
    }
    let count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if count != expected_rows || bytes.len() != 8 + count * 12 {
        return Err(DatasetError::Format("projection cache shape mismatch".into()));
    }
    let mut points = Vec::with_capacity(count);
    for i in 0..count {
        let base = 8 + i * 12;
        let mut p = [0.0f32; 3];
        for a in 0..3 {
            p[a] = f32::from_le_bytes(bytes[base + a * 4..base + a * 4 + 4].try_into().unwrap());
        }
        points.push(p);
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_tags_are_distinct() {
        assert_eq!(ProjectionMethod::Pca.tag(), "pca");
        assert_eq!(ProjectionMethod::Direct.tag(), "direct");
    }

    fn grid_dataset() -> Dataset {
        // 4 rows, 4 distinct columns so column selection is observable.
        let data = vec![
            0.0, 10.0, 20.0, 30.0, //
            1.0, 11.0, 21.0, 31.0, //
            2.0, 12.0, 22.0, 32.0, //
            3.0, 13.0, 23.0, 33.0,
        ];
        Dataset {
            metadata: super::super::metadata::DatasetMetadata::new("g", "builtin", 4, 4),
            source: super::super::FeatureSource::InMemory(data),
            labels: vec![0; 4],
            label_names: vec!["unlabeled".into()],
        }
    }

    #[test]
    fn spec_tags_are_unique_per_config() {
        let a = ProjectionSpec {
            method: ProjectionMethod::Direct,
            dims: 2,
            axes: [0, 3, 2],
        };
        let b = ProjectionSpec {
            method: ProjectionMethod::Direct,
            dims: 2,
            axes: [0, 1, 2],
        };
        assert_ne!(a.tag(), b.tag());
        assert_eq!(ProjectionSpec::full(ProjectionMethod::Pca).tag(), "pca-3");
    }

    #[test]
    fn one_dimensional_projection_flattens_y_and_z() {
        let ds = grid_dataset();
        let spec = ProjectionSpec {
            method: ProjectionMethod::Direct,
            dims: 1,
            axes: [0, 1, 2],
        };
        let proj = project_spec(&ds, &spec, None).unwrap();
        for p in &proj.points {
            assert_eq!(p[1], 0.0);
            assert_eq!(p[2], 0.0);
        }
        // The single active axis still spans the view cube.
        let max = proj.points.iter().map(|p| p[0].abs()).fold(0.0f32, f32::max);
        assert!((max - VIEW_HALF_EXTENT).abs() < 1e-3);
    }

    #[test]
    fn two_dimensional_projection_flattens_z_only() {
        let ds = grid_dataset();
        let spec = ProjectionSpec {
            method: ProjectionMethod::Direct,
            dims: 2,
            axes: [0, 1, 2],
        };
        let proj = project_spec(&ds, &spec, None).unwrap();
        assert!(proj.points.iter().all(|p| p[2] == 0.0));
        assert!(proj.points.iter().any(|p| p[1] != 0.0));
    }

    #[test]
    fn direct_axes_select_the_requested_columns() {
        let ds = grid_dataset();
        // Map column 3 -> X, column 0 -> Y.
        let spec = ProjectionSpec {
            method: ProjectionMethod::Direct,
            dims: 2,
            axes: [3, 0, 1],
        };
        let proj = project_spec(&ds, &spec, None).unwrap();
        // Column 3 is constant-stride like column 0, so after normalization the
        // X and Y spreads must match (both span the cube symmetrically).
        let span = |axis: usize| {
            let mx = proj.points.iter().map(|p| p[axis]).fold(f32::MIN, f32::max);
            let mn = proj.points.iter().map(|p| p[axis]).fold(f32::MAX, f32::min);
            mx - mn
        };
        assert!(span(0) > 0.0 && span(1) > 0.0);
    }

    #[test]
    fn normalize_handles_empty_and_degenerate_clouds() {
        let mut empty: Vec<[f32; 3]> = Vec::new();
        normalize_points(&mut empty); // must not panic

        // All-identical points: centered at origin, no NaNs from /0.
        let mut same = vec![[2.0, 2.0, 2.0]; 4];
        normalize_points(&mut same);
        for p in &same {
            assert_eq!(*p, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn normalize_fills_the_view_cube() {
        let mut pts = vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]];
        normalize_points(&mut pts);
        let max = pts
            .iter()
            .flat_map(|p| p.iter())
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((max - VIEW_HALF_EXTENT).abs() < 1e-3);
    }

    #[test]
    fn cache_load_rejects_corrupt_files() {
        let dir = std::env::temp_dir().join(format!("r3d_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Too small.
        let p1 = dir.join("small.proj");
        std::fs::write(&p1, [1, 2, 3]).unwrap();
        assert!(load_projection_cache(&p1, 1).is_err());

        // Row-count mismatch with the requesting dataset.
        let p2 = dir.join("mismatch.proj");
        save_projection_cache(&p2, &[[1.0, 2.0, 3.0]]).unwrap();
        assert!(load_projection_cache(&p2, 2).is_err());
        // Truncated payload behind a valid count.
        let mut bytes = std::fs::read(&p2).unwrap();
        bytes.pop();
        std::fs::write(&p2, bytes).unwrap();
        assert!(load_projection_cache(&p2, 1).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cache_roundtrip_preserves_points() {
        let dir = std::env::temp_dir().join(format!("r3d_rt_{}", std::process::id()));
        let path = dir.join("ok.proj");
        let points = vec![[1.5, -2.5, 3.25], [0.0, 0.5, -0.5]];
        save_projection_cache(&path, &points).unwrap();
        assert_eq!(load_projection_cache(&path, 2).unwrap(), points);
        std::fs::remove_dir_all(&dir).ok();
    }
}
