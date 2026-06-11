// 3D projection of the feature matrix (PCA), with a persistent disk cache so
// large datasets are only projected once.
//
// The pipeline streams rows through Dataset::row(): the full feature matrix
// is never duplicated in RAM, only the (n_rows x 3) result is materialized.

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

/// Projected 3D coordinates, normalized to fit the view cube.
#[derive(Debug, Clone, PartialEq)]
pub struct Projection {
    pub points: Vec<[f32; 3]>,
    pub method: String,
    /// True when this projection was read back from the disk cache.
    pub from_cache: bool,
}

/// Compute (or load from cache) the 3D projection of a dataset.
///
/// `cache_dir = None` disables caching entirely.
pub fn project(
    dataset: &Dataset,
    method: ProjectionMethod,
    cache_dir: Option<&Path>,
) -> Result<Projection> {
    let cache_path = cache_dir.map(|dir| cache_file_path(dir, dataset, method));
    if let Some(path) = &cache_path {
        if let Ok(points) = load_projection_cache(path, dataset.n_rows()) {
            return Ok(Projection {
                points,
                method: method.tag().to_string(),
                from_cache: true,
            });
        }
    }

    let mut points = match method {
        ProjectionMethod::Direct => project_direct(dataset),
        ProjectionMethod::Pca => project_pca(dataset)?,
    };
    normalize_points(&mut points);

    if let Some(path) = &cache_path {
        // Cache write failures are non-fatal: the projection is still valid.
        let _ = save_projection_cache(path, &points);
    }
    Ok(Projection {
        points,
        method: method.tag().to_string(),
        from_cache: false,
    })
}

fn project_direct(dataset: &Dataset) -> Vec<[f32; 3]> {
    let mut buf = Vec::new();
    (0..dataset.n_rows())
        .map(|i| {
            dataset.row(i, &mut buf);
            [
                buf.first().copied().unwrap_or(0.0),
                buf.get(1).copied().unwrap_or(0.0),
                buf.get(2).copied().unwrap_or(0.0),
            ]
        })
        .collect()
}

fn project_pca(dataset: &Dataset) -> Result<Vec<[f32; 3]>> {
    let d = dataset.n_cols();
    let n = dataset.n_rows();
    if n == 0 {
        return Ok(Vec::new());
    }
    if d <= 3 {
        // Nothing to reduce: fall back to direct axes.
        return Ok(project_direct(dataset));
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

    // Top-3 eigenvectors via power iteration with deflation.
    let mut components: Vec<Vec<f64>> = Vec::with_capacity(3);
    for k in 0..3 {
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
        method.tag().as_bytes(),
    ])
}

pub fn cache_file_path(cache_dir: &Path, dataset: &Dataset, method: ProjectionMethod) -> PathBuf {
    cache_dir.join(format!("{:016x}.proj", cache_key(dataset, method)))
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
