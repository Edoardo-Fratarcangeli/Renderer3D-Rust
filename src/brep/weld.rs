//! Position welding: collapse near-coincident points into shared vertices.
//!
//! This is the single merge-by-position routine in the crate. The composer
//! uses it to discover that two surfaces share an edge: once their endpoints
//! weld to the same vertex ids, a shared edge is just an edge referenced by
//! more than one face.

use std::collections::HashMap;

/// Default welding tolerance (world units).
pub const WELD_EPS: f32 = 1e-4;

/// Merge positions that fall within `eps` of each other.
///
/// Returns `(unique, remap)` where `unique` holds one representative position
/// per cluster and `remap[i]` is the index in `unique` for input position `i`.
/// Clustering snaps each coordinate to an `eps` grid, so it runs in O(n).
pub fn weld_positions(positions: &[[f32; 3]], eps: f32) -> (Vec<[f32; 3]>, Vec<usize>) {
    let eps = eps.max(1e-9);
    let inv = 1.0 / eps;
    let key = |p: [f32; 3]| {
        (
            (p[0] * inv).round() as i64,
            (p[1] * inv).round() as i64,
            (p[2] * inv).round() as i64,
        )
    };

    let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut unique: Vec<[f32; 3]> = Vec::new();
    let mut remap: Vec<usize> = Vec::with_capacity(positions.len());

    for &p in positions {
        let k = key(p);
        match map.get(&k) {
            Some(&idx) => remap.push(idx),
            None => {
                let idx = unique.len();
                map.insert(k, idx);
                unique.push(p);
                remap.push(idx);
            }
        }
    }
    (unique, remap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coincident_points_collapse_to_one() {
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],      // duplicate of #0
            [1.0, 0.000_01, 0.0], // within eps of #1
        ];
        let (unique, remap) = weld_positions(&pts, WELD_EPS);
        assert_eq!(unique.len(), 2);
        assert_eq!(remap[0], remap[2], "duplicates share an id");
        assert_eq!(remap[1], remap[3], "near-coincident share an id");
        assert_ne!(remap[0], remap[1]);
    }

    #[test]
    fn distinct_points_are_preserved() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (unique, remap) = weld_positions(&pts, WELD_EPS);
        assert_eq!(unique.len(), 3);
        assert_eq!(remap, vec![0, 1, 2]);
    }

    #[test]
    fn remap_indexes_into_unique() {
        let pts = [[0.0; 3], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let (unique, remap) = weld_positions(&pts, WELD_EPS);
        assert!(remap.iter().all(|&i| i < unique.len()));
    }
}
