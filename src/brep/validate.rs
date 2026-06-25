//! Topology validation: manifold checks and consistent face orientation.

use std::collections::VecDeque;

use super::solid::{edge_key, Face, Solid};

/// A problem found while checking that a solid is a clean 2-manifold.
#[derive(Debug, Clone, PartialEq)]
pub enum ManifoldIssue {
    /// An edge used by a single face: the surface has a hole / open border.
    BoundaryEdge((usize, usize)),
    /// An edge shared by three or more faces: not a 2-manifold.
    NonManifoldEdge { edge: (usize, usize), faces: usize },
}

/// List every manifold violation. An empty result means a closed 2-manifold.
pub fn manifold_issues(solid: &Solid) -> Vec<ManifoldIssue> {
    let mut issues = Vec::new();
    let mut edges: Vec<((usize, usize), usize)> = solid
        .edge_use()
        .into_iter()
        .map(|(e, faces)| (e, faces.len()))
        .collect();
    edges.sort_unstable();
    for (edge, count) in edges {
        match count {
            1 => issues.push(ManifoldIssue::BoundaryEdge(edge)),
            2 => {}
            n => issues.push(ManifoldIssue::NonManifoldEdge { edge, faces: n }),
        }
    }
    issues
}

/// True when every edge is shared by exactly two faces (closed, watertight).
pub fn is_closed_manifold(solid: &Solid) -> bool {
    manifold_issues(solid).is_empty()
}

/// Reorient faces so neighbours sharing an edge traverse it in opposite
/// directions (a consistent winding across the whole solid). Faces are flipped
/// in place. Returns `false` if the surface is non-orientable (e.g. a Möbius
/// configuration) — in that case `solid` is left untouched.
///
/// Operates per connected component, so disjoint shells are each made
/// consistent relative to their own first face.
pub fn orient_faces(solid: &mut Solid) -> bool {
    let nf = solid.faces.len();
    if nf == 0 {
        return true;
    }
    let edge_use = solid.edge_use();

    // flip[f] == Some(true/false) once f has been assigned an orientation.
    let mut flip: Vec<Option<bool>> = vec![None; nf];

    for seed in 0..nf {
        if flip[seed].is_some() {
            continue;
        }
        flip[seed] = Some(false);
        let mut queue = VecDeque::from([seed]);
        while let Some(fi) = queue.pop_front() {
            let fi_flip = flip[fi].unwrap();
            for (a, b) in solid.faces[fi].directed_edges() {
                let key = edge_key(a, b);
                let Some(neighbors) = edge_use.get(&key) else {
                    continue;
                };
                for &fj in neighbors {
                    if fj == fi {
                        continue;
                    }
                    let Some(dj) = directed_for_key(&solid.faces[fj], key) else {
                        continue;
                    };
                    // fi's effective directed edge along this key.
                    let di_eff = apply_flip((a, b), fi_flip);
                    // For consistency, fj must traverse the edge opposite to fi.
                    // Decide fj's flip so that its effective edge is reversed.
                    let want_flip = apply_flip(dj, false) == di_eff;
                    match flip[fj] {
                        None => {
                            flip[fj] = Some(want_flip);
                            queue.push_back(fj);
                        }
                        Some(existing) => {
                            if existing != want_flip {
                                return false; // contradictory ⇒ non-orientable
                            }
                        }
                    }
                }
            }
        }
    }

    for (fi, f) in solid.faces.iter_mut().enumerate() {
        if flip[fi] == Some(true) {
            f.verts.reverse();
        }
    }
    true
}

/// The directed edge `(from, to)` that face traverses for the given key.
fn directed_for_key(face: &Face, key: (usize, usize)) -> Option<(usize, usize)> {
    face.directed_edges()
        .into_iter()
        .find(|&(a, b)| edge_key(a, b) == key)
}

fn apply_flip(d: (usize, usize), flip: bool) -> (usize, usize) {
    if flip {
        (d.1, d.0)
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::super::solid::Solid;
    use super::super::weld::WELD_EPS;
    use super::*;

    fn cube() -> Solid {
        // Reuse the cube face fixture from the solid module's tests via a local
        // copy (kept identical) so this module's tests are self-contained.
        let v = |x: f32, y: f32, z: f32| [x, y, z];
        let faces = vec![
            vec![
                v(0.0, 0.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(1.0, 1.0, 0.0),
                v(1.0, 0.0, 0.0),
            ],
            vec![
                v(0.0, 0.0, 1.0),
                v(1.0, 0.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(0.0, 1.0, 1.0),
            ],
            vec![
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(1.0, 0.0, 1.0),
                v(0.0, 0.0, 1.0),
            ],
            vec![
                v(0.0, 1.0, 0.0),
                v(0.0, 1.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(1.0, 1.0, 0.0),
            ],
            vec![
                v(0.0, 0.0, 0.0),
                v(0.0, 0.0, 1.0),
                v(0.0, 1.0, 1.0),
                v(0.0, 1.0, 0.0),
            ],
            vec![
                v(1.0, 0.0, 0.0),
                v(1.0, 1.0, 0.0),
                v(1.0, 1.0, 1.0),
                v(1.0, 0.0, 1.0),
            ],
        ];
        Solid::from_face_loops(&faces, WELD_EPS)
    }

    #[test]
    fn closed_cube_is_a_manifold() {
        let solid = cube();
        assert!(is_closed_manifold(&solid));
        assert!(manifold_issues(&solid).is_empty());
    }

    #[test]
    fn open_surface_reports_boundary_edges() {
        // A single quad: all four edges are borders.
        let quad = vec![vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]];
        let solid = Solid::from_face_loops(&quad, WELD_EPS);
        let issues = manifold_issues(&solid);
        assert_eq!(issues.len(), 4);
        assert!(issues
            .iter()
            .all(|i| matches!(i, ManifoldIssue::BoundaryEdge(_))));
        assert!(!is_closed_manifold(&solid));
    }

    #[test]
    fn orientation_makes_neighbours_consistent() {
        let mut solid = cube();
        assert!(orient_faces(&mut solid));
        // After orientation, for every shared edge the two faces traverse it in
        // opposite directions.
        let edge_use = solid.edge_use();
        for (key, faces) in &edge_use {
            if faces.len() == 2 {
                let d0 = directed_for_key(&solid.faces[faces[0]], *key).unwrap();
                let d1 = directed_for_key(&solid.faces[faces[1]], *key).unwrap();
                assert_eq!(d0, (d1.1, d1.0), "edge {key:?} not consistently wound");
            }
        }
    }

    #[test]
    fn orientation_is_idempotent() {
        let mut a = cube();
        orient_faces(&mut a);
        let mut b = a.clone();
        assert!(orient_faces(&mut b));
        assert_eq!(a, b, "already-consistent solid is unchanged");
    }
}
