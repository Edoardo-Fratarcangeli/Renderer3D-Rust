//! A lightweight boundary representation: welded vertices + polygonal faces,
//! with shared-edge discovery. This is the data the composer assembles when it
//! joins surfaces along common edges.

use std::collections::HashMap;

use super::weld::weld_positions;

/// One polygonal face: an ordered loop of vertex ids into [`Solid::verts`].
#[derive(Debug, Clone, PartialEq)]
pub struct Face {
    pub verts: Vec<usize>,
}

impl Face {
    /// Directed boundary edges `(from, to)` of this face, in loop order.
    pub fn directed_edges(&self) -> Vec<(usize, usize)> {
        let n = self.verts.len();
        (0..n)
            .map(|i| (self.verts[i], self.verts[(i + 1) % n]))
            .collect()
    }
}

/// Undirected edge key: vertex ids in ascending order so the same physical edge
/// from two faces collapses to one key regardless of traversal direction.
pub fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Welded vertices plus the faces that reference them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Solid {
    pub verts: Vec<[f32; 3]>,
    pub faces: Vec<Face>,
}

impl Solid {
    /// Assemble a solid from polygonal face loops given in world space. Vertices
    /// within `eps` are welded, so faces that meet along a common edge end up
    /// referencing the same vertex ids (and therefore the same edge key).
    ///
    /// Empty loops (and loops that collapse to fewer than 3 distinct vertices)
    /// are skipped.
    pub fn from_face_loops(loops: &[Vec<[f32; 3]>], eps: f32) -> Self {
        // Flatten all loop points, weld globally, then rebuild per-face loops
        // from the remap so shared vertices are identified across faces.
        let mut flat: Vec<[f32; 3]> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for l in loops {
            let start = flat.len();
            flat.extend_from_slice(l);
            spans.push((start, l.len()));
        }
        let (verts, remap) = weld_positions(&flat, eps);

        let mut faces = Vec::new();
        for (start, len) in spans {
            let mut loop_ids: Vec<usize> = (start..start + len).map(|i| remap[i]).collect();
            // Drop consecutive (and wrap-around) duplicate ids left by welding.
            dedup_cycle(&mut loop_ids);
            if loop_ids.len() >= 3 {
                faces.push(Face { verts: loop_ids });
            }
        }
        Solid { verts, faces }
    }

    /// Map every undirected edge to the faces that use it.
    pub fn edge_use(&self) -> HashMap<(usize, usize), Vec<usize>> {
        let mut map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (fi, face) in self.faces.iter().enumerate() {
            for (a, b) in face.directed_edges() {
                map.entry(edge_key(a, b)).or_default().push(fi);
            }
        }
        map
    }

    /// Edges shared by two or more faces (the "common edges" the composer
    /// stitches along). Sorted for determinism.
    pub fn shared_edges(&self) -> Vec<(usize, usize)> {
        let mut e: Vec<(usize, usize)> = self
            .edge_use()
            .into_iter()
            .filter(|(_, faces)| faces.len() >= 2)
            .map(|(edge, _)| edge)
            .collect();
        e.sort_unstable();
        e
    }

    /// Euler characteristic V − E + F. For a closed genus-0 solid this is 2.
    pub fn euler_characteristic(&self) -> i64 {
        let v = self.verts.len() as i64;
        let e = self.edge_use().len() as i64;
        let f = self.faces.len() as i64;
        v - e + f
    }

    /// Signed volume via the divergence theorem (sum of tetrahedra from the
    /// origin over a fan triangulation of each face). For a closed solid with
    /// consistently *outward* winding this is positive; negate-consistent
    /// (inward) winding makes it negative. Meaningless for open surfaces.
    pub fn signed_volume(&self) -> f32 {
        let mut vol = 0.0;
        for face in &self.faces {
            if face.verts.len() < 3 {
                continue;
            }
            let a = self.verts[face.verts[0]];
            for w in face.verts[1..].windows(2) {
                let b = self.verts[w[0]];
                let c = self.verts[w[1]];
                // a · (b × c) / 6
                let cross = [
                    b[1] * c[2] - b[2] * c[1],
                    b[2] * c[0] - b[0] * c[2],
                    b[0] * c[1] - b[1] * c[0],
                ];
                vol += a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2];
            }
        }
        vol / 6.0
    }

    /// Reverse every face loop (flip the whole shell inside-out).
    pub fn flip(&mut self) {
        for f in &mut self.faces {
            f.verts.reverse();
        }
    }
}

/// Remove consecutive duplicates and a duplicated wrap-around endpoint.
fn dedup_cycle(ids: &mut Vec<usize>) {
    ids.dedup();
    while ids.len() >= 2 && ids.first() == ids.last() {
        ids.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::super::weld::WELD_EPS;
    use super::*;

    /// The six faces of a unit cube as CCW (outward) loops.
    pub(crate) fn cube_faces() -> Vec<Vec<[f32; 3]>> {
        let v = |x: f32, y: f32, z: f32| [x, y, z];
        vec![
            // bottom (z=0), outward normal -Z ⇒ CW when seen from +Z
            vec![
                v(0.0, 0.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(1.0, 1.0, 0.0),
                v(1.0, 0.0, 0.0),
            ],
            // top (z=1)
            vec![
                v(0.0, 0.0, 1.0),
                v(1.0, 0.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(0.0, 1.0, 1.0),
            ],
            // front (y=0)
            vec![
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(1.0, 0.0, 1.0),
                v(0.0, 0.0, 1.0),
            ],
            // back (y=1)
            vec![
                v(0.0, 1.0, 0.0),
                v(0.0, 1.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(1.0, 1.0, 0.0),
            ],
            // left (x=0)
            vec![
                v(0.0, 0.0, 0.0),
                v(0.0, 0.0, 1.0),
                v(0.0, 1.0, 1.0),
                v(0.0, 1.0, 0.0),
            ],
            // right (x=1)
            vec![
                v(1.0, 0.0, 0.0),
                v(1.0, 1.0, 0.0),
                v(1.0, 1.0, 1.0),
                v(1.0, 0.0, 1.0),
            ],
        ]
    }

    #[test]
    fn cube_welds_to_eight_vertices_and_twelve_edges() {
        let solid = Solid::from_face_loops(&cube_faces(), WELD_EPS);
        assert_eq!(solid.verts.len(), 8, "a cube has 8 corners");
        assert_eq!(solid.faces.len(), 6);
        assert_eq!(solid.edge_use().len(), 12, "a cube has 12 edges");
        // Closed solid ⇒ every edge shared by exactly two faces.
        assert_eq!(solid.shared_edges().len(), 12);
        // Euler characteristic of a cube is 2.
        assert_eq!(solid.euler_characteristic(), 2);
    }

    #[test]
    fn two_quads_sharing_an_edge_report_one_shared_edge() {
        // Quads in the XY plane meeting along the x-axis segment (1,0)-(0,0)…
        // actually share the edge from (0,0) to (1,0).
        let a = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let b = vec![
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let solid = Solid::from_face_loops(&[a, b], WELD_EPS);
        assert_eq!(solid.verts.len(), 6, "two quads share two corners");
        assert_eq!(solid.shared_edges(), vec![(0, 1)]);
    }

    #[test]
    fn signed_volume_tracks_orientation_and_flips() {
        use super::super::validate::orient_faces;
        let mut solid = Solid::from_face_loops(&cube_faces(), WELD_EPS);
        orient_faces(&mut solid);
        let v = solid.signed_volume();
        // Unit cube ⇒ |volume| = 1.
        assert!((v.abs() - 1.0).abs() < 1e-4, "volume {v}");
        // Flipping the shell negates the signed volume.
        solid.flip();
        assert!((solid.signed_volume() + v).abs() < 1e-4);
    }

    #[test]
    fn disjoint_faces_share_nothing() {
        let a = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = vec![[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]];
        let solid = Solid::from_face_loops(&[a, b], WELD_EPS);
        assert_eq!(solid.verts.len(), 6);
        assert!(solid.shared_edges().is_empty());
    }
}
