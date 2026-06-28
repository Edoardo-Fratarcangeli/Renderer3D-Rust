//! Boundary-representation composer: assemble 3D solids from 2D surfaces by
//! welding shared edges.
//!
//! The pipeline is: take the world-space boundary loops of several sketch
//! surfaces, [`weld`] their coincident vertices into a shared vertex set,
//! record them as a [`Solid`] of polygonal faces, optionally [`validate`] /
//! reorient them into a consistent manifold, and finally [`to_mesh`] them into
//! a single renderable [`crate::mesh::MeshData`].
//!
//! Pure logic: no egui, no wgpu. Triangulation is delegated to
//! [`crate::sketch::tessellate`] so there is one ear-clipper in the crate.

pub mod solid;
pub mod to_mesh;
pub mod validate;
pub mod weld;

pub use solid::{edge_key, Face, Solid};
pub use to_mesh::solid_to_mesh;
pub use validate::{is_closed_manifold, manifold_issues, orient_faces, ManifoldIssue};
pub use weld::{weld_positions, WELD_EPS};

use crate::mesh::MeshData;

/// One-shot composition: build a solid from face loops, make its winding
/// consistent, and triangulate it into a mesh. Returns the mesh plus whether
/// the result is a closed, watertight manifold (useful for UI feedback).
pub fn compose(loops: &[Vec<[f32; 3]>]) -> (MeshData, bool) {
    let mut solid = Solid::from_face_loops(loops, WELD_EPS);
    orient_faces(&mut solid);
    let closed = is_closed_manifold(&solid);
    // A consistent orientation may still be globally inside-out; for a closed
    // solid, flip the whole shell so normals face outward (correct lighting).
    if closed && solid.signed_volume() < 0.0 {
        solid.flip();
    }
    (solid_to_mesh(&solid), closed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_faces() -> Vec<Vec<[f32; 3]>> {
        let v = |x: f32, y: f32, z: f32| [x, y, z];
        vec![
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
        ]
    }

    #[test]
    fn compose_closed_cube_reports_watertight() {
        let (mesh, closed) = compose(&cube_faces());
        assert!(closed, "a full cube should be watertight");
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn compose_open_box_is_not_watertight() {
        // Drop the top face: still a valid mesh, but not closed.
        let mut faces = cube_faces();
        faces.remove(1);
        let (mesh, closed) = compose(&faces);
        assert!(!closed, "five faces leave an open lid");
        assert_eq!(mesh.triangle_count(), 10);
    }
}
