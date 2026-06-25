//! Convert a [`Solid`] into a renderable [`crate::mesh::MeshData`].
//!
//! Each polygonal face is projected onto its own plane (via Newell's normal),
//! triangulated by the shared [`crate::sketch::tessellate`] ear-clipper, and
//! emitted with flat per-face normals. Faces are duplicated per triangle so the
//! shading stays crisp on the solid's edges.

use cgmath::{InnerSpace, Vector3};

use crate::mesh::MeshData;
use crate::model::Vertex;
use crate::sketch::{newell_normal, tessellate};

use super::solid::Solid;

const SOLID_COLOR: [f32; 3] = [1.0, 1.0, 1.0];

/// Triangulate every face of `solid` into a single mesh with flat normals.
/// Faces that fail to triangulate (degenerate / self-intersecting) are skipped.
pub fn solid_to_mesh(solid: &Solid) -> MeshData {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for face in &solid.faces {
        if face.verts.len() < 3 {
            continue;
        }
        let loop3: Vec<[f32; 3]> = face.verts.iter().map(|&i| solid.verts[i]).collect();
        let normal = newell_normal(&loop3);
        let (u, v) = basis_from_normal(normal);
        let origin = Vector3::from(loop3[0]);

        // Project the face loop to 2D in its own plane (CCW about +normal).
        let loop2: Vec<[f32; 2]> = loop3
            .iter()
            .map(|p| {
                let d = Vector3::from(*p) - origin;
                [d.dot(u), d.dot(v)]
            })
            .collect();

        let Some(tris) = tessellate::triangulate(&loop2) else {
            continue;
        };

        // Each triangle gets its own three vertices carrying the face normal.
        for t in tris.chunks_exact(3) {
            let base = vertices.len() as u32;
            for &li in t {
                vertices.push(Vertex {
                    position: loop3[li],
                    color: SOLID_COLOR,
                    normal,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    }

    MeshData { vertices, indices }
}

/// An orthonormal in-plane basis `(u, v)` with `u × v == normal`.
fn basis_from_normal(normal: [f32; 3]) -> (Vector3<f32>, Vector3<f32>) {
    let n = Vector3::from(normal);
    let reference = if n.x.abs() < 0.9 {
        Vector3::unit_x()
    } else {
        Vector3::unit_y()
    };
    let u = (reference - n * reference.dot(n)).normalize();
    let v = n.cross(u); // u × v == n
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::validate::orient_faces;
    use crate::brep::weld::WELD_EPS;

    fn cube_solid() -> Solid {
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
    fn cube_mesh_has_twelve_triangles() {
        let solid = cube_solid();
        let mesh = solid_to_mesh(&solid);
        // 6 quad faces × 2 triangles.
        assert_eq!(mesh.triangle_count(), 12);
        // Bounding box is the unit cube.
        assert_eq!(mesh.aabb(), ([0.0; 3], [1.0, 1.0, 1.0]));
        // All indices valid.
        let n = mesh.vertices.len() as u32;
        assert!(mesh.indices.iter().all(|&i| i < n));
    }

    #[test]
    fn basis_is_orthonormal_and_right_handed() {
        for n in [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            let (u, v) = basis_from_normal(n);
            assert!((u.magnitude() - 1.0).abs() < 1e-6);
            assert!((v.magnitude() - 1.0).abs() < 1e-6);
            assert!(u.dot(v).abs() < 1e-6);
            let cross = u.cross(v);
            assert!((cross - Vector3::from(n)).magnitude() < 1e-6);
        }
    }

    #[test]
    fn oriented_cube_normals_point_outward() {
        let mut solid = cube_solid();
        assert!(orient_faces(&mut solid));
        let mesh = solid_to_mesh(&solid);
        let center = Vector3::new(0.5, 0.5, 0.5);
        // For an outward-oriented closed solid, each triangle's flat normal
        // should point away from the centroid.
        for t in mesh.indices.chunks_exact(3) {
            let p = Vector3::from(mesh.vertices[t[0] as usize].position);
            let nrm = Vector3::from(mesh.vertices[t[0] as usize].normal);
            let outward = (p - center).dot(nrm);
            assert!(outward > 0.0, "normal not facing outward ({outward})");
        }
    }
}
