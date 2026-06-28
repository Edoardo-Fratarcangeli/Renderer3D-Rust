//! 2D sketches → 3D surfaces.
//!
//! A [`Sketch`] is a [`Profile`] (a chain of straight or curved [`Segment`]s)
//! placed on a [`Plane`] in world space. A closed profile becomes a filled,
//! double-sided surface; an open profile becomes a thin ribbon "stroke". Both
//! produce a [`crate::mesh::MeshData`], so generated geometry travels the exact
//! same render / picking path as imported models — no parallel mesh type.
//!
//! Like [`crate::geometry`], this module is pure logic: no egui, no wgpu. Every
//! function is driven by the headless tests below.
//!
//! - [`segment`] — one boundary edge (line / arc / Bézier) + adaptive flatten
//! - [`profile`] — segment chain, validation, flattening, signed area
//! - [`tessellate`] — ear-clipping triangulation (shared with the B-rep layer)

pub mod profile;
pub mod segment;
pub mod tessellate;

pub use profile::{Profile, ProfileError};
pub use segment::Segment;

use cgmath::InnerSpace;

use crate::mesh::MeshData;
use crate::model::Vertex;

/// A point in 2D sketch space.
pub type Vec2 = [f32; 2];

/// Default chord tolerance for flattening curves (sketch units).
pub const DEFAULT_TOLERANCE: f32 = 0.01;
/// Default half-thickness used when rendering an open polyline as a ribbon.
pub const DEFAULT_STROKE_WIDTH: f32 = 0.05;
/// Color baked into generated surfaces; the per-object instance color tints it.
const SURFACE_COLOR: [f32; 3] = [1.0, 1.0, 1.0];

/// A plane in world space: an origin and an orthonormal basis. `normal` is
/// `u × v`, so a CCW loop in `(u, v)` faces along `+normal`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub origin: [f32; 3],
    pub u: [f32; 3],
    pub v: [f32; 3],
    pub normal: [f32; 3],
}

impl Plane {
    /// The world XY plane (sketch X→world X, Y→world Y, normal +Z).
    pub fn xy() -> Self {
        Self {
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        }
    }

    /// The world XZ plane, sketch X→world X, Y→world Z. The normal is `u × v`
    /// (= −Y); surfaces are double-sided, so the front-face sign is cosmetic.
    pub fn xz() -> Self {
        Self {
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 0.0, 1.0],
            normal: [0.0, -1.0, 0.0],
        }
    }

    /// The world YZ plane (normal +X), sketch X→world Y, Y→world Z.
    pub fn yz() -> Self {
        Self {
            origin: [0.0, 0.0, 0.0],
            u: [0.0, 1.0, 0.0],
            v: [0.0, 0.0, 1.0],
            normal: [1.0, 0.0, 0.0],
        }
    }

    /// Map a 2D sketch point onto this plane in world space.
    pub fn to_world(&self, p: Vec2) -> [f32; 3] {
        [
            self.origin[0] + self.u[0] * p[0] + self.v[0] * p[1],
            self.origin[1] + self.u[1] * p[0] + self.v[1] * p[1],
            self.origin[2] + self.u[2] * p[0] + self.v[2] * p[1],
        ]
    }

    /// The in-plane perpendicular (rotate a 2D direction 90° CCW, then lift).
    fn perp_world(&self, dir2: Vec2) -> [f32; 3] {
        let perp = [-dir2[1], dir2[0]];
        [
            self.u[0] * perp[0] + self.v[0] * perp[1],
            self.u[1] * perp[0] + self.v[1] * perp[1],
            self.u[2] * perp[0] + self.v[2] * perp[1],
        ]
    }
}

/// A profile placed on a plane.
#[derive(Debug, Clone, PartialEq)]
pub struct Sketch {
    pub plane: Plane,
    pub profile: Profile,
}

impl Sketch {
    pub fn new(plane: Plane, profile: Profile) -> Self {
        Self { plane, profile }
    }

    /// Flattened boundary as world-space points. For a closed profile the first
    /// vertex is not repeated at the end.
    pub fn world_polyline(&self, tol: f32) -> Result<Vec<[f32; 3]>, ProfileError> {
        Ok(self
            .profile
            .flatten(tol)?
            .iter()
            .map(|p| self.plane.to_world(*p))
            .collect())
    }

    /// Build the renderable mesh: a filled surface for a closed profile, or a
    /// ribbon stroke for an open one.
    pub fn to_mesh(&self, tol: f32) -> Result<MeshData, ProfileError> {
        if self.profile.closed {
            self.surface_mesh(tol)
        } else {
            self.stroke_mesh(tol, DEFAULT_STROKE_WIDTH)
        }
    }

    /// Filled, double-sided surface for a closed profile.
    pub fn surface_mesh(&self, tol: f32) -> Result<MeshData, ProfileError> {
        let pts2 = self.profile.flatten(tol)?;
        let tris = tessellate::triangulate(&pts2).ok_or(ProfileError::Degenerate)?;
        let n = pts2.len();
        let world: Vec<[f32; 3]> = pts2.iter().map(|p| self.plane.to_world(*p)).collect();
        let nrm = self.plane.normal;
        let back = [-nrm[0], -nrm[1], -nrm[2]];

        let mut vertices = Vec::with_capacity(n * 2);
        // Front vertices (0..n) carry +normal, back vertices (n..2n) carry -normal.
        for p in &world {
            vertices.push(Vertex {
                position: *p,
                color: SURFACE_COLOR,
                normal: nrm,
            });
        }
        for p in &world {
            vertices.push(Vertex {
                position: *p,
                color: SURFACE_COLOR,
                normal: back,
            });
        }

        let mut indices = Vec::with_capacity(tris.len() * 2);
        // Front faces keep the CCW (→ +normal) winding from the tessellator.
        for t in tris.chunks_exact(3) {
            indices.extend_from_slice(&[t[0] as u32, t[1] as u32, t[2] as u32]);
        }
        // Back faces: reversed winding, offset into the -normal vertex block.
        for t in tris.chunks_exact(3) {
            indices.extend_from_slice(&[(t[0] + n) as u32, (t[2] + n) as u32, (t[1] + n) as u32]);
        }
        Ok(MeshData { vertices, indices })
    }

    /// Thin double-sided ribbon along an open polyline, `width` units wide.
    pub fn stroke_mesh(&self, tol: f32, width: f32) -> Result<MeshData, ProfileError> {
        let pts2 = self.profile.flatten(tol)?;
        if pts2.len() < 2 {
            return Err(ProfileError::Degenerate);
        }
        let half = width.max(1e-4) * 0.5;
        let nrm = self.plane.normal;
        let back = [-nrm[0], -nrm[1], -nrm[2]];
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for w in pts2.windows(2) {
            let a = w[0];
            let b = w[1];
            let mut dir = [b[0] - a[0], b[1] - a[1]];
            let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
            if len <= 1e-6 {
                continue;
            }
            dir = [dir[0] / len, dir[1] / len];
            let off = self.plane.perp_world(dir);
            let off = [off[0] * half, off[1] * half, off[2] * half];
            let aw = self.plane.to_world(a);
            let bw = self.plane.to_world(b);
            // Quad corners: a-, a+, b+, b-.
            let corners = [sub(aw, off), add(aw, off), add(bw, off), sub(bw, off)];
            let base = vertices.len() as u32;
            for c in &corners {
                vertices.push(Vertex {
                    position: *c,
                    color: SURFACE_COLOR,
                    normal: nrm,
                });
            }
            for c in &corners {
                vertices.push(Vertex {
                    position: *c,
                    color: SURFACE_COLOR,
                    normal: back,
                });
            }
            // Front (CCW from +normal): 0,1,2 / 0,2,3.
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            // Back (reversed): 4,6,5 / 4,7,6.
            let b2 = base + 4;
            indices.extend_from_slice(&[b2, b2 + 2, b2 + 1, b2, b2 + 3, b2 + 2]);
        }
        if vertices.is_empty() {
            return Err(ProfileError::Degenerate);
        }
        Ok(MeshData { vertices, indices })
    }
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Newell's method: the (unnormalized) normal of a planar 3D polygon loop.
/// Reused by the B-rep layer to find a face's plane. Returns a unit vector,
/// or `[0,0,1]` for a degenerate loop.
pub fn newell_normal(loop_pts: &[[f32; 3]]) -> [f32; 3] {
    let mut n = cgmath::Vector3::new(0.0_f32, 0.0, 0.0);
    let m = loop_pts.len();
    for i in 0..m {
        let a = loop_pts[i];
        let b = loop_pts[(i + 1) % m];
        n.x += (a[1] - b[1]) * (a[2] + b[2]);
        n.y += (a[2] - b[2]) * (a[0] + b[0]);
        n.z += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let len = n.magnitude();
    if len > 1e-12 {
        [n.x / len, n.y / len, n.z / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_have_orthonormal_right_handed_bases() {
        for plane in [Plane::xy(), Plane::xz(), Plane::yz()] {
            let u = cgmath::Vector3::from(plane.u);
            let v = cgmath::Vector3::from(plane.v);
            let n = cgmath::Vector3::from(plane.normal);
            assert!((u.magnitude() - 1.0).abs() < 1e-6);
            assert!((v.magnitude() - 1.0).abs() < 1e-6);
            assert!(u.dot(v).abs() < 1e-6, "u ⟂ v");
            // normal must equal u × v (right-handed).
            let cross = u.cross(v);
            assert!((cross - n).magnitude() < 1e-6, "normal = u×v");
        }
    }

    #[test]
    fn surface_mesh_is_double_sided_and_planar() {
        let sketch = Sketch::new(Plane::xy(), Profile::regular_polygon(5, 1.0).unwrap());
        let mesh = sketch.surface_mesh(0.01).unwrap();
        // Double-sided: 2*(n-2) triangles for an n-gon.
        assert_eq!(mesh.triangle_count(), 2 * (5 - 2));
        // All vertices lie on z = 0 (the XY plane).
        assert!(mesh.vertices.iter().all(|v| v.position[2].abs() < 1e-6));
        // Half the vertices face +Z, half face -Z.
        let up = mesh.vertices.iter().filter(|v| v.normal[2] > 0.5).count();
        let down = mesh.vertices.iter().filter(|v| v.normal[2] < -0.5).count();
        assert_eq!(up, down);
        // Indices in range.
        let n = mesh.vertices.len() as u32;
        assert!(mesh.indices.iter().all(|&i| i < n));
    }

    #[test]
    fn surface_on_xz_plane_lies_flat_on_y() {
        let sketch = Sketch::new(Plane::xz(), Profile::regular_polygon(4, 2.0).unwrap());
        let mesh = sketch.surface_mesh(0.01).unwrap();
        assert!(mesh.vertices.iter().all(|v| v.position[1].abs() < 1e-6));
        assert!(mesh.vertices.iter().all(|v| v.normal[1].abs() > 0.5));
    }

    #[test]
    fn front_faces_point_along_plane_normal() {
        // A CCW square on XY: front triangles must have +Z geometric normal.
        let sketch = Sketch::new(Plane::xy(), Profile::regular_polygon(4, 1.0).unwrap());
        let mesh = sketch.surface_mesh(0.01).unwrap();
        // First triangle is a front face; its geometric normal should be +Z.
        let t = &mesh.indices[0..3];
        let p0 = cgmath::Vector3::from(mesh.vertices[t[0] as usize].position);
        let p1 = cgmath::Vector3::from(mesh.vertices[t[1] as usize].position);
        let p2 = cgmath::Vector3::from(mesh.vertices[t[2] as usize].position);
        let geo = (p1 - p0).cross(p2 - p0);
        assert!(geo.z > 0.0, "front face should wind CCW about +Z");
    }

    #[test]
    fn circle_surface_area_is_close_to_pi_r_squared() {
        let r = 2.0;
        let sketch = Sketch::new(Plane::xy(), Profile::circle(r).unwrap());
        let mesh = sketch.surface_mesh(0.005).unwrap();
        // Sum the front triangle areas (first half of the index list).
        let front = mesh.indices.len() / 2;
        let mut area = 0.0;
        for t in mesh.indices[..front].chunks_exact(3) {
            let p0 = cgmath::Vector3::from(mesh.vertices[t[0] as usize].position);
            let p1 = cgmath::Vector3::from(mesh.vertices[t[1] as usize].position);
            let p2 = cgmath::Vector3::from(mesh.vertices[t[2] as usize].position);
            area += 0.5 * (p1 - p0).cross(p2 - p0).magnitude();
        }
        let expected = std::f32::consts::PI * r * r;
        assert!((area - expected).abs() < 0.1, "area {area} vs {expected}");
    }

    #[test]
    fn open_profile_builds_a_ribbon_stroke() {
        let profile = Profile::from_points(&[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], false).unwrap();
        let sketch = Sketch::new(Plane::xy(), profile);
        let mesh = sketch.to_mesh(0.01).unwrap();
        // Two segments × (2 front + 2 back) triangles.
        assert_eq!(mesh.triangle_count(), 8);
        assert!(mesh.vertices.iter().all(|v| v.position[2].abs() < 1e-6));
    }

    #[test]
    fn world_polyline_maps_through_the_plane() {
        let sketch = Sketch::new(Plane::yz(), Profile::regular_polygon(3, 1.0).unwrap());
        let pts = sketch.world_polyline(0.01).unwrap();
        assert_eq!(pts.len(), 3);
        // YZ plane ⇒ every world point has x = 0.
        assert!(pts.iter().all(|p| p[0].abs() < 1e-6));
    }

    #[test]
    fn newell_normal_of_xy_square_is_unit_z() {
        let sq = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let n = newell_normal(&sq);
        assert!((n[2] - 1.0).abs() < 1e-6 && n[0].abs() < 1e-6 && n[1].abs() < 1e-6);
    }
}
