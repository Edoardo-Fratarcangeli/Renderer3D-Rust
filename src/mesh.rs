//! Import of 3D solid meshes (STL / OBJ / glTF) into renderable geometry.
//!
//! Imported meshes reuse the standard [`crate::model::Vertex`] layout, so they
//! travel through the same render pipeline as the built-in primitives. Unlike
//! the primitives (which fit in 16-bit indices), real models routinely exceed
//! 65 535 vertices, so meshes carry 32-bit indices.
//!
//! Loading runs purely on the CPU and is therefore safe to drive from a worker
//! thread; the host uploads the result to the GPU. STEP (`.step`/`.stp`) is
//! recognised but not yet tessellated.

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::model::Vertex;

/// White vertex color so the per-object instance color drives the final shade
/// (the shader multiplies vertex color × instance color).
const MESH_VERTEX_COLOR: [f32; 3] = [1.0, 1.0, 1.0];

/// File extensions recognised as importable 3D solid models.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["stl", "obj", "gltf", "glb", "step", "stp"];

/// CPU-side triangle mesh: vertices + 32-bit triangle indices.
#[derive(Debug, Clone, Default)]
pub struct MeshData {
    /// Interleaved position / color / normal, ready for the render pipeline.
    pub vertices: Vec<Vertex>,
    /// Triangle indices into [`Self::vertices`].
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Load a mesh, dispatching on the file extension.
    pub fn load(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mesh = match ext.as_str() {
            "stl" => load_stl(path)?,
            "obj" => load_obj(path)?,
            "gltf" | "glb" => load_gltf(path)?,
            "step" | "stp" => {
                return Err(anyhow!(
                    "STEP import is not supported yet (no tessellation backend)"
                ))
            }
            other => {
                return Err(anyhow!(
                    "unsupported 3D model format '{}' (expected stl, obj, gltf or glb)",
                    other
                ))
            }
        };
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return Err(anyhow!("file contains no triangles"));
        }
        Ok(mesh)
    }

    /// Axis-aligned bounding box as `(min, max)`.
    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        if self.vertices.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in &self.vertices {
            for k in 0..3 {
                min[k] = min[k].min(v.position[k]);
                max[k] = max[k].max(v.position[k]);
            }
        }
        (min, max)
    }

    /// Geometric center of the bounding box.
    pub fn center(&self) -> [f32; 3] {
        let (min, max) = self.aabb();
        [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ]
    }

    /// Largest bounding-box extent (used to scale models to a sane size).
    pub fn max_extent(&self) -> f32 {
        let (min, max) = self.aabb();
        (0..3).map(|k| max[k] - min[k]).fold(0.0_f32, f32::max)
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Closest ray/triangle intersection distance (Möller–Trumbore) for a ray
    /// expressed in this mesh's local space; `None` if the ray misses. Used by
    /// click selection.
    pub fn ray_hit(
        &self,
        origin: cgmath::Vector3<f32>,
        dir: cgmath::Vector3<f32>,
    ) -> Option<f32> {
        use cgmath::InnerSpace;
        const EPS: f32 = 1e-7;
        let mut best: Option<f32> = None;
        for tri in self.indices.chunks_exact(3) {
            let v0 = vec3(self.vertices[tri[0] as usize].position);
            let v1 = vec3(self.vertices[tri[1] as usize].position);
            let v2 = vec3(self.vertices[tri[2] as usize].position);
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let h = dir.cross(edge2);
            let a = edge1.dot(h);
            if a.abs() < EPS {
                continue; // ray parallel to triangle
            }
            let f = 1.0 / a;
            let s = origin - v0;
            let u = f * s.dot(h);
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(edge1);
            let v = f * dir.dot(q);
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = f * edge2.dot(q);
            if t > EPS && best.map_or(true, |b| t < b) {
                best = Some(t);
            }
        }
        best
    }
}

fn vec3(p: [f32; 3]) -> cgmath::Vector3<f32> {
    cgmath::Vector3::new(p[0], p[1], p[2])
}

/// Build vertices (with the standard pipeline layout) from raw positions,
/// computing smooth per-vertex normals when the source provides none.
fn build(positions: Vec<[f32; 3]>, normals: Option<Vec<[f32; 3]>>, indices: Vec<u32>) -> MeshData {
    let normals = normals
        .filter(|n| n.len() == positions.len())
        .unwrap_or_else(|| smooth_normals(&positions, &indices));
    let vertices = positions
        .into_iter()
        .zip(normals)
        .map(|(position, normal)| Vertex {
            position,
            color: MESH_VERTEX_COLOR,
            normal,
        })
        .collect();
    MeshData { vertices, indices }
}

/// Area-weighted smooth normals, used when a model ships without normals.
fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    use cgmath::InnerSpace;
    let mut acc = vec![cgmath::Vector3::new(0.0_f32, 0.0, 0.0); positions.len()];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let v0 = vec3(positions[i0]);
        let v1 = vec3(positions[i1]);
        let v2 = vec3(positions[i2]);
        // Cross product magnitude is proportional to triangle area, so summing
        // the raw cross products area-weights the contribution automatically.
        let face = (v1 - v0).cross(v2 - v0);
        acc[i0] += face;
        acc[i1] += face;
        acc[i2] += face;
    }
    acc.into_iter()
        .map(|n| {
            let m = n.magnitude();
            if m > 1e-12 {
                let n = n / m;
                [n.x, n.y, n.z]
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

fn load_stl(path: &Path) -> Result<MeshData> {
    let mut file = std::fs::File::open(path)?;
    let stl = stl_io::read_stl(&mut file)?;
    let mut positions = Vec::with_capacity(stl.faces.len() * 3);
    let mut normals = Vec::with_capacity(stl.faces.len() * 3);
    for face in &stl.faces {
        let n = face.normal;
        for &vi in &face.vertices {
            let v = &stl.vertices[vi];
            positions.push([v[0], v[1], v[2]]);
            normals.push([n[0], n[1], n[2]]);
        }
    }
    let indices = (0..positions.len() as u32).collect();
    // STL ships a per-face normal; honor it directly (flat shading).
    Ok(build(positions, Some(normals), indices))
}

fn load_obj(path: &Path) -> Result<MeshData> {
    let (models, _) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut have_normals = true;
    for model in models {
        let mesh = model.mesh;
        let base = (positions.len()) as u32;
        let n_verts = mesh.positions.len() / 3;
        for i in 0..n_verts {
            positions.push([
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
            ]);
        }
        if mesh.normals.len() == mesh.positions.len() {
            for i in 0..n_verts {
                normals.push([
                    mesh.normals[i * 3],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                ]);
            }
        } else {
            have_normals = false;
        }
        indices.extend(mesh.indices.iter().map(|i| base + i));
    }
    let normals = if have_normals && normals.len() == positions.len() {
        Some(normals)
    } else {
        None // mixed / missing normals → recompute smoothly
    };
    Ok(build(positions, normals, indices))
}

fn load_gltf(path: &Path) -> Result<MeshData> {
    let (doc, buffers, _) = gltf::import(path)?;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut have_normals = true;
    for mesh in doc.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let pos = reader
                .read_positions()
                .ok_or_else(|| anyhow!("glTF primitive has no positions"))?;
            let base = positions.len() as u32;
            let before = positions.len();
            positions.extend(pos);
            let added = positions.len() - before;
            if let Some(norm) = reader.read_normals() {
                normals.extend(norm);
            } else {
                have_normals = false;
            }
            match reader.read_indices() {
                Some(it) => indices.extend(it.into_u32().map(|i| base + i)),
                // No index buffer → triangles are the vertex sequence.
                None => indices.extend((0..added as u32).map(|i| base + i)),
            }
        }
    }
    let normals = if have_normals && normals.len() == positions.len() {
        Some(normals)
    } else {
        None
    };
    Ok(build(positions, normals, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit quad in the z = 0 plane (two triangles), no normals.
    fn quad() -> MeshData {
        build(
            vec![
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            None,
            vec![0, 1, 2, 0, 2, 3],
        )
    }

    #[test]
    fn aabb_center_and_extent() {
        let m = quad();
        assert_eq!(m.aabb(), ([-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]));
        assert_eq!(m.center(), [0.0, 0.0, 0.0]);
        assert_eq!(m.max_extent(), 2.0);
        assert_eq!(m.triangle_count(), 2);
    }

    #[test]
    fn missing_normals_are_synthesized() {
        let m = quad();
        // Quad faces +z, so every computed normal should point along +z.
        for v in &m.vertices {
            assert!((v.normal[2] - 1.0).abs() < 1e-5, "normal = {:?}", v.normal);
        }
    }

    #[test]
    fn ray_hits_and_misses_the_quad() {
        let m = quad();
        // Ray from +z straight down hits the quad at distance 5.
        let t = m
            .ray_hit(
                cgmath::Vector3::new(0.0, 0.0, 5.0),
                cgmath::Vector3::new(0.0, 0.0, -1.0),
            )
            .expect("ray should hit");
        assert!((t - 5.0).abs() < 1e-4);
        // Ray well outside the quad misses.
        assert!(m
            .ray_hit(
                cgmath::Vector3::new(10.0, 10.0, 5.0),
                cgmath::Vector3::new(0.0, 0.0, -1.0),
            )
            .is_none());
    }

    #[test]
    fn unsupported_and_step_formats_error_clearly() {
        let step = MeshData::load(Path::new("model.step")).unwrap_err();
        assert!(step.to_string().contains("STEP"));
        let other = MeshData::load(Path::new("model.xyz")).unwrap_err();
        assert!(other.to_string().contains("unsupported"));
    }
}
