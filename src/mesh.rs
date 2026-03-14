use anyhow::{anyhow, Result};
use parry3d_f64::bounding_volume::Aabb;
use parry3d_f64::na::Point3;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub norm: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow!("No extension"))?
            .to_lowercase();

        match ext.as_str() {
            "stl" => {
                let mut file = std::fs::File::open(path)?;
                let mesh = stl_io::read_stl(&mut file)?;
                let mut vertices = Vec::new();
                for tri in mesh.faces {
                    let n = tri.normal;
                    for &v_idx in &tri.vertices {
                        let v = &mesh.vertices[v_idx];
                        vertices.push(Vertex {
                            pos: [v[0], v[1], v[2]],
                            norm: [n[0], n[1], n[2]],
                        });
                    }
                }
                let indices = (0..vertices.len() as u32).collect();
                Ok(MeshData { vertices, indices })
            }
            "obj" => {
                let (models, _) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;
                let mut vertices = Vec::new();
                let mut indices = Vec::new();
                for m in models {
                    let mesh = m.mesh;
                    for i in 0..mesh.positions.len() / 3 {
                        vertices.push(Vertex {
                            pos: [
                                mesh.positions[i * 3],
                                mesh.positions[i * 3 + 1],
                                mesh.positions[i * 3 + 2],
                            ],
                            norm: if !mesh.normals.is_empty() {
                                [
                                    mesh.normals[i * 3],
                                    mesh.normals[i * 3 + 1],
                                    mesh.normals[i * 3 + 2],
                                ]
                            } else {
                                [0.0, 0.0, 0.0]
                            },
                        });
                    }
                    indices.extend_from_slice(&mesh.indices);
                }
                Ok(MeshData { vertices, indices })
            }
            "gltf" | "glb" => {
                let (doc, buffers, _) = gltf::import(path)?;
                let mut vertices = Vec::new();
                let mut indices = Vec::new();
                for mesh in doc.meshes() {
                    for primitive in mesh.primitives() {
                        let reader = primitive
                            .reader(|buffer| buffers.get(buffer.index()).map(|v| &v.0[..]));
                        let pos = reader
                            .read_positions()
                            .ok_or_else(|| anyhow!("No positions"))?;
                        let norm = reader.read_normals().ok_or_else(|| anyhow!("No normals"))?;

                        let start_idx = vertices.len() as u32;
                        for (p, n) in pos.zip(norm) {
                            vertices.push(Vertex { pos: p, norm: n });
                        }

                        if let Some(iter) = reader.read_indices() {
                            for i in iter.into_u32() {
                                indices.push(start_idx + i);
                            }
                        }
                    }
                }
                Ok(MeshData { vertices, indices })
            }
            "step" | "stp" => {
                // Simplified placeholder for STEP as it requires complex parsing
                Err(anyhow!(
                    "STEP format support requires further implementation with ruststep"
                ))
            }
            _ => Err(anyhow!("Unsupported format: {}", ext)),
        }
    }

    pub fn compute_aabb(&self) -> Aabb {
        let points: Vec<Point3<f64>> = self
            .vertices
            .iter()
            .map(|v| Point3::new(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64))
            .collect();
        Aabb::from_points(&points)
    }
}
