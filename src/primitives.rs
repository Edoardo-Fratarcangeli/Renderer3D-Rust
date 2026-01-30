use crate::model::Vertex;
use std::f32::consts::PI;
use cgmath::InnerSpace;

pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

pub fn create_cube() -> MeshData {
    let c = 0.5;
    // Helper to create quad
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let faces = [
        // Front (Z+)
        ([0.0, 0.0, 1.0], [-c, -c, c], [c, -c, c], [c, c, c], [-c, c, c]),
        // Back (Z-)
        ([0.0, 0.0, -1.0], [c, -c, -c], [-c, -c, -c], [-c, c, -c], [c, c, -c]),
        // Top (Y+)
        ([0.0, 1.0, 0.0], [-c, c, c], [c, c, c], [c, c, -c], [-c, c, -c]),
        // Bottom (Y-)
        ([0.0, -1.0, 0.0], [-c, -c, -c], [c, -c, -c], [c, -c, c], [-c, -c, c]),
        // Right (X+)
        ([1.0, 0.0, 0.0], [c, -c, c], [c, -c, -c], [c, c, -c], [c, c, c]),
        // Left (X-)
        ([-1.0, 0.0, 0.0], [-c, -c, -c], [-c, -c, c], [-c, c, c], [-c, c, -c]),
    ];

    for (normal, p0, p1, p2, p3) in faces.iter() {
        let idx = vertices.len() as u16;
        vertices.push(Vertex { position: *p0, color: [1.0, 1.0, 1.0], normal: *normal }); // BL
        vertices.push(Vertex { position: *p1, color: [1.0, 1.0, 1.0], normal: *normal }); // BR
        vertices.push(Vertex { position: *p2, color: [1.0, 1.0, 1.0], normal: *normal }); // TR
        vertices.push(Vertex { position: *p3, color: [1.0, 1.0, 1.0], normal: *normal }); // TL
        
        indices.push(idx);
        indices.push(idx + 1);
        indices.push(idx + 2);
        
        indices.push(idx + 2);
        indices.push(idx + 3);
        indices.push(idx);
    }

    MeshData { vertices, indices }
}

pub fn create_sphere(radius: f32, lat_segments: u32, long_segments: u32) -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=lat_segments {
        let v = i as f32 / lat_segments as f32;
        let phi = v * PI;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        for j in 0..=long_segments {
            let u = j as f32 / long_segments as f32;
            let theta = u * 2.0 * PI;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            let x = cos_theta * sin_phi;
            let y = cos_phi;
            let z = sin_theta * sin_phi;

            let normal = [x, y, z];
            
            vertices.push(Vertex {
                position: [x * radius, y * radius, z * radius],
                color: [1.0, 1.0, 1.0], 
                normal
            });
        }
    }

    for i in 0..lat_segments {
        for j in 0..long_segments {
            let first = (i * (long_segments + 1)) + j;
            let second = first + long_segments + 1;

            indices.push(first as u16);
            indices.push(second as u16);
            indices.push((first + 1) as u16);

            indices.push(second as u16);
            indices.push((second + 1) as u16);
            indices.push((first + 1) as u16);
        }
    }

    MeshData { vertices, indices }
}

pub fn create_plane(size: f32) -> MeshData {
    let s = size / 2.0;
    // Plane on XZ, facing Y+
    let normal = [0.0, 1.0, 0.0];
    let vertices = vec![
        Vertex { position: [-s, 0.0,  s], color: [1.0, 1.0, 1.0], normal },
        Vertex { position: [ s, 0.0,  s], color: [1.0, 1.0, 1.0], normal },
        Vertex { position: [ s, 0.0, -s], color: [1.0, 1.0, 1.0], normal },
        Vertex { position: [-s, 0.0, -s], color: [1.0, 1.0, 1.0], normal },
    ];
    let indices = vec![0, 1, 2, 2, 3, 0];
    MeshData { vertices, indices }
}

// Just line grid, Normals Up
pub fn create_grid(size: u32, spacing: f32, plane_normal: u8) -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let half = (size as f32 * spacing) / 2.0;
    let color = [0.4, 0.4, 0.4];
    let normal = [0.0, 1.0, 0.0]; // Default up, doesn't matter for line shader mostly

    // plane_normal: 0=Y (XZ grid), 1=Z (XY grid), 2=X (YZ grid)
    
    for i in 0..=size {
        let v = -half + (i as f32 * spacing);
        let idx = vertices.len() as u16;

        if plane_normal == 0 { // XZ Plane (Y is up)
             vertices.push(Vertex { position: [-half, 0.0, v], color, normal });
             vertices.push(Vertex { position: [half, 0.0, v], color, normal });
        } else if plane_normal == 1 { // XY Plane (Z is up)
             vertices.push(Vertex { position: [-half, v, 0.0], color, normal });
             vertices.push(Vertex { position: [half, v, 0.0], color, normal });
        } else { // YZ Plane (X is up)
             vertices.push(Vertex { position: [0.0, -half, v], color, normal });
             vertices.push(Vertex { position: [0.0, half, v], color, normal });
        }
        
        indices.push(idx);
        indices.push(idx + 1);
    }

    for i in 0..=size {
        let v = -half + (i as f32 * spacing);
        let idx = vertices.len() as u16;

         if plane_normal == 0 { // XZ Plane (Y is up)
             vertices.push(Vertex { position: [v, 0.0, -half], color, normal });
             vertices.push(Vertex { position: [v, 0.0, half], color, normal });
        } else if plane_normal == 1 { // XY Plane (Z is up)
             vertices.push(Vertex { position: [v, -half, 0.0], color, normal });
             vertices.push(Vertex { position: [v, half, 0.0], color, normal });
        } else { // YZ Plane (X is up)
             vertices.push(Vertex { position: [0.0, v, -half], color, normal });
             vertices.push(Vertex { position: [0.0, v, half], color, normal });
        }

        indices.push(idx);
        indices.push(idx + 1);
    }
    
    MeshData { vertices, indices }
}

// Create a cylinder along an axis
fn create_cylinder(start: [f32;3], end: [f32;3], radius: f32, color: [f32;3]) -> (Vec<Vertex>, Vec<u16>) {
    let segments = 12;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    let p0 = cgmath::Vector3::from(start);
    let p1 = cgmath::Vector3::from(end);
    let dir = (p1 - p0).normalize();
    
    // Find approximate Up vector to create basis
    let up = if dir.y.abs() > 0.9 { cgmath::Vector3::unit_x() } else { cgmath::Vector3::unit_y() };
    let right = dir.cross(up).normalize();
    let real_up = right.cross(dir).normalize();
    
    // Generate side vertices
    for i in 0..=segments {
        let theta = (i as f32 / segments as f32) * 2.0 * PI;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        
        let offset = right * cos_t * radius + real_up * sin_t * radius;
        let v0 = p0 + offset;
        let v1 = p1 + offset;
        let normal = offset.normalize(); // Flat normal specific to cylinder side
        let n = [normal.x, normal.y, normal.z];

        vertices.push(Vertex { position: [v0.x, v0.y, v0.z], color, normal: n });
        vertices.push(Vertex { position: [v1.x, v1.y, v1.z], color, normal: n });
    }
    
    let base_idx = 0;
    for i in 0..segments {
        let idx = base_idx + i * 2;
        indices.push(idx);
        indices.push(idx + 1);
        indices.push(idx + 2);
        
        indices.push(idx + 2);
        indices.push(idx + 1);
        indices.push(idx + 3);
    }
    
    (vertices, indices)
}

pub fn create_thick_axes(length: f32, thickness: f32) -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    let axes = [
        ([0.0, 0.0, 0.0], [length, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, length, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, length], [0.0, 0.0, 1.0]),
    ];
    
    for (start, end, color) in axes {
        let (mut v, mut i) = create_cylinder(start, end, thickness, color);
        let offset = vertices.len() as u16;
        for idx in i.iter_mut() { *idx += offset; }
        vertices.append(&mut v);
        indices.append(&mut i);
    }
    
    MeshData { vertices, indices }
}
