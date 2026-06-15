#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Instance {
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
    pub scale: cgmath::Vector3<f32>,
}

impl Instance {
    pub fn to_model_matrix(&self) -> cgmath::Matrix4<f32> {
        cgmath::Matrix4::from_translation(self.position)
            * cgmath::Matrix4::from(self.rotation)
            * cgmath::Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z)
    }

    pub fn to_raw(&self) -> InstanceRaw {
        let model = self.to_model_matrix();
        InstanceRaw {
            model: model.into(),
            color: [1.0, 1.0, 1.0, 1.0], // Default white placeholder
        }
    }

    pub fn to_raw_with_color(&self, color: [f32; 3], selected: bool) -> InstanceRaw {
        let model = self.to_model_matrix();
        let alpha = if selected { 2.0 } else { 1.0 };
        InstanceRaw {
            model: model.into(),
            color: [color[0], color[1], color[2], alpha],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::SquareMatrix;

    fn instance() -> Instance {
        Instance {
            position: cgmath::Vector3::new(1.0, 2.0, 3.0),
            rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0), // identity
            scale: cgmath::Vector3::new(2.0, 2.0, 2.0),
        }
    }

    #[test]
    fn model_matrix_translates_and_scales() {
        let m = instance().to_model_matrix();
        // Origin maps to the position.
        let o = m * cgmath::Vector4::new(0.0, 0.0, 0.0, 1.0);
        assert_eq!([o.x, o.y, o.z], [1.0, 2.0, 3.0]);
        // A unit x vector is scaled by 2 then translated.
        let p = m * cgmath::Vector4::new(1.0, 0.0, 0.0, 1.0);
        assert_eq!([p.x, p.y, p.z], [3.0, 2.0, 3.0]);
    }

    #[test]
    fn identity_instance_matches_identity_matrix() {
        let inst = Instance {
            position: cgmath::Vector3::new(0.0, 0.0, 0.0),
            rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
            scale: cgmath::Vector3::new(1.0, 1.0, 1.0),
        };
        let m: [[f32; 4]; 4] = inst.to_model_matrix().into();
        let id: [[f32; 4]; 4] = cgmath::Matrix4::identity().into();
        assert_eq!(m, id);
    }

    #[test]
    fn to_raw_defaults_to_opaque_white() {
        let raw = instance().to_raw();
        assert_eq!(raw.color, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn to_raw_with_color_encodes_selection_in_alpha() {
        let unselected = instance().to_raw_with_color([0.2, 0.4, 0.6], false);
        assert_eq!(unselected.color, [0.2, 0.4, 0.6, 1.0]);
        let selected = instance().to_raw_with_color([0.2, 0.4, 0.6], true);
        assert_eq!(selected.color, [0.2, 0.4, 0.6, 2.0]);
    }

    #[test]
    fn vertex_layout_is_three_float3_attributes() {
        let d = Vertex::desc();
        assert_eq!(d.array_stride, std::mem::size_of::<Vertex>() as u64);
        assert_eq!(d.step_mode, wgpu::VertexStepMode::Vertex);
        assert_eq!(d.attributes.len(), 3);
        assert_eq!(d.attributes[2].shader_location, 2);
    }

    #[test]
    fn instance_layout_is_per_instance_with_five_attributes() {
        let d = InstanceRaw::desc();
        assert_eq!(d.array_stride, std::mem::size_of::<InstanceRaw>() as u64);
        assert_eq!(d.step_mode, wgpu::VertexStepMode::Instance);
        // Four matrix rows (loc 5..=8) + color (loc 9).
        assert_eq!(d.attributes.len(), 5);
        assert_eq!(d.attributes[0].shader_location, 5);
        assert_eq!(d.attributes[4].shader_location, 9);
    }
}

impl InstanceRaw {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Color at loc 9
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}
