use crate::model::Instance;
use cgmath::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometryType {
    Cube,
    Sphere,
    Plane,
    Line, // For adding segment geometry later
}

pub struct SceneObject {
    pub id: usize,
    pub label: String,
    pub instance: Instance,
    pub rotation_euler: [f32; 3],
    pub visible: bool,
    pub selected: bool,
    pub geometry_type: GeometryType,
    pub color: [f32; 3], 
}

impl SceneObject {
    pub fn new(id: usize, label: String, position: [f32; 3], geometry_type: GeometryType) -> Self {
        Self {
            id,
            label,
            instance: Instance {
                position: cgmath::Vector3::new(position[0], position[1], position[2]),
                rotation: cgmath::Quaternion::zero(), 
                scale: cgmath::Vector3::new(1.0, 1.0, 1.0),
            },
            rotation_euler: [0.0, 0.0, 0.0],
            visible: true,
            selected: false,
            geometry_type,
            color: [1.0, 1.0, 1.0], // Default white
        }
    }
    
    pub fn update_rotation(&mut self) {
        let rot = cgmath::Quaternion::from_angle_x(cgmath::Deg(self.rotation_euler[0]))
            * cgmath::Quaternion::from_angle_y(cgmath::Deg(self.rotation_euler[1]))
            * cgmath::Quaternion::from_angle_z(cgmath::Deg(self.rotation_euler[2]));
        self.instance.rotation = rot;
    }
}
