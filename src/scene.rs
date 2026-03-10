use crate::model::Instance;
use cgmath::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometryType {
    Cube,
    Sphere,
    Plane,
    Line, // For adding segment geometry later
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneObject {
    pub id: usize,
    pub label: String,
    pub instance: Instance,
    pub rotation_euler: [f32; 3],
    pub visible: bool,
    pub selected: bool,
    pub geometry_type: GeometryType,
    pub color: [f32; 3],
    pub show_label: bool,
    // Primitive specific properties
    pub plane_surface: f32,
    pub show_normal: bool,
    pub cube_side: f32,
    pub sphere_radius: f32,
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
            show_label: true,
            plane_surface: 1.0,
            show_normal: false,
            cube_side: 1.0,
            sphere_radius: 0.5,
        }
    }

    pub fn update_rotation(&mut self) {
        let rot = cgmath::Quaternion::from_angle_x(cgmath::Deg(self.rotation_euler[0]))
            * cgmath::Quaternion::from_angle_y(cgmath::Deg(self.rotation_euler[1]))
            * cgmath::Quaternion::from_angle_z(cgmath::Deg(self.rotation_euler[2]));
        self.instance.rotation = rot;
    }
}
