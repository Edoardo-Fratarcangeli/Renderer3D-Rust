// use cgmath::SquareMatrix;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[derive(Debug, Clone)]
pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

pub struct CameraController {
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub target: cgmath::Point3<f32>,
    pub min_zoom: f32,
    pub max_zoom: f32,
}

impl CameraController {
    pub fn new(yaw: f32, pitch: f32, dist: f32, target: [f32; 3], min_z: f32, max_z: f32) -> Self {
        Self {
            yaw,
            pitch,
            dist,
            target: cgmath::Point3::new(target[0], target[1], target[2]),
            min_zoom: min_z,
            max_zoom: max_z,
        }
    }

    pub fn target(&self) -> cgmath::Point3<f32> {
        self.target
    }

    pub fn eye(&self) -> cgmath::Point3<f32> {
        use cgmath::prelude::*;
        let yaw = cgmath::Deg(self.yaw);
        let pitch = cgmath::Deg(self.pitch);
        let x = self.dist * yaw.cos() * pitch.cos();
        let y = self.dist * yaw.sin() * pitch.cos();
        let z = self.dist * pitch.sin();
        self.target + cgmath::Vector3::new(x, y, z)
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.5;
        self.yaw -= dx * sensitivity;
        self.pitch += dy * sensitivity;
        self.pitch = self.pitch.clamp(-89.0, 89.0);
    }

    pub fn zoom(&mut self, delta: f32) {
        let zoom_factor = 1.1;
        if delta > 0.0 {
            self.dist /= zoom_factor;
        } else {
            self.dist *= zoom_factor;
        }
        self.dist = self.dist.max(self.min_zoom).min(self.max_zoom);
    }

    pub fn pan(&mut self, dx: f32, dy: f32, camera: &Camera) {
        let sensitivity = self.dist * 0.001;
        use cgmath::InnerSpace;
        let forward = (self.target - camera.eye).normalize();
        let right = forward.cross(camera.up).normalize();
        let up = right.cross(forward).normalize();

        self.target += right * (-dx * sensitivity) + up * (dy * sensitivity);
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        use cgmath::prelude::*;
        let yaw = cgmath::Deg(self.yaw);
        let pitch = cgmath::Deg(self.pitch);

        let x = self.dist * yaw.cos() * pitch.cos();
        let y = self.dist * yaw.sin() * pitch.cos();
        let z = self.dist * pitch.sin();

        camera.up = cgmath::Vector3::unit_z();
        camera.eye = self.target + cgmath::Vector3::new(x, y, z);
        camera.target = self.target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_rotation() {
        let mut controller = CameraController::new(0.0, 0.0, 10.0, [0.0, 0.0, 0.0], 1.0, 100.0);
        controller.rotate(10.0, 5.0);
        assert_eq!(controller.yaw, -5.0);
        assert_eq!(controller.pitch, 2.5);
    }

    #[test]
    fn test_camera_zoom() {
        let mut controller = CameraController::new(0.0, 0.0, 10.0, [0.0, 0.0, 0.0], 1.0, 100.0);
        let init_dist = controller.dist;
        controller.zoom(1.0); // Zoom in
        assert!(controller.dist < init_dist);

        controller.zoom(-1.0); // Zoom out
        assert!((controller.dist - init_dist).abs() < 1e-4);
    }

    #[test]
    fn test_camera_panning() {
        let camera = Camera {
            eye: cgmath::Point3::new(5.0, 5.0, 5.0),
            target: cgmath::Point3::new(0.0, 0.0, 0.0),
            up: cgmath::Vector3::unit_z(),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let mut controller = CameraController::new(45.0, 35.0, 10.0, [0.0, 0.0, 0.0], 1.0, 100.0);

        let old_target = controller.target;
        // Panning right (dx > 0)
        controller.pan(10.0, 0.0, &camera);
        assert!(controller.target.x != old_target.x || controller.target.y != old_target.y);
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

impl Default for Uniforms {
    fn default() -> Self {
        Self::new()
    }
}
