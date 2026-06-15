// use cgmath::SquareMatrix;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

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

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{InnerSpace, SquareMatrix};

    fn camera() -> Camera {
        Camera {
            eye: (0.0, 0.0, 5.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }

    #[test]
    fn opengl_matrix_remaps_clip_depth() {
        // The z row halves and biases depth (OpenGL [-1,1] -> wgpu [0,1]).
        let m = OPENGL_TO_WGPU_MATRIX;
        assert_eq!(m.z.z, 0.5);
        assert_eq!(m.w.z, 0.5);
    }

    #[test]
    fn view_projection_centers_the_target_and_is_invertible() {
        let cam = camera();
        let vp = cam.build_view_projection_matrix();
        // The target projects to the center of the screen (x = y = 0).
        let clip = vp * cgmath::Vector4::new(0.0, 0.0, 0.0, 1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1e-5 && ndc.y.abs() < 1e-5, "ndc = {:?}", ndc);
        // wgpu depth convention keeps the near plane at z in [0, 1].
        assert!((0.0..=1.0).contains(&ndc.z), "depth {} out of range", ndc.z);
        assert!(vp.invert().is_some(), "view-projection must be invertible");
    }

    #[test]
    fn points_left_of_target_land_left_on_screen() {
        let cam = camera();
        let vp = cam.build_view_projection_matrix();
        let p = vp * cgmath::Vector4::new(-1.0, 0.0, 0.0, 1.0);
        let ndc = p.truncate() / p.w;
        assert!(ndc.x < 0.0, "a point at -x must map to the left half");
    }

    #[test]
    fn uniforms_new_is_identity_and_default_matches() {
        let id: [[f32; 4]; 4] = cgmath::Matrix4::identity().into();
        assert_eq!(Uniforms::new().view_proj, id);
        assert_eq!(Uniforms::default().view_proj, Uniforms::new().view_proj);
    }

    #[test]
    fn update_view_proj_overwrites_identity() {
        let mut u = Uniforms::new();
        u.update_view_proj(&camera());
        let id: [[f32; 4]; 4] = cgmath::Matrix4::identity().into();
        assert_ne!(u.view_proj, id, "update must replace the identity matrix");
    }

    #[test]
    fn aspect_ratio_scales_horizontal_extent() {
        // A wider aspect ratio compresses x in NDC for the same world point.
        let mut narrow = camera();
        narrow.aspect = 1.0;
        let mut wide = camera();
        wide.aspect = 2.0;
        let p = cgmath::Vector4::new(1.0, 0.0, 0.0, 1.0);
        let nx = {
            let c = narrow.build_view_projection_matrix() * p;
            (c.truncate() / c.w).x
        };
        let wx = {
            let c = wide.build_view_projection_matrix() * p;
            (c.truncate() / c.w).x
        };
        assert!(wx.abs() < nx.abs(), "wider aspect must shrink x ({wx} vs {nx})");
    }

    #[test]
    fn up_vector_stays_orthogonal_in_view() {
        // Sanity: the camera basis is well-formed (no degenerate look-at).
        let cam = camera();
        let fwd = (cam.target - cam.eye).normalize();
        assert!((fwd.dot(cam.up)).abs() < 1e-6);
    }
}
