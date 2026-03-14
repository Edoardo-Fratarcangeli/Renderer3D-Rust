use crate::camera::Camera;
use cgmath::{InnerSpace, SquareMatrix};
use parry3d_f64::na::{Point3, Vector3 as NaVector3};
use parry3d_f64::query::Ray;

pub fn pick_point(camera: &crate::camera::Camera, objects: &[crate::scene::SceneObject], mouse_ndc: [f32; 2]) -> Option<[f32; 3]> {
    let inv_vp = camera.build_view_projection_matrix().invert()?;
    let near = inv_vp * cgmath::Vector4::new(mouse_ndc[0], mouse_ndc[1], 0.0, 1.0);
    let far = inv_vp * cgmath::Vector4::new(mouse_ndc[0], mouse_ndc[1], 1.0, 1.0);
    let n = near.truncate() / near.w;
    let f = far.truncate() / far.w;
    let dir = (f - n).normalize();

    let mut min_t = f32::MAX;
    let mut hit_pt = None;

    for obj in objects {
        if !obj.visible { continue; }
        
        let model = obj.instance.to_model_matrix();
        let inv_model = model.invert().unwrap_or(cgmath::Matrix4::identity());
        
        // Transform ray to local space
        let local_origin = (inv_model * n.extend(1.0)).truncate();
        let local_dir = (inv_model * dir.extend(0.0)).truncate().normalize();

        let l_origin = Point3::new(local_origin.x as f64, local_origin.y as f64, local_origin.z as f64);
        let l_dir = NaVector3::new(local_dir.x as f64, local_dir.y as f64, local_dir.z as f64);
        let ray = Ray::new(l_origin, l_dir);

        match &obj.geometry_type {
            crate::scene::GeometryType::Mesh { data } => {
                for chunk in data.indices.chunks_exact(3) {
                    let v0 = data.vertices[chunk[0] as usize].pos;
                    let v1 = data.vertices[chunk[1] as usize].pos;
                    let v2 = data.vertices[chunk[2] as usize].pos;
                    let tri = parry3d_f64::shape::Triangle::new(
                        Point3::new(v0[0] as f64, v0[1] as f64, v0[2] as f64),
                        Point3::new(v1[0] as f64, v1[1] as f64, v1[2] as f64),
                        Point3::new(v2[0] as f64, v2[1] as f64, v2[2] as f64),
                    );
                    use parry3d_f64::query::RayCast;
                    if let Some(t) = tri.cast_local_ray(&ray, f64::MAX, true) {
                        let world_hit = model * (local_origin + local_dir * (t as f32)).extend(1.0);
                        let world_t = (world_hit.truncate() - n).magnitude();
                        if world_t < min_t {
                            min_t = world_t;
                            hit_pt = Some([world_hit.x, world_hit.y, world_hit.z]);
                        }
                    }
                }
            }
            _ => {
                // For primitives we can reuse simpler logic if needed, but for now focus on Mesh
            }
        }
    }
    hit_pt
}

pub fn draw_measurement(
    ui: &egui::Ui,
    camera: &Camera,
    p1: [f32; 3],
    p2: [f32; 3],
    size: [f32; 2],
) {
    let s1 = project(camera, p1, size);
    let s2 = project(camera, p2, size);
    let dist = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt();

    let painter = ui.painter();
    painter.line_segment(
        [egui::pos2(s1[0], s1[1]), egui::pos2(s2[0], s2[1])],
        (2.0, egui::Color32::YELLOW),
    );
    painter.text(
        egui::pos2((s1[0] + s2[0]) / 2.0, (s1[1] + s2[1]) / 2.0),
        egui::Align2::CENTER_CENTER,
        format!("{:.2}m", dist),
        egui::FontId::proportional(16.0),
        egui::Color32::WHITE,
    );
}

fn project(camera: &Camera, p: [f32; 3], s: [f32; 2]) -> [f32; 2] {
    let vp = camera.build_view_projection_matrix();
    let pos = vp * cgmath::Vector4::new(p[0], p[1], p[2], 1.0);
    let ndc = pos.truncate() / pos.w;
    [(ndc.x + 1.0) * 0.5 * s[0], (1.0 - ndc.y) * 0.5 * s[1]]
}
