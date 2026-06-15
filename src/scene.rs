use crate::model::Instance;
use cgmath::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometryType {
    Cube,
    Sphere,
    Plane,
    Line, // For adding segment geometry later
    /// An imported 3D solid model (STL/OBJ/glTF). The mesh data and GPU
    /// buffers live in `State::custom_meshes`, keyed by the object id, so this
    /// variant stays `Copy`.
    Mesh,
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

/// A reversible scene edit, stored on the undo/redo stacks.
#[derive(Clone)]
pub enum UndoCommand {
    Add(SceneObject),
    Delete(SceneObject),
    Edit { old: SceneObject, new: SceneObject },
    MultiAction(Vec<UndoCommand>),
}

/// Apply (or reverse, when `is_undo`) an [`UndoCommand`] to a scene's object
/// list. Pure over the object vector so it is unit-testable without a GPU.
pub fn apply_undo_command(objects: &mut Vec<SceneObject>, cmd: &UndoCommand, is_undo: bool) {
    match cmd {
        UndoCommand::Add(obj) => {
            if is_undo {
                objects.retain(|o| o.id != obj.id);
            } else {
                objects.push(obj.clone());
            }
        }
        UndoCommand::Delete(obj) => {
            if is_undo {
                objects.push(obj.clone());
            } else {
                objects.retain(|o| o.id != obj.id);
            }
        }
        UndoCommand::Edit { old, new } => {
            let target = if is_undo { old } else { new };
            if let Some(obj) = objects.iter_mut().find(|o| o.id == target.id) {
                *obj = target.clone();
            }
        }
        UndoCommand::MultiAction(cmds) => {
            if is_undo {
                for c in cmds.iter().rev() {
                    apply_undo_command(objects, c, is_undo);
                }
            } else {
                for c in cmds {
                    apply_undo_command(objects, c, is_undo);
                }
            }
        }
    }
}

/// Ray/primitive intersection in the primitive's local space (unit cube
/// `[-0.5, 0.5]³`, sphere radius 0.5, unit plane on y = 0). Returns the
/// nearest non-negative hit distance along `dir`, or `None` on a miss.
/// `Line`/`Mesh` have no analytic primitive and return `None`.
pub fn intersect_primitive(
    geo_type: GeometryType,
    local_origin: cgmath::Vector3<f32>,
    local_dir: cgmath::Vector3<f32>,
) -> Option<f32> {
    match geo_type {
        GeometryType::Cube => {
            let mut tmin = -f32::INFINITY;
            let mut tmax = f32::INFINITY;
            for i in 0..3 {
                if local_dir[i].abs() < 1e-6 {
                    if local_origin[i] < -0.5 || local_origin[i] > 0.5 {
                        return None;
                    }
                } else {
                    let inv_d = 1.0 / local_dir[i];
                    let mut t1 = (-0.5 - local_origin[i]) * inv_d;
                    let mut t2 = (0.5 - local_origin[i]) * inv_d;
                    if t1 > t2 {
                        std::mem::swap(&mut t1, &mut t2);
                    }
                    tmin = tmin.max(t1);
                    tmax = tmax.min(t2);
                }
            }
            if tmax >= tmin && tmax >= 0.0 {
                Some(tmin.max(0.0))
            } else {
                None
            }
        }
        GeometryType::Sphere => {
            let oc = local_origin;
            let a = local_dir.dot(local_dir);
            let b = 2.0 * oc.dot(local_dir);
            let c = oc.dot(oc) - 0.25; // radius 0.5 matches visuals
            let discriminant = b * b - 4.0 * a * c;
            if discriminant < 0.0 {
                None
            } else {
                let mut t = (-b - discriminant.sqrt()) / (2.0 * a);
                if t < 0.0 {
                    t = (-b + discriminant.sqrt()) / (2.0 * a);
                }
                if t >= 0.0 {
                    Some(t)
                } else {
                    None
                }
            }
        }
        GeometryType::Plane => {
            if local_dir.y.abs() < 1e-6 {
                return None;
            }
            let t = -local_origin.y / local_dir.y;
            if t < 0.0 {
                return None;
            }
            let p = local_origin + local_dir * t;
            if p.x.abs() <= 0.5 && p.z.abs() <= 0.5 {
                Some(t)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::InnerSpace;

    fn vec(x: f32, y: f32, z: f32) -> cgmath::Vector3<f32> {
        cgmath::Vector3::new(x, y, z)
    }

    #[test]
    fn new_sets_expected_defaults() {
        let o = SceneObject::new(7, "obj".into(), [1.0, 2.0, 3.0], GeometryType::Sphere);
        assert_eq!(o.id, 7);
        assert_eq!(o.label, "obj");
        assert_eq!(
            [o.instance.position.x, o.instance.position.y, o.instance.position.z],
            [1.0, 2.0, 3.0]
        );
        assert_eq!(o.instance.scale, cgmath::Vector3::new(1.0, 1.0, 1.0));
        assert!(o.visible);
        assert!(!o.selected);
        assert!(o.show_label);
        assert!(!o.show_normal);
        assert_eq!(o.color, [1.0, 1.0, 1.0]);
        assert_eq!(o.geometry_type, GeometryType::Sphere);
        assert_eq!(o.cube_side, 1.0);
        assert_eq!(o.sphere_radius, 0.5);
        assert_eq!(o.plane_surface, 1.0);
        assert_eq!(o.rotation_euler, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn update_rotation_identity_for_zero_euler() {
        let mut o = SceneObject::new(1, "o".into(), [0.0; 3], GeometryType::Cube);
        o.update_rotation();
        let q = o.instance.rotation;
        assert!((q.s - 1.0).abs() < 1e-6);
        assert!(q.v.magnitude() < 1e-6);
    }

    #[test]
    fn update_rotation_produces_unit_quaternion() {
        let mut o = SceneObject::new(1, "o".into(), [0.0; 3], GeometryType::Cube);
        o.rotation_euler = [90.0, 45.0, 30.0];
        o.update_rotation();
        assert!((o.instance.rotation.magnitude() - 1.0).abs() < 1e-5);
        // A non-trivial rotation must differ from identity.
        assert!((o.instance.rotation.s - 1.0).abs() > 1e-3);
    }

    #[test]
    fn geometry_type_is_copy_and_comparable() {
        let a = GeometryType::Mesh;
        let b = a; // Copy
        assert_eq!(a, b);
        assert_ne!(GeometryType::Cube, GeometryType::Plane);
        assert_ne!(GeometryType::Mesh, GeometryType::Line);
    }

    #[test]
    fn equality_tracks_every_field() {
        let base = SceneObject::new(1, "x".into(), [0.0; 3], GeometryType::Cube);
        let mut other = base.clone();
        assert_eq!(base, other);
        other.cube_side = 2.0;
        assert_ne!(base, other);
    }

    // ---- intersect_primitive --------------------------------------------

    #[test]
    fn cube_ray_hits_front_face_and_misses_when_offset() {
        // Straight-on hit: front face at x = -0.5 from origin at x = -2.
        let t = intersect_primitive(GeometryType::Cube, vec(-2.0, 0.0, 0.0), vec(1.0, 0.0, 0.0));
        assert!((t.unwrap() - 1.5).abs() < 1e-5);
        // Parallel ray passing outside the slab misses.
        assert!(intersect_primitive(GeometryType::Cube, vec(-2.0, 5.0, 0.0), vec(1.0, 0.0, 0.0))
            .is_none());
        // Ray pointing away never reaches the cube.
        assert!(intersect_primitive(GeometryType::Cube, vec(-2.0, 0.0, 0.0), vec(-1.0, 0.0, 0.0))
            .is_none());
    }

    #[test]
    fn cube_ray_from_inside_returns_zero_distance() {
        let t = intersect_primitive(GeometryType::Cube, vec(0.0, 0.0, 0.0), vec(1.0, 0.0, 0.0));
        assert_eq!(t, Some(0.0));
    }

    #[test]
    fn sphere_ray_hits_near_side_and_handles_inside_origin() {
        // From outside: nearest hit at radius 0.5, origin at z = -2 → t = 1.5.
        let t = intersect_primitive(GeometryType::Sphere, vec(0.0, 0.0, -2.0), vec(0.0, 0.0, 1.0));
        assert!((t.unwrap() - 1.5).abs() < 1e-5);
        // From the center, the forward exit point is used (t = radius).
        let t_in =
            intersect_primitive(GeometryType::Sphere, vec(0.0, 0.0, 0.0), vec(0.0, 0.0, 1.0));
        assert!((t_in.unwrap() - 0.5).abs() < 1e-5);
        // A ray that misses entirely.
        assert!(intersect_primitive(GeometryType::Sphere, vec(2.0, 2.0, -2.0), vec(0.0, 0.0, 1.0))
            .is_none());
    }

    #[test]
    fn plane_ray_respects_bounds_and_parallelism() {
        // Hit the unit plane (y = 0) from above, within bounds.
        let t = intersect_primitive(GeometryType::Plane, vec(0.0, 1.0, 0.0), vec(0.0, -1.0, 0.0));
        assert!((t.unwrap() - 1.0).abs() < 1e-5);
        // Within the plane but outside the [-0.5, 0.5] quad → miss.
        assert!(intersect_primitive(GeometryType::Plane, vec(5.0, 1.0, 0.0), vec(0.0, -1.0, 0.0))
            .is_none());
        // Parallel to the plane → miss.
        assert!(intersect_primitive(GeometryType::Plane, vec(0.0, 1.0, 0.0), vec(1.0, 0.0, 0.0))
            .is_none());
        // Pointing up, away from the plane → miss.
        assert!(intersect_primitive(GeometryType::Plane, vec(0.0, 1.0, 0.0), vec(0.0, 1.0, 0.0))
            .is_none());
    }

    #[test]
    fn line_and_mesh_have_no_analytic_primitive() {
        assert!(intersect_primitive(GeometryType::Line, vec(0.0, 0.0, 0.0), vec(1.0, 0.0, 0.0))
            .is_none());
        assert!(intersect_primitive(GeometryType::Mesh, vec(0.0, 0.0, 0.0), vec(1.0, 0.0, 0.0))
            .is_none());
    }

    // ---- apply_undo_command ---------------------------------------------

    fn obj(id: usize) -> SceneObject {
        SceneObject::new(id, format!("o{id}"), [0.0; 3], GeometryType::Cube)
    }

    #[test]
    fn undo_redo_add_and_delete_are_symmetric() {
        let mut objects = vec![obj(1)];
        let add = UndoCommand::Add(obj(2));
        apply_undo_command(&mut objects, &add, false); // redo add
        assert_eq!(objects.len(), 2);
        apply_undo_command(&mut objects, &add, true); // undo add
        assert_eq!(objects.len(), 1);

        let del = UndoCommand::Delete(obj(1));
        apply_undo_command(&mut objects, &del, false); // redo delete
        assert!(objects.is_empty());
        apply_undo_command(&mut objects, &del, true); // undo delete
        assert_eq!(objects.len(), 1);
    }

    #[test]
    fn undo_redo_edit_swaps_old_and_new() {
        let old = obj(1);
        let mut new = old.clone();
        new.label = "renamed".into();
        let mut objects = vec![old.clone()];
        let cmd = UndoCommand::Edit {
            old: old.clone(),
            new: new.clone(),
        };
        apply_undo_command(&mut objects, &cmd, false); // redo edit
        assert_eq!(objects[0].label, "renamed");
        apply_undo_command(&mut objects, &cmd, true); // undo edit
        assert_eq!(objects[0].label, "o1");
    }

    #[test]
    fn multi_action_applies_and_reverses_every_child() {
        let mut objects = Vec::new();
        let cmd = UndoCommand::MultiAction(vec![
            UndoCommand::Add(obj(1)),
            UndoCommand::Add(obj(2)),
        ]);
        apply_undo_command(&mut objects, &cmd, false);
        assert_eq!(objects.len(), 2);
        apply_undo_command(&mut objects, &cmd, true);
        assert!(objects.is_empty());
    }

    #[test]
    fn unit_vec_helper_is_used() {
        // Keep the InnerSpace import meaningful and document the convention.
        assert!((vec(0.0, 3.0, 4.0).magnitude() - 5.0).abs() < 1e-6);
    }
}
