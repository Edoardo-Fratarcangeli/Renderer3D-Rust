//! Point-in-volume tests for spatial dataset filtering.
//!
//! A scene primitive (sphere or cube) doubles as a *filter volume*: dataset
//! points are kept when they fall inside — or outside — the object, in the
//! object's own transformed space (position, rotation, non-uniform scale all
//! respected). Pure math, no UI/GPU; the host resolves a [`FilterVolume`]
//! from the selected [`crate::scene::SceneObject`] each frame.

use cgmath::{InnerSpace, SquareMatrix};

/// The primitive shapes usable as filter volumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeShape {
    /// Unit sphere of radius 0.5 in local space (matches `create_sphere`).
    Sphere,
    /// Unit cube spanning [-0.5, 0.5] in local space (matches `create_cube`).
    Cube,
}

/// Keep points inside or outside the volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpatialMode {
    #[default]
    Inside,
    Outside,
}

/// A world-space filter volume: shape + inverse model matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterVolume {
    shape: VolumeShape,
    /// World → local transform of the source object.
    inv_model: cgmath::Matrix4<f32>,
}

impl FilterVolume {
    /// Build a volume from an object's shape and instance transform.
    /// Returns `None` when the transform is singular (zero scale).
    pub fn new(shape: VolumeShape, model: cgmath::Matrix4<f32>) -> Option<Self> {
        Some(Self {
            shape,
            inv_model: model.invert()?,
        })
    }

    /// Whether the world-space point lies inside the volume.
    pub fn contains(&self, p: [f32; 3]) -> bool {
        let local = self.inv_model * cgmath::Vector4::new(p[0], p[1], p[2], 1.0);
        match self.shape {
            VolumeShape::Sphere => local.truncate().magnitude2() <= 0.25,
            VolumeShape::Cube => {
                local.x.abs() <= 0.5 && local.y.abs() <= 0.5 && local.z.abs() <= 0.5
            }
        }
    }

    /// Whether the point passes the filter in the given mode.
    pub fn keeps(&self, p: [f32; 3], mode: SpatialMode) -> bool {
        self.contains(p) == (mode == SpatialMode::Inside)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Deg, Matrix4, Vector3};

    fn sphere_at(center: [f32; 3], diameter: f32) -> FilterVolume {
        let model = Matrix4::from_translation(Vector3::from(center))
            * Matrix4::from_scale(diameter);
        FilterVolume::new(VolumeShape::Sphere, model).unwrap()
    }

    #[test]
    fn unit_sphere_contains_center_not_far_points() {
        let v = sphere_at([1.0, 0.0, 0.0], 2.0); // radius 1 around (1,0,0)
        assert!(v.contains([1.0, 0.0, 0.0]));
        assert!(v.contains([1.9, 0.0, 0.0]));
        assert!(!v.contains([2.1, 0.0, 0.0]));
        assert!(!v.contains([-1.0, 0.0, 0.0]));
    }

    #[test]
    fn cube_respects_rotation_and_nonuniform_scale() {
        // A 2×1×1 box rotated 90° around Z: its long axis now spans Y.
        let model = Matrix4::from_angle_z(Deg(90.0))
            * Matrix4::from_nonuniform_scale(2.0, 1.0, 1.0);
        let v = FilterVolume::new(VolumeShape::Cube, model).unwrap();
        assert!(v.contains([0.0, 0.9, 0.0]), "long axis along Y after rotation");
        assert!(!v.contains([0.9, 0.0, 0.0]), "short axis along X after rotation");
    }

    #[test]
    fn modes_partition_space() {
        let v = sphere_at([0.0; 3], 1.0);
        for p in [[0.0, 0.0, 0.2], [3.0, 0.0, 0.0]] {
            assert_ne!(v.keeps(p, SpatialMode::Inside), v.keeps(p, SpatialMode::Outside));
        }
    }

    #[test]
    fn degenerate_scale_yields_no_volume() {
        let model = Matrix4::from_nonuniform_scale(0.0, 1.0, 1.0);
        assert!(FilterVolume::new(VolumeShape::Cube, model).is_none());
    }
}
