//! Label id -> point geometry mapping.
//!
//! Cycling through primitive shapes adds a second visual channel on top of
//! color (useful for many labels or colorblind users).

use crate::scene::GeometryType;

const SHAPES: [GeometryType; 3] = [
    GeometryType::Sphere,
    GeometryType::Cube,
    GeometryType::Plane,
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometryPolicy {
    /// Every label uses the same shape.
    Uniform(GeometryType),
    /// Cycle shapes per label id.
    PerLabel,
    /// Bucket the normalized distance from the cloud center into shapes,
    /// turning the radial distance into a second visual channel.
    ByDistance,
}

impl Default for GeometryPolicy {
    fn default() -> Self {
        GeometryPolicy::Uniform(GeometryType::Sphere)
    }
}

pub fn geometry_for_label(policy: GeometryPolicy, label: u32) -> GeometryType {
    match policy {
        GeometryPolicy::Uniform(g) => g,
        GeometryPolicy::PerLabel => SHAPES[(label as usize) % SHAPES.len()],
        // Distance-based assignment needs the scalar, not the label; callers
        // that pick `ByDistance` must use `geometry_for_distance` instead. Fall
        // back to the default shape if it is ever resolved by label.
        GeometryPolicy::ByDistance => SHAPES[0],
    }
}

/// Shape for a normalized distance `t` in `[0, 1]`, split into even thirds:
/// inner → sphere, middle → cube, outer → plane. `t` is clamped.
pub fn geometry_for_distance(t: f32) -> GeometryType {
    let t = t.clamp(0.0, 1.0);
    let bucket = ((t * SHAPES.len() as f32) as usize).min(SHAPES.len() - 1);
    SHAPES[bucket]
}
