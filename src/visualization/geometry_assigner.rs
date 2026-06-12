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
    }
}
