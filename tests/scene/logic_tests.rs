// tests/scene/logic_tests.rs
// Functional logic tests for SceneObject and State manipulations

use cgmath::Vector3;
use rendering_3d::scene::{GeometryType, SceneObject};

#[test]
fn test_plane_scale_calculation() {
    let mut obj = SceneObject::new(1, "Plane".to_string(), [0.0, 0.0, 0.0], GeometryType::Plane);

    // Default surface 1.0 -> scale (1.0, 1.0, 1.0)
    assert_eq!(obj.plane_surface, 1.0);

    // Simulate what happens in the UI
    obj.plane_surface = 4.0;
    let s = obj.plane_surface.sqrt();
    obj.instance.scale = Vector3::new(s, 1.0, s);

    assert_eq!(obj.instance.scale.x, 2.0);
    assert_eq!(obj.instance.scale.z, 2.0);
}

#[test]
fn test_cube_scale_calculation() {
    let mut obj = SceneObject::new(1, "Cube".to_string(), [0.0, 0.0, 0.0], GeometryType::Cube);

    obj.cube_side = 3.0;
    let s = obj.cube_side;
    obj.instance.scale = Vector3::new(s, s, s);

    assert_eq!(obj.instance.scale.x, 3.0);
    assert_eq!(obj.instance.scale.y, 3.0);
    assert_eq!(obj.instance.scale.z, 3.0);
}

#[test]
fn test_sphere_scale_calculation() {
    let mut obj = SceneObject::new(
        1,
        "Sphere".to_string(),
        [0.0, 0.0, 0.0],
        GeometryType::Sphere,
    );

    obj.sphere_radius = 2.0;
    let s = obj.sphere_radius * 2.0;
    obj.instance.scale = Vector3::new(s, s, s);

    // Mesh is 0.5 radius, so scale 4.0 makes it radius 2.0
    assert_eq!(obj.instance.scale.x, 4.0);
}

#[test]
fn test_double_sided_plane_geometry() {
    use rendering_3d::primitives;
    let data = primitives::create_plane(1.0);
    // 4 vertices per side = 8 total. 6 indices per side (2 tris) = 12 total.
    assert_eq!(
        data.vertices.len(),
        8,
        "Plane should have 8 vertices (double-sided)"
    );
    assert_eq!(
        data.indices.len(),
        12,
        "Plane should have 12 indices (double-sided)"
    );
}

#[test]
fn test_arrow_geometry_generation() {
    use rendering_3d::primitives;
    let data = primitives::create_arrow(1.0, 0.1, [1.0, 1.0, 1.0]);
    assert!(!data.vertices.is_empty());
    assert!(!data.indices.is_empty());
}
