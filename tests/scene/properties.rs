// tests/scene/properties.rs
// Tests for primitive-specific properties

use rendering_3d::scene::{GeometryType, SceneObject};

#[test]
fn test_plane_properties_defaults() {
    let obj = SceneObject::new(
        1,
        "Plane 1".to_string(),
        [0.0, 0.0, 0.0],
        GeometryType::Plane,
    );
    assert_eq!(obj.plane_surface, 1.0);
    assert_eq!(obj.show_normal, false);
}

#[test]
fn test_cube_properties_defaults() {
    let obj = SceneObject::new(1, "Cube 1".to_string(), [0.0, 0.0, 0.0], GeometryType::Cube);
    assert_eq!(obj.cube_side, 1.0);
}

#[test]
fn test_sphere_properties_defaults() {
    let obj = SceneObject::new(
        1,
        "Sphere 1".to_string(),
        [0.0, 0.0, 0.0],
        GeometryType::Sphere,
    );
    assert_eq!(obj.sphere_radius, 0.5);
}

#[test]
fn test_naming_initial_id() {
    // This is more about checking if naming logic works as expected if called manually
    let obj = SceneObject::new(
        1,
        format!("Object {}", 1),
        [0.0, 0.0, 0.0],
        GeometryType::Cube,
    );
    assert_eq!(obj.label, "Object 1");
}
