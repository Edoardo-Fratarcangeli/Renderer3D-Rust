// tests/scene/undo_redo.rs
use rendering_3d::scene::{GeometryType, SceneObject};

#[test]
fn test_scene_object_partial_eq() {
    let obj1 = SceneObject::new(1, "Test".into(), [0.0, 0.0, 0.0], GeometryType::Cube);
    let mut obj2 = obj1.clone();

    assert_eq!(obj1, obj2);

    obj2.label = "Changed".into();
    assert_ne!(obj1, obj2);
}

#[test]
fn test_instance_partial_eq() {
    let obj1 = SceneObject::new(1, "Test".into(), [0.0, 0.0, 0.0], GeometryType::Cube);
    let mut obj2 = obj1.clone();

    assert_eq!(obj1.instance, obj2.instance);

    obj2.instance.position.x += 1.0;
    assert_ne!(obj1.instance, obj2.instance);
}
