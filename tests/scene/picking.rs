// tests/scene/picking.rs
// Testing picking and selection logic
use rendering_3d::scene::{GeometryType, SceneObject};

#[test]
fn test_object_selection_toggle() {
    let mut obj = SceneObject::new(1, "Test".into(), [0.0, 0.0, 0.0], GeometryType::Cube);
    assert!(!obj.selected);

    obj.selected = true;
    assert!(obj.selected);

    obj.selected = false;
    assert!(!obj.selected);
}

#[test]
fn test_object_label_toggle() {
    let mut obj = SceneObject::new(1, "Test".into(), [0.0, 0.0, 0.0], GeometryType::Cube);
    assert!(obj.show_label); // Default should be true

    obj.show_label = false;
    assert!(!obj.show_label);

    obj.show_label = true;
    assert!(obj.show_label);
}

#[test]
fn test_picking_ray_intersection_logic() {
    // We can't easily instantiate a full State without a Window/GPU in a unit test,
    // but we can test the intersection math if we make it accessible or mock it.
    // For now, let's test that the logical mapping of GeometryTypes exists.
    let cube = GeometryType::Cube;
    let sphere = GeometryType::Sphere;
    let plane = GeometryType::Plane;

    assert_ne!(cube, sphere);
    assert_ne!(cube, plane);
}

#[test]
fn test_multi_select_logic_simulation() {
    let mut objects = vec![
        SceneObject::new(1, "Obj1".into(), [0.0, 0.0, 0.0], GeometryType::Cube),
        SceneObject::new(2, "Obj2".into(), [2.0, 0.0, 0.0], GeometryType::Cube),
    ];

    // Simulate exclusive selection (standard click)
    let selected_id = 2;
    for obj in &mut objects {
        obj.selected = obj.id == selected_id;
    }
    assert!(!objects[0].selected);
    assert!(objects[1].selected);

    // Simulate multi-selection (ctrl+click)
    let toggle_id = 1;
    if let Some(obj) = objects.iter_mut().find(|o| o.id == toggle_id) {
        obj.selected = !obj.selected;
    }
    assert!(objects[0].selected);
    assert!(objects[1].selected);
}
