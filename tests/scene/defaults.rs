// tests/scene/defaults.rs
// Scene Object default value tests

use rendering_3d::state::{DEFAULT_NEW_OBJ_COLOR, DEFAULT_NEW_OBJ_POS};

#[test]
fn test_new_object_position() {
    assert_eq!(
        DEFAULT_NEW_OBJ_POS,
        [0.0, 0.0, 0.0],
        "New Object Position mismatch"
    );
}

#[test]
fn test_new_object_color() {
    assert_eq!(
        DEFAULT_NEW_OBJ_COLOR,
        [1.0, 0.0, 0.0],
        "New Object Color mismatch (expected Red)"
    );
}
