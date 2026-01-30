// tests/camera/defaults.rs
// Camera and Zoom default value tests

use rendering_3d::state::{
    DEFAULT_CAMERA_YAW,
    DEFAULT_CAMERA_PITCH,
    DEFAULT_CAMERA_DIST,
    DEFAULT_CAMERA_TARGET,
    DEFAULT_MIN_ZOOM,
    DEFAULT_MAX_ZOOM,
};

#[test]
fn test_camera_yaw() {
    assert_eq!(DEFAULT_CAMERA_YAW, 16.0, "Camera Yaw mismatch");
}

#[test]
fn test_camera_pitch() {
    assert_eq!(DEFAULT_CAMERA_PITCH, 36.0, "Camera Pitch mismatch");
}

#[test]
fn test_camera_dist() {
    assert_eq!(DEFAULT_CAMERA_DIST, 21.1, "Camera Dist mismatch");
}

#[test]
fn test_camera_target() {
    assert_eq!(DEFAULT_CAMERA_TARGET, [3.0, 0.15, 0.5], "Camera Target mismatch");
}

#[test]
fn test_min_zoom() {
    assert_eq!(DEFAULT_MIN_ZOOM, 1.0, "Min Zoom mismatch");
}

#[test]
fn test_max_zoom() {
    assert_eq!(DEFAULT_MAX_ZOOM, 1000.0, "Max Zoom mismatch");
}
