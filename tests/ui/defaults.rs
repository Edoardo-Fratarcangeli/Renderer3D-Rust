// tests/ui/defaults.rs
// UI Panel, Grid, and Background default value tests

use rendering_3d::state::{
    // Background
    DEFAULT_BG_COLOR,
    DEFAULT_BOTTOM_PANEL_EXPANDED,
    DEFAULT_SHOW_ADD_PANEL,
    DEFAULT_SHOW_AXES,
    // Grids
    DEFAULT_SHOW_GRID_XY,
    DEFAULT_SHOW_GRID_XZ,
    DEFAULT_SHOW_GRID_YZ,
    // UI Panels
    DEFAULT_SHOW_SETTINGS,
};

// UI Panel Tests
#[test]
fn test_show_settings() {
    assert!(!DEFAULT_SHOW_SETTINGS, "Show Settings mismatch");
}

#[test]
fn test_show_add_panel() {
    assert!(!DEFAULT_SHOW_ADD_PANEL, "Show Add Panel mismatch");
}

#[test]
fn test_bottom_panel_expanded() {
    assert!(
        !DEFAULT_BOTTOM_PANEL_EXPANDED,
        "Bottom Panel Expanded mismatch"
    );
}

// Grid Tests
#[test]
fn test_show_grid_xy() {
    assert!(DEFAULT_SHOW_GRID_XY, "Show Grid XY mismatch");
}

#[test]
fn test_show_grid_xz() {
    assert!(!DEFAULT_SHOW_GRID_XZ, "Show Grid XZ mismatch");
}

#[test]
fn test_show_grid_yz() {
    assert!(!DEFAULT_SHOW_GRID_YZ, "Show Grid YZ mismatch");
}

#[test]
fn test_show_axes() {
    assert!(DEFAULT_SHOW_AXES, "Show Axes mismatch");
}

// Background Test
#[test]
fn test_bg_color() {
    assert_eq!(DEFAULT_BG_COLOR, 0.1_f64, "Background Color mismatch");
}
