// tests/ui/defaults.rs
// UI Panel, Grid, and Background default value tests

use rendering_3d::state::{
    // UI Panels
    DEFAULT_SHOW_SETTINGS,
    DEFAULT_SHOW_ADD_PANEL,
    DEFAULT_BOTTOM_PANEL_EXPANDED,
    // Grids
    DEFAULT_SHOW_GRID_XY,
    DEFAULT_SHOW_GRID_XZ,
    DEFAULT_SHOW_GRID_YZ,
    DEFAULT_SHOW_AXES,
    // Background
    DEFAULT_BG_COLOR,
};

// UI Panel Tests
#[test]
fn test_show_settings() {
    assert_eq!(DEFAULT_SHOW_SETTINGS, false, "Show Settings mismatch");
}

#[test]
fn test_show_add_panel() {
    assert_eq!(DEFAULT_SHOW_ADD_PANEL, false, "Show Add Panel mismatch");
}

#[test]
fn test_bottom_panel_expanded() {
    assert_eq!(DEFAULT_BOTTOM_PANEL_EXPANDED, true, "Bottom Panel Expanded mismatch");
}

// Grid Tests
#[test]
fn test_show_grid_xy() {
    assert_eq!(DEFAULT_SHOW_GRID_XY, true, "Show Grid XY mismatch");
}

#[test]
fn test_show_grid_xz() {
    assert_eq!(DEFAULT_SHOW_GRID_XZ, false, "Show Grid XZ mismatch");
}

#[test]
fn test_show_grid_yz() {
    assert_eq!(DEFAULT_SHOW_GRID_YZ, false, "Show Grid YZ mismatch");
}

#[test]
fn test_show_axes() {
    assert_eq!(DEFAULT_SHOW_AXES, true, "Show Axes mismatch");
}

// Background Test
#[test]
fn test_bg_color() {
    assert_eq!(DEFAULT_BG_COLOR, 0.1_f64, "Background Color mismatch");
}
