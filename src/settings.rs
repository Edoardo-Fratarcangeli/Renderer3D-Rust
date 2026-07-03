//! Persistent application settings saved to the OS config directory as JSON.
//!
//! Call [`AppSettings::load`] at startup and [`AppSettings::save`] whenever a
//! value changes. Unknown keys are silently ignored so old config files survive
//! app upgrades.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── Defaults ─────────────────────────────────────────────────────────────────
// Single source of truth for every persisted visual default. `state.rs`
// re-exports these so UI code and tests read the same values.

pub const DEFAULT_BG_COLOR:            [f32; 3] = [0.1, 0.1, 0.1];
pub const DEFAULT_GRID_CELL_SIZE:      f32      = 1.0;
pub const DEFAULT_GRID_MAX_EXTENT:     f32      = 10.0;
pub const DEFAULT_GRID_LINE_THICKNESS: f32      = 0.02;
pub const DEFAULT_GRID_COLOR:          [f32; 3] = [0.4, 0.4, 0.4];
pub const DEFAULT_SHOW_GRID_XY:        bool     = true;
pub const DEFAULT_SHOW_GRID_XZ:        bool     = false;
pub const DEFAULT_SHOW_GRID_YZ:        bool     = false;
pub const DEFAULT_SHOW_AXES:           bool     = true;
pub const DEFAULT_MIN_ZOOM:            f32      = 1.0;
pub const DEFAULT_MAX_ZOOM:            f32      = 1000.0;

// serde `default = "…"` attributes require functions, so these just forward.
fn default_bg_color()          -> [f32; 3] { DEFAULT_BG_COLOR }
fn default_grid_cell_size()    -> f32       { DEFAULT_GRID_CELL_SIZE }
fn default_grid_max_extent()   -> f32       { DEFAULT_GRID_MAX_EXTENT }
fn default_grid_thickness()    -> f32       { DEFAULT_GRID_LINE_THICKNESS }
fn default_grid_color()        -> [f32; 3] { DEFAULT_GRID_COLOR }
fn default_show_grid_xy()      -> bool      { DEFAULT_SHOW_GRID_XY }
fn default_show_grid_xz()      -> bool      { DEFAULT_SHOW_GRID_XZ }
fn default_show_grid_yz()      -> bool      { DEFAULT_SHOW_GRID_YZ }
fn default_show_axes()         -> bool      { DEFAULT_SHOW_AXES }
fn default_min_zoom()          -> f32       { DEFAULT_MIN_ZOOM }
fn default_max_zoom()          -> f32       { DEFAULT_MAX_ZOOM }

// ── Structs ───────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct GridSettings {
    #[serde(default = "default_grid_cell_size")]
    pub cell_size: f32,
    #[serde(default = "default_grid_max_extent")]
    pub max_extent: f32,
    #[serde(default = "default_grid_thickness")]
    pub line_thickness: f32,
    #[serde(default = "default_grid_color")]
    pub color: [f32; 3],
}

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            cell_size:      default_grid_cell_size(),
            max_extent:     default_grid_max_extent(),
            line_thickness: default_grid_thickness(),
            color:          default_grid_color(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct AppSettings {
    #[serde(default = "default_bg_color")]
    pub bg_color: [f32; 3],
    #[serde(default)]
    pub grid: GridSettings,
    #[serde(default = "default_show_grid_xy")]
    pub show_grid_xy: bool,
    #[serde(default = "default_show_grid_xz")]
    pub show_grid_xz: bool,
    #[serde(default = "default_show_grid_yz")]
    pub show_grid_yz: bool,
    #[serde(default = "default_show_axes")]
    pub show_axes: bool,
    #[serde(default = "default_min_zoom")]
    pub min_zoom: f32,
    #[serde(default = "default_max_zoom")]
    pub max_zoom: f32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            bg_color:     default_bg_color(),
            grid:         GridSettings::default(),
            show_grid_xy: default_show_grid_xy(),
            show_grid_xz: default_show_grid_xz(),
            show_grid_yz: default_show_grid_yz(),
            show_axes:    default_show_axes(),
            min_zoom:     default_min_zoom(),
            max_zoom:     default_max_zoom(),
        }
    }
}

// ── I/O ───────────────────────────────────────────────────────────────────────

fn config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("rendering3d").join("settings.json"))
}

impl AppSettings {
    pub fn load() -> Self {
        let Some(path) = config_path() else { return Self::default() };
        let Ok(text) = std::fs::read_to_string(&path) else { return Self::default() };
        serde_json::from_str(&text).unwrap_or_default()
    }

    pub fn save(&self) {
        let Some(path) = config_path() else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }
}
