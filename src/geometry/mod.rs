//! Universal geometry import: turns external data of many shapes — plain
//! geometry strings, CSV tables, Excel sheets, JSON documents, generic text
//! / XYZ point files — into renderable [`GeometryLayer`]s.
//!
//! Like [`crate::dataset`], this module is pure logic (no egui/wgpu): every
//! parser returns plain [`GeometryRecord`]s, and [`build_batches`] converts
//! visible layers into instanced batches that the renderer draws with one
//! draw call per primitive shape. That is what keeps "very many geometries"
//! fast: a million records are still at most three instanced draw calls.
//!
//! - [`parser`] — geometry-string DSL and XYZ point text
//! - [`table`] — header-mapped tables (CSV and Excel via `calamine`)
//! - [`json`] — JSON arrays of geometry objects
//! - [`loader`] — extension dispatch + auto-detection for pasted text

pub mod json;
pub mod loader;
pub mod parser;
pub mod table;

use std::fmt;

use cgmath::{Deg, Quaternion, Rotation3};

use crate::model::InstanceRaw;
use crate::scene::GeometryType;

/// Errors produced by the geometry import layer.
#[derive(Debug)]
pub enum GeometryError {
    Io(std::io::Error),
    /// Parse failure with a 1-based line/record number for user feedback.
    Parse { line: usize, message: String },
    Format(String),
    Unsupported(String),
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeometryError::Io(e) => write!(f, "I/O error: {}", e),
            GeometryError::Parse { line, message } => {
                write!(f, "line {}: {}", line, message)
            }
            GeometryError::Format(m) => write!(f, "format error: {}", m),
            GeometryError::Unsupported(m) => write!(f, "unsupported: {}", m),
        }
    }
}

impl std::error::Error for GeometryError {}

impl From<std::io::Error> for GeometryError {
    fn from(e: std::io::Error) -> Self {
        GeometryError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, GeometryError>;

/// Default color given to records that do not specify one.
pub const DEFAULT_COLOR: [f32; 3] = [0.8, 0.8, 0.8];
/// Default uniform scale for `point` records (small marker spheres).
pub const POINT_SIZE: f32 = 0.1;

/// One imported geometry: shape + transform + color + optional label.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryRecord {
    pub shape: GeometryType,
    pub position: [f32; 3],
    /// Euler rotation in degrees (X, Y, Z), applied in that order.
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
    pub color: [f32; 3],
    pub label: Option<String>,
}

impl GeometryRecord {
    /// A unit record at `position` with defaults everywhere else.
    pub fn new(shape: GeometryType, position: [f32; 3]) -> Self {
        Self {
            shape,
            position,
            rotation: [0.0; 3],
            scale: [1.0; 3],
            color: DEFAULT_COLOR,
            label: None,
        }
    }

    /// GPU instance for this record (alpha 1.0 = no highlight).
    pub fn to_instance(&self) -> InstanceRaw {
        let rot = Quaternion::from_angle_x(Deg(self.rotation[0]))
            * Quaternion::from_angle_y(Deg(self.rotation[1]))
            * Quaternion::from_angle_z(Deg(self.rotation[2]));
        let model = cgmath::Matrix4::from_translation(self.position.into())
            * cgmath::Matrix4::from(rot)
            * cgmath::Matrix4::from_nonuniform_scale(self.scale[0], self.scale[1], self.scale[2]);
        InstanceRaw {
            model: model.into(),
            color: [self.color[0], self.color[1], self.color[2], 1.0],
        }
    }
}

/// A named group of imported geometries, toggled as a unit in the UI.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryLayer {
    pub name: String,
    pub records: Vec<GeometryRecord>,
    pub visible: bool,
}

impl GeometryLayer {
    pub fn new(name: impl Into<String>, records: Vec<GeometryRecord>) -> Self {
        Self {
            name: name.into(),
            records,
            visible: true,
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Average position of the layer's records (camera focus target).
    pub fn centroid(&self) -> Option<[f32; 3]> {
        if self.records.is_empty() {
            return None;
        }
        let mut acc = [0.0f64; 3];
        for r in &self.records {
            for a in 0..3 {
                acc[a] += r.position[a] as f64;
            }
        }
        let n = self.records.len() as f64;
        Some([
            (acc[0] / n) as f32,
            (acc[1] / n) as f32,
            (acc[2] / n) as f32,
        ])
    }
}

/// Flatten all visible layers into one instanced batch per primitive shape.
///
/// This is the renderer-facing output: each `(shape, instances)` pair maps
/// to a single instanced draw call, regardless of how many records it holds.
pub fn build_batches(layers: &[GeometryLayer]) -> Vec<(GeometryType, Vec<InstanceRaw>)> {
    let mut batches: Vec<(GeometryType, Vec<InstanceRaw>)> = Vec::new();
    for layer in layers.iter().filter(|l| l.visible) {
        for record in &layer.records {
            match batches.iter_mut().find(|(g, _)| *g == record.shape) {
                Some((_, list)) => list.push(record.to_instance()),
                None => batches.push((record.shape, vec![record.to_instance()])),
            }
        }
    }
    batches
}

/// Parse a shape name (with common aliases). `point` maps to a small sphere.
pub fn parse_shape(token: &str) -> Option<(GeometryType, bool)> {
    match token.to_ascii_lowercase().as_str() {
        "cube" | "box" => Some((GeometryType::Cube, false)),
        "sphere" | "ball" => Some((GeometryType::Sphere, false)),
        "plane" | "quad" => Some((GeometryType::Plane, false)),
        "point" | "dot" | "vertex" => Some((GeometryType::Sphere, true)),
        _ => None,
    }
}

/// Parse a color token: `#rgb`, `#rrggbb` or `r,g,b` (0–1 or 0–255).
pub fn parse_color(token: &str) -> Option<[f32; 3]> {
    let t = token.trim();
    if let Some(hex) = t.strip_prefix('#') {
        let (r, g, b) = match hex.len() {
            3 => (
                u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?,
                u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?,
                u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?,
            ),
            6 => (
                u8::from_str_radix(&hex[0..2], 16).ok()?,
                u8::from_str_radix(&hex[2..4], 16).ok()?,
                u8::from_str_radix(&hex[4..6], 16).ok()?,
            ),
            _ => return None,
        };
        return Some([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]);
    }
    let parts: Vec<f32> = t
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<std::result::Result<_, _>>()
        .ok()?;
    if parts.len() != 3 {
        return None;
    }
    Some(normalize_rgb([parts[0], parts[1], parts[2]]))
}

/// Map 0–255 triples down to 0–1; pass 0–1 triples through unchanged.
pub fn normalize_rgb(c: [f32; 3]) -> [f32; 3] {
    if c.iter().any(|v| *v > 1.0) {
        [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0]
    } else {
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_aliases_resolve() {
        assert_eq!(parse_shape("cube"), Some((GeometryType::Cube, false)));
        assert_eq!(parse_shape("BOX"), Some((GeometryType::Cube, false)));
        assert_eq!(parse_shape("ball"), Some((GeometryType::Sphere, false)));
        assert_eq!(parse_shape("Quad"), Some((GeometryType::Plane, false)));
        assert_eq!(parse_shape("point"), Some((GeometryType::Sphere, true)));
        assert_eq!(parse_shape("dodecahedron"), None);
    }

    #[test]
    fn color_parsing_accepts_hex_and_triples() {
        assert_eq!(parse_color("#ff0000"), Some([1.0, 0.0, 0.0]));
        assert_eq!(parse_color("#0f0"), Some([0.0, 1.0, 0.0]));
        let c = parse_color("0.2,0.4,0.8").unwrap();
        assert!((c[1] - 0.4).abs() < 1e-6);
        // 0–255 triples are scaled down.
        let c = parse_color("255, 0, 128").unwrap();
        assert_eq!(c[0], 1.0);
        assert!((c[2] - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(parse_color("#zzz"), None);
        assert_eq!(parse_color("1,2"), None);
        assert_eq!(parse_color("notacolor"), None);
    }

    #[test]
    fn record_instance_carries_transform_and_color() {
        let mut r = GeometryRecord::new(GeometryType::Cube, [1.0, 2.0, 3.0]);
        r.scale = [2.0, 2.0, 2.0];
        r.color = [0.1, 0.2, 0.3];
        let inst = r.to_instance();
        assert_eq!(inst.model[3][0], 1.0);
        assert_eq!(inst.model[3][1], 2.0);
        assert_eq!(inst.model[3][2], 3.0);
        assert_eq!(inst.model[0][0], 2.0);
        assert_eq!(inst.color, [0.1, 0.2, 0.3, 1.0]);
    }

    #[test]
    fn batches_group_by_shape_and_skip_hidden_layers() {
        let cubes = GeometryLayer::new(
            "cubes",
            vec![
                GeometryRecord::new(GeometryType::Cube, [0.0; 3]),
                GeometryRecord::new(GeometryType::Cube, [1.0; 3]),
            ],
        );
        let mixed = GeometryLayer::new(
            "mixed",
            vec![
                GeometryRecord::new(GeometryType::Sphere, [0.0; 3]),
                GeometryRecord::new(GeometryType::Cube, [2.0; 3]),
            ],
        );
        let mut hidden = GeometryLayer::new(
            "hidden",
            vec![GeometryRecord::new(GeometryType::Plane, [0.0; 3])],
        );
        hidden.visible = false;

        let batches = build_batches(&[cubes, mixed, hidden]);
        assert_eq!(batches.len(), 2);
        let cubes = batches.iter().find(|(g, _)| *g == GeometryType::Cube).unwrap();
        assert_eq!(cubes.1.len(), 3);
        let spheres = batches.iter().find(|(g, _)| *g == GeometryType::Sphere).unwrap();
        assert_eq!(spheres.1.len(), 1);
        assert!(!batches.iter().any(|(g, _)| *g == GeometryType::Plane));
    }

    #[test]
    fn centroid_averages_positions() {
        let layer = GeometryLayer::new(
            "l",
            vec![
                GeometryRecord::new(GeometryType::Cube, [0.0, 0.0, 0.0]),
                GeometryRecord::new(GeometryType::Cube, [2.0, 4.0, -6.0]),
            ],
        );
        assert_eq!(layer.centroid(), Some([1.0, 2.0, -3.0]));
        assert_eq!(GeometryLayer::new("e", vec![]).centroid(), None);
        assert!(GeometryLayer::new("e", vec![]).is_empty());
        assert_eq!(layer.len(), 2);
    }
}
