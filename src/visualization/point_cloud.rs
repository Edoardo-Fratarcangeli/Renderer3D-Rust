//! Builds instanced point-cloud batches ([`InstanceRaw`]) from projected
//! points, labels and the active filter.
//!
//! Output plugs straight into the existing instanced mesh pipeline in
//! `state.rs`: one [`PointCloudBatch`] per primitive shape, drawn with a
//! single instanced draw call each.

use cgmath::SquareMatrix;

use super::color_mapper::{color_for_distance, color_for_label, ColorMode};
use super::geometry_assigner::{geometry_for_distance, geometry_for_label, GeometryPolicy};
use crate::model::InstanceRaw;
use crate::scene::GeometryType;

/// Hard cap on rendered instances; above this points are strided evenly so
/// huge datasets still render interactively.
pub const MAX_RENDER_POINTS: usize = 200_000;

#[derive(Debug, Clone)]
pub struct PointCloudSettings {
    pub point_size: f32,
    pub geometry_policy: GeometryPolicy,
    /// How point colors are chosen (per label, or by distance from center).
    pub color_mode: ColorMode,
    /// Row highlighted from the table/search (drawn bigger + selected tint).
    pub highlighted_row: Option<u32>,
}

impl Default for PointCloudSettings {
    fn default() -> Self {
        Self {
            point_size: 0.06,
            geometry_policy: GeometryPolicy::default(),
            color_mode: ColorMode::default(),
            highlighted_row: None,
        }
    }
}

/// One batch = one mesh draw call with N instances.
pub struct PointCloudBatch {
    pub geometry: GeometryType,
    pub instances: Vec<InstanceRaw>,
}

pub struct PointCloudBuildResult {
    pub batches: Vec<PointCloudBatch>,
    /// Total instances after the render cap was applied.
    pub rendered_points: usize,
    /// True when striding was applied because of MAX_RENDER_POINTS.
    pub downsampled: bool,
}

/// Build instance batches for the rows in `visible_rows`.
///
/// `points` is indexed by absolute row id; `labels[row]` gives the label id.
pub fn build_instances(
    points: &[[f32; 3]],
    labels: &[u32],
    visible_rows: &[u32],
    settings: &PointCloudSettings,
) -> PointCloudBuildResult {
    let total = visible_rows.len();
    let stride = if total > MAX_RENDER_POINTS {
        total.div_ceil(MAX_RENDER_POINTS)
    } else {
        1
    };

    // Distance-based channels (color and/or shape) need the radial distance
    // normalized over the visible set, so precompute its range once.
    let needs_distance = settings.color_mode == ColorMode::ByDistance
        || settings.geometry_policy == GeometryPolicy::ByDistance;
    let dist = |row: u32| {
        let p = points[row as usize];
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
    };
    let (min_dist, max_dist) = if needs_distance {
        visible_rows
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &row| {
                let d = dist(row);
                (mn.min(d), mx.max(d))
            })
    } else {
        (0.0, 0.0)
    };
    let span = max_dist - min_dist;

    let mut batches: Vec<PointCloudBatch> = Vec::new();
    let mut rendered = 0usize;
    for (vi, &row) in visible_rows.iter().enumerate() {
        if vi % stride != 0 {
            continue;
        }
        let p = points[row as usize];
        let label = labels[row as usize];
        let t = if needs_distance && span > 1e-12 {
            (dist(row) - min_dist) / span
        } else {
            0.0
        };
        let geometry = match settings.geometry_policy {
            GeometryPolicy::ByDistance => geometry_for_distance(t),
            policy => geometry_for_label(policy, label),
        };
        let color = match settings.color_mode {
            ColorMode::ByDistance => color_for_distance(t),
            ColorMode::ByLabel => color_for_label(label),
        };

        let highlighted = settings.highlighted_row == Some(row);
        let size = if highlighted {
            settings.point_size * 2.5
        } else {
            settings.point_size
        };
        let mut model = cgmath::Matrix4::identity();
        model.w = cgmath::Vector4::new(p[0], p[1], p[2], 1.0);
        model.x.x = size;
        model.y.y = size;
        model.z.z = size;
        // alpha = 2.0 reuses the shader's "selected" highlight path.
        let alpha = if highlighted { 2.0 } else { 1.0 };
        let raw = InstanceRaw {
            model: model.into(),
            color: [color[0], color[1], color[2], alpha],
        };

        match batches.iter_mut().find(|b| b.geometry == geometry) {
            Some(batch) => batch.instances.push(raw),
            None => batches.push(PointCloudBatch {
                geometry,
                instances: vec![raw],
            }),
        }
        rendered += 1;
    }

    PointCloudBuildResult {
        batches,
        rendered_points: rendered,
        downsampled: stride > 1,
    }
}
