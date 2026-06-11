// Builds instanced point-cloud batches (InstanceRaw) from projected points,
// labels and the active filter. Output plugs straight into the existing
// instanced mesh pipeline in state.rs.

use cgmath::SquareMatrix;

use super::color_mapper::color_for_label;
use super::geometry_assigner::{geometry_for_label, GeometryPolicy};
use crate::model::InstanceRaw;
use crate::scene::GeometryType;

/// Hard cap on rendered instances; above this points are strided evenly so
/// huge datasets still render interactively.
pub const MAX_RENDER_POINTS: usize = 200_000;

#[derive(Debug, Clone)]
pub struct PointCloudSettings {
    pub point_size: f32,
    pub geometry_policy: GeometryPolicy,
    /// Row highlighted from the table/search (drawn bigger + selected tint).
    pub highlighted_row: Option<u32>,
}

impl Default for PointCloudSettings {
    fn default() -> Self {
        Self {
            point_size: 0.06,
            geometry_policy: GeometryPolicy::default(),
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
        (total + MAX_RENDER_POINTS - 1) / MAX_RENDER_POINTS
    } else {
        1
    };

    let mut batches: Vec<PointCloudBatch> = Vec::new();
    let mut rendered = 0usize;
    for (vi, &row) in visible_rows.iter().enumerate() {
        if vi % stride != 0 {
            continue;
        }
        let p = points[row as usize];
        let label = labels[row as usize];
        let geometry = geometry_for_label(settings.geometry_policy, label);
        let color = color_for_label(label);

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
