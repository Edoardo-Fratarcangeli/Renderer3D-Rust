// Tests for the visualization layer: color mapping, geometry assignment and
// point-cloud instance building (all GPU-free).

use rendering_3d::scene::GeometryType;
use rendering_3d::visualization::color_mapper::{color_for_label, palette};
use rendering_3d::visualization::geometry_assigner::{geometry_for_label, GeometryPolicy};
use rendering_3d::visualization::point_cloud::{
    build_instances, PointCloudSettings, MAX_RENDER_POINTS,
};

#[test]
fn colors_are_deterministic_and_distinct() {
    let n = 24;
    let colors = palette(n);
    assert_eq!(colors.len(), n);
    // Deterministic
    assert_eq!(colors, palette(n));
    // Pairwise distinct
    for i in 0..n {
        for j in (i + 1)..n {
            assert_ne!(
                colors[i], colors[j],
                "labels {} and {} share a color",
                i, j
            );
        }
    }
    // Valid RGB range
    for c in &colors {
        for ch in c {
            assert!((0.0..=1.0).contains(ch));
        }
    }
    assert_eq!(color_for_label(3), palette(4)[3]);
}

#[test]
fn geometry_policy_maps_labels_to_shapes() {
    let uniform = GeometryPolicy::Uniform(GeometryType::Sphere);
    assert_eq!(geometry_for_label(uniform, 0), GeometryType::Sphere);
    assert_eq!(geometry_for_label(uniform, 7), GeometryType::Sphere);

    let per = GeometryPolicy::PerLabel;
    let g0 = geometry_for_label(per, 0);
    let g1 = geometry_for_label(per, 1);
    let g2 = geometry_for_label(per, 2);
    assert!(g0 != g1 && g1 != g2 && g0 != g2);
    // Cycles after the shape count.
    assert_eq!(geometry_for_label(per, 3), g0);
}

fn sample_points(n: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
    let points = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect();
    let labels = (0..n).map(|i| (i % 3) as u32).collect();
    (points, labels)
}

#[test]
fn point_cloud_renders_only_visible_rows_with_label_colors() {
    let (points, labels) = sample_points(10);
    let visible: Vec<u32> = vec![0, 2, 4];
    let result = build_instances(&points, &labels, &visible, &PointCloudSettings::default());

    assert_eq!(result.rendered_points, 3);
    assert!(!result.downsampled);
    let total: usize = result.batches.iter().map(|b| b.instances.len()).sum();
    assert_eq!(total, 3);

    // Uniform policy -> single sphere batch.
    assert_eq!(result.batches.len(), 1);
    assert_eq!(result.batches[0].geometry, GeometryType::Sphere);

    // Each instance carries its label color and sits at the projected point.
    for (k, &row) in visible.iter().enumerate() {
        let inst = &result.batches[0].instances[k];
        let expected = color_for_label(labels[row as usize]);
        assert_eq!(&inst.color[0..3], &expected);
        assert_eq!(inst.model[3][0], points[row as usize][0]);
    }
}

#[test]
fn point_cloud_groups_batches_per_shape() {
    let (points, labels) = sample_points(9);
    let visible: Vec<u32> = (0..9).collect();
    let settings = PointCloudSettings {
        geometry_policy: GeometryPolicy::PerLabel,
        ..Default::default()
    };
    let result = build_instances(&points, &labels, &visible, &settings);
    assert_eq!(result.batches.len(), 3);
    for batch in &result.batches {
        assert_eq!(batch.instances.len(), 3);
    }
}

#[test]
fn point_cloud_highlight_enlarges_and_tints() {
    let (points, labels) = sample_points(4);
    let settings = PointCloudSettings {
        highlighted_row: Some(2),
        ..Default::default()
    };
    let visible: Vec<u32> = (0..4).collect();
    let result = build_instances(&points, &labels, &visible, &settings);
    let all: Vec<_> = result
        .batches
        .iter()
        .flat_map(|b| b.instances.iter())
        .collect();
    let highlighted = all.iter().find(|i| i.color[3] == 2.0).expect("highlight");
    let normal = all.iter().find(|i| i.color[3] == 1.0).unwrap();
    assert!(highlighted.model[0][0] > normal.model[0][0]);
}

#[test]
fn point_cloud_downsamples_above_cap() {
    let n = MAX_RENDER_POINTS * 2;
    let points = vec![[0.0f32; 3]; n];
    let labels = vec![0u32; n];
    let visible: Vec<u32> = (0..n as u32).collect();
    let result = build_instances(&points, &labels, &visible, &PointCloudSettings::default());
    assert!(result.downsampled);
    assert!(result.rendered_points <= MAX_RENDER_POINTS);
    assert!(result.rendered_points >= MAX_RENDER_POINTS / 2);
}
