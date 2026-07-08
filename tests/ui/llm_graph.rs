// Graph-file import driven through the network visualizer's real render
// path: per-node colors, distance scaling, solid edge beams and the picking
// math all run on the CPU, so they are fully testable headless.

use rendering_3d::llm::graph_import;
use rendering_3d::ui::llm_panel::LlmView;

/// A small colored, weighted graph in GraphML form.
fn sample_graph() -> rendering_3d::llm::network::NetworkGraph {
    let text = r##"<?xml version="1.0"?>
<graphml>
  <key id="d0" for="node" attr.name="color" attr.type="string"/>
  <graph edgedefault="undirected">
    <node id="a"><data key="d0">#ff0000</data></node>
    <node id="b"/>
    <node id="c"/>
    <edge source="a" target="b"/>
    <edge source="b" target="c"/>
  </graph>
</graphml>"##;
    graph_import::load_graph(text.as_bytes(), "graphml", "sample").unwrap()
}

fn view_with_graph() -> LlmView {
    let mut view = LlmView::new();
    view.load_preset(sample_graph());
    view
}

#[test]
fn graph_renders_nodes_and_line_edges_by_default() {
    let mut view = view_with_graph();
    let data = view.build_render_data();
    assert_eq!(data.node_instances.len(), 3);
    assert_eq!(data.edge_vertices.len(), 4, "2 edges × 2 line vertices");
    assert!(data.edge_solid_instances.is_empty());
    // The per-node file color (red, dimmed by the passive factor) survives
    // into the instance stream: node a is redder than the layer-colored rest.
    let reds: Vec<f32> = data.node_instances.iter().map(|i| i.color[0]).collect();
    let max_red = reds.iter().cloned().fold(0.0f32, f32::max);
    assert!(max_red > reds.iter().cloned().fold(f32::INFINITY, f32::min) + 0.1);
}

#[test]
fn edge_width_switches_lines_to_solid_beams() {
    let mut view = view_with_graph();
    view.edge_width = 0.05;
    let data = view.build_render_data();
    assert!(data.edge_vertices.is_empty(), "beams replace line edges");
    assert_eq!(data.edge_solid_instances.len(), 2);
}

#[test]
fn distance_scale_spreads_the_whole_graph() {
    let mut view = view_with_graph();
    let base = view.build_render_data();
    view.distance_scale = 2.0;
    let scaled = view.build_render_data();
    // Instance translation lives in the last matrix row (column-major mat4).
    let pos = |i: &rendering_3d::model::InstanceRaw| [i.model[3][0], i.model[3][1], i.model[3][2]];
    for (a, b) in base.node_instances.iter().zip(&scaled.node_instances) {
        let (pa, pb) = (pos(a), pos(b));
        for axis in 0..3 {
            assert!((pb[axis] - pa[axis] * 2.0).abs() < 1e-4, "positions scale by 2");
        }
    }
    // Centroid follows the same scale.
    let c1 = LlmView::centroid(&{
        let mut v = view_with_graph();
        v.distance_scale = 1.0;
        v
    })
    .unwrap();
    let c2 = view.centroid().unwrap();
    for axis in 0..3 {
        assert!((c2[axis] - c1[axis] * 2.0).abs() < 1e-4);
    }
}

#[test]
fn edge_color_override_tints_the_edges() {
    let mut view = view_with_graph();
    view.edge_color_enabled = true;
    view.edge_color = [1.0, 0.0, 0.0];
    let data = view.build_render_data();
    for v in &data.edge_vertices {
        assert!(v.color[0] > v.color[2], "override edges read red, not blue");
    }
}

#[test]
fn accent_override_beats_per_node_colors() {
    let mut view = view_with_graph();
    view.accent_enabled = true;
    view.accent_color = [0.0, 1.0, 0.0];
    let data = view.build_render_data();
    // Every node gets the same accent-derived color (no red outlier).
    let reds: Vec<f32> = data.node_instances.iter().map(|i| i.color[0]).collect();
    let (min_r, max_r) = reds
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &r| (lo.min(r), hi.max(r)));
    assert!(max_r - min_r < 1e-4);
}

#[test]
fn data_graph_flag_gates_layout_and_table() {
    let g = sample_graph();
    assert!(g.data_graph);
    // layout() must not disturb the imported/graph-laid positions.
    let before: Vec<[f32; 3]> = g.layers[0].nodes.iter().map(|n| n.position).collect();
    let mut g2 = g;
    g2.layout();
    let after: Vec<[f32; 3]> = g2.layers[0].nodes.iter().map(|n| n.position).collect();
    assert_eq!(before, after);
}
