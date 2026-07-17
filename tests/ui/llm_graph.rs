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


/// Flatten the per-geometry instance batches (single-shape graphs produce
/// exactly one batch, but the accessor keeps the tests shape-agnostic).
fn all_instances(
    data: &rendering_3d::ui::llm_panel::LlmRenderData,
) -> Vec<&rendering_3d::model::InstanceRaw> {
    data.node_batches.iter().flat_map(|(_, v)| v.iter()).collect()
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
    assert_eq!(all_instances(&data).len(), 3);
    assert_eq!(data.edge_vertices.len(), 4, "2 edges × 2 line vertices");
    assert!(data.edge_solid_instances.is_empty());
    // The per-node file color (red, dimmed by the passive factor) survives
    // into the instance stream: node a is redder than the layer-colored rest.
    let reds: Vec<f32> = all_instances(&data).iter().map(|i| i.color[0]).collect();
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
    for (a, b) in all_instances(&base).iter().zip(all_instances(&scaled)) {
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
    let reds: Vec<f32> = all_instances(&data).iter().map(|i| i.color[0]).collect();
    let (min_r, max_r) = reds
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &r| (lo.min(r), hi.max(r)));
    assert!(max_r - min_r < 1e-4);
}

#[test]
fn selecting_a_node_highlights_it_and_its_neighbors() {
    let mut view = view_with_graph();
    // Find node "b" (the middle of the a-b-c chain) and select it.
    let bi = view
        .graph
        .as_ref()
        .unwrap()
        .layers[0]
        .nodes
        .iter()
        .position(|n| n.label.as_deref() == Some("b"))
        .unwrap();
    view.selected_node = Some((0, bi));
    let data = view.build_render_data();

    // The selected node uses the golden selected-object shader path
    // (alpha = 2.0) and is drawn larger; the others stay at alpha 1.0.
    let alphas: Vec<f32> = all_instances(&data).iter().map(|i| i.color[3]).collect();
    assert_eq!(alphas.iter().filter(|&&a| (a - 2.0).abs() < 1e-4).count(), 1);
    assert_eq!(alphas.iter().filter(|&&a| (a - 1.0).abs() < 1e-4).count(), 2);
    // Its neighbors (a and c) render brighter than they would unselected.
    view.selected_node = None;
    let plain = view.build_render_data();
    let sum = |d: &rendering_3d::ui::llm_panel::LlmRenderData| -> f32 {
        all_instances(d).iter().map(|i| i.color[0] + i.color[1] + i.color[2]).sum()
    };
    assert!(sum(&data) > sum(&plain), "selection brightens the neighborhood");
}

#[test]
fn graph_nodes_carry_internal_data_and_neighbors() {
    let text = r#"<gexf><graph>
  <attributes class="node"><attribute id="0" title="city"/></attributes>
  <nodes>
    <node id="a" label="Alpha"><attvalues><attvalue for="0" value="Roma"/></attvalues></node>
    <node id="b" label="Beta"/>
  </nodes>
  <edges><edge source="a" target="b"/></edges>
</graph></gexf>"#;
    let g = graph_import::load_graph(text.as_bytes(), "gexf", "t").unwrap();
    let a = g.layers[0]
        .nodes
        .iter()
        .position(|n| n.label.as_deref() == Some("Alpha"))
        .unwrap();
    assert_eq!(
        g.layers[0].nodes[a].attrs,
        vec![("city".to_string(), "Roma".to_string())]
    );
    let nb = g.neighbors(0, a);
    assert_eq!(nb.len(), 1);
    assert_eq!(
        g.layers[0].nodes[nb[0].1].label.as_deref(),
        Some("Beta"),
        "neighbor navigation reaches the connected node"
    );
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

#[test]
fn database_import_renders_one_batch_per_table_shape() {
    let dump = r#"
CREATE TABLE users (id INT PRIMARY KEY, name TEXT);
CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id));
INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');
INSERT INTO orders (id, user_id) VALUES (10, 1), (11, 2), (12, 1);
"#;
    let g = rendering_3d::llm::db_import::load_db(dump.as_bytes(), "sql", "shop").unwrap();
    let mut view = LlmView::new();
    view.load_preset(g);
    let data = view.build_render_data();

    // Two tables → two distinct-geometry batches (sphere + cube), 5 nodes total.
    assert_eq!(data.node_batches.len(), 2);
    let geoms: Vec<_> = data.node_batches.iter().map(|(g, _)| *g).collect();
    assert!(geoms.windows(2).all(|w| w[0] != w[1]));
    assert_eq!(all_instances(&data).len(), 5);
    // The three FK edges reach the render stream too.
    assert_eq!(data.edge_vertices.len(), 6, "3 edges × 2 line vertices");

    // Picking still works across per-table layers: aim a ray at Alice.
    let graph = view.graph.as_ref().unwrap();
    let (li, ni) = (0usize, 0usize);
    let p = graph.layers[li].nodes[ni].position;
    let origin = cgmath::Vector3::new(p[0], p[1], p[2] + 10.0);
    let dir = cgmath::Vector3::new(0.0, 0.0, -1.0);
    assert!(view.pick_node(origin, dir));
    assert!(view.selected_node.is_some());
}

#[test]
fn database_nodes_expose_row_data_and_relations_in_inspector() {
    let dump = r#"
CREATE TABLE users (id INT PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (id INT PRIMARY KEY, user_id INT REFERENCES users(id), item TEXT);
INSERT INTO users (id, name, city) VALUES (1, 'Alice', 'Roma');
INSERT INTO orders (id, user_id, item) VALUES (10, 1, 'Book');
"#;
    let g = rendering_3d::llm::db_import::load_db(dump.as_bytes(), "sql", "shop").unwrap();
    let users_li = g.layers.iter().position(|l| l.name == "users").unwrap();
    let alice_ni = g.layers[users_li]
        .nodes
        .iter()
        .position(|n| n.label.as_deref() == Some("Alice"))
        .unwrap();
    let alice = &g.layers[users_li].nodes[alice_ni];
    assert!(alice.attrs.iter().any(|(k, v)| k == "city" && v == "Roma"));
    // Neighbor navigation crosses table layers via the FK edge.
    let nb = g.neighbors(users_li, alice_ni);
    assert_eq!(nb.len(), 1);
    let (nl, nn, _, _) = nb[0];
    assert_eq!(g.layers[nl].name, "orders");
    assert!(g.layers[nl].nodes[nn]
        .attrs
        .iter()
        .any(|(k, v)| k == "item" && v == "Book"));
}

#[test]
fn data_graphs_neutralize_model_only_view_state() {
    // Compact view is a transformer-block filter left over from a model
    // session; installing a data graph must reset it, or whole database
    // tables would silently disappear from the 3D view.
    let dump = r#"
CREATE TABLE a (id INT PRIMARY KEY);
CREATE TABLE b (id INT PRIMARY KEY, a_id INT REFERENCES a(id));
CREATE TABLE c (id INT PRIMARY KEY, b_id INT REFERENCES b(id));
CREATE TABLE d (id INT PRIMARY KEY);
INSERT INTO a (id) VALUES (1);
INSERT INTO b (id, a_id) VALUES (1, 1);
INSERT INTO c (id, b_id) VALUES (1, 1);
INSERT INTO d (id) VALUES (1);
"#;
    let g = rendering_3d::llm::db_import::load_db(dump.as_bytes(), "sql", "s").unwrap();
    let mut view = LlmView::new();
    view.compact_view = true; // stale state from a previous model session
    view.load_preset(g);
    assert!(!view.compact_view, "installing a data graph resets compact view");
    let data = view.build_render_data();
    assert_eq!(
        all_instances(&data).len(),
        4,
        "every table row renders — none hidden by the block filter"
    );
}

#[test]
fn sampling_note_reaches_the_ui_state() {
    use rendering_3d::llm::db_import::{tables_to_network, DbColumn, DbTable};
    let t = DbTable {
        name: "big".into(),
        columns: vec![DbColumn { name: "id".into(), ty: "INT".into() }],
        pk: Some(0),
        fks: vec![],
        rows: (0..rendering_3d::llm::graph_import::MAX_GRAPH_NODES + 10)
            .map(|i| vec![Some(i.to_string())])
            .collect(),
    };
    let g = tables_to_network(vec![t], "big").unwrap();
    let mut view = LlmView::new();
    view.load_preset(g);
    let note = view.graph.as_ref().unwrap().note.as_ref().unwrap();
    assert!(note.contains("sampled"));
}
