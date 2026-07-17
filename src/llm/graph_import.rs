//! Import of real graph files into the network visualizer.
//!
//! Supported formats (the common interchange set):
//! - **Node-link JSON** (NetworkX `json_graph.node_link_data`)
//! - **Cytoscape JSON** (`.cyjs`, both `elements: {nodes, edges}` and the
//!   flat `elements: [...]` variant)
//! - **GraphML** (`.graphml`)
//! - **GEXF** (`.gexf`, including `viz:color` / `viz:position` / `viz:size`)
//! - **GDF** (`.gdf`, Gephi/GUESS)
//! - **DOT** (`.dot` / `.gv`, Graphviz — the common statement subset)
//! - **TGF** (`.tgf`, Trivial Graph Format)
//! - **Pajek** (`.net` / `.paj`)
//! - **SIF** (`.sif`, Simple Interaction Format)
//! - **JSON-LD / RDF·JSON** (`.jsonld` / `.rj`)
//!
//! Every parser fills the same [`GraphBuilder`] (ids, labels, optional
//! positions / sizes / colors, weighted edges), which then becomes a
//! [`NetworkGraph`] with a single [`LayerKind::Graph`] layer and real edges:
//! - positions from the file are used when present (normalized to the view),
//!   otherwise a deterministic 3D force-directed layout (small graphs) or a
//!   Fibonacci-sphere layout (large graphs) is computed;
//! - node sizes map to `weight_magnitude` (degree-based when absent), so the
//!   existing node-scale slider works unchanged;
//! - per-node colors land in [`Node::color`] and remain editable from the
//!   nodes table in the network window.

use std::collections::HashMap;

use crate::dataset::builtin::Rng;
use crate::llm::network::{Edge, Layer, LayerKind, NetworkGraph, Node};

/// Hard caps: beyond this the file is refused instead of freezing the app.
pub const MAX_GRAPH_NODES: usize = 200_000;
pub const MAX_GRAPH_EDGES: usize = 1_000_000;

/// Force-directed layout is O(n²) per iteration; larger graphs fall back to
/// the Fibonacci sphere.
const FORCE_LAYOUT_MAX_NODES: usize = 1_000;

/// Target bounding radius of the laid-out graph in world units.
const LAYOUT_RADIUS: f32 = 8.0;

/// File extensions routed to this module (besides shape-sniffed `.json`).
pub fn is_graph_extension(ext: &str) -> bool {
    matches!(
        ext,
        "cyjs" | "graphml" | "gexf" | "gdf" | "dot" | "gv" | "tgf" | "net" | "paj" | "sif"
            | "jsonld" | "rj"
    )
}

/// Parse a graph file by extension. `.json` callers should use
/// [`from_json_auto`] after the neural-network JSON loader declined.
pub fn load_graph(bytes: &[u8], ext: &str, name: &str) -> Result<NetworkGraph, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "graph file is not UTF-8".to_string())?;
    let builder = match ext {
        "cyjs" => parse_cyjs(text)?,
        "graphml" => parse_graphml(text)?,
        "gexf" => parse_gexf(text)?,
        "gdf" => parse_gdf(text)?,
        "dot" | "gv" => parse_dot(text)?,
        "tgf" => parse_tgf(text)?,
        "net" | "paj" => parse_pajek(text)?,
        "sif" => parse_sif(text)?,
        "jsonld" => parse_jsonld(text)?,
        "rj" => parse_rdf_json(text)?,
        other => return Err(format!("unsupported graph extension '.{other}'")),
    };
    builder.into_network(name)
}

/// Shape-sniffing entry for `.json` files: node-link (NetworkX), Cytoscape,
/// JSON-LD and RDF·JSON all commonly ship as plain `.json`.
pub fn from_json_auto(bytes: &[u8], name: &str) -> Result<NetworkGraph, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "graph file is not UTF-8".to_string())?;
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let builder = if value.get("elements").is_some() {
        parse_cyjs(text)?
    } else if value.get("nodes").is_some() {
        parse_node_link(text)?
    } else if value.get("@graph").is_some() || value.get("@context").is_some() {
        parse_jsonld(text)?
    } else if looks_like_rdf_json(&value) {
        parse_rdf_json(text)?
    } else {
        return Err(
            "unrecognized JSON graph shape (expected node-link 'nodes'/'links', Cytoscape \
             'elements', JSON-LD '@graph' or RDF/JSON triples)"
                .to_string(),
        );
    };
    builder.into_network(name)
}

// ---------------------------------------------------------------------------
// Shared builder + layout
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct GNode {
    label: Option<String>,
    pos: Option<[f32; 3]>,
    size: Option<f32>,
    color: Option<[f32; 3]>,
    /// Unmapped file attributes, preserved for the node inspector.
    attrs: Vec<(String, String)>,
}

#[derive(Default)]
pub struct GraphBuilder {
    nodes: Vec<GNode>,
    index: HashMap<String, usize>,
    /// (from, to, weight) by node index.
    edges: Vec<(usize, usize, f32)>,
}

impl GraphBuilder {
    /// Get or insert the node with this id.
    fn node(&mut self, id: &str) -> usize {
        if let Some(&i) = self.index.get(id) {
            return i;
        }
        let i = self.nodes.len();
        self.index.insert(id.to_string(), i);
        self.nodes.push(GNode {
            label: Some(id.to_string()),
            ..Default::default()
        });
        i
    }

    fn edge(&mut self, from: &str, to: &str, weight: f32) {
        let (f, t) = (self.node(from), self.node(to));
        self.edges.push((f, t, weight));
    }

    fn set_label(&mut self, id: &str, label: &str) {
        let i = self.node(id);
        self.nodes[i].label = Some(label.to_string());
    }

    fn set_pos(&mut self, id: &str, pos: [f32; 3]) {
        let i = self.node(id);
        self.nodes[i].pos = Some(pos);
    }

    fn set_size(&mut self, id: &str, size: f32) {
        let i = self.node(id);
        self.nodes[i].size = Some(size);
    }

    fn set_color(&mut self, id: &str, color: [f32; 3]) {
        let i = self.node(id);
        self.nodes[i].color = Some(color);
    }

    /// Attach one internal-data attribute (unmapped file field) to a node.
    fn add_attr(&mut self, id: &str, key: &str, value: impl Into<String>) {
        let i = self.node(id);
        self.nodes[i].attrs.push((key.to_string(), value.into()));
    }

    /// Build the final [`NetworkGraph`]: normalize sizes, lay out positions,
    /// normalize edge weights.
    fn into_network(self, name: &str) -> Result<NetworkGraph, String> {
        let n = self.nodes.len();
        if n == 0 {
            return Err("graph file contains no nodes".to_string());
        }
        if n > MAX_GRAPH_NODES {
            return Err(format!("graph has {n} nodes (max {MAX_GRAPH_NODES})"));
        }
        if self.edges.len() > MAX_GRAPH_EDGES {
            return Err(format!(
                "graph has {} edges (max {MAX_GRAPH_EDGES})",
                self.edges.len()
            ));
        }

        // Degree drives the default node size (a classic network-viz default).
        let mut degree = vec![0usize; n];
        for &(f, t, _) in &self.edges {
            degree[f] += 1;
            degree[t] += 1;
        }
        let max_deg = degree.iter().copied().max().unwrap_or(0).max(1) as f32;
        let (size_min, size_max) = self
            .nodes
            .iter()
            .filter_map(|g| g.size)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
                (lo.min(s), hi.max(s))
            });
        let size_span = (size_max - size_min).max(1e-9);

        let positions = layout_positions(&self.nodes, &self.edges);

        let nodes: Vec<Node> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let weight = match g.size {
                    // File sizes → [0.15, 1.0].
                    Some(s) if size_max > size_min => 0.15 + 0.85 * (s - size_min) / size_span,
                    Some(_) => 0.6,
                    // No size attribute: sqrt-scaled degree reads well.
                    None => 0.2 + 0.8 * (degree[i] as f32 / max_deg).sqrt(),
                };
                Node {
                    position: positions[i],
                    weight_magnitude: weight.clamp(0.05, 1.0),
                    color: g.color,
                    label: g.label.clone(),
                    attrs: g.attrs.clone(),
                }
            })
            .collect();

        let max_w = self
            .edges
            .iter()
            .map(|&(_, _, w)| w.abs())
            .fold(0.0f32, f32::max)
            .max(1e-8);
        let edges = self
            .edges
            .iter()
            .map(|&(f, t, w)| Edge {
                from_layer: 0,
                from_node: f,
                to_layer: 0,
                to_node: t,
                // Floor keeps light edges visible against the background.
                importance: (w.abs() / max_w).clamp(0.15, 1.0),
            })
            .collect();

        Ok(NetworkGraph {
            name: name.to_string(),
            layers: vec![Layer {
                name: "nodes".to_string(),
                kind: LayerKind::Graph,
                nodes,
                shape: None,
            }],
            edges,
            estimated_vram_gb: None,
            moe_config: None,
            data_graph: true,
            note: None,
        })
    }
}

/// Positions for every node: file coordinates when at least half the nodes
/// have them, otherwise force-directed (small) / Fibonacci sphere (large).
/// The result is always centered and normalized to [`LAYOUT_RADIUS`].
fn layout_positions(nodes: &[GNode], edges: &[(usize, usize, f32)]) -> Vec<[f32; 3]> {
    let n = nodes.len();
    let with_pos = nodes.iter().filter(|g| g.pos.is_some()).count();
    let mut pts: Vec<[f32; 3]> = if with_pos * 2 >= n && with_pos > 0 {
        // Use the file layout; park position-less nodes on a small jitter so
        // they don't stack at the exact origin.
        let mut rng = Rng::new(7);
        nodes
            .iter()
            .map(|g| {
                g.pos.unwrap_or([
                    (rng.next_f32() - 0.5) * 0.2,
                    (rng.next_f32() - 0.5) * 0.2,
                    (rng.next_f32() - 0.5) * 0.2,
                ])
            })
            .collect()
    } else if n <= FORCE_LAYOUT_MAX_NODES {
        force_layout(n, edges)
    } else {
        fibonacci_sphere(n)
    };

    // Center on the origin and scale the bounding radius to LAYOUT_RADIUS.
    let mut c = [0.0f64; 3];
    for p in &pts {
        for a in 0..3 {
            c[a] += p[a] as f64;
        }
    }
    for a in &mut c {
        *a /= n as f64;
    }
    let mut r_max = 0.0f32;
    for p in &mut pts {
        for a in 0..3 {
            p[a] -= c[a] as f32;
        }
        r_max = r_max.max((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt());
    }
    let s = LAYOUT_RADIUS / r_max.max(1e-6);
    for p in &mut pts {
        for a in p.iter_mut() {
            *a *= s;
        }
    }
    pts
}

/// Deterministic 3D Fruchterman–Reingold: repulsion between all pairs,
/// attraction along edges, cooling step. O(n² · iters) — capped by
/// [`FORCE_LAYOUT_MAX_NODES`].
fn force_layout(n: usize, edges: &[(usize, usize, f32)]) -> Vec<[f32; 3]> {
    let mut pts = fibonacci_sphere(n);
    if n < 2 {
        return pts;
    }
    let k = 1.0f32; // ideal edge length in layout space
    let iters = (200_000 / n).clamp(30, 150);
    let mut temp = 0.8f32;
    let cool = 0.96f32;

    let mut disp = vec![[0.0f32; 3]; n];
    for _ in 0..iters {
        for d in disp.iter_mut() {
            *d = [0.0; 3];
        }
        // Repulsion (all pairs).
        for i in 0..n {
            for j in (i + 1)..n {
                let mut d = [0.0f32; 3];
                let mut dist2 = 0.0;
                for a in 0..3 {
                    d[a] = pts[i][a] - pts[j][a];
                    dist2 += d[a] * d[a];
                }
                let dist = dist2.sqrt().max(1e-4);
                let f = k * k / dist;
                for a in 0..3 {
                    let push = d[a] / dist * f;
                    disp[i][a] += push;
                    disp[j][a] -= push;
                }
            }
        }
        // Attraction (edges).
        for &(u, v, _) in edges {
            if u == v || u >= n || v >= n {
                continue;
            }
            let mut d = [0.0f32; 3];
            let mut dist2 = 0.0;
            for a in 0..3 {
                d[a] = pts[u][a] - pts[v][a];
                dist2 += d[a] * d[a];
            }
            let dist = dist2.sqrt().max(1e-4);
            let f = dist * dist / k;
            for a in 0..3 {
                let pull = d[a] / dist * f;
                disp[u][a] -= pull;
                disp[v][a] += pull;
            }
        }
        // Displace, limited by temperature.
        for i in 0..n {
            let len = (disp[i][0] * disp[i][0]
                + disp[i][1] * disp[i][1]
                + disp[i][2] * disp[i][2])
                .sqrt()
                .max(1e-6);
            let step = len.min(temp);
            for a in 0..3 {
                pts[i][a] += disp[i][a] / len * step;
            }
        }
        temp *= cool;
    }
    pts
}

/// Evenly distributed points on a sphere (deterministic, O(n)). Shared with
/// the database importer's cluster layout.
pub(crate) fn fibonacci_sphere(n: usize) -> Vec<[f32; 3]> {
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    (0..n)
        .map(|i| {
            if n == 1 {
                return [0.0, 0.0, 0.0];
            }
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * i as f32;
            [r * theta.cos(), y, r * theta.sin()]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Small shared parsing helpers
// ---------------------------------------------------------------------------

/// Parse `#rgb`, `#rrggbb` or `r,g,b` (0–255 ints) into linear-ish RGB.
fn parse_color(s: &str) -> Option<[f32; 3]> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix('#') {
        let v = u32::from_str_radix(hex, 16).ok()?;
        return Some(match hex.len() {
            3 => [
                ((v >> 8) & 0xF) as f32 / 15.0,
                ((v >> 4) & 0xF) as f32 / 15.0,
                (v & 0xF) as f32 / 15.0,
            ],
            6 => [
                ((v >> 16) & 0xFF) as f32 / 255.0,
                ((v >> 8) & 0xFF) as f32 / 255.0,
                (v & 0xFF) as f32 / 255.0,
            ],
            _ => return None,
        });
    }
    let parts: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .ok()?;
    if parts.len() == 3 {
        let scale = if parts.iter().any(|&v| v > 1.0) { 255.0 } else { 1.0 };
        return Some([parts[0] / scale, parts[1] / scale, parts[2] / scale]);
    }
    None
}

/// JSON color: "#rrggbb" string or [r,g,b] array (0–1 or 0–255).
fn json_color(v: &serde_json::Value) -> Option<[f32; 3]> {
    match v {
        serde_json::Value::String(s) => parse_color(s),
        serde_json::Value::Array(a) if a.len() == 3 => {
            let vals: Vec<f32> = a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
            if vals.len() != 3 {
                return None;
            }
            let scale = if vals.iter().any(|&v| v > 1.0) { 255.0 } else { 1.0 };
            Some([vals[0] / scale, vals[1] / scale, vals[2] / scale])
        }
        _ => None,
    }
}

/// A JSON id may be a string or a number; normalize to a string key.
fn json_id(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn f32_of(v: &serde_json::Value) -> Option<f32> {
    v.as_f64().map(|f| f as f32)
}

/// Human string for a JSON attribute value (scalars verbatim, containers as
/// compact JSON). None for null.
fn json_attr_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        other => Some(other.to_string()),
    }
}

/// Local name of an IRI (`http://…#label` → `label`, `foaf/name` → `name`).
fn iri_local_name(iri: &str) -> &str {
    iri.rsplit(['#', '/', ':']).next().unwrap_or(iri)
}

// ── Minimal XML scanning (same approach as the VTK XML importer) ──

/// All `<tag ...>inner</tag>` / `<tag .../>` occurrences in `text`, as
/// (attribute region, inner) pairs. Ignores namespace prefixes on lookup:
/// pass the local name and prefixed tags (`viz:color`) match exactly.
fn xml_elements<'a>(text: &'a str, tag: &str) -> Vec<(&'a str, &'a str)> {
    let mut out = Vec::new();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut from = 0;
    while let Some(rel) = text[from..].find(&open) {
        let at = from + rel;
        let after = at + open.len();
        let next = text.as_bytes().get(after).copied();
        if !matches!(next, Some(b) if b.is_ascii_whitespace() || b == b'>' || b == b'/') {
            from = after;
            continue;
        }
        let Some(tag_end_rel) = text[at..].find('>') else { break };
        let tag_end = at + tag_end_rel;
        let attrs = text[after..tag_end].trim_end_matches('/');
        if text[..tag_end].ends_with('/') {
            out.push((attrs, ""));
            from = tag_end + 1;
        } else if let Some(end_rel) = text[tag_end + 1..].find(&close) {
            let inner = &text[tag_end + 1..tag_end + 1 + end_rel];
            out.push((attrs, inner));
            from = tag_end + 1 + end_rel + close.len();
        } else {
            out.push((attrs, ""));
            from = tag_end + 1;
        }
    }
    out
}

/// Attribute lookup in an XML open-tag attribute region.
fn xml_attr(attrs: &str, key: &str) -> Option<String> {
    let bytes = attrs.as_bytes();
    let mut from = 0;
    while let Some(rel) = attrs[from..].find(key) {
        let at = from + rel;
        let before_ok = at == 0 || !bytes[at - 1].is_ascii_alphanumeric() && bytes[at - 1] != b':';
        let mut p = at + key.len();
        while p < attrs.len() && bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        if before_ok && p < attrs.len() && bytes[p] == b'=' {
            p += 1;
            while p < attrs.len() && bytes[p].is_ascii_whitespace() {
                p += 1;
            }
            if p < attrs.len() && (bytes[p] == b'"' || bytes[p] == b'\'') {
                let q = bytes[p];
                let start = p + 1;
                if let Some(end) = attrs[start..].find(q as char) {
                    return Some(xml_unescape(&attrs[start..start + end]));
                }
            }
        }
        from = at + key.len();
    }
    None
}

fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

// ---------------------------------------------------------------------------
// 1. Node-link JSON (NetworkX)
// ---------------------------------------------------------------------------

fn parse_node_link(text: &str) -> Result<GraphBuilder, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let mut b = GraphBuilder::default();

    let nodes = v
        .get("nodes")
        .and_then(|n| n.as_array())
        .ok_or("node-link JSON is missing the 'nodes' array")?;
    // NetworkX links may reference nodes by array index instead of id.
    let mut id_of_index: Vec<String> = Vec::with_capacity(nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let id = node
            .get("id")
            .and_then(json_id)
            .unwrap_or_else(|| i.to_string());
        id_of_index.push(id.clone());
        b.node(&id);
        if let Some(l) = node.get("label").or_else(|| node.get("name")).and_then(|v| v.as_str()) {
            b.set_label(&id, l);
        }
        let (x, y, z) = (
            node.get("x").and_then(f32_of),
            node.get("y").and_then(f32_of),
            node.get("z").and_then(f32_of),
        );
        if let (Some(x), Some(y)) = (x, y) {
            b.set_pos(&id, [x, y, z.unwrap_or(0.0)]);
        }
        if let Some(s) = node.get("size").or_else(|| node.get("value")).and_then(f32_of) {
            b.set_size(&id, s);
        }
        if let Some(c) = node.get("color").and_then(json_color) {
            b.set_color(&id, c);
        }
        // Every unmapped key is internal data for the inspector.
        if let Some(obj) = node.as_object() {
            for (key, value) in obj {
                if matches!(
                    key.as_str(),
                    "id" | "label" | "name" | "x" | "y" | "z" | "size" | "value" | "color"
                ) {
                    continue;
                }
                if let Some(s) = json_attr_string(value) {
                    b.add_attr(&id, key, s);
                }
            }
        }
    }

    let links = v
        .get("links")
        .or_else(|| v.get("edges"))
        .and_then(|l| l.as_array());
    for link in links.into_iter().flatten() {
        let end = |key: &str| -> Option<String> {
            let raw = link.get(key)?;
            let as_id = json_id(raw)?;
            // An integer endpoint that is not a known id is an array index.
            if !b.index.contains_key(&as_id) {
                if let Some(idx) = raw.as_u64() {
                    if let Some(id) = id_of_index.get(idx as usize) {
                        return Some(id.clone());
                    }
                }
            }
            Some(as_id)
        };
        let (Some(s), Some(t)) = (end("source"), end("target")) else {
            continue;
        };
        let w = link.get("weight").and_then(f32_of).unwrap_or(1.0);
        b.edge(&s, &t, w);
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 2. Cytoscape JSON (.cyjs)
// ---------------------------------------------------------------------------

fn parse_cyjs(text: &str) -> Result<GraphBuilder, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let elements = v.get("elements").unwrap_or(&v);
    let mut b = GraphBuilder::default();

    let mut handle = |el: &serde_json::Value, is_edge_hint: Option<bool>| {
        let Some(data) = el.get("data") else { return };
        let is_edge = is_edge_hint
            .unwrap_or_else(|| data.get("source").is_some() && data.get("target").is_some());
        if is_edge {
            let (Some(s), Some(t)) = (
                data.get("source").and_then(json_id),
                data.get("target").and_then(json_id),
            ) else {
                return;
            };
            let w = data.get("weight").and_then(f32_of).unwrap_or(1.0);
            b.edge(&s, &t, w);
        } else {
            let Some(id) = data.get("id").and_then(json_id) else { return };
            b.node(&id);
            if let Some(l) = data
                .get("label")
                .or_else(|| data.get("name"))
                .and_then(|v| v.as_str())
            {
                b.set_label(&id, l);
            }
            if let Some(pos) = el.get("position") {
                if let (Some(x), Some(y)) = (
                    pos.get("x").and_then(f32_of),
                    pos.get("y").and_then(f32_of),
                ) {
                    // Cytoscape y grows downwards; flip so layouts read upright.
                    b.set_pos(&id, [x, -y, pos.get("z").and_then(f32_of).unwrap_or(0.0)]);
                }
            }
            if let Some(s) = data.get("size").and_then(f32_of) {
                b.set_size(&id, s);
            }
            if let Some(c) = data.get("color").and_then(json_color) {
                b.set_color(&id, c);
            }
            if let Some(obj) = data.as_object() {
                for (key, value) in obj {
                    if matches!(key.as_str(), "id" | "label" | "name" | "size" | "color") {
                        continue;
                    }
                    if let Some(s) = json_attr_string(value) {
                        b.add_attr(&id, key, s);
                    }
                }
            }
        }
    };

    if let Some(arr) = elements.as_array() {
        // Flat variant: group tells nodes from edges.
        for el in arr {
            let hint = el.get("group").and_then(|g| g.as_str()).map(|g| g == "edges");
            handle(el, hint);
        }
    } else {
        for el in elements
            .get("nodes")
            .and_then(|n| n.as_array())
            .into_iter()
            .flatten()
        {
            handle(el, Some(false));
        }
        for el in elements
            .get("edges")
            .and_then(|e| e.as_array())
            .into_iter()
            .flatten()
        {
            handle(el, Some(true));
        }
    }
    if b.nodes.is_empty() {
        return Err("Cytoscape JSON has no elements".to_string());
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 3. GraphML
// ---------------------------------------------------------------------------

fn parse_graphml(text: &str) -> Result<GraphBuilder, String> {
    // <key id="d0" for="node" attr.name="label"/> declarations map the
    // per-element <data key="d0"> values onto known attributes.
    let mut key_name: HashMap<String, String> = HashMap::new();
    for (attrs, _) in xml_elements(text, "key") {
        if let (Some(id), Some(name)) = (xml_attr(attrs, "id"), xml_attr(attrs, "attr.name")) {
            key_name.insert(id, name.to_ascii_lowercase());
        }
    }
    let data_of = |inner: &str| -> HashMap<String, String> {
        let mut out = HashMap::new();
        for (attrs, value) in xml_elements(inner, "data") {
            if let Some(k) = xml_attr(attrs, "key") {
                let name = key_name.get(&k).cloned().unwrap_or(k);
                out.insert(name, xml_unescape(value.trim()));
            }
        }
        out
    };

    let mut b = GraphBuilder::default();
    for (attrs, inner) in xml_elements(text, "node") {
        let Some(id) = xml_attr(attrs, "id") else { continue };
        b.node(&id);
        let data = data_of(inner);
        if let Some(l) = data.get("label").or_else(|| data.get("name")) {
            b.set_label(&id, l);
        }
        if let (Some(x), Some(y)) = (
            data.get("x").and_then(|v| v.parse::<f32>().ok()),
            data.get("y").and_then(|v| v.parse::<f32>().ok()),
        ) {
            let z = data.get("z").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
            b.set_pos(&id, [x, y, z]);
        }
        if let Some(s) = data.get("size").and_then(|v| v.parse::<f32>().ok()) {
            b.set_size(&id, s);
        }
        if let Some(c) = data.get("color").and_then(|v| parse_color(v)) {
            b.set_color(&id, c);
        } else if let (Some(r), Some(g), Some(bl)) = (
            data.get("r").and_then(|v| v.parse::<f32>().ok()),
            data.get("g").and_then(|v| v.parse::<f32>().ok()),
            data.get("b").and_then(|v| v.parse::<f32>().ok()),
        ) {
            b.set_color(&id, [r / 255.0, g / 255.0, bl / 255.0]);
        }
        for (key, value) in &data {
            if matches!(
                key.as_str(),
                "label" | "name" | "x" | "y" | "z" | "size" | "color" | "r" | "g" | "b"
            ) {
                continue;
            }
            b.add_attr(&id, key, value.clone());
        }
    }
    for (attrs, inner) in xml_elements(text, "edge") {
        let (Some(s), Some(t)) = (xml_attr(attrs, "source"), xml_attr(attrs, "target")) else {
            continue;
        };
        let w = data_of(inner)
            .get("weight")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0);
        b.edge(&s, &t, w);
    }
    if b.nodes.is_empty() {
        return Err("GraphML file has no <node> elements".to_string());
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 4. GEXF
// ---------------------------------------------------------------------------

fn parse_gexf(text: &str) -> Result<GraphBuilder, String> {
    // <attributes><attribute id=".." title=".."/></attributes> declarations
    // give the per-node <attvalue for=".."> entries their names.
    let mut attr_title: HashMap<String, String> = HashMap::new();
    for (aattrs, _) in xml_elements(text, "attribute") {
        if let Some(id) = xml_attr(aattrs, "id") {
            let title = xml_attr(aattrs, "title").unwrap_or_else(|| id.clone());
            attr_title.insert(id, title);
        }
    }

    let mut b = GraphBuilder::default();
    for (attrs, inner) in xml_elements(text, "node") {
        let Some(id) = xml_attr(attrs, "id") else { continue };
        b.node(&id);
        if let Some(l) = xml_attr(attrs, "label") {
            b.set_label(&id, &l);
        }
        for (vattrs, _) in xml_elements(inner, "attvalue") {
            let key = xml_attr(vattrs, "for").or_else(|| xml_attr(vattrs, "id"));
            let value = xml_attr(vattrs, "value");
            if let (Some(key), Some(value)) = (key, value) {
                let name = attr_title.get(&key).cloned().unwrap_or(key);
                b.add_attr(&id, &name, value);
            }
        }
        for (cattrs, _) in xml_elements(inner, "viz:color") {
            let ch = |k: &str| xml_attr(cattrs, k).and_then(|v| v.parse::<f32>().ok());
            if let (Some(r), Some(g), Some(bl)) = (ch("r"), ch("g"), ch("b")) {
                b.set_color(&id, [r / 255.0, g / 255.0, bl / 255.0]);
            }
        }
        for (pattrs, _) in xml_elements(inner, "viz:position") {
            let ch = |k: &str| xml_attr(pattrs, k).and_then(|v| v.parse::<f32>().ok());
            if let (Some(x), Some(y)) = (ch("x"), ch("y")) {
                b.set_pos(&id, [x, y, ch("z").unwrap_or(0.0)]);
            }
        }
        for (sattrs, _) in xml_elements(inner, "viz:size") {
            if let Some(v) = xml_attr(sattrs, "value").and_then(|v| v.parse::<f32>().ok()) {
                b.set_size(&id, v);
            }
        }
    }
    for (attrs, _) in xml_elements(text, "edge") {
        let (Some(s), Some(t)) = (xml_attr(attrs, "source"), xml_attr(attrs, "target")) else {
            continue;
        };
        let w = xml_attr(attrs, "weight")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0);
        b.edge(&s, &t, w);
    }
    if b.nodes.is_empty() {
        return Err("GEXF file has no <node> elements".to_string());
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 5. GDF
// ---------------------------------------------------------------------------

/// Split one GDF row on commas, honoring single-quoted values ('153,0,0').
fn gdf_split(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '\'' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out.iter().map(|s| s.trim().to_string()).collect()
}

fn parse_gdf(text: &str) -> Result<GraphBuilder, String> {
    #[derive(PartialEq)]
    enum Section {
        None,
        Nodes,
        Edges,
    }
    let mut b = GraphBuilder::default();
    let mut section = Section::None;
    let mut fields: Vec<String> = Vec::new();

    let field_names = |def: &str| -> Vec<String> {
        def.split(',')
            .map(|f| {
                f.split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase()
            })
            .collect()
    };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(def) = line.strip_prefix("nodedef>") {
            section = Section::Nodes;
            fields = field_names(def);
            continue;
        }
        if let Some(def) = line.strip_prefix("edgedef>") {
            section = Section::Edges;
            fields = field_names(def);
            continue;
        }
        let values = gdf_split(line);
        let get = |name: &str| -> Option<&String> {
            fields.iter().position(|f| f == name).and_then(|i| values.get(i))
        };
        match section {
            Section::Nodes => {
                let Some(id) = get("name").cloned() else { continue };
                b.node(&id);
                if let Some(l) = get("label") {
                    b.set_label(&id, l);
                }
                if let (Some(x), Some(y)) = (
                    get("x").and_then(|v| v.parse::<f32>().ok()),
                    get("y").and_then(|v| v.parse::<f32>().ok()),
                ) {
                    let z = get("z").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.0);
                    b.set_pos(&id, [x, y, z]);
                }
                if let Some(s) = get("width")
                    .or_else(|| get("size"))
                    .and_then(|v| v.parse::<f32>().ok())
                {
                    b.set_size(&id, s);
                }
                if let Some(c) = get("color").and_then(|v| parse_color(v)) {
                    b.set_color(&id, c);
                }
                for (fi, field) in fields.iter().enumerate() {
                    if matches!(
                        field.as_str(),
                        "name" | "label" | "x" | "y" | "z" | "width" | "size" | "color"
                    ) {
                        continue;
                    }
                    if let Some(v) = values.get(fi).filter(|v| !v.is_empty()) {
                        b.add_attr(&id, field, v.clone());
                    }
                }
            }
            Section::Edges => {
                let (Some(s), Some(t)) = (get("node1"), get("node2")) else { continue };
                let w = get("weight").and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
                b.edge(&s.clone(), &t.clone(), w);
            }
            Section::None => {}
        }
    }
    if b.nodes.is_empty() {
        return Err("GDF file has no nodedef> section (or it is empty)".to_string());
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 6. DOT (Graphviz)
// ---------------------------------------------------------------------------

/// Tokenize the common DOT subset: identifiers / quoted strings, punctuation,
/// `->` / `--` operators; strips `//`, `#` and `/* */` comments.
fn dot_tokens(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c == '/' && bytes.get(i + 1) == Some(&b'/') || c == '#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if c == '/' && bytes.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
        } else if c == '"' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b'"' {
                if bytes[j] == b'\\' {
                    j += 1;
                }
                j += 1;
            }
            out.push(text[start..j.min(text.len())].replace("\\\"", "\""));
            i = j + 1;
        } else if c == '-' && matches!(bytes.get(i + 1), Some(b'>') | Some(b'-')) {
            out.push(text[i..i + 2].to_string());
            i += 2;
        } else if "{}[];,=".contains(c) {
            out.push(c.to_string());
            i += 1;
        } else {
            let start = i;
            while i < bytes.len() {
                let ch = bytes[i] as char;
                if ch.is_whitespace() || "{}[];,=\"".contains(ch) || (ch == '-' && matches!(bytes.get(i + 1), Some(b'>') | Some(b'-'))) {
                    break;
                }
                i += 1;
            }
            out.push(text[start..i].to_string());
        }
    }
    out
}

/// Parse an optional `[key=value, ...]` attribute list starting at `i`.
fn dot_attrs(tokens: &[String], i: &mut usize) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if tokens.get(*i).map(String::as_str) != Some("[") {
        return out;
    }
    *i += 1;
    while *i < tokens.len() && tokens[*i] != "]" {
        let key = tokens[*i].to_ascii_lowercase();
        if tokens.get(*i + 1).map(String::as_str) == Some("=") {
            if let Some(v) = tokens.get(*i + 2) {
                out.insert(key, v.clone());
            }
            *i += 3;
        } else {
            *i += 1;
        }
        while tokens.get(*i).map(String::as_str) == Some(",")
            || tokens.get(*i).map(String::as_str) == Some(";")
        {
            *i += 1;
        }
    }
    *i += 1; // consume ']'
    out
}

fn parse_dot(text: &str) -> Result<GraphBuilder, String> {
    let tokens = dot_tokens(text);
    let mut b = GraphBuilder::default();
    let mut i = 0;
    let keywords = ["graph", "digraph", "subgraph", "node", "edge", "strict"];

    let apply_node_attrs = |b: &mut GraphBuilder, id: &str, attrs: &HashMap<String, String>| {
        if let Some(l) = attrs.get("label") {
            b.set_label(id, l);
        }
        if let Some(pos) = attrs.get("pos") {
            let nums: Vec<f32> = pos
                .trim_end_matches('!')
                .split(',')
                .filter_map(|v| v.trim().parse::<f32>().ok())
                .collect();
            if nums.len() >= 2 {
                b.set_pos(id, [nums[0], nums[1], nums.get(2).copied().unwrap_or(0.0)]);
            }
        }
        if let Some(c) = attrs
            .get("fillcolor")
            .or_else(|| attrs.get("color"))
            .and_then(|v| parse_color(v))
        {
            b.set_color(id, c);
        }
        if let Some(s) = attrs.get("width").and_then(|v| v.parse::<f32>().ok()) {
            b.set_size(id, s);
        }
        for (key, value) in attrs {
            if matches!(key.as_str(), "label" | "pos" | "fillcolor" | "color" | "width") {
                continue;
            }
            b.add_attr(id, key, value.clone());
        }
    };

    while i < tokens.len() {
        let tok = &tokens[i];
        let low = tok.to_ascii_lowercase();
        // Structure / keyword tokens: skip (braces are transparent; `node`/
        // `edge`/`graph` default-attribute statements consume their list).
        if ["{", "}", ";", ","].contains(&tok.as_str()) {
            i += 1;
            continue;
        }
        if keywords.contains(&low.as_str()) {
            i += 1;
            // Optional graph name, then its default attrs / opening brace.
            if i < tokens.len() && !["{", "[", ";"].contains(&tokens[i].as_str()) {
                i += 1;
            }
            let _ = dot_attrs(&tokens, &mut i);
            continue;
        }
        if tok == "[" {
            let _ = dot_attrs(&tokens, &mut i);
            continue;
        }
        // `id = value` (graph attribute): skip the assignment.
        if tokens.get(i + 1).map(String::as_str) == Some("=") {
            i += 3;
            continue;
        }
        // Node id, possibly with :port (dropped) and an edge chain.
        let id = tok.split(':').next().unwrap_or(tok).to_string();
        i += 1;
        let mut chain = vec![id];
        while matches!(tokens.get(i).map(String::as_str), Some("->") | Some("--")) {
            i += 1;
            match tokens.get(i) {
                // `a -> {b c}` subgraph endpoints are beyond this subset:
                // drop the edge but keep parsing the brace content as nodes.
                Some(next) if next != "{" => {
                    chain.push(next.split(':').next().unwrap_or(next).to_string());
                    i += 1;
                }
                _ => break,
            }
        }
        let attrs = dot_attrs(&tokens, &mut i);
        if chain.len() == 1 {
            b.node(&chain[0]);
            apply_node_attrs(&mut b, &chain[0], &attrs);
        } else {
            let w = attrs
                .get("weight")
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(1.0);
            for pair in chain.windows(2) {
                b.edge(&pair[0], &pair[1], w);
            }
        }
    }
    if b.nodes.is_empty() {
        return Err("DOT file declares no nodes or edges".to_string());
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 7. TGF
// ---------------------------------------------------------------------------

fn parse_tgf(text: &str) -> Result<GraphBuilder, String> {
    let mut b = GraphBuilder::default();
    let mut in_edges = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            in_edges = true;
            continue;
        }
        let mut parts = line.splitn(2, char::is_whitespace);
        let first = parts.next().unwrap_or("");
        let rest = parts.next().map(str::trim);
        if !in_edges {
            b.node(first);
            if let Some(label) = rest.filter(|r| !r.is_empty()) {
                b.set_label(first, label);
            }
        } else {
            let Some(rest) = rest else { continue };
            let mut ep = rest.splitn(2, char::is_whitespace);
            let to = ep.next().unwrap_or("");
            if to.is_empty() {
                continue;
            }
            // A numeric edge label doubles as the weight.
            let w = ep
                .next()
                .and_then(|l| l.trim().parse::<f32>().ok())
                .unwrap_or(1.0);
            b.edge(first, to, w);
        }
    }
    if b.nodes.is_empty() {
        return Err("TGF file has no nodes".to_string());
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 8. Pajek (.net / .paj)
// ---------------------------------------------------------------------------

fn parse_pajek(text: &str) -> Result<GraphBuilder, String> {
    #[derive(PartialEq)]
    enum Section {
        None,
        Vertices,
        Edges,
        EdgeList,
    }
    let mut b = GraphBuilder::default();
    let mut section = Section::None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        if let Some(marker) = line.strip_prefix('*') {
            let m = marker.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
            section = match m.as_str() {
                "vertices" => Section::Vertices,
                "edges" | "arcs" => Section::Edges,
                "edgeslist" | "arcslist" => Section::EdgeList,
                _ => Section::None, // *matrix and friends: skip
            };
            continue;
        }
        match section {
            Section::Vertices => {
                // `id "label with spaces" [x y [z]]` — label may be unquoted.
                let (id, rest) = match line.split_once(char::is_whitespace) {
                    Some((id, rest)) => (id, rest.trim()),
                    None => (line, ""),
                };
                b.node(id);
                let coords_part;
                if let Some(after_quote) = rest.strip_prefix('"') {
                    if let Some(endq) = after_quote.find('"') {
                        b.set_label(id, &after_quote[..endq]);
                        coords_part = after_quote[endq + 1..].trim();
                    } else {
                        coords_part = "";
                    }
                } else if !rest.is_empty() {
                    let mut it = rest.splitn(2, char::is_whitespace);
                    let label = it.next().unwrap_or("");
                    // A purely numeric first token is a coordinate, not a label.
                    if label.parse::<f32>().is_err() {
                        b.set_label(id, label);
                        coords_part = it.next().unwrap_or("").trim();
                    } else {
                        coords_part = rest;
                    }
                } else {
                    coords_part = "";
                }
                let nums: Vec<f32> = coords_part
                    .split_whitespace()
                    .filter_map(|v| v.parse::<f32>().ok())
                    .collect();
                if nums.len() >= 2 {
                    b.set_pos(id, [nums[0], nums[1], nums.get(2).copied().unwrap_or(0.0)]);
                }
            }
            Section::Edges => {
                let toks: Vec<&str> = line.split_whitespace().collect();
                if toks.len() >= 2 {
                    let w = toks.get(2).and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
                    b.edge(toks[0], toks[1], w);
                }
            }
            Section::EdgeList => {
                let mut toks = line.split_whitespace();
                if let Some(from) = toks.next() {
                    for to in toks {
                        b.edge(from, to, 1.0);
                    }
                }
            }
            Section::None => {}
        }
    }
    if b.nodes.is_empty() {
        return Err("Pajek file has no *Vertices section (or it is empty)".to_string());
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 9. SIF
// ---------------------------------------------------------------------------

fn parse_sif(text: &str) -> Result<GraphBuilder, String> {
    let mut b = GraphBuilder::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Tab-delimited when tabs are present (names may contain spaces).
        let toks: Vec<&str> = if line.contains('\t') {
            line.split('\t').map(str::trim).filter(|t| !t.is_empty()).collect()
        } else {
            line.split_whitespace().collect()
        };
        match toks.len() {
            0 => {}
            1 => {
                b.node(toks[0]);
            }
            2 => {
                // Malformed (source + relation, no target): keep the node.
                b.node(toks[0]);
            }
            _ => {
                // source relation target1 [target2 ...]
                for target in &toks[2..] {
                    b.edge(toks[0], target, 1.0);
                }
            }
        }
    }
    if b.nodes.is_empty() {
        return Err("SIF file has no interactions".to_string());
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// 10. JSON-LD / RDF·JSON
// ---------------------------------------------------------------------------

/// RDF/JSON triples: `{ subject: { predicate: [ {type, value} ] } }`.
fn looks_like_rdf_json(v: &serde_json::Value) -> bool {
    v.as_object().is_some_and(|obj| {
        !obj.is_empty()
            && obj.values().all(|preds| {
                preds.as_object().is_some_and(|p| {
                    p.values().all(|objs| {
                        objs.as_array().is_some_and(|arr| {
                            arr.iter().all(|o| o.get("type").is_some() && o.get("value").is_some())
                        })
                    })
                })
            })
    })
}

fn parse_rdf_json(text: &str) -> Result<GraphBuilder, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = v.as_object().ok_or("RDF/JSON root must be an object")?;
    let mut b = GraphBuilder::default();
    for (subject, preds) in obj {
        b.node(subject);
        let Some(preds) = preds.as_object() else { continue };
        for (pred, objects) in preds {
            for o in objects.as_array().into_iter().flatten() {
                let ty = o.get("type").and_then(|t| t.as_str()).unwrap_or("");
                let Some(value) = o.get("value").and_then(|t| t.as_str()) else { continue };
                match ty {
                    "uri" | "bnode" => b.edge(subject, value, 1.0),
                    _ => {
                        // Literal: a *label-ish predicate names the subject,
                        // everything else is internal data.
                        let low = pred.to_ascii_lowercase();
                        if low.ends_with("label") || low.ends_with("name") {
                            b.set_label(subject, value);
                        } else {
                            b.add_attr(subject, iri_local_name(pred), value);
                        }
                    }
                }
            }
        }
    }
    if b.nodes.is_empty() {
        return Err("RDF/JSON document has no triples".to_string());
    }
    Ok(b)
}

fn parse_jsonld(text: &str) -> Result<GraphBuilder, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let mut b = GraphBuilder::default();
    let mut blank = 0usize;

    // Walk objects: everything with an @id (or any nested object) is a node;
    // object-valued properties are edges labeled by the property.
    fn walk(
        v: &serde_json::Value,
        b: &mut GraphBuilder,
        blank: &mut usize,
    ) -> Option<String> {
        match v {
            serde_json::Value::Object(obj) => {
                let id = match obj.get("@id").and_then(|i| i.as_str()) {
                    Some(id) => id.to_string(),
                    None => {
                        // Pure value objects are literals, not nodes.
                        if obj.keys().all(|k| k.starts_with('@')) {
                            return None;
                        }
                        *blank += 1;
                        format!("_:b{blank}")
                    }
                };
                b.node(&id);
                for (key, value) in obj {
                    if key.starts_with('@') {
                        continue;
                    }
                    let klow = key.to_ascii_lowercase();
                    let is_label = klow == "name" || klow.ends_with("label");
                    let children: Vec<&serde_json::Value> = match value {
                        serde_json::Value::Array(a) => a.iter().collect(),
                        other => vec![other],
                    };
                    for child in children {
                        match child {
                            serde_json::Value::Object(_) => {
                                if let Some(cid) = walk(child, b, blank) {
                                    b.edge(&id, &cid, 1.0);
                                }
                            }
                            serde_json::Value::String(s) if is_label => {
                                b.set_label(&id, s);
                            }
                            // Literal property: keep it as internal data.
                            other => {
                                if let Some(s) = json_attr_string(other) {
                                    b.add_attr(&id, iri_local_name(key), s);
                                }
                            }
                        }
                    }
                }
                Some(id)
            }
            _ => None,
        }
    }

    match v.get("@graph").and_then(|g| g.as_array()) {
        Some(items) => {
            for item in items {
                walk(item, &mut b, &mut blank);
            }
        }
        None => match &v {
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item, &mut b, &mut blank);
                }
            }
            other => {
                walk(other, &mut b, &mut blank);
            }
        },
    }
    if b.nodes.is_empty() {
        return Err("JSON-LD document contains no nodes (@id objects)".to_string());
    }
    Ok(b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn net(builder: GraphBuilder) -> NetworkGraph {
        builder.into_network("test").unwrap()
    }

    fn label_index(g: &NetworkGraph, label: &str) -> usize {
        g.layers[0]
            .nodes
            .iter()
            .position(|n| n.label.as_deref() == Some(label))
            .unwrap_or_else(|| panic!("no node labeled '{label}'"))
    }

    #[test]
    fn node_link_json_with_positions_sizes_colors() {
        let text = r##"{
            "directed": false,
            "nodes": [
                {"id": "a", "label": "Alpha", "x": 0.0, "y": 0.0, "size": 1.0, "color": "#ff0000"},
                {"id": "b", "x": 10.0, "y": 0.0, "size": 3.0, "color": [0, 255, 0]},
                {"id": 3, "x": 0.0, "y": 10.0}
            ],
            "links": [
                {"source": "a", "target": "b", "weight": 2.0},
                {"source": "b", "target": 3}
            ]
        }"##;
        let g = from_json_auto(text.as_bytes(), "nl").unwrap();
        assert!(g.data_graph);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edges.len(), 2);
        let a = label_index(&g, "Alpha");
        assert_eq!(g.layers[0].nodes[a].color, Some([1.0, 0.0, 0.0]));
        let b = label_index(&g, "b");
        assert_eq!(g.layers[0].nodes[b].color, Some([0.0, 1.0, 0.0]));
        // Sizes 1 vs 3 must produce different node weights.
        assert!(g.layers[0].nodes[b].weight_magnitude > g.layers[0].nodes[a].weight_magnitude);
        // File positions are used (scaled): a-b along x, a-3 along y.
        let (pa, pb) = (g.layers[0].nodes[a].position, g.layers[0].nodes[b].position);
        assert!((pa[0] - pb[0]).abs() > 1.0, "file layout preserved");
    }

    #[test]
    fn node_link_edges_by_array_index() {
        let text = r#"{"nodes": [{"id": 10}, {"id": 20}], "links": [{"source": 0, "target": 1}]}"#;
        let g = from_json_auto(text.as_bytes(), "idx").unwrap();
        // 0/1 are not ids here, so they resolve as array indices 10 → 20.
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edges.len(), 1);
        assert_ne!(g.edges[0].from_node, g.edges[0].to_node);
    }

    #[test]
    fn cyjs_both_shapes() {
        let nested = r#"{"elements": {
            "nodes": [
                {"data": {"id": "n1", "name": "One"}, "position": {"x": 0, "y": 0}},
                {"data": {"id": "n2"}, "position": {"x": 100, "y": 50}}
            ],
            "edges": [{"data": {"source": "n1", "target": "n2", "weight": 0.5}}]
        }}"#;
        let g = load_graph(nested.as_bytes(), "cyjs", "c").unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.layers[0].nodes[label_index(&g, "One")].label.as_deref(), Some("One"));

        let flat = r#"{"elements": [
            {"group": "nodes", "data": {"id": "x"}},
            {"group": "nodes", "data": {"id": "y"}},
            {"group": "edges", "data": {"source": "x", "target": "y"}}
        ]}"#;
        let g = load_graph(flat.as_bytes(), "cyjs", "c").unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn graphml_with_key_mapped_attributes() {
        let text = r#"<?xml version="1.0"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="node" attr.name="label" attr.type="string"/>
  <key id="d1" for="node" attr.name="color" attr.type="string"/>
  <key id="d2" for="edge" attr.name="weight" attr.type="double"/>
  <graph id="G" edgedefault="undirected">
    <node id="n0"><data key="d0">Zero</data><data key="d1">#0000ff</data></node>
    <node id="n1"/>
    <edge source="n0" target="n1"><data key="d2">2.5</data></edge>
  </graph>
</graphml>"#;
        let g = load_graph(text.as_bytes(), "graphml", "gm").unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edges.len(), 1);
        let n0 = label_index(&g, "Zero");
        assert_eq!(g.layers[0].nodes[n0].color, Some([0.0, 0.0, 1.0]));
    }

    #[test]
    fn gexf_viz_attributes() {
        let text = r#"<?xml version="1.0"?>
<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz">
  <graph defaultedgetype="undirected">
    <nodes>
      <node id="0" label="Hello">
        <viz:color r="255" g="128" b="0"/>
        <viz:position x="15.0" y="40.0" z="0.0"/>
        <viz:size value="2.5"/>
      </node>
      <node id="1" label="World">
        <viz:position x="-15.0" y="-40.0" z="0.0"/>
      </node>
    </nodes>
    <edges><edge id="0" source="0" target="1" weight="3"/></edges>
  </graph>
</gexf>"#;
        let g = load_graph(text.as_bytes(), "gexf", "gx").unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edges.len(), 1);
        let h = label_index(&g, "Hello");
        let c = g.layers[0].nodes[h].color.unwrap();
        assert!((c[0] - 1.0).abs() < 1e-3 && (c[1] - 128.0 / 255.0).abs() < 1e-3);
        // viz positions preserved (opposite corners).
        let (ph, pw) = (
            g.layers[0].nodes[h].position,
            g.layers[0].nodes[label_index(&g, "World")].position,
        );
        assert!(ph[0] > pw[0] && ph[1] > pw[1]);
    }

    #[test]
    fn gdf_with_quoted_colors() {
        let text = "nodedef>name VARCHAR,label VARCHAR,x DOUBLE,y DOUBLE,color VARCHAR\n\
                    s1,Server 1,0,0,'255,0,0'\n\
                    s2,Server 2,10,0,'0,0,255'\n\
                    edgedef>node1 VARCHAR,node2 VARCHAR,weight DOUBLE\n\
                    s1,s2,4\n";
        let g = load_graph(text.as_bytes(), "gdf", "gdf").unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edges.len(), 1);
        let s1 = label_index(&g, "Server 1");
        assert_eq!(g.layers[0].nodes[s1].color, Some([1.0, 0.0, 0.0]));
    }

    #[test]
    fn dot_chains_attributes_and_comments() {
        let text = r##"// a comment
digraph G {
  rankdir=LR; /* block comment */
  a [label="Start", fillcolor="#00ff00", pos="0,0!"];
  b [pos="2,0!"];
  a -> b -> c [weight=2];
  d -- e;
  "quoted node" [label="Q"];
}"##;
        let g = load_graph(text.as_bytes(), "dot", "d").unwrap();
        assert_eq!(g.node_count(), 6, "a b c d e + quoted");
        assert_eq!(g.edges.len(), 3, "a->b, b->c, d--e");
        let a = label_index(&g, "Start");
        assert_eq!(g.layers[0].nodes[a].color, Some([0.0, 1.0, 0.0]));
        assert!(g.layers[0].nodes.iter().any(|n| n.label.as_deref() == Some("Q")));
    }

    #[test]
    fn tgf_nodes_then_edges() {
        let text = "1 First node\n2 Second node\n3\n#\n1 2 1.5\n2 3\n";
        let g = load_graph(text.as_bytes(), "tgf", "t").unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edges.len(), 2);
        assert!(g.layers[0].nodes.iter().any(|n| n.label.as_deref() == Some("First node")));
        // Numeric edge label became the weight → different importances.
        assert!(g.edges[0].importance > g.edges[1].importance);
    }

    #[test]
    fn pajek_vertices_edges_and_edgeslist() {
        let text = "*Vertices 3\n1 \"Node One\" 0.1 0.2 0.0\n2 \"Node Two\" 0.9 0.2 0.0\n3 Plain\n\
                    *Edges\n1 2 2.0\n*Edgeslist\n2 3 1\n";
        let g = load_graph(text.as_bytes(), "net", "p").unwrap();
        assert_eq!(g.node_count(), 3);
        // 1-2 from *Edges, then 2-3 and 2-1 from *Edgeslist.
        assert_eq!(g.edges.len(), 3);
        assert!(g.layers[0].nodes.iter().any(|n| n.label.as_deref() == Some("Node One")));
    }

    #[test]
    fn sif_multi_target_lines() {
        let text = "nodeA\tactivates\tnodeB\tnodeC\nnodeD inhibits nodeA\nlonely\n";
        let g = load_graph(text.as_bytes(), "sif", "s").unwrap();
        assert_eq!(g.node_count(), 5, "A B C D + lonely");
        assert_eq!(g.edges.len(), 3, "A→B, A→C, D→A");
    }

    #[test]
    fn rdf_json_triples() {
        let text = r#"{
            "http://ex.org/alice": {
                "http://xmlns.com/foaf/0.1/knows": [
                    {"type": "uri", "value": "http://ex.org/bob"}
                ],
                "http://www.w3.org/2000/01/rdf-schema#label": [
                    {"type": "literal", "value": "Alice"}
                ]
            }
        }"#;
        let g = load_graph(text.as_bytes(), "rj", "r").unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edges.len(), 1);
        assert!(g.layers[0].nodes.iter().any(|n| n.label.as_deref() == Some("Alice")));
        // Shape sniffing routes the same document from a bare .json too.
        assert!(from_json_auto(text.as_bytes(), "r").is_ok());
    }

    #[test]
    fn jsonld_graph_container() {
        let text = r#"{
            "@context": "https://schema.org",
            "@graph": [
                {"@id": "ex:a", "name": "Anna", "knows": {"@id": "ex:b"}},
                {"@id": "ex:b", "name": "Ben", "knows": [{"@id": "ex:a"}, {"@id": "ex:c"}]}
            ]
        }"#;
        let g = load_graph(text.as_bytes(), "jsonld", "j").unwrap();
        assert_eq!(g.node_count(), 3, "a, b, c");
        assert_eq!(g.edges.len(), 3);
        assert!(g.layers[0].nodes.iter().any(|n| n.label.as_deref() == Some("Anna")));
    }

    #[test]
    fn internal_data_attributes_survive_for_every_format() {
        let attrs_of = |g: &NetworkGraph, label: &str| -> Vec<(String, String)> {
            g.layers[0].nodes[label_index(g, label)].attrs.clone()
        };

        // Node-link JSON: unmapped keys become attributes.
        let nl = r#"{"nodes": [{"id": "a", "group": 3, "role": "hub"}], "links": []}"#;
        let g = from_json_auto(nl.as_bytes(), "nl").unwrap();
        let attrs = attrs_of(&g, "a");
        assert!(attrs.contains(&("group".into(), "3".into())));
        assert!(attrs.contains(&("role".into(), "hub".into())));

        // Cytoscape: data fields beyond id/label.
        let cy = r#"{"elements": {"nodes": [{"data": {"id": "n", "score": 0.7}}], "edges": []}}"#;
        let g = load_graph(cy.as_bytes(), "cyjs", "c").unwrap();
        assert!(attrs_of(&g, "n").contains(&("score".into(), "0.7".into())));

        // GraphML: unmapped <data> keys.
        let gm = r#"<graphml>
  <key id="d0" for="node" attr.name="type" attr.type="string"/>
  <graph><node id="x"><data key="d0">router</data></node></graph>
</graphml>"#;
        let g = load_graph(gm.as_bytes(), "graphml", "g").unwrap();
        assert!(attrs_of(&g, "x").contains(&("type".into(), "router".into())));

        // GEXF: attvalues resolved through the <attribute> titles.
        let gx = r#"<gexf><graph>
  <attributes class="node"><attribute id="0" title="country"/></attributes>
  <nodes><node id="p" label="P"><attvalues><attvalue for="0" value="IT"/></attvalues></node></nodes>
</graph></gexf>"#;
        let g = load_graph(gx.as_bytes(), "gexf", "g").unwrap();
        assert!(attrs_of(&g, "P").contains(&("country".into(), "IT".into())));

        // GDF: extra columns.
        let gdf = "nodedef>name VARCHAR,label VARCHAR,dept VARCHAR\nu1,User 1,Physics\n";
        let g = load_graph(gdf.as_bytes(), "gdf", "g").unwrap();
        assert!(attrs_of(&g, "User 1").contains(&("dept".into(), "Physics".into())));

        // DOT: unmapped attributes.
        let dot = r#"graph { z [shape=box, rank=3]; }"#;
        let g = load_graph(dot.as_bytes(), "dot", "g").unwrap();
        let attrs = attrs_of(&g, "z");
        assert!(attrs.contains(&("shape".into(), "box".into())));
        assert!(attrs.contains(&("rank".into(), "3".into())));

        // RDF/JSON: literal predicates keep their local name.
        let rj = r#"{"http://ex.org/a": {"http://ex.org/age": [{"type": "literal", "value": "42"}]}}"#;
        let g = load_graph(rj.as_bytes(), "rj", "g").unwrap();
        assert!(g.layers[0].nodes[0].attrs.contains(&("age".into(), "42".into())));

        // JSON-LD: literal properties.
        let jl = r#"{"@graph": [{"@id": "ex:n", "name": "N", "schema:year": 2024}]}"#;
        let g = load_graph(jl.as_bytes(), "jsonld", "g").unwrap();
        assert!(attrs_of(&g, "N").contains(&("year".into(), "2024".into())));
    }

    #[test]
    fn neighbors_lists_both_directions() {
        let mut b = GraphBuilder::default();
        b.edge("a", "b", 1.0);
        b.edge("c", "a", 1.0);
        let g = net(b);
        let a = label_index(&g, "a");
        let nb = g.neighbors(0, a);
        assert_eq!(nb.len(), 2);
        let outgoing: Vec<bool> = nb.iter().map(|&(_, _, _, out)| out).collect();
        assert!(outgoing.contains(&true) && outgoing.contains(&false));
    }

    #[test]
    fn layout_is_deterministic_and_normalized() {
        let mut b1 = GraphBuilder::default();
        let mut b2 = GraphBuilder::default();
        for b in [&mut b1, &mut b2] {
            for i in 0..20 {
                b.edge(&format!("n{i}"), &format!("n{}", (i + 1) % 20), 1.0);
            }
        }
        let (g1, g2) = (net(b1), net(b2));
        for (a, b) in g1.layers[0].nodes.iter().zip(&g2.layers[0].nodes) {
            assert_eq!(a.position, b.position, "layout must be deterministic");
        }
        // Bounding radius ≈ LAYOUT_RADIUS.
        let r = g1.layers[0]
            .nodes
            .iter()
            .map(|n| (n.position[0].powi(2) + n.position[1].powi(2) + n.position[2].powi(2)).sqrt())
            .fold(0.0f32, f32::max);
        assert!((r - LAYOUT_RADIUS).abs() < 0.5, "radius {r}");
    }

    #[test]
    fn degree_drives_default_node_size() {
        let mut b = GraphBuilder::default();
        b.edge("hub", "s1", 1.0);
        b.edge("hub", "s2", 1.0);
        b.edge("hub", "s3", 1.0);
        let g = net(b);
        let hub = label_index(&g, "hub");
        let leaf = label_index(&g, "s1");
        assert!(
            g.layers[0].nodes[hub].weight_magnitude > g.layers[0].nodes[leaf].weight_magnitude
        );
    }

    #[test]
    fn empty_and_oversized_inputs_error_cleanly() {
        assert!(load_graph(b"", "sif", "e").is_err());
        assert!(load_graph(b"*Vertices 0\n", "net", "e").is_err());
        assert!(load_graph(b"{}", "cyjs", "e").is_err());
        assert!(from_json_auto(b"{\"foo\": 1}", "e").is_err());
        assert!(load_graph(b"hello", "xyz", "e").is_err());
    }

    #[test]
    fn color_parsing_variants() {
        assert_eq!(parse_color("#ff0000"), Some([1.0, 0.0, 0.0]));
        assert_eq!(parse_color("#f00"), Some([1.0, 0.0, 0.0]));
        assert_eq!(parse_color("255,128,0").map(|c| (c[0] * 255.0) as u8), Some(255));
        assert_eq!(parse_color("0.5,0.5,1.0"), Some([0.5, 0.5, 1.0]));
        assert_eq!(parse_color("nope"), None);
    }

    #[test]
    fn layout_respects_majority_file_positions() {
        let mut b = GraphBuilder::default();
        b.node("a");
        b.node("b");
        b.set_pos("a", [0.0, 0.0, 0.0]);
        b.set_pos("b", [10.0, 0.0, 0.0]);
        b.edge("a", "b", 1.0);
        let g = net(b);
        let (pa, pb) = (g.layers[0].nodes[0].position, g.layers[0].nodes[1].position);
        // Collinear along x after normalization.
        assert!(pa[1].abs() < 1e-4 && pb[1].abs() < 1e-4);
        assert!((pa[0] - pb[0]).abs() > LAYOUT_RADIUS);
    }
}
