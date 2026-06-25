//! End-to-end integration of the sketch + B-rep layers through the public API:
//! build the six faces of a unit cube as 2D sketches on world planes, then
//! compose them into one watertight solid by welding their shared edges.

use rendering_3d::brep;
use rendering_3d::sketch::{Plane, Profile, Sketch, DEFAULT_TOLERANCE};

/// A unit square sketched on `plane` (origin offset baked in), returned as its
/// world-space boundary loop.
fn face(plane: Plane) -> Vec<[f32; 3]> {
    let profile =
        Profile::from_points(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], true).unwrap();
    Sketch::new(plane, profile)
        .world_polyline(DEFAULT_TOLERANCE)
        .unwrap()
}

/// The six faces of the unit cube [0,1]³, each sketched on the right plane.
fn cube_face_loops() -> Vec<Vec<[f32; 3]>> {
    let at = |mut p: Plane, origin: [f32; 3]| {
        p.origin = origin;
        p
    };
    vec![
        face(at(Plane::xy(), [0.0, 0.0, 0.0])), // bottom z=0
        face(at(Plane::xy(), [0.0, 0.0, 1.0])), // top    z=1
        face(at(Plane::xz(), [0.0, 0.0, 0.0])), // front  y=0
        face(at(Plane::xz(), [0.0, 1.0, 0.0])), // back   y=1
        face(at(Plane::yz(), [0.0, 0.0, 0.0])), // left   x=0
        face(at(Plane::yz(), [1.0, 0.0, 0.0])), // right  x=1
    ]
}

#[test]
fn six_sketched_faces_compose_into_a_watertight_cube() {
    let loops = cube_face_loops();
    assert_eq!(loops.len(), 6);

    // The solid layer welds the shared corners: 8 vertices, 12 edges, 6 faces.
    let solid = brep::Solid::from_face_loops(&loops, brep::WELD_EPS);
    assert_eq!(solid.verts.len(), 8, "cube corners welded");
    assert_eq!(solid.faces.len(), 6);
    assert_eq!(solid.edge_use().len(), 12);
    assert_eq!(
        solid.shared_edges().len(),
        12,
        "every edge shared by 2 faces"
    );
    assert_eq!(solid.euler_characteristic(), 2, "V - E + F = 2");

    // The one-shot composer produces a watertight, 12-triangle mesh whose
    // normals all face outward (it flips the shell if it came out inside-out).
    let (mesh, closed) = brep::compose(&loops);
    assert!(closed, "the six faces form a closed solid");
    assert_eq!(mesh.triangle_count(), 12);
    assert_eq!(mesh.aabb(), ([0.0; 3], [1.0, 1.0, 1.0]));

    let center = [0.5_f32, 0.5, 0.5];
    for tri in mesh.indices.chunks_exact(3) {
        let p = mesh.vertices[tri[0] as usize].position;
        let n = mesh.vertices[tri[0] as usize].normal;
        // (p - center) · n > 0  ⇒  the normal points away from the centroid.
        let outward =
            (p[0] - center[0]) * n[0] + (p[1] - center[1]) * n[1] + (p[2] - center[2]) * n[2];
        assert!(outward > 0.0, "face normal should face outward");
    }
}

#[test]
fn dropping_a_face_makes_the_solid_open() {
    let mut loops = cube_face_loops();
    loops.pop(); // remove the right face
    let (mesh, closed) = brep::compose(&loops);
    assert!(!closed, "five faces leave an opening");
    assert_eq!(mesh.triangle_count(), 10);
}

#[test]
fn a_closed_sketch_surface_is_double_sided() {
    // A pentagon surface: 2·(5-2) = 6 triangles (front + back).
    let sketch = Sketch::new(Plane::xy(), Profile::regular_polygon(5, 1.0).unwrap());
    let mesh = sketch.surface_mesh(DEFAULT_TOLERANCE).unwrap();
    assert_eq!(mesh.triangle_count(), 6);
}

#[test]
fn an_open_polyline_becomes_a_ribbon() {
    let profile = Profile::from_points(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], false).unwrap();
    let mesh = Sketch::new(Plane::xy(), profile)
        .to_mesh(DEFAULT_TOLERANCE)
        .unwrap();
    // Two segments, each a double-sided quad → 8 triangles.
    assert_eq!(mesh.triangle_count(), 8);
}
