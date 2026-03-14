use cgmath::Point3;
use rendering_3d::camera::Camera;
use rendering_3d::mesh::{MeshData, Vertex};
use rendering_3d::render::pick_point;
use rendering_3d::scene::{SceneObject, GeometryType};
use std::sync::Arc;

#[test]
fn test_raycast_mock() {
    let mesh = MeshData {
        vertices: vec![
            Vertex {
                pos: [-1.0, -1.0, 0.0],
                norm: [0.0, 0.0, 1.0],
            },
            Vertex {
                pos: [1.0, -1.0, 0.0],
                norm: [0.0, 0.0, 1.0],
            },
            Vertex {
                pos: [0.0, 1.0, 0.0],
                norm: [0.0, 0.0, 1.0],
            },
        ],
        indices: vec![0, 1, 2],
    };

    let obj = SceneObject::new(1, "Test".to_string(), [0.0, 0.0, 0.0], GeometryType::Mesh { data: Arc::new(mesh) });

    let camera = Camera {
        eye: Point3::new(0.0, 0.0, 5.0),
        target: Point3::new(0.0, 0.0, 0.0),
        up: cgmath::Vector3::unit_y(),
        aspect: 1.0,
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
    };

    // Center of NDC should hit the triangle (it's at z=0, looking from z=5)
    let hit = pick_point(&camera, &[obj], [0.0, 0.0]);
    assert!(hit.is_some());
}

#[test]
fn test_mesh_aabb() {
    let mesh = MeshData {
        vertices: vec![
            Vertex {
                pos: [-1.0, -2.0, -3.0],
                norm: [0.0, 0.0, 0.0],
            },
            Vertex {
                pos: [1.0, 2.0, 3.0],
                norm: [0.0, 0.0, 0.0],
            },
        ],
        indices: vec![],
    };
    let aabb = mesh.compute_aabb();
    assert_eq!(aabb.mins.x, -1.0);
    assert_eq!(aabb.maxs.z, 3.0);
}

#[test]
fn test_distance_calculation() {
    let p1: [f32; 3] = [0.0, 0.0, 0.0];
    let p2: [f32; 3] = [3.0, 4.0, 0.0];
    let dist = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt();
    assert_eq!(dist, 5.0);
}

#[test]
fn test_raycast_miss() {
    let mesh = MeshData {
        vertices: vec![
            Vertex {
                pos: [-1.0, -1.0, 0.0],
                norm: [0.0, 0.0, 1.0],
            },
            Vertex {
                pos: [1.0, -1.0, 0.0],
                norm: [0.0, 0.0, 1.0],
            },
            Vertex {
                pos: [0.0, 1.0, 0.0],
                norm: [0.0, 0.0, 1.0],
            },
        ],
        indices: vec![0, 1, 2],
    };

    let obj = SceneObject::new(1, "Test".to_string(), [0.0, 0.0, 0.0], GeometryType::Mesh { data: Arc::new(mesh) });

    let camera = Camera {
        eye: Point3::new(0.0, 0.0, 5.0),
        target: Point3::new(0.0, 0.0, 0.0),
        up: cgmath::Vector3::unit_y(),
        aspect: 1.0,
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
    };

    // Pointing far away from the triangle
    let hit = pick_point(&camera, &[obj], [0.9, 0.9]);
    assert!(hit.is_none());
}
