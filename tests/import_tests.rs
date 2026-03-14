use rendering_3d::mesh::MeshData;
use std::path::Path;

#[test]
fn test_load_stl_small() {
    let path = Path::new("tests/import/lego_heart_keychain.stl");
    if path.exists() {
        let mesh = MeshData::load(path).expect("Failed to load small STL");
        assert!(!mesh.vertices.is_empty(), "Mesh should have vertices");
        assert!(!mesh.indices.is_empty(), "Mesh should have indices");
    }
}

#[test]
fn test_load_stl_medium() {
    let path = Path::new("tests/import/seed-starter-base.stl");
    if path.exists() {
        let mesh = MeshData::load(path).expect("Failed to load medium STL");
        assert!(!mesh.vertices.is_empty());
    }
}

#[test]
fn test_load_obj_man() {
    let path = Path::new("tests/import/man.obj");
    if path.exists() {
        let mesh = MeshData::load(path).expect("Failed to load OBJ man");
        assert!(!mesh.vertices.is_empty());
    }
}

#[test]
fn test_load_obj_large() {
    let path = Path::new("tests/import/game-map.obj");
    if path.exists() {
        let mesh = MeshData::load(path).expect("Failed to load large OBJ");
        assert!(!mesh.vertices.is_empty());
    }
}

#[test]
fn test_load_stl_large() {
    let path = Path::new("tests/import/batman-pen-holder.stl");
    if path.exists() {
        let mesh = MeshData::load(path).expect("Failed to load large STL");
        assert!(!mesh.vertices.is_empty());
    }
}

#[test]
fn test_load_step_error() {
    let path = Path::new("tests/import/build-tray-step.stp");
    if path.exists() {
        let res = MeshData::load(path);
        assert!(res.is_err(), "STEP should return error currently");
        let err_msg = res.err().unwrap().to_string();
        assert!(err_msg.contains("STEP format support requires further implementation"));
    }
}

#[test]
fn test_load_invalid_extension() {
    let res = MeshData::load("tests/import/nonexistent.txt");
    assert!(res.is_err());
}
