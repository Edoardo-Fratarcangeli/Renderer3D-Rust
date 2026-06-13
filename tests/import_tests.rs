//! Integration tests for 3D solid model import (STL / OBJ / glTF).
//!
//! These run against the real sample models under `tests/import/`. Each test
//! is defensive: if the fixture is missing the test is skipped rather than
//! failing, so a lean checkout without the large binaries still passes.

use std::path::PathBuf;

use rendering_3d::mesh::MeshData;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("import")
        .join(name)
}

/// Load a fixture, asserting it produced a non-degenerate mesh. Returns early
/// (skips) when the fixture is not present.
fn assert_loads(name: &str) {
    let path = fixture(name);
    if !path.exists() {
        eprintln!("skipping {name}: fixture not present");
        return;
    }
    let mesh = MeshData::load(&path)
        .unwrap_or_else(|e| panic!("failed to load {name}: {e}"));
    assert!(!mesh.vertices.is_empty(), "{name}: no vertices");
    assert!(!mesh.indices.is_empty(), "{name}: no indices");
    assert_eq!(mesh.indices.len() % 3, 0, "{name}: indices not triangulated");
    // Indices must stay in range.
    let max_idx = *mesh.indices.iter().max().unwrap();
    assert!((max_idx as usize) < mesh.vertices.len(), "{name}: index out of range");
    // A real model has a non-zero bounding box.
    assert!(mesh.max_extent() > 0.0, "{name}: degenerate bounding box");
}

#[test]
fn loads_small_stl() {
    assert_loads("lego_heart_keychain.stl");
}

#[test]
fn loads_medium_stl() {
    assert_loads("seed-starter-base.stl");
    assert_loads("build-tray-stl.stl");
}

#[test]
fn loads_large_stl() {
    assert_loads("dragon.stl");
    assert_loads("batman-pen-holder.stl");
}

#[test]
fn loads_obj_models() {
    assert_loads("man.obj");
    assert_loads("game-map.obj");
}

#[test]
fn step_is_recognized_but_not_yet_supported() {
    let path = fixture("build-tray-step.stp");
    // Even when the fixture exists, STEP must report a clear "not supported"
    // error rather than producing geometry.
    let err = MeshData::load(&path).expect_err("STEP should not load yet");
    assert!(err.to_string().contains("STEP"), "got: {err}");
}

#[test]
fn unknown_extension_errors_clearly() {
    let err = MeshData::load(&fixture("nope.xyz")).unwrap_err();
    assert!(err.to_string().contains("unsupported"), "got: {err}");
}
