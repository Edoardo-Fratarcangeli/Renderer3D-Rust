// Integration tests for the universal geometry import: real files on disk
// for every supported world (CSV, Excel, JSON, XYZ, DSL text) plus the
// layer/batch pipeline the renderer consumes.

use rendering_3d::geometry::{
    build_batches, loader, GeometryLayer, GeometryRecord, DEFAULT_COLOR, POINT_SIZE,
};
use rendering_3d::scene::GeometryType;

#[test]
fn csv_file_imports_as_layer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shapes.csv");
    std::fs::write(
        &path,
        "shape,x,y,z,size,color,name\n\
         cube,0,0,0,2,#ff0000,base\n\
         sphere,1,2,3,0.5,\"0,255,0\",ball\n\
         plane,0,0,-1,4,,floor\n",
    )
    .unwrap();

    let layer = loader::layer_from_path(&path, DEFAULT_COLOR).unwrap();
    assert_eq!(layer.name, "shapes");
    assert_eq!(layer.len(), 3);
    assert_eq!(layer.records[0].shape, GeometryType::Cube);
    assert_eq!(layer.records[0].color, [1.0, 0.0, 0.0]);
    assert_eq!(layer.records[1].color, [0.0, 1.0, 0.0]);
    assert_eq!(layer.records[1].label.as_deref(), Some("ball"));
    // Empty color cell falls back to the layer default.
    assert_eq!(layer.records[2].color, DEFAULT_COLOR);
}

#[test]
fn excel_file_imports_through_calamine() {
    use rust_xlsxwriter::Workbook;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scene.xlsx");

    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    for (col, header) in ["type", "x", "y", "z", "size", "color", "label"]
        .iter()
        .enumerate()
    {
        sheet.write_string(0, col as u16, *header).unwrap();
    }
    let rows: Vec<(&str, f64, f64, f64, f64, &str, &str)> = vec![
        ("cube", 0.0, 0.0, 0.0, 2.0, "#3366ff", "blue box"),
        ("sphere", 5.0, 5.0, 5.0, 1.5, "", "orb"),
    ];
    for (i, (shape, x, y, z, size, color, label)) in rows.iter().enumerate() {
        let r = (i + 1) as u32;
        sheet.write_string(r, 0, *shape).unwrap();
        sheet.write_number(r, 1, *x).unwrap();
        sheet.write_number(r, 2, *y).unwrap();
        sheet.write_number(r, 3, *z).unwrap();
        sheet.write_number(r, 4, *size).unwrap();
        sheet.write_string(r, 5, *color).unwrap();
        sheet.write_string(r, 6, *label).unwrap();
    }
    workbook.save(&path).unwrap();

    let layer = loader::layer_from_path(&path, DEFAULT_COLOR).unwrap();
    assert_eq!(layer.len(), 2);
    assert_eq!(layer.records[0].shape, GeometryType::Cube);
    assert_eq!(layer.records[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(layer.records[0].scale, [2.0; 3]);
    assert_eq!(layer.records[0].label.as_deref(), Some("blue box"));
    assert!((layer.records[0].color[2] - 1.0).abs() < 0.01);
    assert_eq!(layer.records[1].position, [5.0, 5.0, 5.0]);
    assert_eq!(layer.records[1].color, DEFAULT_COLOR);
}

#[test]
fn json_file_imports_as_layer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scene.json");
    std::fs::write(
        &path,
        r##"{"geometries":[
            {"shape":"cube","pos":[1,2,3],"size":2,"color":"#ff8800","label":"a"},
            {"x":9,"y":9,"z":9}
        ]}"##,
    )
    .unwrap();

    let layer = loader::layer_from_path(&path, DEFAULT_COLOR).unwrap();
    assert_eq!(layer.len(), 2);
    assert_eq!(layer.records[0].shape, GeometryType::Cube);
    assert_eq!(layer.records[1].shape, GeometryType::Sphere);
    assert_eq!(layer.records[1].scale, [POINT_SIZE; 3]);
}

#[test]
fn xyz_and_txt_files_import() {
    let dir = tempfile::tempdir().unwrap();

    let xyz = dir.path().join("cloud.xyz");
    std::fs::write(&xyz, "0 0 0\n1 1 1 0.3\n2,2,2\n").unwrap();
    let layer = loader::layer_from_path(&xyz, [1.0, 0.0, 0.0]).unwrap();
    assert_eq!(layer.len(), 3);
    assert_eq!(layer.records[1].scale, [0.3; 3]);
    assert_eq!(layer.records[2].color, [1.0, 0.0, 0.0]);

    // .txt with DSL content auto-detects the DSL...
    let txt = dir.path().join("scene.txt");
    std::fs::write(&txt, "cube 0 0 0 2 #ff0000 base\nsphere 1 1 1 0.5").unwrap();
    let layer = loader::layer_from_path(&txt, DEFAULT_COLOR).unwrap();
    assert_eq!(layer.records[0].shape, GeometryType::Cube);

    // ...and .txt with bare numbers auto-detects XYZ.
    let pts = dir.path().join("points.txt");
    std::fs::write(&pts, "0 0 0\n5 5 5\n").unwrap();
    let layer = loader::layer_from_path(&pts, DEFAULT_COLOR).unwrap();
    assert_eq!(layer.records[0].shape, GeometryType::Sphere);
    assert_eq!(layer.records[0].scale, [POINT_SIZE; 3]);
}

#[test]
fn unsupported_extension_and_missing_file_error_clearly() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("model.step");
    std::fs::write(&bad, "x").unwrap();
    let err = loader::layer_from_path(&bad, DEFAULT_COLOR).unwrap_err();
    assert!(err.to_string().contains("unknown geometry extension"));

    let missing = dir.path().join("nope.csv");
    assert!(loader::layer_from_path(&missing, DEFAULT_COLOR).is_err());
}

#[test]
fn many_records_collapse_into_few_instanced_batches() {
    // 30k mixed records -> exactly 3 batches (one per shape), which the
    // renderer draws with 3 instanced draw calls.
    let shapes = [GeometryType::Cube, GeometryType::Sphere, GeometryType::Plane];
    let records: Vec<GeometryRecord> = (0..30_000)
        .map(|i| GeometryRecord::new(shapes[i % 3], [i as f32, 0.0, 0.0]))
        .collect();
    let layer = GeometryLayer::new("big", records);
    let batches = build_batches(&[layer]);
    assert_eq!(batches.len(), 3);
    let total: usize = batches.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(total, 30_000);
    for (_, instances) in &batches {
        assert_eq!(instances.len(), 10_000);
    }
}

#[test]
#[ignore]
fn bench_batch_build_500k_records() {
    // cargo test --release --test geometry_import -- --ignored --nocapture
    let records: Vec<GeometryRecord> = (0..500_000)
        .map(|i| GeometryRecord::new(GeometryType::Cube, [i as f32, 0.0, 0.0]))
        .collect();
    let layer = GeometryLayer::new("bench", records);
    let t = std::time::Instant::now();
    let batches = build_batches(&[layer]);
    println!("batch build 500k records: {:?}", t.elapsed());
    assert_eq!(batches[0].1.len(), 500_000);
}
