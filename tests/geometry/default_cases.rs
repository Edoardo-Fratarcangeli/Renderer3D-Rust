// Default Geometry Tests
#[test]
fn test_vector_addition() {
    // A simple test simulating vector math from our engine
    let v1 = [1.0, 2.0, 3.0];
    let v2 = [4.0, 5.0, 6.0];
    let res = [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]];
    assert_eq!(res, [5.0, 7.0, 9.0]);
}

#[test]
fn test_sphere_volume() {
    let r = 1.0;
    let vol = (4.0 / 3.0) * std::f32::consts::PI * r * r * r;
    assert!((vol - 4.18879).abs() < 0.0001);
}
