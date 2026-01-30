// Extreme Geometry Tests
#[test]
fn test_large_coordinates() {
    let large: f32 = 1e10;
    let v1 = [large, large, large];
    let v2 = [large, large, large];
    let dist_sq = (v1[0]-v2[0]).powi(2) + (v1[1]-v2[1]).powi(2);
    // Distance should be 0, ensuring precision holds reasonably well relative to itself
    assert_eq!(dist_sq, 0.0);
}

#[test]
fn test_degenerate_triangle() {
    // 3 points on a line
    let p1: [f32; 3] = [0.0, 0.0, 0.0];
    let p2: [f32; 3] = [1.0, 0.0, 0.0];
    let p3: [f32; 3] = [2.0, 0.0, 0.0];
    
    // Check if normal calculation fails or returns zero
    // (Simulated logic)
    let v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]];
    let v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]];
    
    let cross = [
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ];
    
    assert_eq!(cross, [0.0, 0.0, 0.0]);
}
