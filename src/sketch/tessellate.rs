//! Ear-clipping triangulation of a simple polygon.
//!
//! Input is a polyline loop of 2D points (no repeated closing vertex). Output
//! is a flat list of triangle indices into that input, wound counter-clockwise.
//! The algorithm normalizes the winding internally, so callers may pass either
//! orientation; the result is always CCW so a +normal front face is consistent.
//!
//! This is the one tessellator in the crate; the B-rep layer reuses it (after
//! projecting each 3D face to 2D) so there is a single triangulation path.

use super::profile::signed_area;
use super::Vec2;

/// Triangulate a simple polygon. Returns `None` for fewer than 3 points or a
/// degenerate (zero-area) loop. Indices reference `points` and form CCW
/// triangles.
pub fn triangulate(points: &[Vec2]) -> Option<Vec<usize>> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    if signed_area(points).abs() <= 1e-12 {
        return None;
    }

    // Work on an index ring oriented CCW; remember the original indices so the
    // output references the caller's array regardless of input winding.
    let mut ring: Vec<usize> = (0..n).collect();
    if signed_area(points) < 0.0 {
        ring.reverse();
    }

    let mut out = Vec::with_capacity((n - 2) * 3);
    let mut guard = 0;
    // Each successful ear removes one vertex; the guard bounds the worst case.
    while ring.len() > 3 {
        let m = ring.len();
        let mut clipped = false;
        for i in 0..m {
            let prev = ring[(i + m - 1) % m];
            let cur = ring[i];
            let next = ring[(i + 1) % m];
            if is_ear(points, &ring, prev, cur, next) {
                out.push(prev);
                out.push(cur);
                out.push(next);
                ring.remove(i);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // No ear found: the polygon is self-intersecting or numerically
            // tricky. Bail rather than loop forever.
            return None;
        }
        guard += 1;
        if guard > n + 2 {
            return None;
        }
    }
    out.extend_from_slice(&[ring[0], ring[1], ring[2]]);
    Some(out)
}

/// Is the triangle (prev, cur, next) an ear: convex and containing no other
/// ring vertex?
fn is_ear(points: &[Vec2], ring: &[usize], prev: usize, cur: usize, next: usize) -> bool {
    let a = points[prev];
    let b = points[cur];
    let c = points[next];
    // Convex corner (CCW ring ⇒ left turn means convex).
    if cross(a, b, c) <= 0.0 {
        return false;
    }
    // No other vertex may fall inside the candidate triangle.
    for &idx in ring {
        if idx == prev || idx == cur || idx == next {
            continue;
        }
        if point_in_triangle(points[idx], a, b, c) {
            return false;
        }
    }
    true
}

/// Z of the cross product (b-a)×(c-a): >0 is a CCW (left) turn.
fn cross(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Point-in-triangle via consistent half-plane signs (boundary counts as out).
fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = cross(a, b, p);
    let d2 = cross(b, c, p);
    let d3 = cross(c, a, p);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sum of triangle areas from an index list.
    fn triangulated_area(points: &[Vec2], tris: &[usize]) -> f32 {
        tris.chunks_exact(3)
            .map(|t| {
                let a = points[t[0]];
                let b = points[t[1]];
                let c = points[t[2]];
                0.5 * cross(a, b, c).abs()
            })
            .sum()
    }

    #[test]
    fn square_triangulates_into_two_triangles() {
        let sq = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let tris = triangulate(&sq).unwrap();
        assert_eq!(tris.len(), 6, "n-2 = 2 triangles");
        assert!((triangulated_area(&sq, &tris) - 4.0).abs() < 1e-4);
        // Each triangle is CCW (positive area via signed cross).
        for t in tris.chunks_exact(3) {
            assert!(cross(sq[t[0]], sq[t[1]], sq[t[2]]) > 0.0);
        }
    }

    #[test]
    fn area_is_preserved_for_a_concave_polygon() {
        // An "L"/arrow shape with a reflex vertex.
        let poly = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [1.0, 1.0],
            [1.0, 3.0],
            [0.0, 3.0],
        ];
        let tris = triangulate(&poly).unwrap();
        assert_eq!(tris.len(), (poly.len() - 2) * 3);
        // Shoelace area of the L: 4*1 + 1*2 = 6.
        assert!((triangulated_area(&poly, &tris) - 6.0).abs() < 1e-4);
        for t in tris.chunks_exact(3) {
            assert!(cross(poly[t[0]], poly[t[1]], poly[t[2]]) > 0.0);
        }
    }

    #[test]
    fn clockwise_input_is_normalized_to_ccw_output() {
        // Same square wound clockwise: result must still be CCW and area-correct.
        let cw = [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]];
        let tris = triangulate(&cw).unwrap();
        assert!((triangulated_area(&cw, &tris) - 4.0).abs() < 1e-4);
        for t in tris.chunks_exact(3) {
            assert!(cross(cw[t[0]], cw[t[1]], cw[t[2]]) > 0.0);
        }
    }

    #[test]
    fn degenerate_inputs_return_none() {
        assert!(triangulate(&[[0.0, 0.0], [1.0, 1.0]]).is_none());
        // Collinear points → zero area.
        assert!(triangulate(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]).is_none());
    }

    #[test]
    fn indices_are_in_range() {
        let poly = [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.5, 3.0], [0.0, 2.0]];
        let tris = triangulate(&poly).unwrap();
        assert!(tris.iter().all(|&i| i < poly.len()));
    }
}
