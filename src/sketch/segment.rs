//! A single boundary segment of a 2D sketch profile.
//!
//! A [`Segment`] is straight ([`Segment::Line`]) or curved ([`Segment::Arc`],
//! [`Segment::Bezier`]). Every variant knows its endpoints and flattens to a
//! polyline through the single adaptive entry point [`Segment::flatten_into`],
//! so straight and curved segments share one tessellation path downstream.

use super::Vec2;

/// One edge of a profile, in 2D sketch coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Segment {
    /// Straight edge from `a` to `b`.
    Line { a: Vec2, b: Vec2 },
    /// Circular arc, swept from `start` to `end` (radians, CCW positive)
    /// around `center` at `radius`.
    Arc {
        center: Vec2,
        radius: f32,
        start: f32,
        end: f32,
    },
    /// Cubic Bézier from `p0` to `p1` with control points `c1`, `c2`.
    Bezier {
        p0: Vec2,
        c1: Vec2,
        c2: Vec2,
        p1: Vec2,
    },
}

impl Segment {
    /// First point of the segment.
    pub fn start(&self) -> Vec2 {
        match self {
            Segment::Line { a, .. } => *a,
            Segment::Arc {
                center,
                radius,
                start,
                ..
            } => arc_point(*center, *radius, *start),
            Segment::Bezier { p0, .. } => *p0,
        }
    }

    /// Last point of the segment.
    pub fn end(&self) -> Vec2 {
        match self {
            Segment::Line { b, .. } => *b,
            Segment::Arc {
                center,
                radius,
                end,
                ..
            } => arc_point(*center, *radius, *end),
            Segment::Bezier { p1, .. } => *p1,
        }
    }

    /// Append the flattened polyline to `out`, **excluding** the start point
    /// and **including** the end point. Callers seed `out` with the very first
    /// point so consecutive segments join without duplicating shared vertices.
    ///
    /// `tol` is the maximum allowed deviation (chord error) between the true
    /// curve and the polyline, in sketch units. Straight lines ignore it.
    pub fn flatten_into(&self, tol: f32, out: &mut Vec<Vec2>) {
        let tol = tol.max(1e-6);
        match self {
            Segment::Line { b, .. } => out.push(*b),
            Segment::Arc {
                center,
                radius,
                start,
                end,
            } => flatten_arc(*center, *radius, *start, *end, tol, out),
            Segment::Bezier { p0, c1, c2, p1 } => flatten_bezier(*p0, *c1, *c2, *p1, tol, 0, out),
        }
    }
}

/// Point on a circle at `angle` (radians).
fn arc_point(center: Vec2, radius: f32, angle: f32) -> Vec2 {
    [
        center[0] + radius * angle.cos(),
        center[1] + radius * angle.sin(),
    ]
}

/// Adaptive arc flattening: choose the angular step so the chord sagitta stays
/// within `tol`, then emit evenly spaced points (end inclusive).
fn flatten_arc(center: Vec2, radius: f32, start: f32, end: f32, tol: f32, out: &mut Vec<Vec2>) {
    let sweep = end - start;
    if radius <= 1e-6 || sweep.abs() <= 1e-9 {
        out.push(arc_point(center, radius, end));
        return;
    }
    // Sagitta s = r(1 - cos(dθ/2)) ≤ tol  ⇒  dθ = 2·acos(1 - tol/r).
    let ratio = (1.0 - tol / radius).clamp(-1.0, 1.0);
    let max_step = 2.0 * ratio.acos();
    let steps = if max_step <= 1e-6 {
        1
    } else {
        (sweep.abs() / max_step).ceil().max(1.0) as usize
    };
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        out.push(arc_point(center, radius, start + sweep * t));
    }
}

/// Recursive cubic Bézier flattening by flatness test (de Casteljau split).
fn flatten_bezier(
    p0: Vec2,
    c1: Vec2,
    c2: Vec2,
    p1: Vec2,
    tol: f32,
    depth: u32,
    out: &mut Vec<Vec2>,
) {
    // Guard against pathological recursion; 24 levels is far past any sane tol.
    if depth >= 24 || is_flat(p0, c1, c2, p1, tol) {
        out.push(p1);
        return;
    }
    // Split at t = 0.5 via de Casteljau.
    let ab = mid(p0, c1);
    let bc = mid(c1, c2);
    let cd = mid(c2, p1);
    let abc = mid(ab, bc);
    let bcd = mid(bc, cd);
    let abcd = mid(abc, bcd);
    flatten_bezier(p0, ab, abc, abcd, tol, depth + 1, out);
    flatten_bezier(abcd, bcd, cd, p1, tol, depth + 1, out);
}

/// A cubic is "flat enough" when both control points sit within `tol` of the
/// chord `p0`–`p1`.
fn is_flat(p0: Vec2, c1: Vec2, c2: Vec2, p1: Vec2, tol: f32) -> bool {
    dist_point_segment(c1, p0, p1) <= tol && dist_point_segment(c2, p0, p1) <= tol
}

fn mid(a: Vec2, b: Vec2) -> Vec2 {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// Perpendicular distance from `p` to the (infinite-length-safe) segment a–b.
fn dist_point_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len2 = dx * dx + dy * dy;
    if len2 <= 1e-12 {
        // Degenerate chord: fall back to point distance.
        return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
    }
    // Distance from point to the infinite line through a,b (cross / |d|).
    ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn close(a: Vec2, b: Vec2) -> bool {
        (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4
    }

    #[test]
    fn line_endpoints_and_flatten_to_two_points() {
        let s = Segment::Line {
            a: [0.0, 0.0],
            b: [2.0, 1.0],
        };
        assert!(close(s.start(), [0.0, 0.0]));
        assert!(close(s.end(), [2.0, 1.0]));
        let mut out = vec![s.start()];
        s.flatten_into(0.01, &mut out);
        assert_eq!(out.len(), 2, "a line is just its two endpoints");
        assert!(close(out[1], [2.0, 1.0]));
    }

    #[test]
    fn arc_endpoints_are_on_the_circle() {
        let s = Segment::Arc {
            center: [0.0, 0.0],
            radius: 2.0,
            start: 0.0,
            end: PI / 2.0,
        };
        assert!(close(s.start(), [2.0, 0.0]));
        assert!(close(s.end(), [0.0, 2.0]));
    }

    #[test]
    fn arc_flatten_respects_tolerance_and_stays_on_circle() {
        let r = 5.0;
        let s = Segment::Arc {
            center: [0.0, 0.0],
            radius: r,
            start: 0.0,
            end: 2.0 * PI, // full circle
        };
        let tol = 0.05;
        let mut out = vec![s.start()];
        s.flatten_into(tol, &mut out);
        assert!(out.len() > 8, "a full circle needs several chords");
        // Every emitted point lies on the circle.
        for p in &out {
            let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((d - r).abs() < 1e-3, "point off circle: {d}");
        }
        // Chord sagitta must be within tolerance: check the largest gap.
        for w in out.windows(2) {
            let chord = ((w[0][0] - w[1][0]).powi(2) + (w[0][1] - w[1][1]).powi(2)).sqrt();
            // sagitta ≈ chord²/(8r) for small chords.
            let sagitta = chord * chord / (8.0 * r);
            assert!(sagitta <= tol + 1e-3, "sagitta {sagitta} exceeds tol");
        }
    }

    #[test]
    fn finer_tolerance_yields_more_points() {
        let s = Segment::Arc {
            center: [0.0, 0.0],
            radius: 3.0,
            start: 0.0,
            end: PI,
        };
        let mut coarse = vec![s.start()];
        s.flatten_into(0.2, &mut coarse);
        let mut fine = vec![s.start()];
        s.flatten_into(0.01, &mut fine);
        assert!(fine.len() > coarse.len());
    }

    #[test]
    fn straight_bezier_flattens_to_its_endpoint() {
        // Control points on the chord ⇒ already flat ⇒ a single output point.
        let s = Segment::Bezier {
            p0: [0.0, 0.0],
            c1: [1.0, 0.0],
            c2: [2.0, 0.0],
            p1: [3.0, 0.0],
        };
        let mut out = vec![s.start()];
        s.flatten_into(0.01, &mut out);
        assert_eq!(out.len(), 2);
        assert!(close(*out.last().unwrap(), [3.0, 0.0]));
    }

    #[test]
    fn curved_bezier_subdivides_and_passes_through_endpoints() {
        let s = Segment::Bezier {
            p0: [0.0, 0.0],
            c1: [0.0, 4.0],
            c2: [4.0, 4.0],
            p1: [4.0, 0.0],
        };
        let mut out = vec![s.start()];
        s.flatten_into(0.02, &mut out);
        assert!(out.len() > 4, "a real curve needs subdivision");
        assert!(close(out[0], [0.0, 0.0]));
        assert!(close(*out.last().unwrap(), [4.0, 0.0]));
        // The polyline must be reasonably smooth (no huge jumps).
        for w in out.windows(2) {
            let step = ((w[0][0] - w[1][0]).powi(2) + (w[0][1] - w[1][1]).powi(2)).sqrt();
            assert!(step < 2.0, "unexpectedly long chord {step}");
        }
    }
}
