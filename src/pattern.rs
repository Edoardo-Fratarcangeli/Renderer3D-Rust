//! Parametric placement patterns (CAD-style arrays).
//!
//! Pure math turning a pattern description into a list of [`Placement`]s
//! (position + optional orientation). The Objects window's Array tab clones
//! a template object onto them; nothing here touches UI or GPU.
//!
//! Orientations are expressed as the same intrinsic X·Y·Z Euler degrees used
//! by [`crate::scene::SceneObject::update_rotation`], so arrayed objects stay
//! fully editable in the shared object form afterwards.

/// One instance of the pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub position: [f32; 3],
    /// Euler degrees (X·Y·Z order) orienting the object's +X axis along the
    /// pattern direction; `None` keeps the template's own rotation.
    pub rotation_euler: Option<[f32; 3]>,
}

/// Euler degrees (X·Y·Z order, matching `SceneObject::update_rotation`)
/// that rotate the +X axis onto direction `dir`. Returns zeros for a
/// degenerate direction.
pub fn euler_pointing(dir: [f32; 3]) -> [f32; 3] {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if len < 1e-9 {
        return [0.0; 3];
    }
    let t = [dir[0] / len, dir[1] / len, dir[2] / len];
    // q = Qx(0)·Qy(θy)·Qz(θz) maps x̂ to (cosθy·cosθz, sinθz, -sinθy·cosθz).
    let theta_z = t[1].clamp(-1.0, 1.0).asin();
    let theta_y = (-t[2]).atan2(t[0]);
    [0.0, theta_y.to_degrees(), theta_z.to_degrees()]
}

/// Linear array: `count` placements from `base`, stepping by `step`.
pub fn linear(base: [f32; 3], step: [f32; 3], count: usize) -> Vec<Placement> {
    (0..count)
        .map(|i| Placement {
            position: [
                base[0] + step[0] * i as f32,
                base[1] + step[1] * i as f32,
                base[2] + step[2] * i as f32,
            ],
            rotation_euler: None,
        })
        .collect()
}

/// Circular array on the plane spanned by the orthonormal `u`/`v` axes:
/// `count` placements on the circle of `radius` around `center`, from
/// `start_deg` sweeping `sweep_deg`. A full 360° sweep leaves out the
/// duplicate end position. `orient` points each instance along the tangent.
pub fn circular(
    center: [f32; 3],
    u: [f32; 3],
    v: [f32; 3],
    radius: f32,
    start_deg: f32,
    sweep_deg: f32,
    count: usize,
    orient: bool,
) -> Vec<Placement> {
    if count == 0 {
        return Vec::new();
    }
    let full_turn = (sweep_deg.abs() - 360.0).abs() < 1e-3;
    // Closed ring: N evenly spaced. Open arc: include both endpoints.
    let denom = if full_turn {
        count as f32
    } else {
        (count.max(2) - 1) as f32
    };
    (0..count)
        .map(|i| {
            let theta = (start_deg + sweep_deg * i as f32 / denom).to_radians();
            let (sin, cos) = theta.sin_cos();
            let position = [
                center[0] + radius * (u[0] * cos + v[0] * sin),
                center[1] + radius * (u[1] * cos + v[1] * sin),
                center[2] + radius * (u[2] * cos + v[2] * sin),
            ];
            let rotation_euler = orient.then(|| {
                let tangent = [
                    -u[0] * sin + v[0] * cos,
                    -u[1] * sin + v[1] * cos,
                    -u[2] * sin + v[2] * cos,
                ];
                euler_pointing(tangent)
            });
            Placement {
                position,
                rotation_euler,
            }
        })
        .collect()
}

/// Array along a polyline: `count` placements evenly spaced by arc length.
/// Open paths include both endpoints; closed paths distribute evenly around
/// the loop. `orient` points each instance along the local tangent. Returns
/// `None` for degenerate input (fewer than 2 points, zero length, count < 1).
pub fn along_path(
    points: &[[f32; 3]],
    closed: bool,
    count: usize,
    orient: bool,
) -> Option<Vec<Placement>> {
    if points.len() < 2 || count == 0 {
        return None;
    }
    // Segment list (with the closing edge for loops) + cumulative lengths.
    let n_segs = if closed {
        points.len()
    } else {
        points.len() - 1
    };
    let seg = |i: usize| -> ([f32; 3], [f32; 3]) {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        (a, b)
    };
    let seg_len = |i: usize| -> f32 {
        let (a, b) = seg(i);
        ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
    };
    let total: f32 = (0..n_segs).map(seg_len).sum();
    if total < 1e-9 {
        return None;
    }

    let denom = if closed {
        count as f32
    } else {
        (count.max(2) - 1) as f32
    };
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // Clamp the last open-path sample onto the final point exactly.
        let target = (total * i as f32 / denom).min(total);
        let (mut acc, mut si) = (0.0f32, 0usize);
        while si + 1 < n_segs && acc + seg_len(si) < target {
            acc += seg_len(si);
            si += 1;
        }
        let (a, b) = seg(si);
        let l = seg_len(si);
        let f = if l < 1e-9 { 0.0 } else { ((target - acc) / l).clamp(0.0, 1.0) };
        let position = [
            a[0] + (b[0] - a[0]) * f,
            a[1] + (b[1] - a[1]) * f,
            a[2] + (b[2] - a[2]) * f,
        ];
        let rotation_euler = orient
            .then(|| euler_pointing([b[0] - a[0], b[1] - a[1], b[2] - a[2]]));
        out.push(Placement {
            position,
            rotation_euler,
        });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    #[test]
    fn linear_steps_evenly() {
        let p = linear([1.0, 0.0, 0.0], [0.5, 0.0, 2.0], 4);
        assert_eq!(p.len(), 4);
        assert_eq!(p[0].position, [1.0, 0.0, 0.0]);
        assert_eq!(p[3].position, [2.5, 0.0, 6.0]);
        assert!(p.iter().all(|pl| pl.rotation_euler.is_none()));
    }

    #[test]
    fn circular_full_turn_spaces_n_without_duplicate() {
        let p = circular(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            2.0,
            0.0,
            360.0,
            4,
            false,
        );
        assert_eq!(p.len(), 4);
        // 4 points at 0/90/180/270°, radius 2 — all distinct.
        assert!(dist(p[0].position, [2.0, 0.0, 0.0]) < 1e-5);
        assert!(dist(p[1].position, [0.0, 2.0, 0.0]) < 1e-5);
        assert!(dist(p[0].position, p[3].position) > 1.0);
        // Every point sits on the circle.
        for pl in &p {
            assert!((dist(pl.position, [0.0; 3]) - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn circular_open_arc_includes_both_endpoints() {
        let p = circular(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            0.0,
            90.0,
            3,
            true,
        );
        assert!(dist(p[0].position, [1.0, 0.0, 0.0]) < 1e-5);
        assert!(dist(p[2].position, [0.0, 1.0, 0.0]) < 1e-5);
        // Oriented: tangent at θ=0 is +Y → +X must rotate by 90° around Z.
        let e = p[0].rotation_euler.unwrap();
        assert!((e[2] - 90.0).abs() < 1e-3, "euler = {:?}", e);
    }

    #[test]
    fn path_resamples_by_arc_length_with_endpoints() {
        // L-shaped path: 2 units along X then 2 along Y (total 4).
        let pts = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]];
        let p = along_path(&pts, false, 5, true).unwrap();
        assert_eq!(p.len(), 5);
        assert!(dist(p[0].position, [0.0, 0.0, 0.0]) < 1e-5);
        assert!(dist(p[2].position, [2.0, 0.0, 0.0]) < 1e-5, "corner at midpoint");
        assert!(dist(p[4].position, [2.0, 2.0, 0.0]) < 1e-5);
        // First segment points along +X (no rotation), last along +Y (90° Z).
        assert!(p[0].rotation_euler.unwrap()[2].abs() < 1e-3);
        assert!((p[4].rotation_euler.unwrap()[2] - 90.0).abs() < 1e-3);
    }

    #[test]
    fn degenerate_paths_are_rejected() {
        assert!(along_path(&[[0.0; 3]], false, 3, false).is_none());
        assert!(along_path(&[[1.0; 3], [1.0; 3]], false, 3, false).is_none());
        assert!(along_path(&[[0.0; 3], [1.0, 0.0, 0.0]], false, 0, false).is_none());
    }

    #[test]
    fn euler_pointing_matches_update_rotation_convention() {
        use cgmath::{InnerSpace, Quaternion, Rad, Rotation, Rotation3};
        // For a bundle of directions, applying Qx·Qy·Qz (the SceneObject
        // convention) to +X must reproduce the direction.
        for dir in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.5, 0.3],
            [0.2, -0.9, 0.4],
        ] {
            let e = euler_pointing(dir);
            let q = Quaternion::from_angle_x(Rad(e[0].to_radians()))
                * Quaternion::from_angle_y(Rad(e[1].to_radians()))
                * Quaternion::from_angle_z(Rad(e[2].to_radians()));
            let rotated = q.rotate_vector(cgmath::Vector3::unit_x());
            let want = cgmath::Vector3::from(dir).normalize();
            assert!((rotated - want).magnitude() < 1e-4, "dir {:?}", dir);
        }
    }
}
