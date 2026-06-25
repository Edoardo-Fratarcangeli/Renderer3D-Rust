//! A 2D profile: an ordered chain of [`Segment`]s, either a closed loop (which
//! bounds a fillable surface) or an open polyline.
//!
//! The profile owns validation (endpoint continuity, closure) and flattening
//! to a polyline; it has no knowledge of 3D, the GPU or egui.

use super::segment::Segment;
use super::Vec2;

/// Maximum gap between consecutive endpoints still considered "joined".
const JOIN_EPS: f32 = 1e-4;

/// Why a profile could not be used.
#[derive(Debug, Clone, PartialEq)]
pub enum ProfileError {
    /// No segments at all.
    Empty,
    /// Segment `index`'s start does not meet the previous segment's end.
    Discontinuous { index: usize, gap: f32 },
    /// A closed profile whose last point does not return to the first.
    NotClosed { gap: f32 },
    /// Fewer than 3 distinct points: nothing to fill.
    Degenerate,
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfileError::Empty => write!(f, "profile has no segments"),
            ProfileError::Discontinuous { index, gap } => write!(
                f,
                "segment {index} does not connect to the previous one (gap {gap:.4})"
            ),
            ProfileError::NotClosed { gap } => {
                write!(
                    f,
                    "closed profile does not return to its start (gap {gap:.4})"
                )
            }
            ProfileError::Degenerate => write!(f, "profile has fewer than 3 distinct points"),
        }
    }
}

impl std::error::Error for ProfileError {}

/// An ordered chain of segments plus a closed/open flag.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    pub segments: Vec<Segment>,
    pub closed: bool,
}

impl Profile {
    /// Wrap a segment list. Call [`Profile::validate`] before use.
    pub fn new(segments: Vec<Segment>, closed: bool) -> Self {
        Self { segments, closed }
    }

    /// A regular `n`-gon (`n >= 3`) of `radius`, centered at the origin, wound
    /// counter-clockwise and closed. First vertex sits on +X.
    pub fn regular_polygon(n: usize, radius: f32) -> Result<Self, ProfileError> {
        if n < 3 {
            return Err(ProfileError::Degenerate);
        }
        let pts: Vec<Vec2> = (0..n)
            .map(|i| {
                let a = std::f32::consts::TAU * (i as f32) / (n as f32);
                [radius * a.cos(), radius * a.sin()]
            })
            .collect();
        Self::from_points(&pts, true)
    }

    /// A circle of `radius` centered at the origin, built from four CCW arc
    /// quadrants (a genuinely curved closed profile).
    pub fn circle(radius: f32) -> Result<Self, ProfileError> {
        use std::f32::consts::PI;
        if radius <= 1e-6 {
            return Err(ProfileError::Degenerate);
        }
        let q = |start: f32, end: f32| Segment::Arc {
            center: [0.0, 0.0],
            radius,
            start,
            end,
        };
        let segments = vec![
            q(0.0, PI / 2.0),
            q(PI / 2.0, PI),
            q(PI, 3.0 * PI / 2.0),
            q(3.0 * PI / 2.0, 2.0 * PI),
        ];
        Ok(Self::new(segments, true))
    }

    /// Build a straight-edged profile from explicit points. When `closed`, the
    /// final point is joined back to the first automatically (do not repeat it).
    pub fn from_points(points: &[Vec2], closed: bool) -> Result<Self, ProfileError> {
        if points.len() < 2 {
            return Err(ProfileError::Degenerate);
        }
        let mut segments = Vec::with_capacity(points.len());
        for w in points.windows(2) {
            segments.push(Segment::Line { a: w[0], b: w[1] });
        }
        if closed {
            segments.push(Segment::Line {
                a: *points.last().unwrap(),
                b: points[0],
            });
        }
        Ok(Self::new(segments, closed))
    }

    /// Check endpoint continuity (and closure, for closed profiles).
    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.segments.is_empty() {
            return Err(ProfileError::Empty);
        }
        for i in 1..self.segments.len() {
            let prev = self.segments[i - 1].end();
            let cur = self.segments[i].start();
            let gap = dist(prev, cur);
            if gap > JOIN_EPS {
                return Err(ProfileError::Discontinuous { index: i, gap });
            }
        }
        if self.closed {
            let gap = dist(
                self.segments.last().unwrap().end(),
                self.segments[0].start(),
            );
            if gap > JOIN_EPS {
                return Err(ProfileError::NotClosed { gap });
            }
        }
        Ok(())
    }

    /// Flatten to a polyline of 2D points. For a closed profile the returned
    /// loop does **not** repeat the first vertex at the end.
    pub fn flatten(&self, tol: f32) -> Result<Vec<Vec2>, ProfileError> {
        self.validate()?;
        let mut pts = vec![self.segments[0].start()];
        for seg in &self.segments {
            seg.flatten_into(tol, &mut pts);
        }
        // A closed loop's last flattened point coincides with the first; drop it.
        if self.closed && pts.len() >= 2 && dist(*pts.last().unwrap(), pts[0]) <= JOIN_EPS {
            pts.pop();
        }
        // Remove any consecutive duplicates that survived flattening.
        dedup_consecutive(&mut pts);
        if self.closed && pts.len() < 3 {
            return Err(ProfileError::Degenerate);
        }
        Ok(pts)
    }
}

/// Signed area of a 2D polygon (shoelace formula). Positive = CCW.
pub fn signed_area(points: &[Vec2]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        acc += a[0] * b[1] - b[0] * a[1];
    }
    acc * 0.5
}

fn dist(a: Vec2, b: Vec2) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn dedup_consecutive(pts: &mut Vec<Vec2>) {
    pts.dedup_by(|a, b| dist(*a, *b) <= JOIN_EPS);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_polygon_has_n_points_and_positive_area() {
        let p = Profile::regular_polygon(6, 1.0).unwrap();
        let pts = p.flatten(0.01).unwrap();
        assert_eq!(pts.len(), 6, "hexagon has six corners");
        // Generated CCW, so signed area is positive.
        assert!(signed_area(&pts) > 0.0);
        // Area of a regular hexagon with r=1 is 3√3/2 ≈ 2.598.
        assert!((signed_area(&pts) - 2.598).abs() < 0.05);
    }

    #[test]
    fn too_few_sides_is_degenerate() {
        assert_eq!(
            Profile::regular_polygon(2, 1.0),
            Err(ProfileError::Degenerate)
        );
    }

    #[test]
    fn circle_profile_is_closed_and_round() {
        let p = Profile::circle(2.0).unwrap();
        p.validate().unwrap();
        let pts = p.flatten(0.01).unwrap();
        // Every point is on the circle of radius 2.
        for q in &pts {
            assert!(((q[0] * q[0] + q[1] * q[1]).sqrt() - 2.0).abs() < 1e-2);
        }
        // Area approaches πr² = 4π ≈ 12.566.
        assert!((signed_area(&pts).abs() - 12.566).abs() < 0.2);
    }

    #[test]
    fn from_points_closed_joins_back_to_start() {
        let p =
            Profile::from_points(&[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], true).unwrap();
        assert_eq!(p.segments.len(), 4, "closing edge added automatically");
        p.validate().unwrap();
        let pts = p.flatten(0.01).unwrap();
        assert_eq!(pts.len(), 4);
        assert!(
            (signed_area(&pts) - 4.0).abs() < 1e-4,
            "unit-ish square area"
        );
    }

    #[test]
    fn open_polyline_is_not_required_to_close() {
        let p = Profile::from_points(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], false).unwrap();
        assert_eq!(p.segments.len(), 2);
        p.validate().unwrap();
        let pts = p.flatten(0.01).unwrap();
        assert_eq!(pts.len(), 3);
    }

    #[test]
    fn discontinuous_chain_is_rejected() {
        let segs = vec![
            Segment::Line {
                a: [0.0, 0.0],
                b: [1.0, 0.0],
            },
            // Starts away from the previous end (0.5 gap).
            Segment::Line {
                a: [1.5, 0.0],
                b: [2.0, 0.0],
            },
        ];
        let p = Profile::new(segs, false);
        assert!(matches!(
            p.validate(),
            Err(ProfileError::Discontinuous { index: 1, .. })
        ));
    }

    #[test]
    fn closed_flag_without_closure_is_rejected() {
        let segs = vec![
            Segment::Line {
                a: [0.0, 0.0],
                b: [1.0, 0.0],
            },
            Segment::Line {
                a: [1.0, 0.0],
                b: [1.0, 1.0],
            },
        ];
        let p = Profile::new(segs, true);
        assert!(matches!(p.validate(), Err(ProfileError::NotClosed { .. })));
    }

    #[test]
    fn signed_area_sign_tracks_winding() {
        let ccw = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let cw = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        assert!(signed_area(&ccw) > 0.0);
        assert!(signed_area(&cw) < 0.0);
    }
}
