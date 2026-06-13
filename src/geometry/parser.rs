//! Plain-text geometry parsers: the geometry-string DSL and XYZ point text.
//!
//! # DSL grammar
//!
//! One record per line (or `;`-separated). Comments start with `#` or `//`.
//!
//! ```text
//! <shape> <x> <y> <z> [size | sx sy sz] [options...]
//! ```
//!
//! - `shape`: `cube|box`, `sphere|ball`, `plane|quad`, `point|dot|vertex`
//! - bare numbers after the position: one = uniform scale, three = per-axis
//! - options, in any order:
//!   - `#rrggbb` / `#rgb` / `r,g,b` — color (0–1 or 0–255)
//!   - `rot=rx,ry,rz` — Euler rotation in degrees
//!   - `size=v` / `radius=v` / `scale=sx,sy,sz` — explicit scale
//!   - `color=...`, `label=...`, `name=...`
//!   - any other bare word(s) — label
//!
//! Example:
//!
//! ```text
//! # a small scene
//! cube   0 0 0  2        #ff8800  base
//! sphere 0 0 2  0.5      color=0,1,0 label=marker
//! plane  0 0 -1 4 4 1    rot=0,0,45  floor
//! point  1 1 1
//! ```
//!
//! # XYZ point text
//!
//! Lines of `x y z [size]` (commas or whitespace), rendered as point
//! spheres. [`looks_like_xyz`] auto-detects this format for pasted text and
//! `.txt` files whose lines start with numbers.

use crate::scene::GeometryType;

use super::{parse_color, parse_shape, GeometryError, GeometryRecord, Result, POINT_SIZE};

/// Split free text into logical records: newline or `;`, comments stripped.
fn logical_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .flat_map(|(i, line)| line.split(';').map(move |part| (i + 1, part)))
        .map(|(n, part)| (n, part.trim()))
        .filter(|(_, part)| {
            !part.is_empty() && !part.starts_with('#') && !part.starts_with("//")
        })
}

/// Parse the geometry-string DSL (see module docs for the grammar).
pub fn parse_dsl(text: &str, default_color: [f32; 3]) -> Result<Vec<GeometryRecord>> {
    let mut records = Vec::new();
    for (line_no, line) in logical_lines(text) {
        records.push(parse_dsl_record(line, line_no, default_color)?);
    }
    Ok(records)
}

fn parse_dsl_record(
    line: &str,
    line_no: usize,
    default_color: [f32; 3],
) -> Result<GeometryRecord> {
    let err = |message: String| GeometryError::Parse {
        line: line_no,
        message,
    };
    let mut tokens = line.split_whitespace().peekable();

    let shape_token = tokens.next().ok_or_else(|| err("empty record".into()))?;
    let (shape, is_point) = parse_shape(shape_token)
        .ok_or_else(|| err(format!("unknown shape '{}'", shape_token)))?;

    // Position: exactly three numbers.
    let mut pos = [0.0f32; 3];
    for p in &mut pos {
        let t = tokens
            .next()
            .ok_or_else(|| err("expected x y z after the shape".into()))?;
        *p = t
            .parse()
            .map_err(|_| err(format!("'{}' is not a coordinate", t)))?;
    }
    let mut record = GeometryRecord::new(shape, pos);
    record.color = default_color;
    if is_point {
        record.scale = [POINT_SIZE; 3];
    }

    // Bare numeric run after the position = scale (1 uniform or 3 per-axis).
    let mut scale_run: Vec<f32> = Vec::new();
    while let Some(t) = tokens.peek() {
        match t.parse::<f32>() {
            Ok(v) => {
                scale_run.push(v);
                tokens.next();
            }
            Err(_) => break,
        }
    }
    match scale_run.len() {
        0 => {}
        1 => record.scale = [scale_run[0]; 3],
        3 => record.scale = [scale_run[0], scale_run[1], scale_run[2]],
        n => return Err(err(format!("expected 1 or 3 scale values, found {}", n))),
    }

    // Remaining tokens: options and label words.
    let mut label_words: Vec<&str> = Vec::new();
    for t in tokens {
        if t.starts_with('#') {
            record.color = parse_color(t).ok_or_else(|| err(format!("bad color '{}'", t)))?;
        } else if let Some((key, value)) = t.split_once('=') {
            match key.to_ascii_lowercase().as_str() {
                "color" | "colour" => {
                    record.color =
                        parse_color(value).ok_or_else(|| err(format!("bad color '{}'", value)))?
                }
                "rot" | "rotation" => {
                    record.rotation = parse_triple(value)
                        .ok_or_else(|| err(format!("bad rotation '{}'", value)))?
                }
                "scale" => {
                    record.scale = parse_triple(value)
                        .ok_or_else(|| err(format!("bad scale '{}'", value)))?
                }
                "size" | "radius" | "r" => {
                    let v: f32 = value
                        .parse()
                        .map_err(|_| err(format!("bad size '{}'", value)))?;
                    record.scale = [v; 3];
                }
                "label" | "name" => label_words.push(value),
                other => return Err(err(format!("unknown option '{}'", other))),
            }
        } else if t.contains(',') {
            record.color = parse_color(t).ok_or_else(|| err(format!("bad color '{}'", t)))?;
        } else {
            label_words.push(t);
        }
    }
    if !label_words.is_empty() {
        record.label = Some(label_words.join(" "));
    }
    Ok(record)
}

fn parse_triple(value: &str) -> Option<[f32; 3]> {
    let parts: Vec<f32> = value
        .split(',')
        .map(|p| p.trim().parse::<f32>())
        .collect::<std::result::Result<_, _>>()
        .ok()?;
    (parts.len() == 3).then(|| [parts[0], parts[1], parts[2]])
}

/// True when the text looks like bare XYZ point data (every non-comment
/// line starts with a number) rather than the shape-first DSL.
pub fn looks_like_xyz(text: &str) -> bool {
    let mut any = false;
    for (_, line) in logical_lines(text) {
        any = true;
        let first = line
            .split(|c: char| c.is_whitespace() || c == ',')
            .find(|t| !t.is_empty());
        match first {
            Some(t) if t.parse::<f32>().is_ok() => {}
            _ => return false,
        }
    }
    any
}

/// Parse XYZ point text: `x y z [size]` per line, commas or whitespace.
pub fn parse_xyz(text: &str, color: [f32; 3]) -> Result<Vec<GeometryRecord>> {
    let mut records = Vec::new();
    for (line_no, line) in logical_lines(text) {
        let nums: Vec<f32> = line
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|t| !t.is_empty())
            .map(|t| {
                t.parse::<f32>().map_err(|_| GeometryError::Parse {
                    line: line_no,
                    message: format!("'{}' is not a number", t),
                })
            })
            .collect::<Result<_>>()?;
        if nums.len() < 3 {
            return Err(GeometryError::Parse {
                line: line_no,
                message: format!("expected at least x y z, found {} values", nums.len()),
            });
        }
        let mut record =
            GeometryRecord::new(GeometryType::Sphere, [nums[0], nums[1], nums[2]]);
        record.scale = [nums.get(3).copied().unwrap_or(POINT_SIZE); 3];
        record.color = color;
        records.push(record);
    }
    Ok(records)
}

/// Parse pasted text, auto-detecting XYZ vs DSL.
pub fn parse_auto(text: &str, default_color: [f32; 3]) -> Result<Vec<GeometryRecord>> {
    if looks_like_xyz(text) {
        parse_xyz(text, default_color)
    } else {
        parse_dsl(text, default_color)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::DEFAULT_COLOR;

    #[test]
    fn dsl_parses_full_scene() {
        let text = "\
# comment
cube   0 0 0  2        #ff8800  base
sphere 0 0 2  0.5      color=0,1,0 label=marker
plane  0 0 -1 4 4 1    rot=0,0,45  floor tile
point  1 1 1; ball 5 5 5 radius=2
";
        let recs = parse_dsl(text, DEFAULT_COLOR).unwrap();
        assert_eq!(recs.len(), 5);
        assert_eq!(recs[0].shape, GeometryType::Cube);
        assert_eq!(recs[0].scale, [2.0; 3]);
        assert_eq!(recs[0].label.as_deref(), Some("base"));
        assert!((recs[0].color[0] - 1.0).abs() < 1e-3);
        assert_eq!(recs[1].color, [0.0, 1.0, 0.0]);
        assert_eq!(recs[2].scale, [4.0, 4.0, 1.0]);
        assert_eq!(recs[2].rotation, [0.0, 0.0, 45.0]);
        assert_eq!(recs[2].label.as_deref(), Some("floor tile"));
        assert_eq!(recs[3].scale, [POINT_SIZE; 3]);
        assert_eq!(recs[4].scale, [2.0; 3]);
    }

    #[test]
    fn dsl_reports_line_numbers_on_errors() {
        let bad = "cube 0 0 0\nspherex 1 2 3";
        let err = parse_dsl(bad, DEFAULT_COLOR).unwrap_err();
        assert!(err.to_string().contains("line 2"));
        assert!(err.to_string().contains("spherex"));

        for bad in [
            "cube 0 zero 0",
            "cube 0 0",
            "cube 0 0 0 1 2",
            "cube 0 0 0 #nothex",
            "cube 0 0 0 rot=1,2",
            "cube 0 0 0 wrench=5",
            "cube 0 0 0 size=big",
        ] {
            assert!(parse_dsl(bad, DEFAULT_COLOR).is_err(), "{}", bad);
        }
    }

    #[test]
    fn xyz_detection_and_parsing() {
        let xyz = "1 2 3\n4,5,6,0.5\n# note\n7 8 9";
        assert!(looks_like_xyz(xyz));
        assert!(!looks_like_xyz("cube 0 0 0"));
        assert!(!looks_like_xyz(""));
        assert!(!looks_like_xyz("# only comments"));

        let recs = parse_xyz(xyz, [1.0, 0.0, 0.0]).unwrap();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(recs[0].scale, [POINT_SIZE; 3]);
        assert_eq!(recs[1].scale, [0.5; 3]);
        assert_eq!(recs[2].color, [1.0, 0.0, 0.0]);

        assert!(parse_xyz("1 2", DEFAULT_COLOR).is_err());
        assert!(parse_xyz("1 2 x", DEFAULT_COLOR).is_err());
    }

    #[test]
    fn auto_routes_between_formats() {
        let pts = parse_auto("0 0 0\n1 1 1", DEFAULT_COLOR).unwrap();
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].shape, GeometryType::Sphere);

        let shapes = parse_auto("cube 0 0 0", DEFAULT_COLOR).unwrap();
        assert_eq!(shapes[0].shape, GeometryType::Cube);
    }
}
