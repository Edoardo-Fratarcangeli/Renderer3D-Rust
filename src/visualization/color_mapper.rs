//! Deterministic label id -> RGB color mapping.
//!
//! The base palette is Okabe-Ito (colorblind friendly) extended to twelve
//! entries; ids beyond the palette get darkened variants so any number of
//! labels still maps to distinct, stable colors.

/// Qualitative base palette (colorblind-friendly Okabe-Ito + extras).
const PALETTE: [[f32; 3]; 12] = [
    [0.902, 0.624, 0.000], // orange
    [0.337, 0.706, 0.914], // sky blue
    [0.000, 0.620, 0.451], // bluish green
    [0.941, 0.894, 0.259], // yellow
    [0.000, 0.447, 0.698], // blue
    [0.835, 0.369, 0.000], // vermillion
    [0.800, 0.475, 0.655], // reddish purple
    [0.580, 0.404, 0.741], // violet
    [0.549, 0.337, 0.294], // brown
    [0.890, 0.467, 0.761], // pink
    [0.498, 0.498, 0.498], // grey
    [0.090, 0.745, 0.812], // cyan
];

/// Color for a label id. Ids beyond the palette get hue-rotated variants,
/// so any number of labels still maps to distinct, stable colors.
pub fn color_for_label(label: u32) -> [f32; 3] {
    let base = PALETTE[(label as usize) % PALETTE.len()];
    let round = (label as usize) / PALETTE.len();
    if round == 0 {
        return base;
    }
    // Darken successive rounds so repeats stay distinguishable.
    let factor = 1.0 / (1.0 + round as f32 * 0.45);
    [base[0] * factor, base[1] * factor, base[2] * factor]
}

/// Full color table for `n` labels.
pub fn palette(n: usize) -> Vec<[f32; 3]> {
    (0..n as u32).map(color_for_label).collect()
}

/// How point colors are chosen.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Qualitative color per label id (the default).
    #[default]
    ByLabel,
    /// Sequential gradient driven by the distance from the cloud center,
    /// turning the radial distance into an extra visual dimension.
    ByDistance,
}

/// Sequential viridis-like gradient for a normalized scalar `t` in `[0, 1]`
/// (used to encode the distance from the center). `t` is clamped.
pub fn color_for_distance(t: f32) -> [f32; 3] {
    const STOPS: [[f32; 3]; 5] = [
        [0.267, 0.005, 0.329], // deep purple (near center)
        [0.231, 0.320, 0.545], // blue
        [0.128, 0.567, 0.551], // teal
        [0.369, 0.788, 0.383], // green
        [0.993, 0.906, 0.144], // yellow (far edge)
    ];
    let t = t.clamp(0.0, 1.0);
    let scaled = t * (STOPS.len() - 1) as f32;
    let i = (scaled.floor() as usize).min(STOPS.len() - 2);
    let f = scaled - i as f32;
    let a = STOPS[i];
    let b = STOPS[i + 1];
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}
