// Deterministic label id -> RGB color mapping.

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
