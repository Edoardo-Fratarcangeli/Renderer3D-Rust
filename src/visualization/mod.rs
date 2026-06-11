// Visualization layer: turns a projected dataset into GPU-ready instance
// batches. Pure data transformation — no wgpu device access here, so the
// whole layer is unit-testable without a GPU.

pub mod color_mapper;
pub mod geometry_assigner;
pub mod point_cloud;
