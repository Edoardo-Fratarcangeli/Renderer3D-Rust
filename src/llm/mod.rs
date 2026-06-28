//! LLM / SLM parameter visualization.
//!
//! Sub-modules:
//!  - [`network`]    — graph data structures and 3D layout.
//!  - [`loader`]     — JSON and GGUF file parsers.
//!  - [`tokenizer`]  — lightweight whitespace tokenizer.
//!  - [`activation`] — activation-wave simulation and per-frame glow values.
//!
//! The [`LlmView`] struct lives in [`crate::ui::llm_panel`] (same pattern as
//! [`crate::ui::geometry_panel::GeometryView`]) so the rendering code can
//! use egui directly.

pub mod activation;
pub mod loader;
pub mod network;
pub mod tokenizer;
