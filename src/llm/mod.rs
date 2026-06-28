//! LLM / SLM parameter visualization.
//!
//! Sub-modules:
//!  - [`network`]    — graph data structures and 3D layout.
//!  - [`arch`]       — generic architecture registry (LLaMA, Mistral, Gemma, …).
//!  - [`loader`]     — JSON and GGUF file parsers.
//!  - [`ollama`]     — Ollama REST client (list models, architecture, inference).
//!  - [`tokenizer`]  — lightweight whitespace tokenizer.
//!  - [`activation`] — activation-wave simulation and per-frame glow values.
//!
//! The [`LlmView`] struct lives in [`crate::ui::llm_panel`] (same pattern as
//! [`crate::ui::geometry_panel::GeometryView`]) so the rendering code can
//! use egui directly.

pub mod activation;
pub mod arch;
pub mod loader;
pub mod network;
pub mod ollama;
pub mod tokenizer;
