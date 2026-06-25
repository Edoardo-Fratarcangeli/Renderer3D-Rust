// Library entry point - exposes modules for integration testing

#[macro_use]
extern crate rust_i18n;

// Embeds every translation file under locales/ at compile time and registers
// them with a global backend. English is the fallback for missing keys.
rust_i18n::i18n!("locales", fallback = "en");

#[macro_use]
pub mod logger;
pub mod brep;
pub mod camera;
pub mod dataset;
pub mod geometry;
pub mod i18n;
pub mod mesh;
pub mod model;
pub mod primitives;
pub mod scene;
pub mod sketch;
pub mod state;
pub mod ui;
pub mod updater;
pub(crate) mod util;
pub mod visualization;
