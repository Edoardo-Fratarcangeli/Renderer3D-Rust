// Main entry point for `tests/ui` integration test suite
mod dataset_panels;
// `defaults` deliberately asserts compile-time-known default constants as
// regression guards, which trips clippy::assertions_on_constants by design.
#[allow(clippy::assertions_on_constants)]
mod defaults;
mod geometry_panel_tests;
mod icon_test;
mod layout_tests;
mod panel_helpers;
