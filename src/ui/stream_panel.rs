//! Stream tab: configure and drive a runtime data stream.
//!
//! Like every other panel this is a pure function over explicit state
//! ([`StreamState`]); it never opens sockets itself, it only returns a
//! [`StreamUiAction`] for [`super::DatasetView`] to act on. The live session
//! handle lives in [`StreamState::handle`] and is polled each frame by the
//! view.

use std::time::Duration;

use super::import_dialog::method_radio;
use super::{projection_spec_from, StatusMessage};
use crate::dataset::preprocessor::ProjectionSpec;
use crate::dataset::stream::{StreamFormat, StreamHandle, StreamStatus};

/// "Receiving" window: data seen within this is shown as the blue state.
pub const RECEIVING_WINDOW: Duration = Duration::from_millis(500);

/// State of the Stream tab (config form + live handle + status).
pub struct StreamState {
    pub format: StreamFormat,
    /// Bind address the app listens on (producers connect to it).
    pub addr: String,
    /// Rolling buffer cap.
    pub max_rows: usize,
    /// Projection flags (mirrors the import form; shared spec builder).
    pub use_pca: bool,
    pub use_radial: bool,
    pub dims: u8,
    pub axes: [usize; 3],
    /// Last user-facing status line.
    pub status: Option<StatusMessage>,
    /// The live session, when streaming.
    pub handle: Option<StreamHandle>,
    /// Last buffer version the view rendered (change detection).
    pub seen_version: u64,
    /// Number of label classes already enabled (so new ones can be revealed).
    pub known_labels: usize,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            format: StreamFormat::Ndjson,
            addr: "127.0.0.1:8765".to_string(),
            max_rows: 50_000,
            // Radial is the natural default for live data: it is stateless
            // per-row, so the cloud does not reorient as rows arrive (unlike
            // PCA, which refits each refresh).
            use_pca: false,
            use_radial: true,
            dims: 3,
            axes: [0, 1, 2],
            status: None,
            handle: None,
            seen_version: 0,
            known_labels: 0,
        }
    }
}

impl StreamState {
    /// Whether a session is currently running.
    pub fn is_active(&self) -> bool {
        self.handle.is_some()
    }

    /// Projection used for the live preview.
    pub fn projection(&self) -> ProjectionSpec {
        projection_spec_from(self.use_pca, self.use_radial, self.dims, self.axes)
    }

    /// Status-dot color: gray idle, green active, blue receiving, red error.
    pub fn dot_color(&self) -> egui::Color32 {
        match &self.handle {
            None => egui::Color32::from_gray(110),
            Some(h) => match h.status() {
                StreamStatus::Error => egui::Color32::from_rgb(230, 80, 80),
                StreamStatus::Active if h.is_receiving(RECEIVING_WINDOW) => {
                    egui::Color32::from_rgb(70, 140, 255) // blue: receiving
                }
                StreamStatus::Active => egui::Color32::from_rgb(80, 210, 110), // green: active
                StreamStatus::Idle | StreamStatus::Stopped => egui::Color32::from_gray(110),
            },
        }
    }
}

/// What the Stream tab asks the view to do this frame.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamUiAction {
    None,
    /// Start a session with the current configuration.
    Start,
    /// Stop the running session.
    Stop,
}

/// Draw the Stream tab; returns the requested action.
pub fn show(ui: &mut egui::Ui, state: &mut StreamState) -> StreamUiAction {
    let mut action = StreamUiAction::None;
    let active = state.is_active();

    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new(t!("stream.heading").to_string()).heading());
        ui.label(egui::RichText::new(t!("stream.subheading").to_string()).weak());
    });
    ui.add_space(8.0);

    // Configuration is locked while a session runs.
    ui.add_enabled_ui(!active, |ui| {
        egui::Grid::new("stream_form_grid")
            .num_columns(2)
            .spacing([12.0, 8.0])
            .show(ui, |ui| {
                ui.label(t!("stream.format").to_string());
                ui.horizontal(|ui| {
                    ui.radio_value(
                        &mut state.format,
                        StreamFormat::Ndjson,
                        t!("stream.format_ndjson").to_string(),
                    );
                    ui.radio_value(
                        &mut state.format,
                        StreamFormat::ArrowIpc,
                        t!("stream.format_arrow").to_string(),
                    );
                });
                ui.end_row();

                ui.label(t!("stream.address").to_string());
                ui.add(
                    egui::TextEdit::singleline(&mut state.addr)
                        .desired_width(f32::INFINITY)
                        .hint_text("127.0.0.1:8765"),
                );
                ui.end_row();

                ui.label(t!("stream.max_rows").to_string());
                ui.add(egui::DragValue::new(&mut state.max_rows).range(100..=5_000_000));
                ui.end_row();

                ui.label(t!("dataset.method").to_string());
                ui.horizontal(|ui| {
                    method_radio(ui, &mut state.use_pca, &mut state.use_radial);
                });
                ui.end_row();

                ui.label(t!("dataset.dimensions").to_string());
                ui.horizontal(|ui| {
                    ui.radio_value(&mut state.dims, 3, "3D");
                    ui.radio_value(&mut state.dims, 2, "2D");
                    ui.radio_value(&mut state.dims, 1, "1D");
                });
                ui.end_row();
            });
    });

    ui.add_space(10.0);

    // Start / Stop with the colored status dot.
    ui.vertical_centered(|ui| {
        let dot = egui::RichText::new("⏺").color(state.dot_color()).size(16.0);
        let label = if active {
            t!("stream.stop").to_string()
        } else {
            t!("stream.start").to_string()
        };
        ui.horizontal(|ui| {
            ui.label(dot);
            let button = egui::Button::new(egui::RichText::new(label).size(14.0))
                .min_size(egui::vec2(140.0, 28.0));
            if ui.add(button).clicked() {
                action = if active {
                    StreamUiAction::Stop
                } else {
                    StreamUiAction::Start
                };
            }
        });
    });

    // Live counters.
    if let Some(h) = &state.handle {
        ui.add_space(6.0);
        ui.vertical_centered(|ui| {
            let received = h.total_received();
            let retained = h.with_buffer(|b| b.n_rows());
            ui.label(
                t!(
                    "stream.stats",
                    addr = h.addr().to_string(),
                    received = received.to_string(),
                    retained = retained.to_string()
                )
                .to_string(),
            );
        });
    }

    if let Some(status) = &state.status {
        ui.add_space(8.0);
        ui.separator();
        ui.vertical_centered(|ui| status.show(ui));
    }

    action
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_idle_radial() {
        let s = StreamState::default();
        assert!(!s.is_active());
        assert_eq!(s.format, StreamFormat::Ndjson);
        assert!(s.use_radial);
        assert_eq!(s.dot_color(), egui::Color32::from_gray(110));
    }

    #[test]
    fn projection_follows_flags() {
        let s = StreamState {
            use_radial: false,
            use_pca: true,
            dims: 2,
            ..Default::default()
        };
        assert_eq!(s.projection().dims, 2);
        assert!(matches!(
            s.projection().method,
            crate::dataset::preprocessor::ProjectionMethod::Pca
        ));
    }
}
