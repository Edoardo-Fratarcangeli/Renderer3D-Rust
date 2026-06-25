//! Dataset visualizer UI.
//!
//! This layer only renders state and forwards user intent; parsing, indexing
//! and projection live in [`crate::dataset`] / [`crate::visualization`] and
//! run on a background thread so the UI never blocks on large files.
//!
//! Structure:
//! - [`DatasetView`] is the single owner of all visualizer UI state and is
//!   embedded in `State`. Each frame, `State` calls [`DatasetView::show`].
//! - The window is organized in tabs ([`DatasetTab`]); each tab delegates to
//!   a focused panel module ([`import_dialog`], [`label_filter`],
//!   [`distribution_chart`], [`export_panel`]). Imported datasets are listed
//!   in the shared object list of the main window (see `state`), not in a
//!   dedicated tab. [`dataset_table`] / [`search_panel`] remain as reusable
//!   helpers (e.g. [`dataset_table::row_text`]).
//! - Panels are plain functions over explicit state, so they can be driven
//!   headless (no GPU) by the integration tests in `tests/ui`.
//!
//! The universal geometry import window lives in [`geometry_panel`] and
//! follows the same architecture (worker thread + dirty flag).

pub mod dataset_table;
pub mod distribution_chart;
pub mod export_panel;
pub mod geometry_panel;
pub mod import_dialog;
pub mod label_filter;
pub mod search_panel;
pub mod stream_panel;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use crate::dataset::index::{apply_filter, FilterSpec, SearchQuery};
use crate::dataset::preprocessor::{self, Projection, ProjectionMethod, ProjectionSpec};
use crate::dataset::stream::{StreamConfig, StreamEvent, StreamSession};
use crate::dataset::{builtin, loader, Dataset, DatasetIndex};
use crate::visualization::point_cloud::{self, PointCloudSettings};

/// Directory holding metadata / index / projection caches.
pub const CACHE_DIR: &str = ".r3d_cache";

/// A dataset ready for display: data + label index + 3D projection.
pub struct LoadedDataset {
    /// The decoded dataset (features may stay memory mapped on disk).
    pub dataset: Dataset,
    /// Label -> rows index used by filters.
    pub index: DatasetIndex,
    /// Normalized 3D coordinates, one per row.
    pub projection: Projection,
}

/// What the dataset UI asks the host (`State`) to do this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DatasetAction {
    /// Nothing to do.
    None,
    /// Move the camera target onto this world-space point.
    FocusPoint([f32; 3]),
}

/// Where an import should read from.
#[derive(Debug, Clone, PartialEq)]
pub enum ImportSource {
    /// A dataset file on disk (NPY/NPZ/CSV/IDX/Parquet).
    Path(PathBuf),
    /// One of the synthetic benchmark generators (see [`builtin`]).
    Builtin(&'static str),
}

/// A fully specified import request produced by the import dialog.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportRequest {
    /// File path or builtin generator name.
    pub source: ImportSource,
    /// Optional hard cap on imported rows.
    pub max_rows: Option<usize>,
    /// Projection (method + dimensions + axes) used for the 3D preview.
    pub projection: ProjectionSpec,
}

/// Severity of a [`StatusMessage`]; controls its color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusKind {
    /// Neutral progress/info text.
    Info,
    /// A completed operation.
    Success,
    /// A failed operation.
    Error,
}

/// A colored, user-facing status line shown under a panel.
#[derive(Debug, Clone, PartialEq)]
pub struct StatusMessage {
    /// Severity (drives the color).
    pub kind: StatusKind,
    /// Human readable text.
    pub text: String,
}

impl StatusMessage {
    /// Neutral informational message.
    pub fn info(text: impl Into<String>) -> Self {
        Self {
            kind: StatusKind::Info,
            text: text.into(),
        }
    }

    /// Green success message.
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            kind: StatusKind::Success,
            text: text.into(),
        }
    }

    /// Red error message.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            kind: StatusKind::Error,
            text: text.into(),
        }
    }

    /// Color used to render this message in the current theme.
    pub fn color(&self, visuals: &egui::Visuals) -> egui::Color32 {
        match self.kind {
            StatusKind::Info => visuals.text_color(),
            StatusKind::Success => egui::Color32::from_rgb(120, 220, 120),
            StatusKind::Error => egui::Color32::from_rgb(255, 120, 120),
        }
    }

    /// Render the message as a colored label.
    pub fn show(&self, ui: &mut egui::Ui) {
        let color = self.color(ui.visuals());
        ui.colored_label(color, &self.text);
    }
}

/// Tabs of the visualizer window.
///
/// The per-row table ("Explore") was removed: imported datasets now appear as
/// a single entry in the shared object list at the bottom of the main window,
/// alongside scene objects and solid layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetTab {
    /// File / benchmark import.
    Import,
    /// Runtime data streaming (TCP NDJSON / Arrow IPC).
    Stream,
    /// Per-label visibility filters + distribution chart.
    Labels,
    /// Point size and shape policy.
    View,
    /// CSV export of the filtered subset.
    Export,
}

impl DatasetTab {
    /// All tabs in display order.
    pub const ALL: [DatasetTab; 5] = [
        DatasetTab::Import,
        DatasetTab::Stream,
        DatasetTab::Labels,
        DatasetTab::View,
        DatasetTab::Export,
    ];

    /// Icon + title shown on the tab button.
    pub fn title(&self) -> String {
        match self {
            DatasetTab::Import => t!("dataset.tab_import").to_string(),
            DatasetTab::Stream => t!("dataset.tab_stream").to_string(),
            DatasetTab::Labels => t!("dataset.tab_labels").to_string(),
            DatasetTab::View => t!("dataset.tab_view").to_string(),
            DatasetTab::Export => t!("dataset.tab_export").to_string(),
        }
    }

    /// Whether the tab is usable before any dataset is loaded.
    pub fn needs_dataset(&self) -> bool {
        !matches!(self, DatasetTab::Import | DatasetTab::Stream)
    }
}

/// State of the import form (path, row cap, projection config, progress).
#[derive(Default)]
pub struct ImportState {
    /// Path typed in the file field.
    pub path_text: String,
    /// Whether the row cap is active.
    pub limit_rows: bool,
    /// Row cap value (used when `limit_rows`).
    pub max_rows: usize,
    /// PCA (true) vs direct column projection (false). Ignored when
    /// [`use_radial`](Self::use_radial) is set.
    pub use_pca: bool,
    /// Multidimensional radial ("star coordinates") projection. Takes
    /// precedence over [`use_pca`](Self::use_pca) when set.
    pub use_radial: bool,
    /// Output spatial dimensions: 1, 2 or 3.
    pub dims: u8,
    /// For direct projection, the feature-column index mapped to X, Y, Z.
    pub axes: [usize; 3],
    /// Last import outcome shown to the user.
    pub status: Option<StatusMessage>,
    /// True while the worker thread is importing.
    pub loading: bool,
}

/// Build a [`ProjectionSpec`] from the PCA/Radial flags + dims + axes. Shared
/// by the import form, the View tab and the Stream panel so the method mapping
/// lives in exactly one place.
pub fn projection_spec_from(
    use_pca: bool,
    use_radial: bool,
    dims: u8,
    axes: [usize; 3],
) -> ProjectionSpec {
    ProjectionSpec {
        method: if use_radial {
            ProjectionMethod::Radial
        } else if use_pca {
            ProjectionMethod::Pca
        } else {
            ProjectionMethod::Direct
        },
        dims: dims.clamp(1, 3),
        axes,
    }
}

impl ImportState {
    /// Build a [`ProjectionSpec`] from the current form selections.
    pub fn projection(&self) -> ProjectionSpec {
        projection_spec_from(self.use_pca, self.use_radial, self.dims, self.axes)
    }
}

/// State of the export form.
#[derive(Default)]
pub struct ExportState {
    /// Destination path typed by the user.
    pub path_text: String,
    /// Last export outcome shown to the user.
    pub status: Option<StatusMessage>,
}

/// Root state of the dataset visualizer window.
///
/// Owns the loaded dataset, the filter/search state, the rendering settings
/// and the background import worker. The host (`State`) drains the dirty
/// flag via [`DatasetView::take_render_dirty`] to rebuild GPU buffers only
/// when the visible point set actually changed.
pub struct DatasetView {
    /// Whether the window is visible.
    pub show_window: bool,
    /// Currently selected tab.
    pub tab: DatasetTab,
    /// Import form state.
    pub import: ImportState,
    /// Export form state.
    pub export: ExportState,
    /// The active dataset, if any.
    pub loaded: Option<LoadedDataset>,
    /// Whether the loaded dataset's point cloud is drawn (toggled from the
    /// shared object list at the bottom of the main window).
    pub visible: bool,

    /// Label ids currently visible.
    pub enabled_labels: HashSet<u32>,
    /// Raw search text (parsed by [`SearchQuery::parse`]).
    pub search_text: String,
    /// Parse error of the current search text, if any.
    pub search_error: Option<String>,
    /// Rows matching filter + search, ascending.
    pub visible_rows: Vec<u32>,

    /// Point cloud rendering settings.
    pub settings: PointCloudSettings,
    /// Human readable summary of the last instance build.
    pub last_build_info: String,
    /// Runtime streaming tab state (config + live session).
    pub stream: stream_panel::StreamState,

    render_dirty: bool,
    worker: Option<Receiver<Result<LoadedDataset, String>>>,
}

impl Default for DatasetView {
    fn default() -> Self {
        Self::new()
    }
}

impl DatasetView {
    /// Fresh view with sensible defaults (PCA on, 100k row cap suggestion).
    pub fn new() -> Self {
        Self {
            show_window: false,
            tab: DatasetTab::Import,
            import: ImportState {
                use_pca: true,
                max_rows: 100_000,
                dims: 3,
                axes: [0, 1, 2],
                ..Default::default()
            },
            export: ExportState {
                path_text: "export.csv".to_string(),
                ..Default::default()
            },
            loaded: None,
            visible: true,
            enabled_labels: HashSet::new(),
            search_text: String::new(),
            search_error: None,
            visible_rows: Vec::new(),
            settings: PointCloudSettings::default(),
            last_build_info: String::new(),
            stream: stream_panel::StreamState::default(),
            render_dirty: false,
            worker: None,
        }
    }

    /// True once per change to the visible point set; the host should
    /// rebuild its GPU instance buffers when this fires.
    pub fn take_render_dirty(&mut self) -> bool {
        std::mem::take(&mut self.render_dirty)
    }

    /// Force an instance-buffer rebuild on the next frame.
    pub fn mark_dirty(&mut self) {
        self.render_dirty = true;
    }

    /// True while a background import is in flight.
    pub fn is_loading(&self) -> bool {
        self.import.loading
    }

    /// Install a freshly loaded dataset, reset filters/selection and jump
    /// to the Labels tab.
    pub fn install(&mut self, loaded: LoadedDataset) {
        self.enabled_labels = (0..loaded.dataset.label_names.len() as u32).collect();
        self.search_text.clear();
        self.search_error = None;
        self.settings.highlighted_row = None;
        self.visible = true;
        self.loaded = Some(loaded);
        self.recompute_visible();
        let l = self.loaded.as_ref().unwrap();
        let mmap = if l.dataset.source.is_memory_mapped() {
            t!("dataset.memory_mapped_suffix").to_string()
        } else {
            String::new()
        };
        let cache = if l.projection.from_cache {
            t!("dataset.cache_suffix").to_string()
        } else {
            String::new()
        };
        self.import.status = Some(StatusMessage::success(
            t!(
                "dataset.loaded_summary",
                name = l.dataset.metadata.name,
                rows = l.dataset.n_rows().to_string(),
                cols = l.dataset.n_cols().to_string(),
                labels = l.dataset.label_names.len().to_string(),
                mmap = mmap,
                cache = cache
            )
            .to_string(),
        ));
        self.tab = DatasetTab::Labels;
    }

    /// Centroid of the loaded projection, used to focus the camera on the
    /// dataset from the shared object list.
    pub fn centroid(&self) -> Option<[f32; 3]> {
        let points = &self.loaded.as_ref()?.projection.points;
        if points.is_empty() {
            return None;
        }
        let mut sum = [0.0f64; 3];
        for p in points {
            sum[0] += p[0] as f64;
            sum[1] += p[1] as f64;
            sum[2] += p[2] as f64;
        }
        let n = points.len() as f64;
        Some([
            (sum[0] / n) as f32,
            (sum[1] / n) as f32,
            (sum[2] / n) as f32,
        ])
    }

    /// Remove the loaded dataset and clear the rendered point cloud.
    pub fn clear_dataset(&mut self) {
        self.loaded = None;
        self.visible_rows.clear();
        self.settings.highlighted_row = None;
        self.render_dirty = true;
    }

    // --- Runtime streaming ------------------------------------------------

    /// Start a streaming session from the current Stream-tab configuration.
    pub fn start_stream(&mut self) {
        if self.stream.is_active() {
            return;
        }
        let config = StreamConfig {
            format: self.stream.format,
            addr: self.stream.addr.trim().to_string(),
            max_rows: self.stream.max_rows,
            projection: self.stream.projection(),
        };
        match StreamSession::start(config) {
            Ok(handle) => {
                self.stream.seen_version = 0;
                self.stream.known_labels = 0;
                self.stream.status = Some(StatusMessage::info(
                    t!("stream.listening", addr = handle.addr().to_string()).to_string(),
                ));
                self.stream.handle = Some(handle);
            }
            Err(e) => {
                self.stream.status = Some(StatusMessage::error(
                    t!("stream.start_failed", msg = e.to_string()).to_string(),
                ));
            }
        }
    }

    /// Stop the running streaming session (keeps the last snapshot on screen).
    pub fn stop_stream(&mut self) {
        if let Some(mut handle) = self.stream.handle.take() {
            handle.stop();
        }
        self.stream.status = Some(StatusMessage::info(t!("stream.stopped").to_string()));
    }

    /// Per-frame stream poll: drain lifecycle events for the status line and
    /// rebuild the dataset snapshot whenever the rolling buffer changed.
    pub fn poll_stream(&mut self) {
        let Some(handle) = &self.stream.handle else {
            return;
        };
        // Lifecycle events → status text.
        for ev in handle.drain_events() {
            self.stream.status = Some(match ev {
                StreamEvent::Listening(addr) => {
                    StatusMessage::info(t!("stream.listening", addr = addr).to_string())
                }
                StreamEvent::Connected(peer) => {
                    StatusMessage::success(t!("stream.connected", peer = peer).to_string())
                }
                StreamEvent::Disconnected => {
                    StatusMessage::info(t!("stream.disconnected").to_string())
                }
                StreamEvent::Error(msg) => {
                    StatusMessage::error(t!("stream.error", msg = msg).to_string())
                }
            });
        }
        // Rebuild the snapshot only when new rows arrived.
        let version = self.stream.handle.as_ref().unwrap().buffer_version();
        if version != self.stream.seen_version {
            self.stream.seen_version = version;
            self.refresh_stream_snapshot();
        }
    }

    /// Project the current stream buffer and install it as the live dataset,
    /// reusing the standard projection pipeline (no disk cache).
    fn refresh_stream_snapshot(&mut self) {
        let spec = self.stream.projection();
        let dataset = self
            .stream
            .handle
            .as_ref()
            .and_then(|h| h.with_buffer(|b| b.to_dataset()));
        let Some(dataset) = dataset else {
            return;
        };
        match prepare_dataset_spec(dataset, &spec, None) {
            Ok(loaded) => {
                let n_labels = loaded.dataset.label_names.len();
                // Reveal any newly-appeared label classes without resetting the
                // user's existing visibility choices.
                for id in self.stream.known_labels..n_labels {
                    self.enabled_labels.insert(id as u32);
                }
                self.stream.known_labels = n_labels;
                self.settings.highlighted_row = None;
                self.visible = true;
                self.loaded = Some(loaded);
                self.recompute_visible();
            }
            Err(e) => {
                self.stream.status = Some(StatusMessage::error(
                    t!("stream.error", msg = e.to_string()).to_string(),
                ));
            }
        }
    }

    /// Toggle whether the dataset's point cloud is drawn.
    pub fn set_visible(&mut self, visible: bool) {
        if self.visible != visible {
            self.visible = visible;
            self.render_dirty = true;
        }
    }

    /// Re-evaluate filter + search into [`Self::visible_rows`].
    pub fn recompute_visible(&mut self) {
        let Some(loaded) = &self.loaded else {
            self.visible_rows.clear();
            self.render_dirty = true;
            return;
        };
        let query = match SearchQuery::parse(&self.search_text) {
            Ok(q) => {
                self.search_error = None;
                q
            }
            Err(e) => {
                self.search_error = Some(e.to_string());
                SearchQuery::All
            }
        };
        let spec = FilterSpec {
            enabled_labels: self.enabled_labels.clone(),
            query,
        };
        self.visible_rows = apply_filter(&loaded.dataset, &loaded.index, &spec);
        // Drop the highlight if it was filtered out.
        if let Some(h) = self.settings.highlighted_row {
            if self.visible_rows.binary_search(&h).is_err() {
                self.settings.highlighted_row = None;
            }
        }
        self.render_dirty = true;
    }

    /// Re-run the projection on the already-loaded dataset with a new spec
    /// (used by the View tab to switch between 1D/2D/3D, PCA/Direct and axes
    /// without re-importing). Row membership is unchanged, so filters and the
    /// label index are preserved.
    pub fn reproject(&mut self, spec: ProjectionSpec) {
        let cache_dir = PathBuf::from(CACHE_DIR);
        let result = match self.loaded.as_ref() {
            Some(loaded) => preprocessor::project_spec(&loaded.dataset, &spec, Some(&cache_dir)),
            None => return,
        };
        match result {
            Ok(proj) => {
                if let Some(loaded) = self.loaded.as_mut() {
                    loaded.projection = proj;
                }
                self.recompute_visible();
            }
            Err(e) => {
                self.import.status = Some(StatusMessage::error(
                    t!("dataset.reprojection_failed", msg = e.to_string()).to_string(),
                ));
            }
        }
    }

    /// Build the instance batches for the current visible set. Returns an
    /// empty result when no dataset is loaded or the dataset is hidden.
    pub fn build_point_cloud(&mut self) -> point_cloud::PointCloudBuildResult {
        let empty = || point_cloud::PointCloudBuildResult {
            batches: Vec::new(),
            rendered_points: 0,
            downsampled: false,
        };
        if !self.visible {
            return empty();
        }
        let Some(loaded) = &self.loaded else {
            return empty();
        };
        let result = point_cloud::build_instances(
            &loaded.projection.points,
            &loaded.dataset.labels,
            &self.visible_rows,
            &self.settings,
        );
        let extra = if result.downsampled {
            t!("dataset.downsampled_suffix").to_string()
        } else {
            String::new()
        };
        self.last_build_info = t!(
            "dataset.points_rendered",
            rendered = result.rendered_points.to_string(),
            total = self.visible_rows.len().to_string(),
            extra = extra
        )
        .to_string();
        result
    }

    /// Poll the loader thread, draw the window, return the host action.
    pub fn show(&mut self, ctx: &egui::Context) -> DatasetAction {
        self.poll_worker();
        self.poll_stream();
        if !self.show_window {
            return DatasetAction::None;
        }
        // Keep the live status dot (and counters) repainting while streaming,
        // even when the pointer is idle.
        if self.stream.is_active() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        let mut action = DatasetAction::None;
        let mut open = true;
        let screen_center = ctx.screen_rect().center();
        egui::Window::new(t!("dataset.window_title").to_string())
            .open(&mut open)
            // Fixed footprint: the window must not grow when a dataset is
            // loaded. Long content scrolls inside instead of widening.
            .fixed_size([540.0, 520.0])
            // Spawn centered on screen; remains draggable afterwards.
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(screen_center)
            .show(ctx, |ui| {
                action = self.contents(ui);
            });
        if !open {
            self.show_window = false;
        }
        action
    }

    /// Window body: tab bar, summary strip and the active tab's panel.
    fn contents(&mut self, ui: &mut egui::Ui) -> DatasetAction {
        let action = DatasetAction::None;

        // --- Tab bar (centered) ---
        ui.vertical_centered(|ui| {
            ui.horizontal(|ui| {
                for tab in DatasetTab::ALL {
                    let enabled = !tab.needs_dataset() || self.loaded.is_some();
                    let selected = self.tab == tab;
                    // The Stream tab shows a live status dot (green active /
                    // blue receiving / red error) right on its label.
                    let title = if tab == DatasetTab::Stream && self.stream.is_active() {
                        format!("⏺ {}", tab.title())
                    } else {
                        tab.title()
                    };
                    let mut label = egui::RichText::new(title).size(14.0);
                    if tab == DatasetTab::Stream && self.stream.is_active() {
                        label = label.color(self.stream.dot_color());
                    }
                    if ui
                        .add_enabled(enabled, egui::SelectableLabel::new(selected, label))
                        .clicked()
                    {
                        self.tab = tab;
                    }
                }
            });
        });
        ui.separator();

        // --- Summary strip ---
        if let Some(loaded) = &self.loaded {
            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new(
                        t!(
                            "dataset.summary",
                            name = loaded.dataset.metadata.name,
                            rows = loaded.dataset.n_rows().to_string(),
                            cols = loaded.dataset.n_cols().to_string(),
                            format = loaded.dataset.metadata.format.to_string()
                        )
                        .to_string(),
                    )
                    .strong(),
                );
                if !self.last_build_info.is_empty() {
                    ui.label(egui::RichText::new(&self.last_build_info).weak());
                }
            });
            ui.separator();
        }

        // --- Active tab ---
        if self.tab.needs_dataset() && self.loaded.is_none() {
            // Defensive: tabs are disabled in this case, but keep a friendly
            // empty state in case the tab was preselected.
            empty_state(ui);
            return action;
        }
        match self.tab {
            DatasetTab::Import => {
                if let Some(req) = import_dialog::show(ui, &mut self.import) {
                    self.start_import(req);
                }
            }
            DatasetTab::Stream => match stream_panel::show(ui, &mut self.stream) {
                stream_panel::StreamUiAction::Start => self.start_stream(),
                stream_panel::StreamUiAction::Stop => self.stop_stream(),
                stream_panel::StreamUiAction::None => {}
            },
            DatasetTab::Labels => {
                let loaded = self.loaded.as_ref().unwrap();
                let stats = loaded.dataset.metadata.labels.clone();
                if label_filter::show(ui, &stats, &mut self.enabled_labels) {
                    self.recompute_visible();
                }
                ui.add_space(8.0);
                ui.vertical_centered(|ui| {
                    ui.label(egui::RichText::new(t!("dataset.distribution").to_string()).heading());
                });
                distribution_chart::show(ui, &stats, &self.enabled_labels);
            }
            DatasetTab::View => {
                let mut changed = false;
                egui::Grid::new("view_settings_grid")
                    .num_columns(2)
                    .spacing([12.0, 8.0])
                    .show(ui, |ui| {
                        ui.label(t!("dataset.point_size").to_string());
                        changed |= ui
                            .add(egui::Slider::new(&mut self.settings.point_size, 0.01..=0.5))
                            .changed();
                        ui.end_row();

                        use crate::visualization::color_mapper::ColorMode;
                        use crate::visualization::geometry_assigner::GeometryPolicy;

                        // Shape channel: uniform, per label, or by distance.
                        ui.label(t!("dataset.shape_channel").to_string());
                        ui.horizontal(|ui| {
                            let policy = self.settings.geometry_policy;
                            let uniform = matches!(policy, GeometryPolicy::Uniform(_));
                            let per_label = matches!(policy, GeometryPolicy::PerLabel);
                            let by_dist = matches!(policy, GeometryPolicy::ByDistance);
                            if ui
                                .selectable_label(uniform, t!("dataset.shape_uniform").to_string())
                                .clicked()
                            {
                                self.settings.geometry_policy = GeometryPolicy::default();
                                changed = true;
                            }
                            if ui
                                .selectable_label(per_label, t!("dataset.cycle_shapes").to_string())
                                .clicked()
                            {
                                self.settings.geometry_policy = GeometryPolicy::PerLabel;
                                changed = true;
                            }
                            if ui
                                .selectable_label(by_dist, t!("dataset.by_distance").to_string())
                                .clicked()
                            {
                                self.settings.geometry_policy = GeometryPolicy::ByDistance;
                                changed = true;
                            }
                        });
                        ui.end_row();

                        // Color channel: per label, or by distance from center.
                        ui.label(t!("dataset.color_channel").to_string());
                        ui.horizontal(|ui| {
                            let by_label = self.settings.color_mode == ColorMode::ByLabel;
                            if ui
                                .selectable_label(by_label, t!("dataset.color_by_label").to_string())
                                .clicked()
                            {
                                self.settings.color_mode = ColorMode::ByLabel;
                                changed = true;
                            }
                            if ui
                                .selectable_label(!by_label, t!("dataset.by_distance").to_string())
                                .clicked()
                            {
                                self.settings.color_mode = ColorMode::ByDistance;
                                changed = true;
                            }
                        });
                        ui.end_row();
                    });
                if changed {
                    self.render_dirty = true;
                }

                // --- Reconfigure the projection on the loaded dataset ---
                ui.add_space(8.0);
                ui.separator();
                ui.vertical_centered(|ui| {
                    ui.label(egui::RichText::new(t!("dataset.projection").to_string()).heading());
                });

                let col_names: Vec<String> = self
                    .loaded
                    .as_ref()
                    .map(|l| l.dataset.metadata.column_names.clone())
                    .unwrap_or_default();
                let n_cols = col_names.len();
                let mut apply = false;
                egui::Grid::new("projection_grid")
                    .num_columns(2)
                    .spacing([12.0, 8.0])
                    .show(ui, |ui| {
                        ui.label(t!("dataset.method").to_string());
                        ui.horizontal(|ui| {
                            import_dialog::method_radio(
                                ui,
                                &mut self.import.use_pca,
                                &mut self.import.use_radial,
                            );
                        });
                        ui.end_row();

                        ui.label(t!("dataset.dimensions").to_string());
                        ui.horizontal(|ui| {
                            ui.radio_value(&mut self.import.dims, 1, "1D");
                            ui.radio_value(&mut self.import.dims, 2, "2D");
                            ui.radio_value(&mut self.import.dims, 3, "3D");
                        });
                        ui.end_row();

                        // Direct projection: pick the column feeding each axis
                        // from the real column names.
                        if !self.import.use_pca && !self.import.use_radial && n_cols > 0 {
                            let dims = self.import.dims.clamp(1, 3) as usize;
                            for (a, axis) in ["X", "Y", "Z"].iter().enumerate().take(dims) {
                                ui.label(t!("dataset.axis_column", axis = axis.to_string()).to_string());
                                let sel = self.import.axes[a].min(n_cols - 1);
                                egui::ComboBox::from_id_source(format!("axis_combo_{}", a))
                                    .selected_text(col_names[sel].clone())
                                    .show_ui(ui, |ui| {
                                        for (ci, name) in col_names.iter().enumerate() {
                                            ui.selectable_value(&mut self.import.axes[a], ci, name);
                                        }
                                    });
                                ui.end_row();
                            }
                        }
                    });
                ui.vertical_centered(|ui| {
                    if ui.button(t!("dataset.apply_projection").to_string()).clicked() {
                        apply = true;
                    }
                });
                if apply {
                    let spec = self.import.projection();
                    self.reproject(spec);
                }
            }
            DatasetTab::Export => {
                let n_visible = self.visible_rows.len();
                if let Some(path) = export_panel::show(ui, &mut self.export, n_visible) {
                    let loaded = self.loaded.as_ref().unwrap();
                    self.export.status = match crate::dataset::export::export_csv(
                        &loaded.dataset,
                        &self.visible_rows,
                        &path,
                    ) {
                        Ok(n) => Some(StatusMessage::success(
                            t!(
                                "dataset.export_ok",
                                count = n.to_string(),
                                path = path.display().to_string()
                            )
                            .to_string(),
                        )),
                        Err(e) => Some(StatusMessage::error(
                            t!("dataset.export_failed", msg = e.to_string()).to_string(),
                        )),
                    };
                }
            }
        }
        action
    }

    /// Drain the background worker channel, if an import is in flight.
    fn poll_worker(&mut self) {
        let Some(rx) = &self.worker else { return };
        match rx.try_recv() {
            Ok(Ok(loaded)) => {
                self.worker = None;
                self.import.loading = false;
                self.install(loaded);
            }
            Ok(Err(msg)) => {
                self.worker = None;
                self.import.loading = false;
                self.import.status = Some(StatusMessage::error(
                    t!("status.import_failed", msg = msg).to_string(),
                ));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.worker = None;
                self.import.loading = false;
                self.import.status =
                    Some(StatusMessage::error(t!("status.worker_died").to_string()));
            }
        }
    }

    /// Kick off an import: builtins are generated synchronously (they are
    /// tiny), files are loaded + projected on a worker thread.
    pub fn start_import(&mut self, req: ImportRequest) {
        match req.source {
            ImportSource::Builtin(name) => match builtin::BuiltinDataset::default_of(name) {
                Some(kind) => {
                    let loaded =
                        prepare_dataset_spec(builtin::generate(kind, 42), &req.projection, None);
                    match loaded {
                        Ok(l) => self.install(l),
                        Err(e) => {
                            self.import.status = Some(StatusMessage::error(
                                t!("status.import_failed", msg = e.to_string()).to_string(),
                            ))
                        }
                    }
                }
                None => {
                    self.import.status = Some(StatusMessage::error(
                        t!("dataset.unknown_builtin", name = name.to_string()).to_string(),
                    ))
                }
            },
            ImportSource::Path(path) => {
                self.import.loading = true;
                self.import.status = Some(StatusMessage::info(
                    t!("status.loading", path = path.display().to_string()).to_string(),
                ));
                let (tx, rx) = std::sync::mpsc::channel();
                self.worker = Some(rx);
                let spec = req.projection;
                let max_rows = req.max_rows;
                std::thread::spawn(move || {
                    let cache_dir = PathBuf::from(CACHE_DIR);
                    let result =
                        load_dataset_pipeline_spec(&path, max_rows, &spec, Some(&cache_dir))
                            .map_err(|e| e.to_string());
                    let _ = tx.send(result);
                });
            }
        }
    }
}

/// Centered placeholder shown when a data tab is opened with no dataset.
fn empty_state(ui: &mut egui::Ui) {
    ui.add_space(40.0);
    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new("🗄").size(48.0));
        ui.add_space(8.0);
        ui.label(egui::RichText::new(t!("dataset.no_dataset").to_string()).heading());
        ui.label(egui::RichText::new(t!("dataset.no_dataset_hint").to_string()).weak());
    });
    ui.add_space(40.0);
}

/// Full import pipeline used by the worker thread: load file, build/reuse
/// the label index cache, compute/reuse the 3D projection cache, persist
/// metadata JSON. Convenience wrapper using a full 3D projection.
pub fn load_dataset_pipeline(
    path: &std::path::Path,
    max_rows: Option<usize>,
    method: ProjectionMethod,
    cache_dir: Option<&std::path::Path>,
) -> crate::dataset::Result<LoadedDataset> {
    load_dataset_pipeline_spec(path, max_rows, &ProjectionSpec::full(method), cache_dir)
}

/// Full import pipeline for an explicit [`ProjectionSpec`] (dims + axes).
pub fn load_dataset_pipeline_spec(
    path: &std::path::Path,
    max_rows: Option<usize>,
    spec: &ProjectionSpec,
    cache_dir: Option<&std::path::Path>,
) -> crate::dataset::Result<LoadedDataset> {
    let opts = loader::LoadOptions {
        max_rows,
        label_column: None,
    };
    let dataset = loader::load(path, &opts)?;
    prepare_dataset_spec(dataset, spec, cache_dir)
}

/// Index + projection for an already-loaded dataset, with optional caching.
/// Convenience wrapper using a full 3D projection.
pub fn prepare_dataset(
    dataset: Dataset,
    method: ProjectionMethod,
    cache_dir: Option<&std::path::Path>,
) -> crate::dataset::Result<LoadedDataset> {
    prepare_dataset_spec(dataset, &ProjectionSpec::full(method), cache_dir)
}

/// Index + projection for an already-loaded dataset using an explicit
/// [`ProjectionSpec`], with optional caching.
pub fn prepare_dataset_spec(
    dataset: Dataset,
    spec: &ProjectionSpec,
    cache_dir: Option<&std::path::Path>,
) -> crate::dataset::Result<LoadedDataset> {
    // The label index is independent of the projection, so it is keyed only by
    // the dataset content (PCA-3 tag) and shared across projection configs.
    let index = match cache_dir {
        Some(dir) => {
            let key = preprocessor::cache_key(&dataset, ProjectionMethod::Pca);
            let index_path = dir.join(format!("{:016x}.index.json", key));
            match DatasetIndex::load_json(&index_path) {
                Ok(idx) if idx.n_rows == dataset.n_rows() => idx,
                _ => {
                    let idx = DatasetIndex::build(&dataset.labels, dataset.label_names.len());
                    let _ = idx.save_json(&index_path);
                    idx
                }
            }
        }
        None => DatasetIndex::build(&dataset.labels, dataset.label_names.len()),
    };

    let projection = preprocessor::project_spec(&dataset, spec, cache_dir)?;

    if let Some(dir) = cache_dir {
        let key = preprocessor::cache_key_spec(&dataset, spec);
        let _ = dataset
            .metadata
            .save_json(&dir.join(format!("{:016x}.meta.json", key)));
    }

    Ok(LoadedDataset {
        dataset,
        index,
        projection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_message_constructors_set_kind() {
        assert_eq!(StatusMessage::info("a").kind, StatusKind::Info);
        assert_eq!(StatusMessage::success("b").kind, StatusKind::Success);
        assert_eq!(StatusMessage::error("c").kind, StatusKind::Error);
        assert_eq!(StatusMessage::error("c").text, "c");
    }

    #[test]
    fn status_message_colors_distinguish_outcomes() {
        let visuals = egui::Visuals::dark();
        let ok = StatusMessage::success("ok").color(&visuals);
        let err = StatusMessage::error("no").color(&visuals);
        let info = StatusMessage::info("hm").color(&visuals);
        assert_ne!(ok, err);
        assert_ne!(ok, info);
        assert_ne!(err, info);
    }

    #[test]
    fn tab_metadata_is_consistent() {
        assert_eq!(DatasetTab::ALL.len(), 5);
        // Titles unique and non-empty.
        for (i, a) in DatasetTab::ALL.iter().enumerate() {
            assert!(!a.title().is_empty());
            for b in &DatasetTab::ALL[i + 1..] {
                assert_ne!(a.title(), b.title());
            }
        }
        // Import and Stream work without data; the rest need a dataset.
        assert!(!DatasetTab::Import.needs_dataset());
        assert!(!DatasetTab::Stream.needs_dataset());
        for tab in [DatasetTab::Labels, DatasetTab::View, DatasetTab::Export] {
            assert!(tab.needs_dataset());
        }
    }

    #[test]
    fn dataset_visibility_gates_the_point_cloud() {
        let loaded = prepare_dataset(
            builtin::generate(builtin::BuiltinDataset::default_of("blobs").unwrap(), 1),
            ProjectionMethod::Direct,
            None,
        )
        .unwrap();
        let mut view = DatasetView::new();
        view.install(loaded);
        assert!(view.build_point_cloud().rendered_points > 0);
        view.set_visible(false);
        assert_eq!(view.build_point_cloud().rendered_points, 0);
        assert!(view.centroid().is_some());
    }

    #[test]
    fn new_view_defaults() {
        let view = DatasetView::new();
        assert!(!view.show_window);
        assert_eq!(view.tab, DatasetTab::Import);
        assert!(view.import.use_pca);
        assert!(view.loaded.is_none());
        assert!(view.visible_rows.is_empty());
        assert!(!view.is_loading());
        assert_eq!(view.export.path_text, "export.csv");
    }

    #[test]
    fn dirty_flag_is_drained_once() {
        let mut view = DatasetView::new();
        assert!(!view.take_render_dirty());
        view.mark_dirty();
        assert!(view.take_render_dirty());
        assert!(!view.take_render_dirty());
    }
}
