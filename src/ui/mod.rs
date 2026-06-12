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
//!   a focused panel module ([`import_dialog`], [`dataset_table`],
//!   [`label_filter`], [`search_panel`], [`distribution_chart`],
//!   [`export_panel`]).
//! - Panels are plain functions over explicit state, so they can be driven
//!   headless (no GPU) by the integration tests in `tests/ui`.

pub mod dataset_table;
pub mod distribution_chart;
pub mod export_panel;
pub mod import_dialog;
pub mod label_filter;
pub mod search_panel;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use crate::dataset::index::{apply_filter, FilterSpec, SearchQuery};
use crate::dataset::preprocessor::{self, Projection, ProjectionMethod};
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
    /// Projection used for the 3D preview.
    pub method: ProjectionMethod,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetTab {
    /// File / benchmark import.
    Import,
    /// Search bar + virtual-scrolling table.
    Explore,
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
        DatasetTab::Explore,
        DatasetTab::Labels,
        DatasetTab::View,
        DatasetTab::Export,
    ];

    /// Icon + title shown on the tab button.
    pub fn title(&self) -> &'static str {
        match self {
            DatasetTab::Import => "📂 Import",
            DatasetTab::Explore => "🔍 Explore",
            DatasetTab::Labels => "🏷 Labels",
            DatasetTab::View => "🎨 View",
            DatasetTab::Export => "💾 Export",
        }
    }

    /// Whether the tab is usable before any dataset is loaded.
    pub fn needs_dataset(&self) -> bool {
        !matches!(self, DatasetTab::Import)
    }
}

/// State of the import form (path, row cap, projection method, progress).
#[derive(Default)]
pub struct ImportState {
    /// Path typed in the file field.
    pub path_text: String,
    /// Whether the row cap is active.
    pub limit_rows: bool,
    /// Row cap value (used when `limit_rows`).
    pub max_rows: usize,
    /// PCA (true) vs direct first-three-columns projection (false).
    pub use_pca: bool,
    /// Last import outcome shown to the user.
    pub status: Option<StatusMessage>,
    /// True while the worker thread is importing.
    pub loading: bool,
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
                ..Default::default()
            },
            export: ExportState {
                path_text: "export.csv".to_string(),
                ..Default::default()
            },
            loaded: None,
            enabled_labels: HashSet::new(),
            search_text: String::new(),
            search_error: None,
            visible_rows: Vec::new(),
            settings: PointCloudSettings::default(),
            last_build_info: String::new(),
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
    /// to the Explore tab.
    pub fn install(&mut self, loaded: LoadedDataset) {
        self.enabled_labels = (0..loaded.dataset.label_names.len() as u32).collect();
        self.search_text.clear();
        self.search_error = None;
        self.settings.highlighted_row = None;
        self.loaded = Some(loaded);
        self.recompute_visible();
        let l = self.loaded.as_ref().unwrap();
        self.import.status = Some(StatusMessage::success(format!(
            "Loaded '{}': {} rows × {} cols, {} labels{}{}",
            l.dataset.metadata.name,
            l.dataset.n_rows(),
            l.dataset.n_cols(),
            l.dataset.label_names.len(),
            if l.dataset.source.is_memory_mapped() {
                " (memory mapped)"
            } else {
                ""
            },
            if l.projection.from_cache {
                " (projection from cache)"
            } else {
                ""
            },
        )));
        self.tab = DatasetTab::Explore;
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

    /// Build the instance batches for the current visible set.
    pub fn build_point_cloud(&mut self) -> point_cloud::PointCloudBuildResult {
        let Some(loaded) = &self.loaded else {
            return point_cloud::PointCloudBuildResult {
                batches: Vec::new(),
                rendered_points: 0,
                downsampled: false,
            };
        };
        let result = point_cloud::build_instances(
            &loaded.projection.points,
            &loaded.dataset.labels,
            &self.visible_rows,
            &self.settings,
        );
        self.last_build_info = format!(
            "{} / {} points rendered{}",
            result.rendered_points,
            self.visible_rows.len(),
            if result.downsampled {
                " (downsampled)"
            } else {
                ""
            }
        );
        result
    }

    /// Poll the loader thread, draw the window, return the host action.
    pub fn show(&mut self, ctx: &egui::Context) -> DatasetAction {
        self.poll_worker();
        if !self.show_window {
            return DatasetAction::None;
        }

        let mut action = DatasetAction::None;
        let mut open = true;
        let screen_center = ctx.screen_rect().center();
        egui::Window::new("📊 Dataset Visualizer")
            .open(&mut open)
            .default_size([540.0, 520.0])
            // Spawn centered on screen; remains draggable afterwards.
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(screen_center)
            .resizable(true)
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
        let mut action = DatasetAction::None;

        // --- Tab bar (centered) ---
        ui.vertical_centered(|ui| {
            ui.horizontal(|ui| {
                for tab in DatasetTab::ALL {
                    let enabled = !tab.needs_dataset() || self.loaded.is_some();
                    let selected = self.tab == tab;
                    let label = egui::RichText::new(tab.title()).size(14.0);
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
                    egui::RichText::new(format!(
                        "{} — {} rows × {} cols ({})",
                        loaded.dataset.metadata.name,
                        loaded.dataset.n_rows(),
                        loaded.dataset.n_cols(),
                        loaded.dataset.metadata.format
                    ))
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
            DatasetTab::Explore => {
                if search_panel::show(ui, &mut self.search_text, &self.search_error) {
                    self.recompute_visible();
                }
                ui.add_space(4.0);
                let loaded = self.loaded.as_ref().unwrap();
                if let Some(focus) = dataset_table::show(
                    ui,
                    &loaded.dataset,
                    &loaded.projection.points,
                    &self.visible_rows,
                    &mut self.settings.highlighted_row,
                ) {
                    action = DatasetAction::FocusPoint(focus);
                    self.render_dirty = true;
                }
            }
            DatasetTab::Labels => {
                let loaded = self.loaded.as_ref().unwrap();
                let stats = loaded.dataset.metadata.labels.clone();
                if label_filter::show(ui, &stats, &mut self.enabled_labels) {
                    self.recompute_visible();
                }
                ui.add_space(8.0);
                ui.vertical_centered(|ui| {
                    ui.label(egui::RichText::new("Distribution").heading());
                });
                distribution_chart::show(ui, &stats, &self.enabled_labels);
            }
            DatasetTab::View => {
                let mut changed = false;
                egui::Grid::new("view_settings_grid")
                    .num_columns(2)
                    .spacing([12.0, 8.0])
                    .show(ui, |ui| {
                        ui.label("Point size");
                        changed |= ui
                            .add(egui::Slider::new(&mut self.settings.point_size, 0.01..=0.5))
                            .changed();
                        ui.end_row();

                        ui.label("Shape per label");
                        let mut per_label = matches!(
                            self.settings.geometry_policy,
                            crate::visualization::geometry_assigner::GeometryPolicy::PerLabel
                        );
                        if ui
                            .checkbox(&mut per_label, "cycle sphere / cube / plane")
                            .changed()
                        {
                            self.settings.geometry_policy = if per_label {
                                crate::visualization::geometry_assigner::GeometryPolicy::PerLabel
                            } else {
                                crate::visualization::geometry_assigner::GeometryPolicy::default()
                            };
                            changed = true;
                        }
                        ui.end_row();
                    });
                if changed {
                    self.render_dirty = true;
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
                        Ok(n) => Some(StatusMessage::success(format!(
                            "Exported {} rows to {}",
                            n,
                            path.display()
                        ))),
                        Err(e) => Some(StatusMessage::error(format!("Export failed: {}", e))),
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
                self.import.status = Some(StatusMessage::error(format!("Import failed: {}", msg)));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.worker = None;
                self.import.loading = false;
                self.import.status =
                    Some(StatusMessage::error("Import worker died unexpectedly"));
            }
        }
    }

    /// Kick off an import: builtins are generated synchronously (they are
    /// tiny), files are loaded + projected on a worker thread.
    pub fn start_import(&mut self, req: ImportRequest) {
        match req.source {
            ImportSource::Builtin(name) => match builtin::BuiltinDataset::default_of(name) {
                Some(kind) => {
                    let loaded = prepare_dataset(builtin::generate(kind, 42), req.method, None);
                    match loaded {
                        Ok(l) => self.install(l),
                        Err(e) => {
                            self.import.status =
                                Some(StatusMessage::error(format!("Import failed: {}", e)))
                        }
                    }
                }
                None => {
                    self.import.status =
                        Some(StatusMessage::error(format!("Unknown builtin '{}'", name)))
                }
            },
            ImportSource::Path(path) => {
                self.import.loading = true;
                self.import.status = Some(StatusMessage::info(format!(
                    "Loading {} ...",
                    path.display()
                )));
                let (tx, rx) = std::sync::mpsc::channel();
                self.worker = Some(rx);
                let method = req.method;
                let max_rows = req.max_rows;
                std::thread::spawn(move || {
                    let cache_dir = PathBuf::from(CACHE_DIR);
                    let result =
                        load_dataset_pipeline(&path, max_rows, method, Some(&cache_dir))
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
        ui.label(egui::RichText::new("No dataset loaded").heading());
        ui.label(egui::RichText::new("Import a file or a benchmark from the Import tab.").weak());
    });
    ui.add_space(40.0);
}

/// Full import pipeline used by the worker thread: load file, build/reuse
/// the label index cache, compute/reuse the 3D projection cache, persist
/// metadata JSON.
pub fn load_dataset_pipeline(
    path: &std::path::Path,
    max_rows: Option<usize>,
    method: ProjectionMethod,
    cache_dir: Option<&std::path::Path>,
) -> crate::dataset::Result<LoadedDataset> {
    let opts = loader::LoadOptions {
        max_rows,
        label_column: None,
    };
    let dataset = loader::load(path, &opts)?;
    prepare_dataset_cached(dataset, method, cache_dir)
}

/// Index + projection for an already-loaded dataset, with optional caching.
pub fn prepare_dataset(
    dataset: Dataset,
    method: ProjectionMethod,
    cache_dir: Option<&std::path::Path>,
) -> crate::dataset::Result<LoadedDataset> {
    prepare_dataset_cached(dataset, method, cache_dir)
}

fn prepare_dataset_cached(
    dataset: Dataset,
    method: ProjectionMethod,
    cache_dir: Option<&std::path::Path>,
) -> crate::dataset::Result<LoadedDataset> {
    let index = match cache_dir {
        Some(dir) => {
            let key = preprocessor::cache_key(&dataset, method);
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

    let projection = preprocessor::project(&dataset, method, cache_dir)?;

    if let Some(dir) = cache_dir {
        let key = preprocessor::cache_key(&dataset, method);
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
        // Only Import works without data.
        assert!(!DatasetTab::Import.needs_dataset());
        for tab in &DatasetTab::ALL[1..] {
            assert!(tab.needs_dataset());
        }
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
