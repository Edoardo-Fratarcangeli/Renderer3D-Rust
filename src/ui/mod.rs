// Dataset visualizer UI. This layer only renders state and forwards user
// intent; parsing, indexing and projection live in `dataset`/`visualization`
// and run on a background thread so the UI never blocks on large files.

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

pub struct LoadedDataset {
    pub dataset: Dataset,
    pub index: DatasetIndex,
    pub projection: Projection,
}

/// What the dataset UI asks the host (State) to do this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DatasetAction {
    None,
    /// Move the camera target onto this world-space point.
    FocusPoint([f32; 3]),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportSource {
    Path(PathBuf),
    Builtin(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportRequest {
    pub source: ImportSource,
    pub max_rows: Option<usize>,
    pub method: ProjectionMethod,
}

#[derive(Default)]
pub struct ImportState {
    pub path_text: String,
    pub limit_rows: bool,
    pub max_rows: usize,
    pub use_pca: bool,
    pub status: String,
    pub loading: bool,
}

#[derive(Default)]
pub struct ExportState {
    pub path_text: String,
    pub status: String,
}

pub struct DatasetView {
    pub show_window: bool,
    pub import: ImportState,
    pub export: ExportState,
    pub loaded: Option<LoadedDataset>,

    // Filter / search state
    pub enabled_labels: HashSet<u32>,
    pub search_text: String,
    pub search_error: Option<String>,
    pub visible_rows: Vec<u32>,

    pub settings: PointCloudSettings,
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
    pub fn new() -> Self {
        Self {
            show_window: false,
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

    pub fn mark_dirty(&mut self) {
        self.render_dirty = true;
    }

    /// Install a freshly loaded dataset and reset filters/selection.
    pub fn install(&mut self, loaded: LoadedDataset) {
        self.enabled_labels = (0..loaded.dataset.label_names.len() as u32).collect();
        self.search_text.clear();
        self.search_error = None;
        self.settings.highlighted_row = None;
        self.loaded = Some(loaded);
        self.recompute_visible();
        let l = self.loaded.as_ref().unwrap();
        self.import.status = format!(
            "Loaded '{}': {} rows x {} cols, {} labels{}{}",
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
        );
    }

    /// Re-evaluate filter + search into `visible_rows`.
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

    /// Poll the loader thread; draw the window; return host action.
    pub fn show(&mut self, ctx: &egui::Context) -> DatasetAction {
        self.poll_worker();
        let mut action = DatasetAction::None;
        if !self.show_window {
            return action;
        }

        let mut open = true;
        egui::Window::new("📊 Dataset Visualizer")
            .open(&mut open)
            .default_width(440.0)
            .resizable(true)
            .vscroll(true)
            .show(ctx, |ui| {
                // --- Import ---
                egui::CollapsingHeader::new("Import")
                    .default_open(self.loaded.is_none())
                    .show(ui, |ui| {
                        if let Some(req) = import_dialog::show(ui, &mut self.import) {
                            self.start_import(req);
                        }
                    });

                let Some(loaded) = &self.loaded else { return };

                ui.separator();
                ui.label(format!(
                    "{} — {} rows x {} cols ({})",
                    loaded.dataset.metadata.name,
                    loaded.dataset.n_rows(),
                    loaded.dataset.n_cols(),
                    loaded.dataset.metadata.format
                ));
                if !self.last_build_info.is_empty() {
                    ui.label(&self.last_build_info);
                }

                // --- View settings ---
                egui::CollapsingHeader::new("View").show(ui, |ui| {
                    let mut changed = false;
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.settings.point_size, 0.01..=0.5)
                                .text("Point size"),
                        )
                        .changed();
                    let mut per_label = matches!(
                        self.settings.geometry_policy,
                        crate::visualization::geometry_assigner::GeometryPolicy::PerLabel
                    );
                    if ui
                        .checkbox(&mut per_label, "Shape per label (sphere/cube/plane)")
                        .changed()
                    {
                        self.settings.geometry_policy = if per_label {
                            crate::visualization::geometry_assigner::GeometryPolicy::PerLabel
                        } else {
                            crate::visualization::geometry_assigner::GeometryPolicy::default()
                        };
                        changed = true;
                    }
                    if changed {
                        self.render_dirty = true;
                    }
                });

                // --- Filters ---
                let mut filters_changed = false;
                egui::CollapsingHeader::new("Label Filter")
                    .default_open(true)
                    .show(ui, |ui| {
                        filters_changed |= label_filter::show(
                            ui,
                            &loaded.dataset.metadata.labels,
                            &mut self.enabled_labels,
                        );
                    });

                // --- Search ---
                egui::CollapsingHeader::new("Search").show(ui, |ui| {
                    filters_changed |=
                        search_panel::show(ui, &mut self.search_text, &self.search_error);
                });

                // --- Distribution ---
                egui::CollapsingHeader::new("Label Distribution").show(ui, |ui| {
                    distribution_chart::show(
                        ui,
                        &loaded.dataset.metadata.labels,
                        &self.enabled_labels,
                    );
                });

                // --- Table ---
                egui::CollapsingHeader::new("Table")
                    .default_open(true)
                    .show(ui, |ui| {
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
                    });

                // --- Export ---
                egui::CollapsingHeader::new("Export").show(ui, |ui| {
                    if let Some(path) = export_panel::show(ui, &mut self.export) {
                        match crate::dataset::export::export_csv(
                            &loaded.dataset,
                            &self.visible_rows,
                            &path,
                        ) {
                            Ok(n) => {
                                self.export.status =
                                    format!("Exported {} rows to {}", n, path.display())
                            }
                            Err(e) => self.export.status = format!("Export failed: {}", e),
                        }
                    }
                });

                if filters_changed {
                    self.recompute_visible();
                }
            });
        if !open {
            self.show_window = false;
        }
        action
    }

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
                self.import.status = format!("Import failed: {}", msg);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.worker = None;
                self.import.loading = false;
                self.import.status = "Import worker died unexpectedly".to_string();
            }
        }
    }

    pub fn start_import(&mut self, req: ImportRequest) {
        match req.source {
            ImportSource::Builtin(name) => {
                // Synthetic datasets are tiny: generate synchronously.
                match builtin::BuiltinDataset::default_of(name) {
                    Some(kind) => {
                        let loaded = prepare_dataset(builtin::generate(kind, 42), req.method, None);
                        match loaded {
                            Ok(l) => self.install(l),
                            Err(e) => self.import.status = format!("Import failed: {}", e),
                        }
                    }
                    None => self.import.status = format!("Unknown builtin '{}'", name),
                }
            }
            ImportSource::Path(path) => {
                self.import.loading = true;
                self.import.status = format!("Loading {} ...", path.display());
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
                    let idx =
                        DatasetIndex::build(&dataset.labels, dataset.label_names.len());
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
