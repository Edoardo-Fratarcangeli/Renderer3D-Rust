# ML Dataset 3D Visualizer

Extension of Renderer3D-Rust into a 3D visualizer for ML datasets: import,
3D preview, label filters, search and export — built on the existing
`wgpu` instanced pipeline and `egui` UI.

This is the **ML data block** entry of the universal import system. Tabular
data — **CSV and Excel (xlsx/xls/ods)** — as well as NPY/NPZ/IDX/Parquet is
imported here. For geometry strings, JSON/XYZ text and **3D solid models
(STL/OBJ/glTF)** see [GEOMETRY_IMPORT.md](GEOMETRY_IMPORT.md).

## 1. Architectural plan

Three strictly separated layers (UI never parses or indexes; data layers
never touch egui/wgpu):

```
┌────────────────────────────────────────────────────────────┐
│ ui/            egui panels, background import thread       │
│   import_dialog · label_filter · distribution_chart        │
│   export_panel   (dataset_table/search_panel: helpers)     │
├────────────────────────────────────────────────────────────┤
│ visualization/ pure data → GPU-ready instance batches      │
│   color_mapper · geometry_assigner · point_cloud           │
├────────────────────────────────────────────────────────────┤
│ dataset/       parsing, indexing, projection, export       │
│   loader · metadata · index · preprocessor · export        │
│   builtin (synthetic benchmark generators)                 │
└────────────────────────────────────────────────────────────┘
```

`state.rs` hosts a `DatasetView` and consumes its instance batches with the
pre-existing instanced mesh pipeline (one draw call per point shape). A
dirty flag rebuilds GPU buffers only when the visible selection changes.

### Window layout

The visualizer opens centered on screen with a **fixed footprint** (it never
widens when data loads) and is organized in four tabs — `DatasetTab` — with a
persistent summary strip (dataset name, shape, rendered-point count) under the
tab bar:

| Tab | Content |
|-----|---------|
| 📂 Import | grid form (file, row cap, projection method/dimensions/columns) + one-click benchmarks |
| 🏷 Labels | per-label toggles (two columns) + distribution chart with hover % |
| 🎨 View | point size slider, shape-per-label policy, **re-projection controls** (method, 1D/2D/3D, axis columns) |
| 💾 Export | destination + row count preview, colored result status |

There is **no "Explore" tab**: the imported dataset appears as a single entry
in the shared object list at the bottom of the main window (with visibility,
camera-focus and remove controls), alongside scene objects and imported
solids — one unified list instead of a per-window table.

Tabs that need data are disabled until a dataset is loaded; a centered
empty state guides the user to Import. Status lines use `StatusMessage`
(info/success/error → themed colors). All panels are plain functions over
explicit state, so the whole UI runs headless in tests (no GPU).

### Configurable projection

The 3D preview is fully configurable through `ProjectionSpec`
(`method` + `dims` + `axes`):

- **Method** — `Pca` (top principal components) or `Direct` (raw columns).
- **Dimensions** — 1D, 2D or 3D. Unused axes are held at 0, so a 2D
  projection lies on the `z = 0` plane and a 1D projection on the `x` axis.
- **Axes (Direct only)** — which feature column feeds X / Y / Z. In the Import
  dialog these are chosen by 0-based index (column names are unknown before
  load); the **View tab** then offers dropdowns over the real column names and
  an *Apply* button that re-projects the loaded dataset in place (row
  membership, filters and the label index are preserved).

### Large-file strategy

- **NPY / IDX**: memory mapped (`memmap2`); rows are decoded lazily from the
  mapping (`FeatureSource::Mmap`). The feature matrix is never copied to RAM.
- **CSV / Parquet**: streamed record by record; optional row cap.
- **Excel (xlsx/xls/ods)**: first sheet read via `calamine`; header row →
  feature/label columns, sharing label resolution with the CSV loader
  (`finish_table_dataset`).
- **NPZ**: stream-decompressed per entry (zip + deflate).
- **Rendering**: instance count capped at 200k with even striding, so huge
  datasets stay interactive.
- **PCA**: mean/covariance estimated on ≤5000 evenly-strided rows
  (projection itself covers every row), three passes total, O(d²) memory.

### Persistent caches (`.r3d_cache/`)

Keyed by FNV-1a of source path + size + mtime + shape + projection tag (e.g.
`pca-3`, `direct-2-0_4_…`), so both source edits and projection changes
invalidate naturally. The label index is projection-independent and keyed by
the dataset content alone:

| File | Content |
|------|---------|
| `<key>.meta.json` | `DatasetMetadata` (JSON) |
| `<key>.index.json` | `DatasetIndex` label → rows (JSON) |
| `<key>.proj` | row count + raw little-endian f32 3D points |

## 2. Files created / modified

Created: `src/dataset/{mod,loader,metadata,index,preprocessor,export,builtin}.rs`,
`src/visualization/{mod,color_mapper,geometry_assigner,point_cloud}.rs`,
`src/ui/{mod,import_dialog,dataset_table,label_filter,search_panel,distribution_chart,export_panel}.rs`,
`tests/dataset/*`, `tests/visualization/*`, this document.

Modified: `Cargo.toml` (memmap2, csv, serde_json, zip; optional `parquet`),
`src/lib.rs` (module exports), `src/state.rs` (DatasetView field, toolbar
button, focus action, instance-buffer rebuild + draw), `.gitignore`.

## 3. Implementation steps (as landed)

1. `Dataset` / `DatasetMetadata` / `DatasetIndex` structs + `FeatureSource`
   (in-memory vs mmap row decoding).
2. Loaders: NPY (mmap, sibling `*_labels.npy`), NPZ (X/y entries), CSV
   (label-column auto-detection), Excel (xlsx/xls/ods via `calamine`, sharing
   `finish_table_dataset` with CSV), IDX (MNIST pairing), Parquet behind the
   `parquet-support` feature.
3. Preprocessor: streaming PCA (power iteration + deflation) for 1–3
   components, configurable Direct column→axis mapping, view-cube
   normalization, binary projection cache (keyed by `ProjectionSpec`) + JSON
   index/metadata caches.
4. Visualization: Okabe-Ito-based deterministic label palette, per-label
   shape policy, instanced batch builder with highlight + downsampling.
5. UI: import dialog (file or builtin benchmark, row cap, projection
   method/dimensions/columns, background thread), label filter, distribution
   chart, View-tab re-projection controls, CSV export of the filtered subset.
   The imported dataset is listed in the shared bottom object list of the main
   window. (`dataset_table::row_text` and `search_panel` remain as reusable
   helpers / the filter grammar `substring | row:N | c<i> <op> <v>`.)
6. Tests + ignored benchmarks (`cargo test --release --test dataset -- --ignored --nocapture`).

## 4. Tests

Integration suites:

- `tests/dataset/loader_tests.rs` — every format, mmap flag, row caps, label
  auto-detection (incl. the `y` vs `label` precedence regression), explicit
  label column, unlabeled fallbacks, fortran/big-endian/dtype rejections.
- `tests/dataset/metadata_tests.rs` — JSON roundtrip, label stats.
- `tests/dataset/index_tests.rs` — index build/persistence, filter, search grammar.
- `tests/dataset/preprocessor_tests.rs` — PCA correctness, normalization, cache hit/miss.
- `tests/dataset/export_tests.rs` — filtered export re-imports identically.
- `tests/dataset/smoke.rs` — full import→cache→filter→render→export pipeline.
- `tests/visualization/main.rs` — palette determinism/distinctness, shape policy, instance batches, highlight, downsampling.
- `tests/ui/dataset_panels.rs` — headless egui frames over the real
  `DatasetView`: every tab renders (loaded and empty), background-worker
  import success/failure, search/filter/highlight invariants, table text,
  and projection dimensions (1D/2D/3D) controlling the active axes.
- `tests/dataset/bench.rs` — `#[ignore]` timing benchmarks
  (`cargo test --release --test dataset -- --ignored --nocapture`).

In-module unit tests (`cargo test --lib`) cover the private helpers:
NPY/IDX header parsing, dtype decoding for every `ElemType`, FNV hashing,
CSV value formatting/escaping, comparison operators, projection cache
corruption handling, RNG determinism, status/tab metadata.

Coverage is measured with `cargo llvm-cov --summary-only`. To report only the
unit-testable surface (i.e. excluding the GPU/window code, see below):

```
cargo llvm-cov --summary-only --ignore-filename-regex 'state.rs|main.rs'
```

Region coverage of the library modules:

| Module | Regions covered |
|--------|-----------------|
| `camera`, `model`, `primitives`, `scene` | 100% |
| `visualization/*` | 100% |
| `dataset/mod`, `dataset/builtin` | 100% |
| `dataset/preprocessor` | 98% |
| `mesh` | 96% |
| `geometry/*` | 93–96% |
| `dataset/export`, `dataset/loader`, `dataset/index` | 92–94% |
| `dataset/metadata` | 90% |
| `ui/*` | 90–99% |
| **Library modules (excluding `state.rs`/`main.rs`)** | **≈ 95%** |

**GPU / window-bound code is excluded by necessity.** `state.rs` (the wgpu
device, surface, render pass and the egui draw/`input` closures) and `main.rs`
(the winit event loop and SVG icon rendering) need a real GPU adapter and an
OS window, which are unavailable in headless CI. To keep this logic testable,
the pure parts have been extracted into covered modules:

- ray/primitive intersection → `scene::intersect_primitive`,
- undo/redo application → `scene::apply_undo_command` + `UndoCommand`,
- mesh loading, AABB and ray-pick → `mesh`,
- projection, normalization and caching → `dataset::preprocessor`.

What remains in `state.rs`/`main.rs` is GPU resource setup and event wiring,
exercised manually by running the app. The other small gaps are egui
click-handler closures that only fire on real pointer input and a few
`unwrap_or(0)` fallbacks on system-clock errors.

## 5. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Huge files exhaust RAM | mmap + lazy row decode; CSV/Parquet streaming; row cap option |
| PCA too slow on big n×d | estimation subsampling (≤5000 rows); `MAX_DIMS` guard (4096) |
| UI freeze during import | loader+projection run on a worker thread (mpsc) |
| Millions of points stall GPU | 200k instance cap with even striding (reported in UI) |
| Stale caches after file edits / projection changes | cache key includes size + mtime + shape + projection tag |
| Parquet dependency weight | optional `parquet-support` feature, off by default |
| Big-endian / fortran NPY, >1-byte IDX | rejected with explicit errors instead of silent corruption |

## 6. Completion criteria (verified)

- ✅ Small dataset imports without errors (loader tests, smoke test).
- ✅ Large dataset path uses mmap + caches (mmap asserts, cache roundtrip test).
- ✅ 3D view shows points colored per label (point_cloud + color tests; instanced draw in `state.rs`).
- ✅ Filters change the visible selection (`label_filter_changes_visible_rows`, UI recompute).
- ✅ Projection is configurable to 1D/2D/3D over chosen columns/components, re-projectable in place (`projection_dims_control_the_active_axes`, preprocessor spec tests).
- ✅ Export produces a subset consistent with filters (`export_writes_filtered_subset_consistent_with_filter`).
- ✅ All automated tests green (`cargo test`).
