# ML Dataset 3D Visualizer

Extension of Renderer3D-Rust into a 3D visualizer for ML datasets: import,
3D preview, label filters, search and export — built on the existing
`wgpu` instanced pipeline and `egui` UI.

## 1. Architectural plan

Three strictly separated layers (UI never parses or indexes; data layers
never touch egui/wgpu):

```
┌────────────────────────────────────────────────────────────┐
│ ui/            egui panels, background import thread       │
│   import_dialog · dataset_table · label_filter             │
│   search_panel · distribution_chart · export_panel         │
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

### Large-file strategy

- **NPY / IDX**: memory mapped (`memmap2`); rows are decoded lazily from the
  mapping (`FeatureSource::Mmap`). The feature matrix is never copied to RAM.
- **CSV / Parquet**: streamed record by record; optional row cap.
- **NPZ**: stream-decompressed per entry (zip + deflate).
- **Rendering**: instance count capped at 200k with even striding, so huge
  datasets stay interactive.
- **PCA**: mean/covariance estimated on ≤5000 evenly-strided rows
  (projection itself covers every row), three passes total, O(d²) memory.

### Persistent caches (`.r3d_cache/`)

Keyed by FNV-1a of source path + size + mtime + shape + method, so source
edits invalidate naturally:

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
   (label-column auto-detection), IDX (MNIST pairing), Parquet behind the
   `parquet-support` feature.
3. Preprocessor: streaming PCA (power iteration + deflation), view-cube
   normalization, binary projection cache + JSON index/metadata caches.
4. Visualization: Okabe-Ito-based deterministic label palette, per-label
   shape policy, instanced batch builder with highlight + downsampling.
5. UI: import dialog (file or builtin benchmark, row cap, PCA toggle,
   background thread), virtual-scrolling table (click = focus camera),
   label filter, search (`substring | row:N | c<i> <op> <v>`), distribution
   chart, CSV export of the filtered subset.
6. Tests (33) + ignored benchmarks (`cargo test --release --test dataset -- --ignored --nocapture`).

## 4. Tests

- `tests/dataset/loader_tests.rs` — every format, mmap flag, row cap, error paths.
- `tests/dataset/metadata_tests.rs` — JSON roundtrip, label stats.
- `tests/dataset/index_tests.rs` — index build/persistence, filter, search grammar.
- `tests/dataset/preprocessor_tests.rs` — PCA correctness, normalization, cache hit/miss.
- `tests/dataset/export_tests.rs` — filtered export re-imports identically.
- `tests/dataset/smoke.rs` — full import→cache→filter→render→export pipeline.
- `tests/visualization/main.rs` — palette determinism/distinctness, shape policy, instance batches, highlight, downsampling.
- `tests/dataset/bench.rs` — `#[ignore]` timing benchmarks.

## 5. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Huge files exhaust RAM | mmap + lazy row decode; CSV/Parquet streaming; row cap option |
| PCA too slow on big n×d | estimation subsampling (≤5000 rows); `MAX_DIMS` guard (4096) |
| UI freeze during import | loader+projection run on a worker thread (mpsc) |
| Millions of points stall GPU | 200k instance cap with even striding (reported in UI) |
| Stale caches after file edits | cache key includes size + mtime + shape + method |
| Parquet dependency weight | optional `parquet-support` feature, off by default |
| Big-endian / fortran NPY, >1-byte IDX | rejected with explicit errors instead of silent corruption |

## 6. Completion criteria (verified)

- ✅ Small dataset imports without errors (loader tests, smoke test).
- ✅ Large dataset path uses mmap + caches (mmap asserts, cache roundtrip test).
- ✅ 3D view shows points colored per label (point_cloud + color tests; instanced draw in `state.rs`).
- ✅ Filters change the visible selection (`label_filter_changes_visible_rows`, UI recompute).
- ✅ Export produces a subset consistent with filters (`export_writes_filtered_subset_consistent_with_filter`).
- ✅ All automated tests green (`cargo test`: 71 passing).
