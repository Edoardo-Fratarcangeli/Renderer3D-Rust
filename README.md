# Rust 3D Renderer

A **complete, high-performance 3D renderer** written in Rust with `wgpu`
(WebGPU) and `egui`. It visualizes **very large numbers of geometries
spatially and fast** — grouped into instanced batches, one draw call per
primitive shape, with automatic LOD meshes for huge sphere clouds — and it
**imports those geometries from many worlds**:

| Source | How |
|--------|-----|
| ✍️ Plain geometry strings | paste a DSL like `cube 0 0 0 2 #ff8800` straight into the app |
| 📊 Excel tables | `.xlsx` / `.xlsm` / `.xls` / `.ods`, header-mapped columns |
| 🗃 CSV tables | same header mapping as Excel |
| 🧾 JSON documents | arrays of geometry objects, tolerant field names |
| 📄 Text files of many kinds | `.txt` / `.geo` / `.dsl` / `.xyz`, DSL-vs-points auto-detected |
| 🤖 ML data blocks | NPY / NPZ / CSV / Parquet / IDX datasets with labels, PCA, filters |

→ Format reference: [docs/GEOMETRY_IMPORT.md](docs/GEOMETRY_IMPORT.md) ·
ML pipeline: [docs/ML_VISUALIZER.md](docs/ML_VISUALIZER.md)

On top of that it is a full interactive scene editor: instanced rendering,
Euler-angle rotation, a CAD-like camera, picking, undo/redo and a polished
egui UI.

## 🚀 Features

- **Measure tool** (📏 button in the toolbar): toggle it, then click two surface points (on meshes or primitives) to read the straight-line distance between them, drawn as a labelled segment in the viewport.
- **Solids Import** (🧊 button in the toolbar):
  - **Import 3D models**: STL, OBJ and glTF/GLB solid meshes load on a background thread and appear as selectable objects in the scene (auto-scaled and centered). STEP (`.step`/`.stp`) is recognised but not yet tessellated.
  - **Paste anything**: geometry DSL, XYZ point lists or JSON — auto-detected and parsed into a layer.
  - **Import files**: JSON, XYZ and generic text/DSL files; parsing runs on a background thread. (Tabular CSV/Excel data import now lives in the Dataset window.)
  - **Layers**: every import is a named layer with visibility toggle, camera focus (🎯), removal and a distinct default color; per-record colors/rotations/scales/labels supported everywhere.
  - **Fast at scale**: records collapse into one instanced batch per shape (a million geometries ≈ 3 draw calls); buffers rebuild only when layers change; sphere batches >2000 instances switch to a low-poly LOD mesh.
  - Clear errors with line numbers (`line 2: unknown shape 'spherex'`).
- **ML Dataset 3D Visualizer** (📊 button in the toolbar):
  - **Polished tabbed window**: opens centered on screen, with a fixed footprint, and four tabs (Import / Labels / View / Export), a persistent dataset summary strip, colored status messages and friendly empty states. Imported datasets appear in the shared object list at the bottom of the main window (with visibility, focus and remove controls), alongside scene objects.
  - **Multi-format import**: NPY (memory mapped), NPZ, CSV (streamed), Excel (xlsx/xls/ods, first sheet), MNIST-style IDX, Parquet (optional `parquet-support` feature), plus builtin synthetic benchmarks (blobs, spirals, swiss roll).
  - **Configurable projection**: choose the method (PCA or direct columns), the number of spatial dimensions (1D / 2D / 3D), and — for direct projection — which feature column feeds each axis. Reconfigure it from the View tab (with column-name dropdowns) to re-project the loaded dataset in place. Computed on a background thread.
  - **3D point cloud**: instanced rendering with deterministic per-label colors and optional per-label shapes.
  - **Persistent caches**: metadata JSON, label index and projection cached under `.r3d_cache/`, keyed by file content fingerprint + projection config.
  - **Filters & search**: per-label visibility toggles, query search (`substring`, `row:N`, `c0 > 0.5`), label distribution chart.
  - **Export**: writes the currently filtered subset to CSV.
  - See [docs/ML_VISUALIZER.md](docs/ML_VISUALIZER.md) for the architecture.

- **Multilingual UI**: English, Italian, Spanish, French and German, auto-detected from the OS and switchable live from **Settings → Language** (see [docs/I18N.md](docs/I18N.md)).
- **Native installers**: self-contained installers for Windows (NSIS wizard), macOS (`.dmg` + localized `.pkg`) and Linux (`.AppImage`/`.deb`), built per-platform in CI (see [docs/PACKAGING.md](docs/PACKAGING.md)).
- **Instanced Rendering**: Efficiently renders multiple instances of objects with low overhead.
- **WGPU Graphics**: Uses the modern `wgpu` crate for cross-platform, type-safe graphics programming.
- **Advanced Selection System**:
  - **3D Picking**: Select objects directly by clicking on them in the viewport via raycasting.
  - **Visual Feedback**: Selected objects emit a subtle **golden glow** for clear identification.
  - **Multi-Selection**: Support for standard selection, or multi-selection using `CTRL`, `SHIFT`, or `CMD` modifiers.
- **Undo/Redo System**:
  - Full support for undoing and redoing actions (Add, Delete, Edit, Duplicate, Paste).
  - Maintains a history stack for reliable scene recovery.
- **Object Labels**:
  - **3D World Space Labels**: Names appear as floating labels above objects.
  - **Visibility Toggles**: Control label visibility per-object via a dedicated icon (🏷️) in the list.
- **Interactive Modern UI**: Integrated `egui` panel custom-styled with a transparent layout, vector graphics tabs, and dynamic coloring based on item states.
- **Advanced Camera Controls**:
  - **Orbit**: Left Mouse Drag to rotate around the target point.
  - **Pan**: Middle Mouse Drag to translate the view across planes.
  - **Zoom**: Scroll Wheel to zoom smoothly in and out.
  - **Focus**: Option to focus the camera instantly on the bounds of selected objects.
- **Double-Click Workflow**:
  - **Quick Edit**: Double-click any object in the 3D viewport or the list to instantly open its property editor.
- **Scene Management**:
  - **Draft Creation**: Draft new objects (Cube, Sphere, Plane) with **context-specific properties** (Side Length, Radius, Surface Area) that update the scale in real-time.
  - **Geometry Properties**: Toggle visualization aids like **Normal Vectors** for planes, with color matching and independent scaling.
  - **Non-Destructive Editing**: Property editor uses a draft system; changes are only applied when clicking **Confirm**. Clicking **Cancel** reverts to the previous state.
  - **Visibility & Deletion**: Toggle visibility with eye/sunglass icons, or delete objects directly via the UI list.
- **Keyboard Shortcuts**:
  - `CTRL + Z`: Undo last action.
  - `CTRL + Shift + Z` or `CTRL + Y`: Redo last undone action.
  - `CTRL + C`: Copy selected objects to clipboard.
  - `CTRL + V`: Paste objects from clipboard to the scene.
  - `CTRL + D`: Duplicate currently selected objects.
  - `N`: Open "Add New Object" window.
  - `ENTER`: Confirm draft/edit actions.
  - `ESC`: Cancel draft/edit actions.
  - `CANC` (Delete): Delete selected objects immediately.

## 🧪 Testing & Verification

The project includes a robust testing suite (GUI and integration tests) and a dedicated GUI tool for managing them.

- **Integrated Tests**:
  - `tests/scene/`: Logic for picking, selection, and object defaults.
  - `tests/ui/`: UI layout, icons, defaults, plus **headless egui tests** that drive the real Dataset Visualizer window (every tab, background imports, filters) without a GPU.
  - `tests/dataset/`: loaders for every format, index/filter/search, PCA + caches, export roundtrips, end-to-end smoke tests and `#[ignore]` benchmarks.
  - `tests/geometry_import/`: real CSV/Excel/JSON/XYZ/TXT files round-tripped through the universal geometry importer, batch grouping, error paths and a 500k-record benchmark.
  - `tests/visualization/`: color palette, geometry policy and instanced point-cloud batches.
  - In-module unit tests (`cargo test --lib`) cover private parsing/format helpers.
- **Coverage**: measured with [`cargo llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov):
  ```bash
  cargo llvm-cov --tests --summary-only
  ```
- **Test Manager**: A Python-based GUI (`tests/test_manager.py`) allows you to select, run, and review test results with detailed summary reports.

## 📂 Project Structure

```
src/
├── main.rs       # Entry point, Window initialization, and Event Loop handling
├── state.rs      # Engine state, WGPU rendering loop, UI drawing, Input & Shortcuts
├── camera.rs     # Camera struct, View/Projection math, and Uniforms
├── model.rs      # Vertex layout and Instancing Definitions
├── primitives.rs # Generates meshes like Cubes, Spheres, Planes, Grids, and Axes
├── scene.rs      # SceneObject struct for entity transform and metadata
├── i18n.rs       # UI language detection, persistence and runtime switching
├── updater.rs    # Signed in-place auto-update client (inert until configured)
├── ui/           # egui panels: dataset visualizer, geometry import, filters…
└── shader.wgsl   # WebGPU (WGSL) shader code for lighting and highlighting
locales/          # Translations (app/dataset/geometry .yml) — see docs/I18N.md
packaging/        # Installer assets (NSIS, macOS .pkg, Linux .desktop/AppStream)
scripts/          # Icon generation, update-signing key, release helpers
.github/workflows # CI matrix that builds the native installers
docs/             # PACKAGING.md, I18N.md, GEOMETRY_IMPORT.md, ML_VISUALIZER.md
tests/            # Integration tests and Python Test Manager
```

## 📥 Installation (end users)

Download the installer for your platform from the
[Releases page](https://github.com/Edoardo-Fratarcangeli/Renderer3D-Rust/releases):

- **Windows** — run the `*-setup.exe` wizard (pick your language on the first page).
- **macOS** — open the `.dmg` and drag the app to Applications, or run the `.pkg`
  wizard. Universal build (Apple Silicon & Intel).
- **Linux** — make the `.AppImage` executable and run it, or install the `.deb`.

The app is self-contained and its interface is available in English, Italian,
Spanish, French and German (auto-detected, switchable in Settings). Installers
are currently unsigned, so you may see a one-time security prompt on first launch.

## 🛠️ Usage (developers)

### Prerequisites

- [Rust Toolchain](https://www.rust-lang.org/tools/install) (latest stable)
- A GPU with Vulkan (Linux/Windows), Metal (macOS) or DX12 (Windows) support
- [Python 3.x](https://www.python.org/downloads/) (for the Test Manager)

### Running

To run the renderer natively:

```bash
cargo run --release
```

To run the Test Manager:

```bash
python tests/test_manager.py
```

### Building the installers

Native installers are built per-platform in CI on every `v*` tag; see
[docs/PACKAGING.md](docs/PACKAGING.md) for the full flow and for building them
locally with `cargo-packager` / `makensis`.

### Navigating the UI

- **➕ Object**: Top left area, opens the draft window to prepare a new geometry.
- **⚙ Settings**: Top right area, global settings for background, grids, camera and **interface language**.
- **Bottom Panel**: Collapsible list of everything in the scene — objects and any imported dataset.
- **Object Controls**: Each item in the list has icons for:
  - `✏ Edit`: Open the property editor.
  - `🏷 Label`: Toggle the 3D label in the viewport.
  - `🗑 Delete`: Remove the object.
  - `👁 Visibility`: Toggle rendering state.

## 📦 Dependencies

- `wgpu`: Core Graphics API
- `winit`: Window creation and OS event handling
- `egui`, `egui-wgpu`, `egui-winit`: Immediate mode graphical user interface
- `cgmath`: Comprehensive linear algebra
- `bytemuck`: Safe casting of raw bytes for GPU buffers
- `pollster`: Blocking async executor for the main thread
- `rust-i18n`, `sys-locale`, `dirs`: Multilingual UI (translations, OS-language detection, persisted choice)
- `cargo-packager-updater`: Signed in-place auto-update client

## 🛠️ Next steps

- Real STEP (`.step`/`.stp`) import via a BREP tessellation backend (e.g. `truck`) — currently recognised but not yet tessellated.
- **Code signing** of the installers (macOS notarization, Windows Authenticode) to remove first-launch security prompts — see [docs/PACKAGING.md](docs/PACKAGING.md).
- **Enable auto-update**: generate the signing key and embed the public key — see [docs/PACKAGING.md](docs/PACKAGING.md) › Auto-update.

