# Rust 3D Renderer

A modern, high-performance 3D renderer written in Rust using `wgpu` (WebGPU) and `egui` for the user interface. This project demonstrates instanced rendering, Euler angle rotation, a CAD-like camera control system, and a highly polished, interactive UI.

## 🚀 Features

- **ML Dataset 3D Visualizer** (📊 button in the toolbar):
  - **Polished tabbed window**: opens centered on screen with five tabs (Import / Explore / Labels / View / Export), a persistent dataset summary strip, colored status messages and friendly empty states.
  - **Multi-format import**: NPY (memory mapped), NPZ, CSV (streamed), MNIST-style IDX, Parquet (optional `parquet-support` feature), plus builtin synthetic benchmarks (blobs, spirals, swiss roll).
  - **3D point cloud**: instanced rendering with deterministic per-label colors and optional per-label shapes; PCA (or direct-axis) projection computed on a background thread.
  - **Persistent caches**: metadata JSON, label index and 3D projection cached under `.r3d_cache/`, keyed by file content fingerprint.
  - **Table, filters & search**: virtual-scrolling table (click a row to focus the camera), per-label visibility toggles, query search (`substring`, `row:N`, `c0 > 0.5`), label distribution chart.
  - **Export**: writes the currently filtered subset to CSV.
  - See [docs/ML_VISUALIZER.md](docs/ML_VISUALIZER.md) for the architecture.

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
└── shader.wgsl   # WebGPU (WGSL) shader code for lighting and highlighting
tests/            # Integration tests and Python Test Manager
```

## 🛠️ Usage

### Prerequisites

- [Rust Toolchain](https://www.rust-lang.org/tools/install) (latest stable)
- [Python 3.x](https://www.python.org/downloads/) (for Test Manager)

### Running

To run the renderer natively:

```bash
cargo run --release
```

To run the Test Manager:

```bash
python tests/test_manager.py
```

### Navigating the UI

- **➕ New Object**: Top left area, opens the draft window to prepare a new geometry.
- **⚙ Settings**: Top right area, global settings for background, grids, and camera.
- **Bottom Panel**: Collapsible list of all objects in the scene.
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
