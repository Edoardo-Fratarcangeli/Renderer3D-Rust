# Rust 3D Renderer

[![Rust CI](https://github.com/Edoardo-Fratarcangeli/Renderer3D-Rust/actions/workflows/ci.yml/badge.svg)](https://github.com/Edoardo-Fratarcangeli/Renderer3D-Rust/actions/workflows/ci.yml)

A modern, high-performance 3D renderer written in Rust using `wgpu` (WebGPU) and `egui` for the user interface. This project demonstrates instanced rendering, Euler angle rotation, a CAD-like camera control system, and a highly polished, interactive UI.

## 🚀 Features

- **Batch Model Loading**: Load entire folders or single files (STL, OBJ, GLTF/GLB) with threaded background loading.
- **3D Measurement Tool**: Precise Euclidean distance measurement between points picked on any scene object (primitives or meshes) via raycasting.
- **Transform Controls**: Direct manipulation of **Position**, **Rotation** (Euler angles), and **Scale** for any selected object via the interactive **Analysis Window**.
- **Mesh Pivot Recentering**: Automatically realign an imported model's origin to its geometric center to ensure intuitive rotation.
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
- **Modern Unified UI**:
  - **Smart Toolbar**: Top-left floating toolbar with grouped icons for Add, Import, Load Folder, and Measurement.
  - **Dynamic Analysis Window**: Horizontally draggable panel for measurement results and selection transforms.
  - **Custom Styling**: Integrated `egui` panel with transparent layouts and dynamic coloring.
- **High-Precision Camera System**:
  - **Refactored Controller**: Dedicated `CameraController` with unit tests for rotation, zoom, and panning.
  - **Orbit (Drag)**: Left Mouse Drag to rotate around the target point.
  - **Pan (Middle Mouse)**: Middle Mouse Drag to translate the view parallel to the camera plane.
  - **Zoom (Scroll)**: Scroll Wheel for smooth multiplicative zoom.
  - **Focus & Reset**: Quickly focus on selection or reset to origin.
- **Double-Click Workflow**:
  - **Quick Edit**: Double-click any object in the 3D viewport or the list to instantly open its property editor.
- **Scene Management**:
  - **Draft Creation**: Draft new objects (Cube, Sphere, Plane) with real-time property updates.
  - **Geometry Properties**: Toggle visualization aids like **Normal Vectors** for planes.
  - **Non-Destructive Editing**: Property editor with a draft system; changes only applied on **Confirm**.
- **Keyboard Shortcuts**:
  - `CTRL + Z`: Undo last action.
  - `CTRL + Shift + Z` or `CTRL + Y`: Redo last undone action.
  - `CTRL + C`: Copy selected objects.
  - `CTRL + V`: Paste objects.
  - `CTRL + D`: Duplicate objects.
  - `N`: New Object Draft.
  - `M`: Toggle Measure Tool.
  - `CANC` (Delete): Delete selection.

## 🧪 Testing & Verification

The project includes a robust testing suite (GUI and integration tests) and a dedicated GUI tool for managing them.

- **Integrated Tests**:
  - `tests/scene/`: Logic for picking, selection, and object defaults.
  - `tests/ui/`: UI layout, icons, and default configuration verification.
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
