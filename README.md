# Rust 3D Renderer

A modern, high-performance 3D renderer written in Rust using `wgpu` (WebGPU) and `egui` for the user interface. This project demonstrates instanced rendering, Euler angle rotation, and a CAD-like camera control system.

## 🚀 Features

*   **Instanced Rendering**: Efficiently renders multiple instances of objects with low overhead.
*   **WGPU Graphics**: Uses the modern `wgpu` crate for cross-platform, type-safe graphics programming.
*   **Interactive UI**: Integrated `egui` panel for real-time control over the scene.
*   **Camera Controls**:
    *   **Orbit**: Left Mouse Drag to rotate around the target.
    *   **Pan**: Middle Mouse Drag to translate the view.
    *   **Zoom**: Scroll Wheel to zoom in/out.
*   **Scene Management**:
    *   Add/Remove objects.
    *   Toggle visibility.
    *   Rename objects.
    *   Control position and rotation.

## 📂 Project Structure

The project has been refactored into a modular architecture for better maintainability and scalability:

```
src/
├── main.rs       # Entry point and Event Loop handling
├── state.rs      # Core engine state, WGPU rendering loop, and Input handling
├── camera.rs     # Camera struct, View/Projection logic, and Uniforms
├── model.rs      # Vertex format and Instance data definitions
├── scene.rs      # SceneObject logic for managing entities
└── shader.wgsl   # WebGPU Shading Language code
```

*   **`main.rs`**: Initializes the application window and runs the event loop.
*   **`state.rs`**: The heart of the application. Manages the `wgpu::Device`, `wgpu::Queue`, rendering pipeline, and high-level logical updates.
*   **`camera.rs`**: Encapsulates camera math using `cgmath`. Handles coordinate system transformations.
*   **`model.rs`**: Defines the data layout for Vertices (`[x,y,z, r,g,b]`) and raw Instance matrices sent to the GPU.
*   **`scene.rs`**: High-level representation of objects in the world (`SceneObject`), holding their transform state and metadata.

## 🛠️ Usage

### Prerequisites
*   [Rust Toolchain](https://www.rust-lang.org/tools/install) (latest stable)

### Running
```bash
cargo run
```

### Controls
*   **Left Click + Drag**: Rotate Camera
*   **Middle Click + Drag**: Pan Camera
*   **Scroll**: Zoom
*   **UI Panel**: Use the side panel to add cubes and modify their properties.

## 📦 Dependencies
*   `wgpu`: Graphics API
*   `winit`: Window handling
*   `egui`, `egui-wgpu`, `egui-winit`: Immediate mode GUI
*   `cgmath`: Linear algebra library
*   `bytemuck`: Casting raw bytes for buffers
*   `pollster`: Blocking async executor for main
