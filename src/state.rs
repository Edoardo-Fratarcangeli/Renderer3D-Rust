use cgmath::prelude::*;
use wgpu::util::DeviceExt;
use winit::{event::*, window::Window};

use crate::camera::{Camera, Uniforms};
use crate::model::{InstanceRaw, Vertex};
use crate::primitives;
use crate::scene::{apply_undo_command, intersect_primitive, GeometryType, SceneObject, UndoCommand};

// Camera Default Values (used for initialization and reset)
pub const DEFAULT_CAMERA_YAW: f32 = 16.0;
pub const DEFAULT_CAMERA_PITCH: f32 = 36.0;
pub const DEFAULT_CAMERA_DIST: f32 = 21.1;
pub const DEFAULT_CAMERA_TARGET: [f32; 3] = [3.0, 0.15, 0.5];

// Zoom Limits
pub const DEFAULT_MIN_ZOOM: f32 = 1.0;
pub const DEFAULT_MAX_ZOOM: f32 = 1000.0;

// New Object Defaults
pub const DEFAULT_NEW_OBJ_POS: [f32; 3] = [0.0, 0.0, 0.0];
pub const DEFAULT_NEW_OBJ_COLOR: [f32; 3] = [1.0, 0.0, 0.0]; // Red

// UI Panel Defaults
pub const DEFAULT_SHOW_SETTINGS: bool = false;
pub const DEFAULT_SHOW_ADD_PANEL: bool = false;
pub const DEFAULT_BOTTOM_PANEL_EXPANDED: bool = false;

// Grid Defaults
pub const DEFAULT_SHOW_GRID_XY: bool = true;
pub const DEFAULT_SHOW_GRID_XZ: bool = false;
pub const DEFAULT_SHOW_GRID_YZ: bool = false;
pub const DEFAULT_SHOW_AXES: bool = true;

// Background
pub const DEFAULT_BG_COLOR: f64 = 0.1;

pub struct MeshBuffers {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

/// GPU buffers + CPU geometry for an imported 3D model. Unlike the built-in
/// [`MeshBuffers`] (16-bit indices), imported meshes use 32-bit indices. The
/// CPU [`crate::mesh::MeshData`] is retained for ray-pick selection.
pub struct CustomMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub data: std::sync::Arc<crate::mesh::MeshData>,
}

pub struct State {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub window: std::sync::Arc<Window>,

    // Render Pipelines
    render_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,

    // Geometry Resources
    cube_mesh: MeshBuffers,
    sphere_mesh: MeshBuffers,
    plane_mesh: MeshBuffers,
    grid_xy_mesh: MeshBuffers,
    grid_xz_mesh: MeshBuffers,
    grid_yz_mesh: MeshBuffers,
    axes_mesh: MeshBuffers,
    normal_arrow_mesh: MeshBuffers,

    // Scene
    pub objects: Vec<SceneObject>,
    pub next_id: usize,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    pub camera: Camera,

    // Egui
    pub egui_renderer: egui_wgpu::Renderer,
    pub egui_state: egui_winit::State,

    // Camera Control State
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_dist: f32,
    pub min_zoom: f32,
    pub max_zoom: f32,

    // UI State
    pub new_obj_pos: [f32; 3],
    pub new_obj_type: GeometryType,
    pub new_obj_color: [f32; 3],
    pub show_settings: bool,
    pub show_add_panel: bool,

    // Grid Settings
    pub show_grid_xy: bool,
    pub show_grid_xz: bool,
    pub show_grid_yz: bool,
    pub show_axes: bool,

    // Background
    pub bg_color: f64,

    pub bottom_panel_expanded: bool,
    pub last_click_time: std::time::Instant,

    // Mouse
    pub is_drag_active: bool,
    pub is_pan_active: bool,
    pub camera_target: cgmath::Point3<f32>,
    pub mouse_pos: [f32; 2],

    // Measurement tool: when active, clicking surfaces records up to two
    // world-space points and reports the distance between them.
    pub measure_mode: bool,
    pub measure_points: Vec<[f32; 3]>,

    // Editor State
    pub editing_obj_id: Option<usize>,
    pub editing_obj_draft: Option<SceneObject>,

    // Draft state for new object creation
    pub draft_object: Option<SceneObject>,

    // Clipboard
    pub clipboard: Vec<SceneObject>,

    // Undo/Redo
    pub undo_stack: Vec<UndoCommand>,
    pub redo_stack: Vec<UndoCommand>,

    pub should_focus_name: bool,

    // ML Dataset visualizer
    pub dataset_view: crate::ui::DatasetView,
    dataset_point_batches: Vec<(GeometryType, wgpu::Buffer, u32)>,

    // Universal geometry import (layers of instanced shapes)
    pub geometry_view: crate::ui::geometry_panel::GeometryView,
    geometry_batches: Vec<(GeometryType, wgpu::Buffer, u32)>,

    // Imported 3D solid models (STL/OBJ/glTF), keyed by scene-object id.
    pub custom_meshes: std::collections::HashMap<usize, CustomMesh>,
    mesh_worker: Option<std::sync::mpsc::Receiver<Result<(crate::mesh::MeshData, String), String>>>,
    // Low-poly sphere used for instanced batches above LOD_SPHERE_THRESHOLD
    // instances, so huge point clouds stay fast.
    sphere_lod_mesh: MeshBuffers,
}

/// Above this many instances in one batch, spheres use the low-poly mesh.
pub const LOD_SPHERE_THRESHOLD: u32 = 2000;

/// Small transparent icon button used by every row of the bottom object list
/// (scene objects, imported datasets, …) so they share one look and one
/// implementation instead of repeating the `Button`/`RichText` boilerplate.
fn list_icon_button(ui: &mut egui::Ui, icon: &str, color: egui::Color32, hover: &str) -> bool {
    ui.add(
        egui::Button::new(egui::RichText::new(icon).color(color))
            .fill(egui::Color32::TRANSPARENT),
    )
    .on_hover_text(hover)
    .clicked()
}

impl State {
    pub async fn new(window: Window) -> Self {
        let window = std::sync::Arc::new(window);
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // --- Helper to Upload Mesh ---
        let upload_mesh = |data: crate::primitives::MeshData| -> MeshBuffers {
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&data.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&data.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            MeshBuffers {
                vertex_buffer,
                index_buffer,
                num_indices: data.indices.len() as u32,
            }
        };

        // --- Initialize Meshes ---
        let cube_mesh = upload_mesh(primitives::create_cube());
        let sphere_mesh = upload_mesh(primitives::create_sphere(0.5, 32, 32));
        let sphere_lod_mesh = upload_mesh(primitives::create_sphere(0.5, 12, 8));
        let plane_mesh = upload_mesh(primitives::create_plane(1.0));

        // Grids
        let grid_size = 20;
        let spacing = 1.0;
        let grid_xy_mesh = upload_mesh(primitives::create_grid(grid_size, spacing, 1));
        let grid_xz_mesh = upload_mesh(primitives::create_grid(grid_size, spacing, 0));
        let grid_yz_mesh = upload_mesh(primitives::create_grid(grid_size, spacing, 2));

        // Axes (Thick)
        let axes_mesh = upload_mesh(primitives::create_thick_axes(3.0, 0.05));

        // Normal Arrow
        let normal_arrow_mesh = upload_mesh(primitives::create_arrow(1.0, 0.04, [1.0, 1.0, 0.0]));

        // --- Camera & Uniforms ---
        let camera = Camera {
            eye: (5.0, 5.0, 5.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_z(), // Z-Up
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 1000.0, // Increased zfar for large scenes
        };

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        // --- Pipelines ---
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Triangle Pipeline (Mesh)
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Line Pipeline (Grid)
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let egui_context = egui::Context::default();

        let mut style = (*egui_context.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.window_rounding = egui::Rounding::same(8.0);
        style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(6.0);
        style.visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);
        style.visuals.widgets.hovered.rounding = egui::Rounding::same(6.0);
        style.visuals.widgets.active.rounding = egui::Rounding::same(6.0);
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.window_margin = egui::Margin::same(12.0);
        egui_context.set_style(style);

        let egui_state = egui_winit::State::new(
            egui_context,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            Some(wgpu::TextureFormat::Depth32Float),
            1,
        );

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            line_pipeline,
            cube_mesh,
            sphere_mesh,
            plane_mesh,
            grid_xy_mesh,
            grid_xz_mesh,
            grid_yz_mesh,
            axes_mesh,
            uniform_buffer,
            uniform_bind_group,
            camera,
            objects: Vec::new(),
            next_id: 1,
            egui_renderer,
            egui_state,
            // User requested default (matches Reset View)
            camera_yaw: DEFAULT_CAMERA_YAW,
            camera_pitch: DEFAULT_CAMERA_PITCH,
            camera_dist: DEFAULT_CAMERA_DIST,
            min_zoom: DEFAULT_MIN_ZOOM,
            max_zoom: DEFAULT_MAX_ZOOM,
            new_obj_pos: DEFAULT_NEW_OBJ_POS,
            new_obj_type: GeometryType::Cube,
            new_obj_color: DEFAULT_NEW_OBJ_COLOR,
            show_settings: DEFAULT_SHOW_SETTINGS,
            show_add_panel: DEFAULT_SHOW_ADD_PANEL,
            // Grids
            show_grid_xy: DEFAULT_SHOW_GRID_XY,
            show_grid_xz: DEFAULT_SHOW_GRID_XZ,
            show_grid_yz: DEFAULT_SHOW_GRID_YZ,
            show_axes: DEFAULT_SHOW_AXES,
            bg_color: DEFAULT_BG_COLOR,

            bottom_panel_expanded: DEFAULT_BOTTOM_PANEL_EXPANDED,
            is_drag_active: false,
            is_pan_active: false,
            camera_target: cgmath::Point3::new(
                DEFAULT_CAMERA_TARGET[0],
                DEFAULT_CAMERA_TARGET[1],
                DEFAULT_CAMERA_TARGET[2],
            ),
            editing_obj_id: None,
            editing_obj_draft: None,
            draft_object: None,
            clipboard: Vec::new(),
            mouse_pos: [0.0, 0.0],
            measure_mode: false,
            measure_points: Vec::new(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            normal_arrow_mesh,
            last_click_time: std::time::Instant::now(),
            should_focus_name: false,
            dataset_view: crate::ui::DatasetView::new(),
            dataset_point_batches: Vec::new(),
            geometry_view: crate::ui::geometry_panel::GeometryView::new(),
            geometry_batches: Vec::new(),
            custom_meshes: std::collections::HashMap::new(),
            mesh_worker: None,
            sphere_lod_mesh,
        }
    }

    /// Spawn a worker thread that loads a 3D model file off the UI thread.
    pub fn spawn_mesh_load(&mut self, path: std::path::PathBuf) {
        self.geometry_view.mesh_loading = true;
        self.geometry_view.status = Some(crate::ui::StatusMessage::info(format!(
            "Loading {} ...",
            path.display()
        )));
        let (tx, rx) = std::sync::mpsc::channel();
        self.mesh_worker = Some(rx);
        std::thread::spawn(move || {
            let label = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string();
            let result = crate::mesh::MeshData::load(&path)
                .map(|mesh| (mesh, label))
                .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
    }

    /// Drain the mesh-load worker, adding the model to the scene on success.
    fn poll_mesh_worker(&mut self) {
        let Some(rx) = &self.mesh_worker else { return };
        match rx.try_recv() {
            Ok(Ok((mesh, label))) => {
                self.mesh_worker = None;
                self.geometry_view.mesh_loading = false;
                self.add_mesh_object(mesh, label);
            }
            Ok(Err(msg)) => {
                self.mesh_worker = None;
                self.geometry_view.mesh_loading = false;
                self.geometry_view.status =
                    Some(crate::ui::StatusMessage::error(format!("Import failed: {}", msg)));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.mesh_worker = None;
                self.geometry_view.mesh_loading = false;
                self.geometry_view.status =
                    Some(crate::ui::StatusMessage::error("Import worker died unexpectedly"));
            }
        }
    }

    /// Upload a loaded mesh to the GPU and add it as a selected scene object,
    /// auto-scaled and centered at the origin so it is immediately visible.
    pub fn add_mesh_object(&mut self, mesh: crate::mesh::MeshData, label: String) {
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Custom Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Custom Mesh Index Buffer"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let num_indices = mesh.indices.len() as u32;
        let tri_count = mesh.triangle_count();

        // Fit the model into roughly a 6-unit box centered at the origin.
        let extent = mesh.max_extent();
        let scale = if extent > 1e-6 { 6.0 / extent } else { 1.0 };
        let center = mesh.center();

        let id = self.next_id;
        self.next_id += 1;

        for obj in &mut self.objects {
            obj.selected = false;
        }
        let mut obj = SceneObject::new(id, label, [0.0, 0.0, 0.0], GeometryType::Mesh);
        obj.instance.scale = cgmath::Vector3::new(scale, scale, scale);
        obj.instance.position =
            cgmath::Vector3::new(-center[0] * scale, -center[1] * scale, -center[2] * scale);
        obj.selected = true;

        self.custom_meshes.insert(
            id,
            CustomMesh {
                vertex_buffer,
                index_buffer,
                num_indices,
                data: std::sync::Arc::new(mesh),
            },
        );
        self.push_undo(UndoCommand::Add(obj.clone()));
        self.objects.push(obj);

        // Center the camera on the freshly imported model.
        self.camera_target = cgmath::Point3::new(0.0, 0.0, 0.0);
        self.geometry_view.status = Some(crate::ui::StatusMessage::success(format!(
            "Imported model with {} triangles",
            tri_count
        )));
    }

    pub fn push_undo(&mut self, cmd: UndoCommand) {
        self.undo_stack.push(cmd);
        self.redo_stack.clear(); // Redo stack clears on a new action
        if self.undo_stack.len() > 100 {
            self.undo_stack.remove(0);
        }
    }

    pub fn undo(&mut self) {
        if let Some(cmd) = self.undo_stack.pop() {
            self.apply_undo_cmd(cmd.clone(), true);
            self.redo_stack.push(cmd);
        }
    }

    pub fn redo(&mut self) {
        if let Some(cmd) = self.redo_stack.pop() {
            self.apply_undo_cmd(cmd.clone(), false);
            self.undo_stack.push(cmd);
        }
    }

    fn apply_undo_cmd(&mut self, cmd: UndoCommand, is_undo: bool) {
        apply_undo_command(&mut self.objects, &cmd, is_undo);
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        // Pass event to egui first
        let response = self.egui_state.on_window_event(&self.window, event);
        if response.consumed {
            return true;
        }

        // Handle Keyboard Shortcuts
        if let WindowEvent::KeyboardInput {
            event:
                winit::event::KeyEvent {
                    state: ElementState::Pressed,
                    logical_key,
                    ..
                },
            ..
        } = event
        {
            let modifiers = self.egui_state.egui_ctx().input(|i| i.modifiers);
            let is_ctrl = modifiers.ctrl || modifiers.mac_cmd;

            match logical_key {
                winit::keyboard::Key::Character(c) => match c.as_str() {
                    "c" | "C" if is_ctrl => {
                        self.clipboard = self
                            .objects
                            .iter()
                            .filter(|o| o.selected)
                            .cloned()
                            .collect();
                        return true;
                    }
                    "v" | "V" if is_ctrl => {
                        let mut new_objs = Vec::new();
                        for mut obj in self.clipboard.clone() {
                            obj.id = self.next_id;
                            self.next_id += 1;
                            obj.selected = true; // Select newly pasted
                            new_objs.push(obj);
                        }

                        // Deselect old ones
                        for obj in &mut self.objects {
                            obj.selected = false;
                        }

                        let mut undo_cmds = Vec::new();
                        for obj in &new_objs {
                            undo_cmds.push(UndoCommand::Add(obj.clone()));
                        }
                        self.push_undo(UndoCommand::MultiAction(undo_cmds));

                        self.objects.extend(new_objs);
                        return true;
                    }
                    "d" | "D" if is_ctrl => {
                        let mut duplicated = Vec::new();
                        for obj in self.objects.iter().filter(|o| o.selected) {
                            let mut new_obj = obj.clone();
                            new_obj.id = self.next_id;
                            self.next_id += 1;
                            duplicated.push(new_obj);
                        }

                        let mut undo_cmds = Vec::new();
                        for obj in &duplicated {
                            undo_cmds.push(UndoCommand::Add(obj.clone()));
                        }
                        self.push_undo(UndoCommand::MultiAction(undo_cmds));

                        self.objects.extend(duplicated);
                        return true;
                    }
                    "z" | "Z" if is_ctrl => {
                        if modifiers.shift {
                            self.redo();
                        } else {
                            self.undo();
                        }
                        return true;
                    }
                    "y" | "Y" if is_ctrl => {
                        self.redo();
                        return true;
                    }
                    "n" | "N" if !is_ctrl => {
                        if self.draft_object.is_none() {
                            let id = self.next_id;
                            let default_obj = SceneObject::new(
                                id,
                                format!("Object {}", id),
                                [0.0, 0.0, 0.0],
                                GeometryType::Cube,
                            );
                            self.draft_object = Some(default_obj);
                            self.should_focus_name = true;
                        }
                        return true;
                    }
                    _ => {}
                },
                winit::keyboard::Key::Named(winit::keyboard::NamedKey::Delete) => {
                    let to_delete: Vec<SceneObject> = self
                        .objects
                        .iter()
                        .filter(|o| o.selected)
                        .cloned()
                        .collect();
                    if !to_delete.is_empty() {
                        let mut undo_cmds = Vec::new();
                        for obj in &to_delete {
                            undo_cmds.push(UndoCommand::Delete(obj.clone()));
                        }
                        self.push_undo(UndoCommand::MultiAction(undo_cmds));
                        self.objects.retain(|o| !o.selected);
                    }
                    return true;
                }
                _ => {}
            }
        }

        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                match button {
                    MouseButton::Left => {
                        if *state == ElementState::Pressed {
                            let is_over_ui = self.egui_state.egui_ctx().is_pointer_over_area()
                                || self.egui_state.egui_ctx().wants_pointer_input();

                            // Only begin an orbit drag (and run picking) when the
                            // press lands on the 3D viewport. Clicking a button or
                            // panel must never rotate the camera.
                            if !is_over_ui {
                                let x = (2.0 * self.mouse_pos[0]) / self.size.width as f32 - 1.0;
                                let y = 1.0 - (2.0 * self.mouse_pos[1]) / self.size.height as f32;

                                if self.measure_mode {
                                    // Measurement: record the surface point under
                                    // the cursor, keeping only the last two so the
                                    // distance always reflects a single segment.
                                    // Does not orbit the camera.
                                    if let Some(p) = self.raycast_world(x, y) {
                                        if self.measure_points.len() >= 2 {
                                            self.measure_points.clear();
                                        }
                                        self.measure_points.push(p);
                                    }
                                } else {
                                    self.is_drag_active = true;

                                    // Deselect panels
                                    self.show_add_panel = false;
                                    self.show_settings = false;

                                    let now = std::time::Instant::now();
                                    let is_double_click = now
                                        .duration_since(self.last_click_time)
                                        .as_millis()
                                        < 300;
                                    self.last_click_time = now;

                                    if let Some(hit_id) = self.select_object_at_ndc(x, y) {
                                        if is_double_click {
                                            // Initialize Edit from 3D Double Click
                                            self.editing_obj_id = Some(hit_id);
                                            if let Some(obj) =
                                                self.objects.iter().find(|o| o.id == hit_id)
                                            {
                                                self.editing_obj_draft = Some(obj.clone());
                                                self.should_focus_name = true;
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            // Releasing the button always ends any active drag.
                            self.is_drag_active = false;
                        }
                    }
                    MouseButton::Middle => self.is_pan_active = *state == ElementState::Pressed,
                    _ => {}
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Multiplicative Zoom for consistent feel at all distances
                let zoom_factor = 1.1;
                match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        if *y > 0.0 {
                            self.camera_dist /= zoom_factor;
                        } else {
                            self.camera_dist *= zoom_factor;
                        }
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        if pos.y > 0.0 {
                            self.camera_dist /= zoom_factor;
                        } else {
                            self.camera_dist *= zoom_factor;
                        }
                    }
                }
                self.camera_dist = self.camera_dist.max(self.min_zoom).min(self.max_zoom);
                true
            }
            _ => false,
        }
    }

    // Calculate centroid of selected objects
    fn get_selected_centroid(&self) -> Option<cgmath::Point3<f32>> {
        let selected: Vec<&SceneObject> = self.objects.iter().filter(|o| o.selected).collect();
        if selected.is_empty() {
            return None;
        }

        let mut sum = cgmath::Vector3::zero();
        for obj in &selected {
            sum += obj.instance.position;
        }
        let center = sum / selected.len() as f32;
        Some(cgmath::Point3::new(center.x, center.y, center.z))
    }

    /// Build a world-space ray (origin, normalized direction) from a point in
    /// normalized device coordinates.
    fn build_ray_ndc(&self, x: f32, y: f32) -> (cgmath::Vector3<f32>, cgmath::Vector3<f32>) {
        let inv_vp = self
            .camera
            .build_view_projection_matrix()
            .invert()
            .unwrap_or(cgmath::Matrix4::identity());
        let near = inv_vp * cgmath::Vector4::new(x, y, 0.0, 1.0);
        let far = inv_vp * cgmath::Vector4::new(x, y, 1.0, 1.0);
        let near_world = near.truncate() / near.w;
        let far_world = far.truncate() / far.w;
        (near_world, (far_world - near_world).normalize())
    }

    /// Closest world-space surface hit under the NDC point, across all visible
    /// objects (imported meshes and primitives). Used by the measurement tool.
    pub fn raycast_world(&self, x: f32, y: f32) -> Option<[f32; 3]> {
        let (ray_origin, ray_dir) = self.build_ray_ndc(x, y);
        let mut best: Option<(f32, [f32; 3])> = None;
        for obj in &self.objects {
            if !obj.visible {
                continue;
            }
            let model = obj.instance.to_model_matrix();
            let inv_model = model.invert().unwrap_or(cgmath::Matrix4::identity());
            let local_origin = (inv_model * ray_origin.extend(1.0)).truncate();
            let local_dir = (inv_model * ray_dir.extend(0.0)).truncate().normalize();
            let hit = if obj.geometry_type == GeometryType::Mesh {
                self.custom_meshes
                    .get(&obj.id)
                    .and_then(|m| m.data.ray_hit(local_origin, local_dir))
            } else {
                intersect_primitive(obj.geometry_type, local_origin, local_dir)
            };
            if let Some(t) = hit {
                let p_local = local_origin + local_dir * t;
                let p_world = (model * p_local.extend(1.0)).truncate();
                let dist = (p_world - ray_origin).magnitude2();
                if best.is_none_or(|(d, _)| dist < d) {
                    best = Some((dist, [p_world.x, p_world.y, p_world.z]));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    fn select_object_at_ndc(&mut self, x: f32, y: f32) -> Option<usize> {
        let modifiers = self.egui_state.egui_ctx().input(|i| i.modifiers);
        let multi_select = modifiers.ctrl || modifiers.shift || modifiers.mac_cmd;

        let (ray_origin, ray_dir) = self.build_ray_ndc(x, y);

        let mut closest_hit: Option<(usize, f32)> = None;

        for obj in &self.objects {
            if !obj.visible {
                continue;
            }

            let model = obj.instance.to_model_matrix();
            let inv_model = model.invert().unwrap_or(cgmath::Matrix4::identity());

            let local_origin = (inv_model * ray_origin.extend(1.0)).truncate();
            let local_dir = (inv_model * ray_dir.extend(0.0)).truncate().normalize();

            // Imported meshes ray-test against their triangles; primitives use
            // their analytic intersector.
            let hit = if obj.geometry_type == GeometryType::Mesh {
                self.custom_meshes
                    .get(&obj.id)
                    .and_then(|m| m.data.ray_hit(local_origin, local_dir))
            } else {
                intersect_primitive(obj.geometry_type, local_origin, local_dir)
            };
            if let Some(t) = hit {
                let world_dir = (model * local_dir.extend(0.0)).truncate();
                let world_t = t * world_dir.magnitude();

                if closest_hit.is_none() || world_t < closest_hit.as_ref().unwrap().1 {
                    closest_hit = Some((obj.id, world_t));
                }
            }
        }

        if !multi_select {
            for obj in &mut self.objects {
                obj.selected = false;
            }
        }

        if let Some((hit_id, _)) = closest_hit {
            if let Some(obj) = self.objects.iter_mut().find(|o| o.id == hit_id) {
                if multi_select {
                    obj.selected = !obj.selected; // Toggle in multi
                } else {
                    obj.selected = true;
                }
            }
            Some(hit_id)
        } else {
            None
        }
    }

    pub fn update(&mut self) {
        // Update Camera from parameters
        let yaw = cgmath::Deg(self.camera_yaw);
        let pitch = cgmath::Deg(self.camera_pitch);
        let dist = self.camera_dist;

        // Z-Up Spherical Coordinates
        // Pitch lifts Z
        // Yaw rotates around Z
        let x = dist * yaw.cos() * pitch.cos();
        let y = dist * yaw.sin() * pitch.cos();
        let z = dist * pitch.sin();

        // Ensure camera UP is Z
        self.camera.up = cgmath::Vector3::unit_z();

        let offset = cgmath::Vector3::new(x, y, z);
        self.camera.eye = self.camera_target + offset;
        self.camera.target = self.camera_target;

        // Uniforms
        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&self.camera);
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Drain any in-flight 3D-model import before drawing this frame.
        self.poll_mesh_worker();

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("Depth Texture"),
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // --- EGUI UI DRAWING ---
        let raw_input = self.egui_state.take_egui_input(&self.window);

        // Deferred actions
        let mut action_reset_view = false;
        let mut action_focus_selected = false;
        let mut create_object = None;

        let mut action_edit_obj_id = None;
        let mut action_close_editor = false;
        let mut action_delete_obj_id = None;
        let mut action_confirm_edit = false;
        let mut dataset_action = crate::ui::DatasetAction::None;
        let mut geometry_focus: Option<[f32; 3]> = None;
        // Bottom-list actions for the imported dataset entry.
        let mut dataset_focus: Option<[f32; 3]> = None;
        let mut remove_dataset = false;

        let full_output = self.egui_state.egui_ctx().run(raw_input, |ctx| {
            // Because we can't easily borrow fields disjointly multiple times in complex closure,
            // We'll use the 'self' accessible in the closure but be careful.

            // Top Left Panel: Add Object Button
            egui::Area::new("add_obj_area".into())
                .anchor(egui::Align2::LEFT_TOP, [10.0, 10.0])
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("➕ Object").clicked() {
                            let id = self.next_id;
                            let default_obj = SceneObject::new(
                                id,
                                format!("Object {}", id),
                                [0.0, 0.0, 0.0],
                                GeometryType::Cube,
                            );
                            self.draft_object = Some(default_obj);
                            self.should_focus_name = true;
                        }
                        if ui.button("📊 Dataset").clicked() {
                            self.dataset_view.show_window = !self.dataset_view.show_window;
                        }
                        if ui.button("🧊 Solids").clicked() {
                            self.geometry_view.show_window = !self.geometry_view.show_window;
                        }
                        // Measurement tool toggle. While active, clicking two
                        // surface points reports the distance between them.
                        if ui
                            .selectable_label(self.measure_mode, "📏 Measure")
                            .on_hover_text("Click two surface points to measure the distance")
                            .clicked()
                        {
                            self.measure_mode = !self.measure_mode;
                            self.measure_points.clear();
                        }
                    });
                });

            // Measurement read-out + clear, shown only while the tool is on.
            if self.measure_mode {
                egui::Area::new("measure_hud".into())
                    .anchor(egui::Align2::LEFT_TOP, [10.0, 48.0])
                    .show(ctx, |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            match self.measure_points.as_slice() {
                                [] => {
                                    ui.label("📏 Click the first point");
                                }
                                [_] => {
                                    ui.label("📏 Click the second point");
                                }
                                [a, b, ..] => {
                                    let d = ((a[0] - b[0]).powi(2)
                                        + (a[1] - b[1]).powi(2)
                                        + (a[2] - b[2]).powi(2))
                                    .sqrt();
                                    ui.label(
                                        egui::RichText::new(format!("📏 Distance: {:.3}", d))
                                            .strong(),
                                    );
                                }
                            }
                            if ui.button("Clear").clicked() {
                                self.measure_points.clear();
                            }
                        });
                    });
            }

            // Dataset visualizer window (import, table, filters, search, export)
            dataset_action = self.dataset_view.show(ctx);
            // Geometry import window (paste / files / layers)
            geometry_focus = self.geometry_view.show(ctx);

            // Top Right Panel: Settings
            egui::Window::new(t!("settings.window_title").to_string())
                .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
                .collapsible(true)
                .open(&mut self.show_settings)
                .show(ctx, |ui| {
                    ui.heading(t!("settings.global_heading").to_string());

                    // Interface language picker (persists the choice).
                    ui.horizontal(|ui| {
                        ui.label(format!("{}:", t!("settings.language")));
                        let current = crate::i18n::current();
                        egui::ComboBox::from_id_source("language_combo")
                            .selected_text(crate::i18n::display_name(&current))
                            .show_ui(ui, |ui| {
                                for &(code, name) in crate::i18n::LANGUAGES {
                                    if ui.selectable_label(current == code, name).clicked() {
                                        crate::i18n::set_language(code);
                                    }
                                }
                            });
                    });

                    ui.add(
                        egui::Slider::new(&mut self.bg_color, 0.0..=1.0)
                            .text(t!("common.background").to_string()),
                    );

                    ui.separator();
                    ui.heading(t!("settings.camera_view").to_string());
                    if ui.button(t!("settings.focus_selected").to_string()).clicked() {
                        action_focus_selected = true;
                    }
                    if ui.button(t!("settings.reset_view").to_string()).clicked() {
                        action_reset_view = true;
                    }

                    ui.separator();
                    ui.label(t!("settings.zoom_limits").to_string());
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.min_zoom)
                                .speed(0.1)
                                .prefix(t!("common.min_prefix").to_string()),
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.max_zoom)
                                .speed(1.0)
                                .prefix(t!("common.max_prefix").to_string()),
                        );
                    });

                    ui.separator();
                    ui.heading(t!("settings.grid_options").to_string());
                    ui.checkbox(&mut self.show_grid_xy, t!("settings.grid_xy").to_string());
                    ui.checkbox(&mut self.show_grid_xz, t!("settings.grid_xz").to_string());
                    ui.checkbox(&mut self.show_grid_yz, t!("settings.grid_yz").to_string());
                    ui.checkbox(&mut self.show_axes, t!("settings.show_axes").to_string());
                });

            // Gear Button
            if !self.show_settings {
                egui::Area::new("settings_btn_area".into())
                    .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
                    .show(ctx, |ui| {
                        if ui.button(egui::RichText::new("⚙").size(20.0)).clicked() {
                            self.show_settings = true;
                        }
                    });
            }

            // Status Bar (Very Bottom)
            egui::TopBottomPanel::bottom("status_bar")
                .resizable(false)
                .min_height(20.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "{}: {:.2}, {:.2}, {:.2}",
                            t!("status.camera_eye"),
                            self.camera.eye.x,
                            self.camera.eye.y,
                            self.camera.eye.z
                        ));
                        ui.separator();
                        ui.label(format!(
                            "{}: {:.2}, {:.2}, {:.2}",
                            t!("status.target"),
                            self.camera_target.x,
                            self.camera_target.y,
                            self.camera_target.z
                        ));
                        ui.separator();
                        ui.label(format!(
                            "{}: {:.1}° {}: {:.1}°",
                            t!("status.yaw"),
                            self.camera_yaw,
                            t!("status.pitch"),
                            self.camera_pitch
                        ));
                    });
                });

            // Bottom Panel: Object List
            let mut panel_expanded = self.bottom_panel_expanded;
            egui::TopBottomPanel::bottom("bottom_panel")
                .frame(egui::Frame::none()) // Remove horizontal band when closed
                .resizable(false)
                .show(ctx, |ui| {
                    ui.spacing_mut().item_spacing.y = 0.0; // Remove spacing below tab
                                                           // Central Tab
                    ui.vertical_centered(|ui| {
                        let tab_bg = ctx.style().visuals.window_fill;
                        egui::Frame::none()
                            .fill(tab_bg)
                            .rounding(egui::Rounding {
                                nw: 12.0,
                                ne: 12.0,
                                sw: 0.0,
                                se: 0.0,
                            })
                            .inner_margin(egui::Margin::symmetric(24.0, 6.0))
                            .show(ui, |ui| {
                                let (rect, response) = ui.allocate_exact_size(
                                    egui::vec2(16.0, 8.0),
                                    egui::Sense::click(),
                                );

                                let color = if response.hovered() {
                                    ctx.style().visuals.widgets.hovered.text_color()
                                } else {
                                    ctx.style().visuals.text_color()
                                };

                                let mut points = vec![];
                                if panel_expanded {
                                    // Point downwards
                                    points.push(rect.left_top());
                                    points.push(rect.right_top());
                                    points.push(rect.center_bottom());
                                } else {
                                    // Point upwards
                                    points.push(rect.center_top());
                                    points.push(rect.right_bottom());
                                    points.push(rect.left_bottom());
                                }
                                ui.painter().add(egui::Shape::convex_polygon(
                                    points,
                                    color,
                                    egui::Stroke::NONE,
                                ));

                                if response.hovered() {
                                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                }
                                if response.clicked() {
                                    panel_expanded = !panel_expanded;
                                }
                            });
                    });

                    if panel_expanded {
                        let panel_bg = ctx.style().visuals.window_fill;
                        egui::Frame::none()
                            .fill(panel_bg)
                            .inner_margin(egui::Margin {
                                left: 12.0,
                                right: 12.0,
                                top: 4.0,
                                bottom: 12.0,
                            })
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width()); // expand to full width
                                egui::ScrollArea::vertical()
                                    .max_height(250.0)
                                    .show(ui, |ui| {
                                        ui.spacing_mut().item_spacing.y = 2.0;
                                        let mut select_exclusive_id = None;
                                        for obj in self.objects.iter_mut() {
                                            ui.horizontal(|ui| {
                                                // Unique Name (Selectable)
                                                let response = ui.selectable_label(
                                                    obj.selected,
                                                    egui::RichText::new(&obj.label)
                                                        .strong()
                                                        .size(14.0),
                                                );
                                                if response.clicked() {
                                                    let modifiers = ui.ctx().input(|i| i.modifiers);
                                                    if modifiers.ctrl
                                                        || modifiers.shift
                                                        || modifiers.mac_cmd
                                                    {
                                                        obj.selected = !obj.selected;
                                                    } else {
                                                        select_exclusive_id = Some(obj.id);
                                                    }
                                                }
                                                if response.double_clicked() {
                                                    action_edit_obj_id = Some(obj.id);
                                                }

                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        // Edit / delete / label / visibility icons
                                                        // share the same look via `list_icon_button`.
                                                        let is_editing = action_edit_obj_id
                                                            == Some(obj.id)
                                                            || self.editing_obj_id == Some(obj.id);
                                                        let edit_color = if is_editing {
                                                            egui::Color32::from_rgb(0, 150, 255)
                                                        } else {
                                                            ui.visuals().text_color()
                                                        };
                                                        if list_icon_button(ui, "✏", edit_color, "Edit")
                                                        {
                                                            action_edit_obj_id = Some(obj.id);
                                                            self.editing_obj_draft =
                                                                Some(obj.clone());
                                                            self.should_focus_name = true;
                                                        }

                                                        if list_icon_button(
                                                            ui,
                                                            "🗑",
                                                            egui::Color32::from_rgb(255, 100, 100),
                                                            "Delete",
                                                        ) {
                                                            action_delete_obj_id = Some(obj.id);
                                                        }

                                                        let label_color = if obj.show_label {
                                                            egui::Color32::from_rgb(0, 200, 255)
                                                        } else {
                                                            egui::Color32::from_rgb(150, 150, 150)
                                                        };
                                                        if list_icon_button(
                                                            ui,
                                                            "🏷",
                                                            label_color,
                                                            "Toggle Label",
                                                        ) {
                                                            obj.show_label = !obj.show_label;
                                                        }

                                                        // Visibility (opposite colors when hidden).
                                                        let (vis_icon, vis_color) = if obj.visible {
                                                            ("👁", egui::Color32::from_rgb(100, 255, 100))
                                                        } else {
                                                            ("🕶", egui::Color32::from_rgb(255, 100, 100))
                                                        };
                                                        if list_icon_button(
                                                            ui, vis_icon, vis_color, "Visibility",
                                                        ) {
                                                            obj.visible = !obj.visible;
                                                        }
                                                    },
                                                );
                                            });
                                        }

                                        if let Some(id) = select_exclusive_id {
                                            for obj in &mut self.objects {
                                                obj.selected = obj.id == id;
                                            }
                                        }

                                        // --- Imported dataset ---
                                        // The dataset visualizer no longer owns
                                        // an "Explore" list; its imported data
                                        // lives here, in the same list as every
                                        // other object.
                                        let dataset_entry =
                                            self.dataset_view.loaded.as_ref().map(|l| {
                                                (
                                                    l.dataset.metadata.name.clone(),
                                                    l.dataset.n_rows(),
                                                )
                                            });
                                        if let Some((name, n_rows)) = dataset_entry {
                                            let visible = self.dataset_view.visible;
                                            let centroid = self.dataset_view.centroid();
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    egui::RichText::new(format!("📊 {}", name))
                                                        .strong()
                                                        .size(14.0),
                                                );
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "({} rows)",
                                                        n_rows
                                                    ))
                                                    .weak(),
                                                );
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(egui::Align::Center),
                                                    |ui| {
                                                        if list_icon_button(
                                                            ui,
                                                            "🗑",
                                                            egui::Color32::from_rgb(255, 100, 100),
                                                            "Remove dataset",
                                                        ) {
                                                            remove_dataset = true;
                                                        }
                                                        if list_icon_button(
                                                            ui,
                                                            "🎯",
                                                            ui.visuals().text_color(),
                                                            "Focus camera on dataset",
                                                        ) {
                                                            dataset_focus = centroid;
                                                        }
                                                        let (vis_icon, vis_color) = if visible {
                                                            (
                                                                "👁",
                                                                egui::Color32::from_rgb(
                                                                    100, 255, 100,
                                                                ),
                                                            )
                                                        } else {
                                                            (
                                                                "🕶",
                                                                egui::Color32::from_rgb(
                                                                    255, 100, 100,
                                                                ),
                                                            )
                                                        };
                                                        if list_icon_button(
                                                            ui,
                                                            vis_icon,
                                                            vis_color,
                                                            "Visibility",
                                                        ) {
                                                            self.dataset_view.set_visible(!visible);
                                                        }
                                                    },
                                                );
                                            });
                                        }
                                    });
                            });
                    }
                });
            self.bottom_panel_expanded = panel_expanded;

            // Draft Create Object Window
            if let Some(mut draft) = self.draft_object.take() {
                let mut open = true;
                let mut action_discard = false;
                let mut action_confirm = false;
                egui::Window::new(t!("add.window_title").to_string())
                    .open(&mut open)
                    .resizable(true)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(t!("common.name").to_string());
                            let res = ui.text_edit_singleline(&mut draft.label);
                            if self.should_focus_name {
                                res.request_focus();
                                self.should_focus_name = false;
                            }
                        });
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label(t!("common.type").to_string());
                            let type_label = match draft.geometry_type {
                                GeometryType::Cube => t!("shape.cube").to_string(),
                                GeometryType::Sphere => t!("shape.sphere").to_string(),
                                GeometryType::Plane => t!("shape.plane").to_string(),
                                _ => format!("{:?}", draft.geometry_type),
                            };
                            egui::ComboBox::from_id_source("draft_type_combo")
                                .selected_text(type_label)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut draft.geometry_type,
                                        GeometryType::Cube,
                                        t!("shape.cube").to_string(),
                                    );
                                    ui.selectable_value(
                                        &mut draft.geometry_type,
                                        GeometryType::Sphere,
                                        t!("shape.sphere").to_string(),
                                    );
                                    ui.selectable_value(
                                        &mut draft.geometry_type,
                                        GeometryType::Plane,
                                        t!("shape.plane").to_string(),
                                    );
                                });
                        });

                        ui.heading(t!("add.transform").to_string());
                        ui.horizontal(|ui| {
                            ui.label(t!("common.position").to_string());
                            ui.add(egui::DragValue::new(&mut draft.instance.position.x).speed(0.1));
                            ui.add(egui::DragValue::new(&mut draft.instance.position.y).speed(0.1));
                            ui.add(egui::DragValue::new(&mut draft.instance.position.z).speed(0.1));
                        });
                        ui.horizontal(|ui| {
                            ui.label(t!("common.rotation").to_string());
                            let mut changed = false;
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut draft.rotation_euler[0])
                                        .speed(1.0)
                                        .suffix("°"),
                                )
                                .changed();
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut draft.rotation_euler[1])
                                        .speed(1.0)
                                        .suffix("°"),
                                )
                                .changed();
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut draft.rotation_euler[2])
                                        .speed(1.0)
                                        .suffix("°"),
                                )
                                .changed();
                            if changed {
                                draft.update_rotation();
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(t!("common.scale").to_string());
                            ui.add(
                                egui::DragValue::new(&mut draft.instance.scale.x)
                                    .speed(0.01)
                                    .max_decimals(2),
                            );
                            ui.add(
                                egui::DragValue::new(&mut draft.instance.scale.y)
                                    .speed(0.01)
                                    .max_decimals(2),
                            );
                            ui.add(
                                egui::DragValue::new(&mut draft.instance.scale.z)
                                    .speed(0.01)
                                    .max_decimals(2),
                            );
                        });

                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label(t!("common.color").to_string());
                            ui.color_edit_button_rgb(&mut draft.color);
                        });
                        ui.checkbox(&mut draft.show_label, t!("add.show_label").to_string());

                        ui.separator();
                        ui.heading(t!("add.geometry_props").to_string());
                        match draft.geometry_type {
                            GeometryType::Cube => {
                                ui.horizontal(|ui| {
                                    ui.label(t!("add.side_length").to_string());
                                    if ui
                                        .add(egui::DragValue::new(&mut draft.cube_side).speed(0.1))
                                        .changed()
                                    {
                                        let s = draft.cube_side;
                                        draft.instance.scale = cgmath::Vector3::new(s, s, s);
                                    }
                                });
                            }
                            GeometryType::Sphere => {
                                ui.horizontal(|ui| {
                                    ui.label(t!("add.radius").to_string());
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut draft.sphere_radius)
                                                .speed(0.1),
                                        )
                                        .changed()
                                    {
                                        let s = draft.sphere_radius * 2.0; // Mesh is radius 0.5, so scale=2*radius
                                        draft.instance.scale = cgmath::Vector3::new(s, s, s);
                                    }
                                });
                            }
                            GeometryType::Plane => {
                                ui.horizontal(|ui| {
                                    ui.label(t!("add.surface_area").to_string());
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut draft.plane_surface)
                                                .speed(0.1)
                                                .range(0.1..=100.0),
                                        )
                                        .changed()
                                    {
                                        let s = draft.plane_surface.sqrt();
                                        draft.instance.scale = cgmath::Vector3::new(s, 1.0, s);
                                    }
                                });
                                ui.checkbox(&mut draft.show_normal, t!("add.show_normal").to_string());
                            }
                            _ => {}
                        }

                        ui.separator();
                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui
                                .add_sized(
                                    [80.0, 24.0],
                                    egui::Button::new(t!("common.cancel").to_string()),
                                )
                                .clicked()
                                || ui.input(|i| i.key_pressed(egui::Key::Escape))
                            {
                                action_discard = true;
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add_sized(
                                            [80.0, 24.0],
                                            egui::Button::new(t!("common.add").to_string()),
                                        )
                                        .clicked()
                                        || ui.input(|i| i.key_pressed(egui::Key::Enter))
                                    {
                                        action_confirm = true;
                                    }
                                },
                            );
                        });
                    });

                if action_discard {
                    open = false;
                }
                if action_confirm {
                    create_object = Some(draft.clone());
                    open = false;
                }

                if open {
                    self.draft_object = Some(draft); // Put it back if still open
                }
            }

            // Editor Popup Window
            if self.editing_obj_id.is_some() {
                let mut open = true;
                if let Some(mut obj) = self.editing_obj_draft.take() {
                    egui::Window::new("Object Properties")
                        .open(&mut open)
                        .resizable(true)
                        .show(ctx, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Name:");
                                let res = ui.text_edit_singleline(&mut obj.label);
                                if self.should_focus_name {
                                    res.request_focus();
                                    self.should_focus_name = false;
                                }
                            });
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label("Type:");
                                egui::ComboBox::from_id_source(format!("type_combo_{}", obj.id))
                                    .selected_text(format!("{:?}", obj.geometry_type))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut obj.geometry_type,
                                            GeometryType::Cube,
                                            "Cube",
                                        );
                                        ui.selectable_value(
                                            &mut obj.geometry_type,
                                            GeometryType::Sphere,
                                            "Sphere",
                                        );
                                        ui.selectable_value(
                                            &mut obj.geometry_type,
                                            GeometryType::Plane,
                                            "Plane",
                                        );
                                    });
                            });

                            ui.heading("Transform");
                            ui.horizontal(|ui| {
                                ui.label("Pos:");
                                ui.add(
                                    egui::DragValue::new(&mut obj.instance.position.x).speed(0.1),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut obj.instance.position.y).speed(0.1),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut obj.instance.position.z).speed(0.1),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.label("Rot:");
                                let mut changed = false;
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut obj.rotation_euler[0])
                                            .speed(1.0)
                                            .suffix("°"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut obj.rotation_euler[1])
                                            .speed(1.0)
                                            .suffix("°"),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut obj.rotation_euler[2])
                                            .speed(1.0)
                                            .suffix("°"),
                                    )
                                    .changed();
                                if changed {
                                    obj.update_rotation();
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Scale:");
                                ui.add(
                                    egui::DragValue::new(&mut obj.instance.scale.x)
                                        .speed(0.01)
                                        .max_decimals(2),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut obj.instance.scale.y)
                                        .speed(0.01)
                                        .max_decimals(2),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut obj.instance.scale.z)
                                        .speed(0.01)
                                        .max_decimals(2),
                                );
                            });

                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label("Color:");
                                ui.color_edit_button_rgb(&mut obj.color);
                            });
                            ui.checkbox(&mut obj.show_label, "Show Label in Viewport");

                            ui.separator();
                            ui.heading("Geometry Properties");
                            match obj.geometry_type {
                                GeometryType::Cube => {
                                    ui.horizontal(|ui| {
                                        ui.label("Side Length:");
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut obj.cube_side).speed(0.1),
                                            )
                                            .changed()
                                        {
                                            let s = obj.cube_side;
                                            obj.instance.scale = cgmath::Vector3::new(s, s, s);
                                        }
                                    });
                                }
                                GeometryType::Sphere => {
                                    ui.horizontal(|ui| {
                                        ui.label("Radius:");
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut obj.sphere_radius)
                                                    .speed(0.1),
                                            )
                                            .changed()
                                        {
                                            let s = obj.sphere_radius * 2.0;
                                            obj.instance.scale = cgmath::Vector3::new(s, s, s);
                                        }
                                    });
                                }
                                GeometryType::Plane => {
                                    ui.horizontal(|ui| {
                                        ui.label("Surface Area:");
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut obj.plane_surface)
                                                    .speed(0.1)
                                                    .range(0.1..=100.0),
                                            )
                                            .changed()
                                        {
                                            let s = obj.plane_surface.sqrt();
                                            obj.instance.scale = cgmath::Vector3::new(s, 1.0, s);
                                        }
                                    });
                                    ui.checkbox(&mut obj.show_normal, "Show Normal Arrow");
                                }
                                _ => {}
                            }

                            ui.separator();
                            ui.horizontal(|ui| {
                                if ui
                                    .add_sized([80.0, 24.0], egui::Button::new("Cancel"))
                                    .clicked()
                                    || ui.input(|i| i.key_pressed(egui::Key::Escape))
                                {
                                    action_close_editor = true;
                                }
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .add_sized([80.0, 24.0], egui::Button::new("Confirm"))
                                            .clicked()
                                            || ui.input(|i| i.key_pressed(egui::Key::Enter))
                                        {
                                            action_confirm_edit = true;
                                        }
                                    },
                                );
                            });
                        });

                    if open {
                        self.editing_obj_draft = Some(obj);
                    }
                }
                if !open {
                    action_close_editor = true;
                }
            }

            // 3D Object Labels
            let vp = self.camera.build_view_projection_matrix();
            for obj in &self.objects {
                if obj.visible && obj.show_label {
                    let world_pos = obj.instance.position;
                    // Draw label slightly above the object
                    let offset = match obj.geometry_type {
                        GeometryType::Plane => 0.1,
                        _ => 0.6 * obj.instance.scale.z,
                    };
                    let label_pos = world_pos + cgmath::Vector3::new(0.0, 0.0, offset);
                    let clip_pos = vp * label_pos.extend(1.0);

                    if clip_pos.w > 0.0 {
                        let ndc = clip_pos.truncate() / clip_pos.w;
                        if ndc.x.abs() <= 1.1 && ndc.y.abs() <= 1.1 {
                            let screen_x = (ndc.x + 1.0) * 0.5 * self.size.width as f32;
                            let screen_y = (1.0 - ndc.y) * 0.5 * self.size.height as f32;

                            let ppp = ctx.pixels_per_point();
                            let egui_pos = egui::pos2(screen_x / ppp, screen_y / ppp);

                            egui::Area::new(format!("label_area_{}", obj.id).into())
                                .fixed_pos(egui_pos)
                                .pivot(egui::Align2::CENTER_BOTTOM)
                                .show(ctx, |ui| {
                                    ui.label(
                                        egui::RichText::new(&obj.label)
                                            .color(egui::Color32::WHITE)
                                            .background_color(egui::Color32::from_black_alpha(160))
                                            .strong(),
                                    );
                                });
                        }
                    }
                }
            }

            // Measurement overlay: markers, the segment and its length.
            if self.measure_mode && !self.measure_points.is_empty() {
                let ppp = ctx.pixels_per_point();
                let w = self.size.width as f32;
                let h = self.size.height as f32;
                let vp = self.camera.build_view_projection_matrix();
                let project = |world: [f32; 3]| -> Option<egui::Pos2> {
                    let clip = vp * cgmath::Vector4::new(world[0], world[1], world[2], 1.0);
                    if clip.w <= 0.0 {
                        return None;
                    }
                    let ndc = clip.truncate() / clip.w;
                    Some(egui::pos2(
                        ((ndc.x + 1.0) * 0.5 * w) / ppp,
                        ((1.0 - ndc.y) * 0.5 * h) / ppp,
                    ))
                };
                let painter = ctx.layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    egui::Id::new("measure_overlay"),
                ));
                let pts: Vec<egui::Pos2> =
                    self.measure_points.iter().filter_map(|p| project(*p)).collect();
                for p in &pts {
                    painter.circle_filled(*p, 4.0, egui::Color32::YELLOW);
                }
                if pts.len() == 2 {
                    painter.line_segment(
                        [pts[0], pts[1]],
                        egui::Stroke::new(2.0, egui::Color32::YELLOW),
                    );
                    let a = self.measure_points[0];
                    let b = self.measure_points[1];
                    let d = ((a[0] - b[0]).powi(2)
                        + (a[1] - b[1]).powi(2)
                        + (a[2] - b[2]).powi(2))
                    .sqrt();
                    let mid = egui::pos2((pts[0].x + pts[1].x) * 0.5, (pts[0].y + pts[1].y) * 0.5);
                    painter.text(
                        mid,
                        egui::Align2::CENTER_BOTTOM,
                        format!("{:.3}", d),
                        egui::FontId::proportional(14.0),
                        egui::Color32::WHITE,
                    );
                }
            }
        });

        // Apply deferred actions
        if let Some(obj) = create_object {
            self.objects.push(obj);
            self.next_id += 1;
        }

        if action_reset_view {
            self.camera_target = cgmath::Point3::new(
                DEFAULT_CAMERA_TARGET[0],
                DEFAULT_CAMERA_TARGET[1],
                DEFAULT_CAMERA_TARGET[2],
            );
            self.camera_yaw = DEFAULT_CAMERA_YAW;
            self.camera_pitch = DEFAULT_CAMERA_PITCH;
            self.camera_dist = DEFAULT_CAMERA_DIST;
        }

        if action_focus_selected {
            if let Some(target) = self.get_selected_centroid() {
                self.camera_target = target;
            }
        }

        if let crate::ui::DatasetAction::FocusPoint(p) = dataset_action {
            self.camera_target = cgmath::Point3::new(p[0], p[1], p[2]);
        }
        if let Some(p) = geometry_focus {
            self.camera_target = cgmath::Point3::new(p[0], p[1], p[2]);
        }
        if let Some(p) = dataset_focus {
            self.camera_target = cgmath::Point3::new(p[0], p[1], p[2]);
        }
        if remove_dataset {
            self.dataset_view.clear_dataset();
        }

        // Kick off a 3D-model import requested from the Solids window.
        if let Some(path) = self.geometry_view.take_mesh_request() {
            self.spawn_mesh_load(path);
        }

        // Rebuild geometry-layer instance buffers only when layers changed.
        if self.geometry_view.take_render_dirty() {
            self.geometry_batches.clear();
            for (geometry, instances) in self.geometry_view.build_geometry_batches() {
                let buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Geometry Layer Instance Buffer"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                self.geometry_batches
                    .push((geometry, buffer, instances.len() as u32));
            }
        }

        // Rebuild dataset point-cloud instance buffers only when the visible
        // selection / settings changed.
        if self.dataset_view.take_render_dirty() {
            self.dataset_point_batches.clear();
            let result = self.dataset_view.build_point_cloud();
            for batch in result.batches {
                let buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Dataset Point Instance Buffer"),
                        contents: bytemuck::cast_slice(&batch.instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                self.dataset_point_batches.push((
                    batch.geometry,
                    buffer,
                    batch.instances.len() as u32,
                ));
            }
        }

        // Editor Actions
        if let Some(id) = action_edit_obj_id {
            self.editing_obj_id = Some(id);
        }

        if action_close_editor {
            self.editing_obj_id = None;
            self.editing_obj_draft = None;
        }

        if action_confirm_edit {
            if let Some(new_draft) = self.editing_obj_draft.take() {
                let mut old_state = None;
                if let Some(obj) = self.objects.iter_mut().find(|o| o.id == new_draft.id) {
                    if *obj != new_draft {
                        old_state = Some(obj.clone());
                        *obj = new_draft.clone();
                    }
                }
                if let Some(old) = old_state {
                    self.push_undo(UndoCommand::Edit {
                        old,
                        new: new_draft,
                    });
                }
            }
            self.editing_obj_id = None;
        }

        if let Some(id_to_delete) = action_delete_obj_id {
            if let Some(index) = self.objects.iter().position(|o| o.id == id_to_delete) {
                // Record the deletion so it is undoable, consistent with the
                // Delete-key path. Any imported-mesh GPU buffers in
                // `custom_meshes` are intentionally retained so undo can
                // restore the object without re-uploading.
                let removed = self.objects.remove(index);
                self.push_undo(UndoCommand::Delete(removed));
            }
            if self.editing_obj_id == Some(id_to_delete) {
                self.editing_obj_id = None;
            }
        }

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        let tris = self.egui_state.egui_ctx().tessellate(
            full_output.shapes,
            self.egui_state.egui_ctx().pixels_per_point(),
        );
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.egui_state.egui_ctx().pixels_per_point(),
        };

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &tris,
            &screen_descriptor,
        );

        // --- Prepare Buffers ---
        let mut draw_list: Vec<(GeometryType, wgpu::Buffer, u32)> = Vec::new();

        // Identity Matrix for Grids/Axes
        let identity_instance = [InstanceRaw {
            model: cgmath::Matrix4::identity().into(),
            color: [1.0; 4],
        }];
        let single_instance_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Single Instance Buffer"),
                    contents: bytemuck::cast_slice(&identity_instance),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        for geo_type in [
            GeometryType::Cube,
            GeometryType::Sphere,
            GeometryType::Plane,
        ] {
            let instances = self
                .objects
                .iter()
                .filter(|o| o.visible && o.geometry_type == geo_type)
                .map(|o| o.instance.to_raw_with_color(o.color, o.selected))
                .collect::<Vec<_>>();

            if !instances.is_empty() {
                let buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Instance Buffer"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                draw_list.push((geo_type, buffer, instances.len() as u32));
            }
        }

        // Imported 3D models: one draw per object (each has its own mesh), so
        // build a single-instance buffer per visible mesh object up front to
        // keep the buffers alive for the whole render pass.
        let mesh_draws: Vec<(usize, wgpu::Buffer)> = self
            .objects
            .iter()
            .filter(|o| {
                o.visible
                    && o.geometry_type == GeometryType::Mesh
                    && self.custom_meshes.contains_key(&o.id)
            })
            .map(|o| {
                let instance = [o.instance.to_raw_with_color(o.color, o.selected)];
                let buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Custom Mesh Instance Buffer"),
                        contents: bytemuck::cast_slice(&instance),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                (o.id, buffer)
            })
            .collect();

        // Prepare Normal Arrow Buffer outside render pass to avoid lifetime issues
        let normal_arrows: Vec<InstanceRaw> = self
            .objects
            .iter()
            .filter(|o| o.visible && o.geometry_type == GeometryType::Plane && o.show_normal)
            .map(|o| {
                // Arrow should be centered on plane and follow its position/rotation
                // BUT it must NOT inherit the plane's scale (otherwise it gets squashed)
                let mut arrow_instance = o.instance.clone();
                arrow_instance.scale = cgmath::Vector3::new(1.0, 1.0, 1.0);

                // Use a slightly different shade of the plane color for the arrow
                let arrow_color = [
                    (o.color[0] * 0.8).min(1.0),
                    (o.color[1] * 0.8).min(1.0),
                    (o.color[2] * 0.8).min(1.0),
                ];

                arrow_instance.to_raw_with_color(arrow_color, false)
            })
            .collect();

        let normal_arrow_instance_buffer = if !normal_arrows.is_empty() {
            Some(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Normal Arrow Instance Buffer"),
                        contents: bytemuck::cast_slice(&normal_arrows),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
            )
        } else {
            None
        };

        // --- Render Pass ---
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bg_color,
                            g: self.bg_color,
                            b: self.bg_color,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            // Draw Grid (Lines)
            render_pass.set_pipeline(&self.line_pipeline);
            render_pass.set_vertex_buffer(1, single_instance_buffer.slice(..));

            if self.show_grid_xy {
                render_pass.set_vertex_buffer(0, self.grid_xy_mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    self.grid_xy_mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                render_pass.draw_indexed(0..self.grid_xy_mesh.num_indices, 0, 0..1);
            }
            if self.show_grid_xz {
                render_pass.set_vertex_buffer(0, self.grid_xz_mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    self.grid_xz_mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                render_pass.draw_indexed(0..self.grid_xz_mesh.num_indices, 0, 0..1);
            }
            if self.show_grid_yz {
                render_pass.set_vertex_buffer(0, self.grid_yz_mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    self.grid_yz_mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                render_pass.draw_indexed(0..self.grid_yz_mesh.num_indices, 0, 0..1);
            }

            // Draw Objects (Triangles) & AXES
            render_pass.set_pipeline(&self.render_pipeline);

            // Draw Axes
            if self.show_axes {
                render_pass.set_vertex_buffer(0, self.axes_mesh.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, single_instance_buffer.slice(..));
                render_pass.set_index_buffer(
                    self.axes_mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                render_pass.draw_indexed(0..self.axes_mesh.num_indices, 0, 0..1);
            }

            for (geo_type, instance_buf, count) in &draw_list {
                let mesh = match geo_type {
                    GeometryType::Cube => &self.cube_mesh,
                    GeometryType::Sphere => &self.sphere_mesh,
                    GeometryType::Plane => &self.plane_mesh,
                    _ => &self.cube_mesh,
                };

                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, instance_buf.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..mesh.num_indices, 0, 0..*count);
            }

            // Draw imported 3D models (32-bit indices, one instance each).
            for (id, instance_buf) in &mesh_draws {
                if let Some(custom) = self.custom_meshes.get(id) {
                    render_pass.set_vertex_buffer(0, custom.vertex_buffer.slice(..));
                    render_pass.set_vertex_buffer(1, instance_buf.slice(..));
                    render_pass.set_index_buffer(
                        custom.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..custom.num_indices, 0, 0..1);
                }
            }

            // Draw bulk instanced batches: dataset point cloud + geometry
            // layers. Spheres switch to the low-poly LOD mesh above the
            // threshold so very large imports stay interactive.
            for (geo_type, instance_buf, count) in self
                .dataset_point_batches
                .iter()
                .chain(self.geometry_batches.iter())
            {
                let mesh = match geo_type {
                    GeometryType::Cube => &self.cube_mesh,
                    GeometryType::Sphere if *count > LOD_SPHERE_THRESHOLD => {
                        &self.sphere_lod_mesh
                    }
                    GeometryType::Sphere => &self.sphere_mesh,
                    GeometryType::Plane => &self.plane_mesh,
                    _ => &self.sphere_lod_mesh,
                };
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, instance_buf.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..mesh.num_indices, 0, 0..*count);
            }

            // Draw Normal Arrows for Planes
            if let Some(instance_buffer) = &normal_arrow_instance_buffer {
                render_pass.set_vertex_buffer(0, self.normal_arrow_mesh.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
                render_pass.set_index_buffer(
                    self.normal_arrow_mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                render_pass.draw_indexed(
                    0..self.normal_arrow_mesh.num_indices,
                    0,
                    0..normal_arrows.len() as u32,
                );
            }

            // Render UI
            self.egui_renderer
                .render(&mut render_pass, &tris, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
