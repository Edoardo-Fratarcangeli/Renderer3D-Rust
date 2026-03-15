use crate::render::{draw_measurement, pick_point};
use cgmath::prelude::*;
use wgpu::util::DeviceExt;
use winit::{event::*, window::Window};

use crate::camera::{Camera, Uniforms};
use crate::model::{InstanceRaw, Vertex};
use crate::primitives;
use crate::scene::{GeometryType, SceneObject};

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

#[derive(Clone)]
pub enum UndoCommand {
    Add(SceneObject),
    Delete(SceneObject),
    Edit { old: SceneObject, new: SceneObject },
    MultiAction(Vec<UndoCommand>),
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
    pub camera_controller: crate::camera::CameraController,

    // Egui
    pub egui_renderer: egui_wgpu::Renderer,
    pub egui_state: egui_winit::State,

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
    pub mouse_pos: [f32; 2],

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
    pub custom_meshes: std::collections::HashMap<usize, MeshBuffers>,

    // Measure & Analysis (Moved from App)
    pub measure_mode: bool,
    pub measure_points: Vec<[f32; 3]>,
    pub measure_dist: f32,
    pub analysis_x: f32,
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
        let camera_controller = crate::camera::CameraController::new(
            DEFAULT_CAMERA_YAW,
            DEFAULT_CAMERA_PITCH,
            DEFAULT_CAMERA_DIST,
            DEFAULT_CAMERA_TARGET,
            DEFAULT_MIN_ZOOM,
            DEFAULT_MAX_ZOOM,
        );

        let camera = Camera {
            eye: camera_controller.eye(),
            target: camera_controller.target(),
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
            camera_controller: crate::camera::CameraController::new(
                DEFAULT_CAMERA_YAW,
                DEFAULT_CAMERA_PITCH,
                DEFAULT_CAMERA_DIST,
                DEFAULT_CAMERA_TARGET,
                DEFAULT_MIN_ZOOM,
                DEFAULT_MAX_ZOOM,
            ),
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
            editing_obj_id: None,
            editing_obj_draft: None,
            draft_object: None,
            clipboard: Vec::new(),
            mouse_pos: [0.0, 0.0],
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            normal_arrow_mesh,
            last_click_time: std::time::Instant::now(),
            should_focus_name: false,
            custom_meshes: std::collections::HashMap::new(),
            measure_mode: false,
            measure_points: Vec::new(),
            measure_dist: 0.0,
            analysis_x: 800.0,
        }
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
        match cmd {
            UndoCommand::Add(obj) => {
                if is_undo {
                    self.objects.retain(|o| o.id != obj.id);
                } else {
                    self.objects.push(obj);
                }
            }
            UndoCommand::Delete(obj) => {
                if is_undo {
                    self.objects.push(obj);
                } else {
                    self.objects.retain(|o| o.id != obj.id);
                }
            }
            UndoCommand::Edit { old, new } => {
                let target = if is_undo { &old } else { &new };
                if let Some(obj) = self.objects.iter_mut().find(|o| o.id == target.id) {
                    *obj = target.clone();
                }
            }
            UndoCommand::MultiAction(cmds) => {
                if is_undo {
                    for c in cmds.iter().rev() {
                        self.apply_undo_cmd(c.clone(), is_undo);
                    }
                } else {
                    for c in cmds {
                        self.apply_undo_cmd(c.clone(), is_undo);
                    }
                }
            }
        }
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
                            self.custom_meshes.remove(&obj.id);
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
                        self.is_drag_active = *state == ElementState::Pressed;
                        // If clicking background (not on UI), close panels
                        if *state == ElementState::Pressed {
                            let is_over_ui = self.egui_state.egui_ctx().is_pointer_over_area()
                                || self.egui_state.egui_ctx().wants_pointer_input();

                            // Log for debugging if picking fails again
                            // println!("Click at physical: {:?}, is_over_ui: {}", self.mouse_pos, is_over_ui);

                            if !is_over_ui {
                                // Deselect panels
                                self.show_add_panel = false;
                                self.show_settings = false;

                                // Selection Picking
                                let x = (2.0 * self.mouse_pos[0]) / self.size.width as f32 - 1.0;
                                let y = 1.0 - (2.0 * self.mouse_pos[1]) / self.size.height as f32;

                                let now = std::time::Instant::now();
                                let is_double_click =
                                    now.duration_since(self.last_click_time).as_millis() < 300;
                                self.last_click_time = now;

                                let ctx = self.egui_state.egui_ctx().clone();
                                if self.measure_mode && !ctx.is_pointer_over_area() {
                                    if let Some(pt) =
                                        pick_point(&self.camera, &self.objects, [x, y])
                                    {
                                        self.measure_points.push(pt);
                                        if self.measure_points.len() > 2 {
                                            self.measure_points.remove(0);
                                        }
                                        if self.measure_points.len() == 2 {
                                            let p1 = self.measure_points[0];
                                            let p2 = self.measure_points[1];
                                            self.measure_dist = ((p1[0] - p2[0]).powi(2)
                                                + (p1[1] - p2[1]).powi(2)
                                                + (p1[2] - p2[2]).powi(2))
                                            .sqrt();
                                        }
                                    }
                                } else if let Some(hit_id) = self.select_object_at_ndc(x, y) {
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
                    }
                    MouseButton::Middle => self.is_pan_active = *state == ElementState::Pressed,
                    _ => {}
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    MouseScrollDelta::LineDelta(_, y) => self.camera_controller.zoom(*y),
                    MouseScrollDelta::PixelDelta(pos) => self.camera_controller.zoom(pos.y as f32),
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current_pos = [position.x as f32, position.y as f32];
                let dx = current_pos[0] - self.mouse_pos[0];
                let dy = current_pos[1] - self.mouse_pos[1];

                if self.is_drag_active {
                    self.camera_controller.rotate(dx, dy);
                }
                if self.is_pan_active {
                    self.camera_controller.pan(dx, dy, &self.camera);
                }
                self.mouse_pos = current_pos;
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

    fn select_object_at_ndc(&mut self, x: f32, y: f32) -> Option<usize> {
        let modifiers = self.egui_state.egui_ctx().input(|i| i.modifiers);
        let multi_select = modifiers.ctrl || modifiers.shift || modifiers.mac_cmd;

        let inv_vp = self
            .camera
            .build_view_projection_matrix()
            .invert()
            .unwrap_or(cgmath::Matrix4::identity());

        let near_point = inv_vp * cgmath::Vector4::new(x, y, 0.0, 1.0);
        let far_point = inv_vp * cgmath::Vector4::new(x, y, 1.0, 1.0);

        let near_world = near_point.truncate() / near_point.w;
        let far_world = far_point.truncate() / far_point.w;

        let ray_origin = near_world;
        let ray_dir = (far_world - near_world).normalize();

        let mut closest_hit: Option<(usize, f32)> = None;

        for obj in &self.objects {
            if !obj.visible {
                continue;
            }

            let model = obj.instance.to_model_matrix();
            let inv_model = model.invert().unwrap_or(cgmath::Matrix4::identity());

            let local_origin = (inv_model * ray_origin.extend(1.0)).truncate();
            let local_dir = (inv_model * ray_dir.extend(0.0)).truncate().normalize();

            if let Some(t) = self.intersect_primitive(&obj.geometry_type, local_origin, local_dir) {
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

    fn intersect_primitive(
        &self,
        geo_type: &GeometryType,
        local_origin: cgmath::Vector3<f32>,
        local_dir: cgmath::Vector3<f32>,
    ) -> Option<f32> {
        match geo_type {
            GeometryType::Cube => {
                let mut tmin = -f32::INFINITY;
                let mut tmax = f32::INFINITY;
                for i in 0..3 {
                    if local_dir[i].abs() < 1e-6 {
                        if local_origin[i] < -0.5 || local_origin[i] > 0.5 {
                            return None;
                        }
                    } else {
                        let inv_d = 1.0 / local_dir[i];
                        let mut t1 = (-0.5 - local_origin[i]) * inv_d;
                        let mut t2 = (0.5 - local_origin[i]) * inv_d;
                        if t1 > t2 {
                            std::mem::swap(&mut t1, &mut t2);
                        }
                        tmin = tmin.max(t1);
                        tmax = tmax.min(t2);
                    }
                }
                if tmax >= tmin && tmax >= 0.0 {
                    Some(tmin.max(0.0))
                } else {
                    None
                }
            }
            GeometryType::Sphere => {
                let oc = local_origin;
                let a = local_dir.dot(local_dir);
                let b = 2.0 * oc.dot(local_dir);
                let c = oc.dot(oc) - 0.25; // radius 0.5 matches visuals
                let discriminant = b * b - 4.0 * a * c;
                if discriminant < 0.0 {
                    None
                } else {
                    let mut t = (-b - discriminant.sqrt()) / (2.0 * a);
                    if t < 0.0 {
                        t = (-b + discriminant.sqrt()) / (2.0 * a);
                    }
                    if t >= 0.0 {
                        Some(t)
                    } else {
                        None
                    }
                }
            }
            GeometryType::Plane => {
                if local_dir.y.abs() < 1e-6 {
                    return None;
                }
                let t = -local_origin.y / local_dir.y;
                if t < 0.0 {
                    return None;
                }
                let p = local_origin + local_dir * t;
                if p.x.abs() <= 0.5 && p.z.abs() <= 0.5 {
                    Some(t)
                } else {
                    None
                }
            }
            GeometryType::Mesh { data } => {
                let ray = parry3d_f64::query::Ray::new(
                    parry3d_f64::na::Point3::new(
                        local_origin.x as f64,
                        local_origin.y as f64,
                        local_origin.z as f64,
                    ),
                    parry3d_f64::na::Vector3::new(
                        local_dir.x as f64,
                        local_dir.y as f64,
                        local_dir.z as f64,
                    ),
                );

                let mut min_t = f64::MAX;
                let mut found = false;

                for chunk in data.indices.chunks_exact(3) {
                    let v0 = data.vertices[chunk[0] as usize].pos;
                    let v1 = data.vertices[chunk[1] as usize].pos;
                    let v2 = data.vertices[chunk[2] as usize].pos;
                    let tri = parry3d_f64::shape::Triangle::new(
                        parry3d_f64::na::Point3::new(v0[0] as f64, v0[1] as f64, v0[2] as f64),
                        parry3d_f64::na::Point3::new(v1[0] as f64, v1[1] as f64, v1[2] as f64),
                        parry3d_f64::na::Point3::new(v2[0] as f64, v2[1] as f64, v2[2] as f64),
                    );
                    use parry3d_f64::query::RayCast;
                    if let Some(t) = tri.cast_local_ray(&ray, f64::MAX, true) {
                        if t < min_t {
                            min_t = t;
                            found = true;
                        }
                    }
                }

                if found {
                    Some(min_t as f32)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);

        // Uniforms
        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&self.camera);
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.render_with_callback(|_| {})
    }

    pub fn add_default_object(&mut self) {
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

    pub fn add_mesh_object(&mut self, mesh_data: crate::mesh::MeshData, label: String) {
        let id = self.next_id;
        self.next_id += 1;

        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Custom Mesh Vertex Buffer {}", id)),
                contents: bytemuck::cast_slice(&mesh_data.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Custom Mesh Index Buffer {}", id)),
                contents: bytemuck::cast_slice(mesh_data.indices.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            });

        self.custom_meshes.insert(
            id,
            MeshBuffers {
                vertex_buffer,
                index_buffer,
                num_indices: mesh_data.indices.len() as u32,
            },
        );

        let mut obj = SceneObject::new(
            id,
            label,
            [0.0, 0.0, 0.0],
            GeometryType::Mesh {
                data: std::sync::Arc::new(mesh_data),
            },
        );
        obj.selected = true; // Select newly added
        self.objects.push(obj);
    }

    pub fn recenter_mesh_pivot(&mut self, obj_id: usize) {
        if let Some(obj) = self.objects.iter_mut().find(|o| o.id == obj_id) {
            if let GeometryType::Mesh { data } = &obj.geometry_type {
                if data.vertices.is_empty() {
                    return;
                }

                let mut sum = cgmath::Vector3::new(0.0, 0.0, 0.0);
                for v in &data.vertices {
                    sum += cgmath::Vector3::from(v.pos);
                }
                let centroid = sum / data.vertices.len() as f32;

                let mut new_vertices = data.vertices.clone();
                for v in &mut new_vertices {
                    v.pos[0] -= centroid.x;
                    v.pos[1] -= centroid.y;
                    v.pos[2] -= centroid.z;
                }

                let new_data = crate::mesh::MeshData {
                    vertices: new_vertices,
                    indices: data.indices.clone(),
                };

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!(
                                "Custom Mesh Vertex Buffer (Recentered) {}",
                                obj_id
                            )),
                            contents: bytemuck::cast_slice(&new_data.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                if let Some(mesh_buf) = self.custom_meshes.get_mut(&obj_id) {
                    mesh_buf.vertex_buffer = vertex_buffer;
                }

                obj.geometry_type = GeometryType::Mesh {
                    data: std::sync::Arc::new(new_data),
                };

                // Adjust translation to keep object in same world position
                let scale = obj.instance.scale;
                let scaled_centroid = cgmath::Vector3::new(
                    centroid.x * scale.x,
                    centroid.y * scale.y,
                    centroid.z * scale.z,
                );
                let offset = obj.instance.rotation * scaled_centroid;
                obj.instance.position += offset;
            }
        }
    }

    pub fn render_with_callback<F>(&mut self, custom_ui: F) -> Result<(), wgpu::SurfaceError>
    where
        F: FnOnce(&egui::Context),
    {
        let mut action_add_primitive = false;
        let mut action_recenter_id = None;

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
        let mut action_add_mesh = None;
        let mut action_load_folder = None;

        let full_output = self.egui_state.egui_ctx().run(raw_input, |ctx| {
            // Consolidation of App UI features inside State
            let screen_size = [self.size.width as f32, self.size.height as f32];
            let any_mesh = self
                .objects
                .iter()
                .any(|o| matches!(o.geometry_type, GeometryType::Mesh { .. }));

            // 1. Toolbar area
            egui::Area::new("toolbar_area_state".into())
                .anchor(egui::Align2::LEFT_TOP, [10.0, 10.0])
                .show(ctx, |ui| {
                    egui::Frame::window(&ctx.style())
                        .rounding(10.0)
                        .inner_margin(6.0)
                        .fill(ctx.style().visuals.window_fill())
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui
                                    .add(
                                        egui::Button::new(egui::RichText::new("➕").size(18.0))
                                            .frame(false),
                                    )
                                    .on_hover_text("Add new primitive object")
                                    .clicked()
                                {
                                    action_add_primitive = true;
                                }

                                ui.add(egui::Separator::default().vertical());

                                if ui
                                    .add(
                                        egui::Button::new(egui::RichText::new("📄").size(18.0))
                                            .frame(false),
                                    )
                                    .on_hover_text("Import single 3D model")
                                    .clicked()
                                {
                                    if let Some(path) = rfd::FileDialog::new()
                                        .add_filter("3D Models", &["stl", "obj", "gltf", "glb"])
                                        .pick_file()
                                    {
                                        let label = path
                                            .file_name()
                                            .unwrap_or_default()
                                            .to_string_lossy()
                                            .into_owned();
                                        if let Ok(m) = crate::mesh::MeshData::load(path) {
                                            action_add_mesh = Some((m, label));
                                        }
                                    }
                                }

                                if ui
                                    .add(
                                        egui::Button::new(egui::RichText::new("📂").size(18.0))
                                            .frame(false),
                                    )
                                    .on_hover_text("Load entire folder of 3D models")
                                    .clicked()
                                {
                                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                        action_load_folder = Some(path);
                                    }
                                }

                                ui.add(egui::Separator::default().vertical());

                                let m_color = if self.measure_mode {
                                    egui::Color32::LIGHT_BLUE
                                } else {
                                    ui.visuals().widgets.noninteractive.text_color()
                                };
                                if ui
                                    .add(
                                        egui::Button::new(
                                            egui::RichText::new("📏").size(18.0).color(m_color),
                                        )
                                        .frame(false),
                                    )
                                    .on_hover_text("Measure tool: click points on meshes")
                                    .clicked()
                                {
                                    self.measure_mode = !self.measure_mode;
                                }
                            });
                        });
                });

            // 2. Analysis / Transform Window
            let sel_indices: Vec<usize> = self
                .objects
                .iter()
                .enumerate()
                .filter(|(_, o)| o.selected)
                .map(|(i, _)| i)
                .collect();

            if self.measure_mode || !sel_indices.is_empty() || any_mesh {
                let mut ax = self.analysis_x;
                egui::Window::new("⚙ Analysis")
                    .id(egui::Id::new("analysis_win"))
                    .fixed_pos([ax, 80.0])
                    .collapsible(true)
                    .resizable(false)
                    .show(ctx, |ui| {
                        let d_resp =
                            ui.interact(ui.max_rect(), ui.id().with("drag"), egui::Sense::drag());
                        if d_resp.dragged() {
                            ax += d_resp.drag_delta().x;
                        }

                        if self.measure_mode {
                            ui.heading("📏 Measurement");
                            ui.label(format!("Distance: {:.3}m", self.measure_dist));
                            if ui.button("Clear Points").clicked() {
                                self.measure_points.clear();
                                self.measure_dist = 0.0;
                            }
                            ui.separator();
                        }

                        if !sel_indices.is_empty() {
                            ui.heading("🎯 Selection");
                            if sel_indices.len() == 1 {
                                let idx = sel_indices[0];
                                let obj = &mut self.objects[idx];
                                ui.label(format!("Editing: {}", obj.label));

                                ui.add_space(4.0);
                                ui.label("Position:");
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::DragValue::new(&mut obj.instance.position.x)
                                            .speed(0.1)
                                            .prefix("X:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut obj.instance.position.y)
                                            .speed(0.1)
                                            .prefix("Y:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut obj.instance.position.z)
                                            .speed(0.1)
                                            .prefix("Z:"),
                                    );
                                });

                                ui.add_space(4.0);
                                ui.label("Rotation:");
                                ui.horizontal(|ui| {
                                    let mut changed = false;
                                    changed |= ui
                                        .add(
                                            egui::DragValue::new(&mut obj.rotation_euler[0])
                                                .speed(1.0)
                                                .prefix("X:")
                                                .suffix("°"),
                                        )
                                        .changed();
                                    changed |= ui
                                        .add(
                                            egui::DragValue::new(&mut obj.rotation_euler[1])
                                                .speed(1.0)
                                                .prefix("Y:")
                                                .suffix("°"),
                                        )
                                        .changed();
                                    changed |= ui
                                        .add(
                                            egui::DragValue::new(&mut obj.rotation_euler[2])
                                                .speed(1.0)
                                                .prefix("Z:")
                                                .suffix("°"),
                                        )
                                        .changed();
                                    if changed {
                                        obj.update_rotation();
                                    }
                                });

                                ui.add_space(4.0);
                                ui.label("Scale:");
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::DragValue::new(&mut obj.instance.scale.x)
                                            .speed(0.01)
                                            .prefix("X:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut obj.instance.scale.y)
                                            .speed(0.01)
                                            .prefix("Y:"),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut obj.instance.scale.z)
                                            .speed(0.01)
                                            .prefix("Z:"),
                                    );
                                });

                                if let GeometryType::Mesh { .. } = obj.geometry_type {
                                    ui.add_space(8.0);
                                    if ui
                                        .button("🎯 Recenter Pivot")
                                        .on_hover_text(
                                            "Align the object's origin to its geometric center",
                                        )
                                        .clicked()
                                    {
                                        action_recenter_id = Some(obj.id);
                                    }
                                }
                            } else {
                                ui.label(format!("{} objects selected", sel_indices.len()));
                            }
                        } else if any_mesh {
                            ui.label("Select a model to edit its transform.");
                        }
                    });
                self.analysis_x = ax;
            }

            // 3. Measurement overlay
            if self.measure_mode && self.measure_points.len() == 2 {
                egui::Area::new("measure_overlay_state".into())
                    .interactable(false)
                    .anchor(egui::Align2::LEFT_TOP, [0.0, 0.0])
                    .show(ctx, |ui| {
                        draw_measurement(
                            ui,
                            &self.camera,
                            self.measure_points[0],
                            self.measure_points[1],
                            screen_size,
                        );
                    });
            }

            // Top Right Panel: Settings
            egui::Window::new("Settings")
                .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
                .collapsible(true)
                .open(&mut self.show_settings)
                .show(ctx, |ui| {
                    ui.heading("Global Settings");
                    ui.add(egui::Slider::new(&mut self.bg_color, 0.0..=1.0).text("Background"));

                    ui.separator();
                    ui.heading("Camera & View");
                    if ui.button("Focus Selected").clicked() {
                        action_focus_selected = true;
                    }
                    if ui.button("Reset View (0,0,0)").clicked() {
                        action_reset_view = true;
                    }

                    ui.separator();
                    ui.label("Zoom Limits:");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.camera_controller.min_zoom)
                                .speed(0.1)
                                .prefix("Min: "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.camera_controller.max_zoom)
                                .speed(1.0)
                                .prefix("Max: "),
                        );
                    });

                    ui.separator();
                    ui.heading("Grid Options");
                    ui.checkbox(&mut self.show_grid_xy, "XY Plane");
                    ui.checkbox(&mut self.show_grid_xz, "XZ Plane");
                    ui.checkbox(&mut self.show_grid_yz, "YZ Plane");
                    ui.checkbox(&mut self.show_axes, "Show Axes");
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
                            "Camera Eye: {:.2}, {:.2}, {:.2}",
                            self.camera.eye.x, self.camera.eye.y, self.camera.eye.z
                        ));
                        ui.separator();
                        ui.label(format!(
                            "Target: {:.2}, {:.2}, {:.2}",
                            self.camera_controller.target.x,
                            self.camera_controller.target.y,
                            self.camera_controller.target.z
                        ));
                        ui.separator();
                        ui.label(format!(
                            "Yaw: {:.1}° Pitch: {:.1}°",
                            self.camera_controller.yaw, self.camera_controller.pitch
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
                                                        // Edit icon
                                                        let is_editing = action_edit_obj_id
                                                            == Some(obj.id)
                                                            || self.editing_obj_id == Some(obj.id);
                                                        let edit_color = if is_editing {
                                                            egui::Color32::from_rgb(0, 150, 255)
                                                        } else {
                                                            ui.visuals().text_color()
                                                        };
                                                        if ui
                                                            .add(
                                                                egui::Button::new(
                                                                    egui::RichText::new("✏")
                                                                        .color(edit_color),
                                                                )
                                                                .fill(egui::Color32::TRANSPARENT),
                                                            )
                                                            .on_hover_text("Edit")
                                                            .clicked()
                                                        {
                                                            action_edit_obj_id = Some(obj.id);
                                                            self.editing_obj_draft =
                                                                Some(obj.clone());
                                                            self.should_focus_name = true;
                                                        }

                                                        // Delete icon
                                                        if ui
                                                            .add(
                                                                egui::Button::new(
                                                                    egui::RichText::new("🗑").color(
                                                                        egui::Color32::from_rgb(
                                                                            255, 100, 100,
                                                                        ),
                                                                    ),
                                                                )
                                                                .fill(egui::Color32::TRANSPARENT),
                                                            )
                                                            .on_hover_text("Delete")
                                                            .clicked()
                                                        {
                                                            action_delete_obj_id = Some(obj.id);
                                                        }

                                                        // Label toggle icon
                                                        let label_icon = "🏷";
                                                        let label_color = if obj.show_label {
                                                            egui::Color32::from_rgb(0, 200, 255)
                                                        } else {
                                                            egui::Color32::from_rgb(150, 150, 150)
                                                        };
                                                        if ui
                                                            .add(
                                                                egui::Button::new(
                                                                    egui::RichText::new(label_icon)
                                                                        .color(label_color),
                                                                )
                                                                .fill(egui::Color32::TRANSPARENT),
                                                            )
                                                            .on_hover_text("Toggle Label")
                                                            .clicked()
                                                        {
                                                            obj.show_label = !obj.show_label;
                                                        }

                                                        // Visibility icon (Colori opposti)
                                                        let vis_icon = if obj.visible {
                                                            "👁"
                                                        } else {
                                                            "🕶"
                                                        };
                                                        let vis_color = if obj.visible {
                                                            egui::Color32::from_rgb(100, 255, 100)
                                                        } else {
                                                            egui::Color32::from_rgb(255, 100, 100)
                                                        };
                                                        if ui
                                                            .add(
                                                                egui::Button::new(
                                                                    egui::RichText::new(vis_icon)
                                                                        .color(vis_color),
                                                                )
                                                                .fill(egui::Color32::TRANSPARENT),
                                                            )
                                                            .on_hover_text("Visibility")
                                                            .clicked()
                                                        {
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
                egui::Window::new("Add New Object")
                    .open(&mut open)
                    .resizable(true)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            let res = ui.text_edit_singleline(&mut draft.label);
                            if self.should_focus_name {
                                res.request_focus();
                                self.should_focus_name = false;
                            }
                        });
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Type:");
                            egui::ComboBox::from_id_source("draft_type_combo")
                                .selected_text(format!("{:?}", draft.geometry_type))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut draft.geometry_type,
                                        GeometryType::Cube,
                                        "Cube",
                                    );
                                    ui.selectable_value(
                                        &mut draft.geometry_type,
                                        GeometryType::Sphere,
                                        "Sphere",
                                    );
                                    ui.selectable_value(
                                        &mut draft.geometry_type,
                                        GeometryType::Plane,
                                        "Plane",
                                    );
                                });
                        });

                        ui.heading("Transform");
                        ui.horizontal(|ui| {
                            ui.label("Pos:");
                            ui.add(egui::DragValue::new(&mut draft.instance.position.x).speed(0.1));
                            ui.add(egui::DragValue::new(&mut draft.instance.position.y).speed(0.1));
                            ui.add(egui::DragValue::new(&mut draft.instance.position.z).speed(0.1));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Rot:");
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
                            ui.label("Scale:");
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
                            ui.label("Color:");
                            ui.color_edit_button_rgb(&mut draft.color);
                        });
                        ui.checkbox(&mut draft.show_label, "Show Label in Viewport");

                        ui.separator();
                        ui.heading("Geometry Properties");
                        match draft.geometry_type {
                            GeometryType::Cube => {
                                ui.horizontal(|ui| {
                                    ui.label("Side Length:");
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
                                    ui.label("Radius:");
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
                                    ui.label("Surface Area:");
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
                                ui.checkbox(&mut draft.show_normal, "Show Normal Arrow");
                            }
                            _ => {}
                        }

                        ui.separator();
                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui
                                .add_sized([80.0, 24.0], egui::Button::new("Cancel"))
                                .clicked()
                                || ui.input(|i| i.key_pressed(egui::Key::Escape))
                            {
                                action_discard = true;
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add_sized([80.0, 24.0], egui::Button::new("Add"))
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

            custom_ui(ctx);
        });

        // Apply deferred actions
        if let Some(obj) = create_object {
            self.objects.push(obj);
            self.next_id += 1;
        }

        if action_add_primitive {
            self.add_default_object();
        }
        if let Some(id) = action_recenter_id {
            self.recenter_mesh_pivot(id);
        }

        if let Some((m, label)) = action_add_mesh {
            self.add_mesh_object(m, label);
        }
        if let Some(path) = action_load_folder {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    let label = p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned();
                    if let Ok(m) = crate::mesh::MeshData::load(p) {
                        self.add_mesh_object(m, label);
                    }
                }
            }
        }

        if action_reset_view {
            self.camera_controller.target = cgmath::Point3::new(
                DEFAULT_CAMERA_TARGET[0],
                DEFAULT_CAMERA_TARGET[1],
                DEFAULT_CAMERA_TARGET[2],
            );
            self.camera_controller.yaw = DEFAULT_CAMERA_YAW;
            self.camera_controller.pitch = DEFAULT_CAMERA_PITCH;
            self.camera_controller.dist = DEFAULT_CAMERA_DIST;
        }

        if action_focus_selected {
            if let Some(target) = self.get_selected_centroid() {
                self.camera_controller.target = target;
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
                self.objects.remove(index);
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

        // --- Prepare Custom Meshes Draw List ---
        let mut custom_draw_list: Vec<(usize, wgpu::Buffer, u32)> = Vec::new(); // (obj_id, instance_buffer, instances_len)
        for obj in &self.objects {
            if let GeometryType::Mesh { .. } = &obj.geometry_type {
                if obj.visible {
                    let instance_raw = [obj.instance.to_raw_with_color(obj.color, obj.selected)];
                    let instance_buf =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Custom Mesh Instance Buffer"),
                                contents: bytemuck::cast_slice(&instance_raw),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    custom_draw_list.push((obj.id, instance_buf, 1));
                }
            }
        }

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

            // Draw Custom Meshes
            for (obj_id, instance_buf, _) in &custom_draw_list {
                if let Some(mesh_buf) = self.custom_meshes.get(obj_id) {
                    render_pass.set_vertex_buffer(0, mesh_buf.vertex_buffer.slice(..));
                    render_pass.set_vertex_buffer(1, instance_buf.slice(..));
                    render_pass.set_index_buffer(
                        mesh_buf.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..mesh_buf.num_indices, 0, 0..1);
                }
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
