use wgpu::util::DeviceExt;
use winit::{
    event::*,
    window::Window,
};
use cgmath::prelude::*;

use crate::camera::{Camera, Uniforms};
use crate::model::{Vertex, InstanceRaw};
use crate::scene::{SceneObject, GeometryType};
use crate::primitives;

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
pub const DEFAULT_BOTTOM_PANEL_EXPANDED: bool = true;

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

    // Mouse
    pub is_drag_active: bool,
    pub is_pan_active: bool,
    pub camera_target: cgmath::Point3<f32>,
    
    // Editor State
    pub editing_obj_id: Option<usize>,
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

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
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

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        let egui_state = egui_winit::State::new(
            egui_context, 
            egui::ViewportId::ROOT, 
            &window,
            Some(window.scale_factor() as f32), 
            None
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, Some(wgpu::TextureFormat::Depth32Float), 1);

        // Initial Objects
        let objects = Vec::new();

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
            objects,
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
            camera_target: cgmath::Point3::new(DEFAULT_CAMERA_TARGET[0], DEFAULT_CAMERA_TARGET[1], DEFAULT_CAMERA_TARGET[2]),
            editing_obj_id: None,
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

        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                match button {
                     MouseButton::Left => {
                         self.is_drag_active = *state == ElementState::Pressed;
                         // If clicking background (not on UI), close panels
                         if *state == ElementState::Pressed {
                             if !self.egui_state.egui_ctx().is_pointer_over_area() {
                                 // Deselect panels
                                 self.show_add_panel = false;
                                 self.show_settings = false;
                             }
                         }
                     },
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
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // --- EGUI UI DRAWING ---
        let raw_input = self.egui_state.take_egui_input(&self.window);
        
        // Deferred actions
        let mut action_reset_view = false;
        let mut action_focus_selected = false;
        let mut next_id_increment = false;
        let mut create_object = None;
        
        let mut action_edit_obj_id = None;
        let mut action_close_editor = false;
        let mut action_delete_edited = false;

        let full_output = self.egui_state.egui_ctx().run(raw_input, |ctx| {
            // Because we can't easily borrow fields disjointly multiple times in complex closure,
            // We'll use the 'self' accessible in the closure but be careful.
            
            // Top Left Panel: Add Objects
            egui::Window::new("Add Panel")
                .anchor(egui::Align2::LEFT_TOP, [10.0, 10.0])
                .collapsible(true)
                .open(&mut self.show_add_panel)
                .show(ctx, |ui| {
                    ui.label("Create New Object");
                    
                    egui::ComboBox::from_label("Type")
                        .selected_text(format!("{:?}", self.new_obj_type))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.new_obj_type, GeometryType::Cube, "Cube");
                            ui.selectable_value(&mut self.new_obj_type, GeometryType::Sphere, "Sphere");
                            ui.selectable_value(&mut self.new_obj_type, GeometryType::Plane, "Plane");
                        });

                    ui.horizontal(|ui| {
                        ui.label("Pos:");
                        ui.add(egui::DragValue::new(&mut self.new_obj_pos[0]).speed(0.1));
                        ui.add(egui::DragValue::new(&mut self.new_obj_pos[1]).speed(0.1));
                        ui.add(egui::DragValue::new(&mut self.new_obj_pos[2]).speed(0.1));
                    });

                    ui.horizontal(|ui| {
                         ui.label("Color:");
                         ui.color_edit_button_rgb(&mut self.new_obj_color);
                    });
                    
                    if ui.button("Create").clicked() {
                        create_object = Some((self.new_obj_type, self.new_obj_pos, self.new_obj_color));
                        next_id_increment = true;
                    }
                });
            
            // "Add" Toggle Button
            if !self.show_add_panel {
                 egui::Window::new("Add Button")
                    .anchor(egui::Align2::LEFT_TOP, [10.0, 10.0])
                    .title_bar(false)
                    .resizable(false)
                    .show(ctx, |ui| {
                        if ui.button("➕ Add Object").clicked() {
                            self.show_add_panel = true;
                        }
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
                         ui.add(egui::DragValue::new(&mut self.min_zoom).speed(0.1).prefix("Min: "));
                         ui.add(egui::DragValue::new(&mut self.max_zoom).speed(1.0).prefix("Max: "));
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
                 egui::Window::new("Settings Button")
                    .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
                    .title_bar(false)
                    .resizable(false)
                    .show(ctx, |ui| {
                        if ui.button("⚙").clicked() {
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
                         ui.label(format!("Camera Eye: {:.2}, {:.2}, {:.2}", self.camera.eye.x, self.camera.eye.y, self.camera.eye.z));
                         ui.separator();
                         ui.label(format!("Target: {:.2}, {:.2}, {:.2}", self.camera_target.x, self.camera_target.y, self.camera_target.z));
                         ui.separator();
                         ui.label(format!("Yaw: {:.1}° Pitch: {:.1}°", self.camera_yaw, self.camera_pitch));
                    });
                });

            // Bottom Panel: Object List
            egui::TopBottomPanel::bottom("bottom_panel")
                .resizable(true)
                .min_height(30.0)
                .show_animated(ctx, self.bottom_panel_expanded, |ui| {
                     ui.horizontal(|ui| {
                         ui.heading("Scene Objects");
                         ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                             if ui.button(if self.bottom_panel_expanded { "▼" } else { "▲" }).clicked() {
                                 // Expand logic could be here
                             }
                         });
                     });
                     ui.separator();
                     
                     egui::ScrollArea::vertical().show(ui, |ui| {
                         for obj in self.objects.iter_mut() {
                             ui.horizontal(|ui| {
                                 ui.checkbox(&mut obj.selected, "");
                                 ui.label(&obj.label);
                                 ui.checkbox(&mut obj.visible, "Vis");
                                 if ui.button("Edit").clicked() {
                                     action_edit_obj_id = Some(obj.id);
                                 }
                             });
                         }
                     });
                });
            
            // Editor Popup Window
            if let Some(edit_id) = self.editing_obj_id {
                let mut open = true;
                if let Some(obj) = self.objects.iter_mut().find(|o| o.id == edit_id) {
                    egui::Window::new(format!("Edit Object {}", edit_id))
                        .open(&mut open)
                        .resizable(true)
                        .show(ctx, |ui| {
                            ui.text_edit_singleline(&mut obj.label);
                            ui.separator();
                            ui.label(format!("Type: {:?}", obj.geometry_type));
                            
                            ui.heading("Transform");
                            ui.horizontal(|ui| {
                                ui.label("Pos:");
                                ui.add(egui::DragValue::new(&mut obj.instance.position.x).speed(0.1));
                                ui.add(egui::DragValue::new(&mut obj.instance.position.y).speed(0.1));
                                ui.add(egui::DragValue::new(&mut obj.instance.position.z).speed(0.1));
                            });
                            ui.horizontal(|ui| {
                                ui.label("Rot:");
                                let mut changed = false;
                                changed |= ui.add(egui::DragValue::new(&mut obj.rotation_euler[0]).speed(1.0).suffix("°")).changed();
                                changed |= ui.add(egui::DragValue::new(&mut obj.rotation_euler[1]).speed(1.0).suffix("°")).changed();
                                changed |= ui.add(egui::DragValue::new(&mut obj.rotation_euler[2]).speed(1.0).suffix("°")).changed();
                                if changed {
                                    obj.update_rotation();
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Scale:");
                                ui.add(egui::DragValue::new(&mut obj.instance.scale.x).speed(0.01).max_decimals(2));
                                ui.add(egui::DragValue::new(&mut obj.instance.scale.y).speed(0.01).max_decimals(2));
                                ui.add(egui::DragValue::new(&mut obj.instance.scale.z).speed(0.01).max_decimals(2));
                            });
                            
                            ui.separator();
                            ui.horizontal(|ui| {
                                 ui.label("Color:");
                                 ui.color_edit_button_rgb(&mut obj.color);
                            });
                            
                            ui.separator();
                            if ui.button("🗑 Delete Object").clicked() {
                                action_delete_edited = true;
                                action_close_editor = true;
                            }
                        });
                }
                if !open {
                    action_close_editor = true;
                }
            }
        });

        // Apply deferred actions
        if let Some((g_type, pos, col)) = create_object {
            let id = self.next_id;
            if next_id_increment { self.next_id += 1; }
            let mut obj = SceneObject::new(
                 id,
                 format!("{:?} {}", g_type, id),
                 pos,
                 g_type
            );
            obj.color = col;
            self.objects.push(obj);
        }
        
        if action_reset_view {
             self.camera_target = cgmath::Point3::new(DEFAULT_CAMERA_TARGET[0], DEFAULT_CAMERA_TARGET[1], DEFAULT_CAMERA_TARGET[2]);
             self.camera_yaw = DEFAULT_CAMERA_YAW;   
             self.camera_pitch = DEFAULT_CAMERA_PITCH; 
             self.camera_dist = DEFAULT_CAMERA_DIST;
        }
        
        if action_focus_selected {
            if let Some(target) = self.get_selected_centroid() {
                self.camera_target = target;
            }
        }
        
        // Editor Actions
        if let Some(id) = action_edit_obj_id {
            self.editing_obj_id = Some(id);
        }
        
        if action_delete_edited {
             if let Some(id) = self.editing_obj_id {
                if let Some(index) = self.objects.iter().position(|o| o.id == id) {
                    self.objects.remove(index);
                }
                self.editing_obj_id = None;
             }
        } else if action_close_editor {
             self.editing_obj_id = None;
        }

        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        let tris = self.egui_state.egui_ctx().tessellate(full_output.shapes, self.egui_state.egui_ctx().pixels_per_point());
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.egui_state.egui_ctx().pixels_per_point(),
        };
        
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &tris, &screen_descriptor);

        // --- Prepare Buffers ---
        let mut draw_list: Vec<(GeometryType, wgpu::Buffer, u32)> = Vec::new();
        
        // Identity Matrix for Grids/Axes
        let identity_instance = [InstanceRaw {
            model: cgmath::Matrix4::identity().into(),
            color: [1.0; 4],
        }];
        let single_instance_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
             label: Some("Single Instance Buffer"),
             contents: bytemuck::cast_slice(&identity_instance),
             usage: wgpu::BufferUsages::VERTEX,
        });

        for geo_type in [GeometryType::Cube, GeometryType::Sphere, GeometryType::Plane] {
             let instances = self.objects.iter()
                .filter(|o| o.visible && o.geometry_type == geo_type)
                .map(|o| o.instance.to_raw_with_color(o.color))
                .collect::<Vec<_>>();
             
             if !instances.is_empty() {
                 let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer"),
                    contents: bytemuck::cast_slice(&instances),
                    usage: wgpu::BufferUsages::VERTEX,
                 });
                 draw_list.push((geo_type, buffer, instances.len() as u32));
             }
        }

        // --- Render Pass ---
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bg_color, g: self.bg_color, b: self.bg_color, a: 1.0,
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
                    render_pass.set_index_buffer(self.grid_xy_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..self.grid_xy_mesh.num_indices, 0, 0..1);
            }
            if self.show_grid_xz {
                    render_pass.set_vertex_buffer(0, self.grid_xz_mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(self.grid_xz_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..self.grid_xz_mesh.num_indices, 0, 0..1);
            }
            if self.show_grid_yz {
                    render_pass.set_vertex_buffer(0, self.grid_yz_mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(self.grid_yz_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..self.grid_yz_mesh.num_indices, 0, 0..1);
            }

            // Draw Objects (Triangles) & AXES
            render_pass.set_pipeline(&self.render_pipeline);
            
            // Draw Axes
            if self.show_axes {
                render_pass.set_vertex_buffer(0, self.axes_mesh.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, single_instance_buffer.slice(..));
                render_pass.set_index_buffer(self.axes_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
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
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..mesh.num_indices, 0, 0..*count);
            }
            
            // Render UI
            self.egui_renderer.render(&mut render_pass, &tris, &screen_descriptor);
        }
        
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
