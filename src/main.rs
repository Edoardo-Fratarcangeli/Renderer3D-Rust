use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};

use cgmath::{Angle, InnerSpace};
use rendering_3d::log_info;
use rendering_3d::logger;
use rendering_3d::state::State;

fn load_icon() -> winit::window::Icon {
    let svg_data = include_bytes!("../assets/icon.svg");
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_data(svg_data, &opt).expect("Failed to parse SVG");

    // Windows taskbar works best with standard sizes like 32, 48, 64.
    // We render at 64x64 for high density.
    let width = 64;
    let height = 64;

    let mut pixmap = tiny_skia::Pixmap::new(width, height).expect("Failed to create pixmap");

    // Calculate scale to fit the 200x200 SVG into 64x64
    let sx = width as f32 / tree.size().width();
    let sy = height as f32 / tree.size().height();

    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(sx, sy),
        &mut pixmap.as_mut(),
    );

    let mut rgba_data = pixmap.take();
    // Tiny-skia uses premultiplied alpha, winit expects non-premultiplied alpha.
    // Convert it back:
    for pixel in rgba_data.chunks_exact_mut(4) {
        let a = pixel[3] as f32 / 255.0;
        if a > 0.0 {
            pixel[0] = (pixel[0] as f32 / a).min(255.0) as u8;
            pixel[1] = (pixel[1] as f32 / a).min(255.0) as u8;
            pixel[2] = (pixel[2] as f32 / a).min(255.0) as u8;
        }
    }

    winit::window::Icon::from_rgba(rgba_data, width, height).expect("Failed to create winit icon")
}

pub async fn run() {
    logger::init(logger::LogLevel::Info);
    log_info!("Application starting...");

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Rust 3D Renderer")
        .with_window_icon(Some(load_icon()))
        .with_inner_size(winit::dpi::LogicalSize::new(1200.0, 800.0))
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(window).await;

    _ = event_loop.run(move |event, elwt| {
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if state.is_drag_active {
                    let sensitivity = 0.5;
                    state.camera_yaw -= (delta.0 as f32) * sensitivity;
                    state.camera_pitch += (delta.1 as f32) * sensitivity;

                    // Clamp pitch to prevent singularity at poles (90 degrees)
                    state.camera_pitch = state.camera_pitch.clamp(-89.0, 89.0);

                    state.window.request_redraw();
                } else if state.is_pan_active {
                    let sensitivity = state.camera_dist * 0.001;

                    // Calculate right vector
                    // Z-Up Spherical Coordinates
                    // x = cos(pitch) * cos(yaw)
                    // y = cos(pitch) * sin(yaw)
                    // z = sin(pitch)
                    let yaw = cgmath::Deg(state.camera_yaw);
                    let pitch = cgmath::Deg(state.camera_pitch);

                    let forward = cgmath::Vector3::new(
                        pitch.cos() * yaw.cos(),
                        pitch.cos() * yaw.sin(),
                        pitch.sin(),
                    )
                    .normalize();

                    let up = cgmath::Vector3::unit_z();
                    let right = up.cross(forward).normalize();
                    let cam_up = forward.cross(right).normalize(); // Screen up

                    // Move target opposite to mouse motion
                    state.camera_target += right * (-delta.0 as f32 * sensitivity);
                    state.camera_target += cam_up * (delta.1 as f32 * sensitivity);
                    state.window.request_redraw();
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CursorMoved { position, .. } => {
                            state.mouse_pos = [position.x as f32, position.y as f32];
                        }
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { .. } => {
                            // Handled by egui or ignored for now
                        }
                        WindowEvent::RedrawRequested => {
                            state.update();
                            match state.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                state.window.request_redraw();
            }
            _ => {}
        }
    });
}

fn main() {
    pollster::block_on(run());
}
