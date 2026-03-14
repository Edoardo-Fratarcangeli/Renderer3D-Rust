use rendering_3d::state::State;
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};

pub async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Rust 3D Renderer")
        .with_inner_size(winit::dpi::LogicalSize::new(1200.0, 800.0))
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(window).await;

    _ = event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(size) => state.resize(*size),
                    WindowEvent::RedrawRequested => {
                        state.update();
                        if let Err(e) = state.render_with_callback(|_| {}) {
                            eprintln!("{:?}", e);
                        }
                    }
                    _ => {}
                }
            }
        }
        Event::AboutToWait => state.window.request_redraw(),
        _ => {}
    });
}

fn main() {
    pollster::block_on(run());
}
