fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("ProductName", "Rust 3D Renderer");
        res.set("FileDescription", "High performance 3D renderer using WGPU");
        res.set("LegalCopyright", "Copyright © 2026");
        res.compile().unwrap();
    }
}
