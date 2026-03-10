use image::{ImageBuffer, RgbaImage};

#[test]
fn test_generate_ico_from_svg() {
    let svg_data = include_bytes!("../../assets/icon.svg");
    let tree = usvg::Tree::from_data(svg_data, &usvg::Options::default()).unwrap();
    let size = tree.size().to_int_size();
    let mut pixmap = tiny_skia::Pixmap::new(size.width(), size.height()).unwrap();
    resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());
    let data = pixmap.take();

    // Save to ICO via image crate
    let img: RgbaImage = ImageBuffer::from_raw(size.width(), size.height(), data).unwrap();
    img.save_with_format("assets/icon.ico", image::ImageFormat::Ico)
        .unwrap();
    println!("Saved assets/icon.ico!");
}
