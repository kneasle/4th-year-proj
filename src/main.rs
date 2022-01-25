use image_ed::{
    context::Context,
    tree::{Effect, Layer, Tree},
};

fn main() {
    // Init logging
    env_logger::init();

    // Create the singleton `Context`, and load an example image effect
    let mut context = Context::new();
    let invert_id = context.load_wgsl_effect(
        "Invert".to_owned(),
        include_str!("../shader/invert.wgsl").to_owned(),
    );

    // Load an image, taking the path as the first arg if provided
    let file_path = std::env::args()
        .skip(1) // First arg is path to this executable
        .next()
        .unwrap_or_else(|| "img/mc-skin.png".to_owned());
    let image = Tree {
        effects: vec![Effect { id: invert_id }],
        layer: Layer::from_file(&file_path).unwrap(),
    };

    // Create output textures for the image
    let rgb_image = context.render_to_image(&image);
    rgb_image.save("out.png").unwrap();
}
