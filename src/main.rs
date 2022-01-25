use image_ed::{
    context::Context,
    tree::{Effect, Layer, Tree},
};

fn main() {
    env_logger::init();

    // Create the singleton `Context`, and load an example image effect
    let mut context = Context::new();
    let invert_id = context.load_wgsl_effect(
        "Invert".to_owned(),
        include_str!("../shader/invert.wgsl").to_owned(),
    );

    // Load an image
    let image = Tree {
        effects: vec![Effect { id: invert_id }],
        layer: Layer::from_file("img/mc-skin.png").unwrap(),
    };

    // Create output textures for the image
    let rgb_image = context.render_to_image(&image);
    rgb_image.save("out.png").unwrap();
}
