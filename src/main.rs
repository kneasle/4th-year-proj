use std::time::Instant;

use image_ed::{
    context::Context,
    tree::{Effect, Tree},
};

fn main() {
    // Init logging
    env_logger::init();

    // Create the singleton `Context`, and load an example image effect
    println!("Loading FX");
    let mut context = Context::with_builtin_effects();
    // Load an image, taking the path as the first arg if provided
    println!("Loading file");
    let file_path = std::env::args()
        .nth(1) // 0th arg is path to this executable
        .unwrap_or_else(|| "img/mc-skin.png".to_owned());
    let image = Tree {
        effects: vec![
            Effect {
                id: Context::ID_VALUE_INVERT.to_owned(),
            };
            2
        ],
        layer_id: context.load_layer_from_file(&file_path).unwrap(),
    };
    // Create output textures for the image
    println!("Processing image");
    let start = Instant::now();
    let rgb_image = context.render_to_image(&image);
    println!("Processed in {:?}", Instant::now() - start);
    println!("Saving image");
    rgb_image.save("out.png").unwrap();
}
