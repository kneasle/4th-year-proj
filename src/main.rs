use image_ed::{
    instance::Instance,
    tree::{Effect, Layer, Tree},
};

fn main() {
    env_logger::init();

    // Spawn a new Instance
    let mut instance = Instance::new();
    let invert_id = instance.load_wgsl_effect(
        "Invert".to_owned(),
        include_str!("../shader/invert.wgsl").to_owned(),
    );

    // Load an image
    let image = Tree {
        effects: vec![Effect { id: invert_id }],
        layer: Layer::from_file("img/mc-skin.png").unwrap(),
    };

    // Create output textures for the image
    let rgb_image = instance.render_to_image(&image);
    rgb_image.save("out.png").unwrap();
}
