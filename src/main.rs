use image_ed::{
    instance::{Instance, PlugIdx, Plugin},
    tree::{Effect, Layer, Tree},
};

fn main() {
    env_logger::init();

    // Spawn a new Instance
    let mut instance = Instance::new(vec![Plugin {
        name: "Invert".to_owned(),
        wgsl_source: include_str!("../shader/invert.wgsl").to_owned(),
    }]);

    dbg!(&instance);

    // Load an image
    let image = Tree {
        effects: vec![/* Effect {
            plugin_idx: PlugIdx::new(0),
        } */],
        layer: Layer::from_file("img/mc-skin.png").unwrap(),
    };

    dbg!(&image);

    // Create output textures for the image
    let rgb_image = instance.render_image(&image);
    rgb_image.save("out.png").unwrap();
}
