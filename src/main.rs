use image_ed::{
    ctx::{Ctx, PlugIdx, Plugin},
    Effect, Image, Layer,
};

fn main() {
    env_logger::init();

    let ctx = Ctx::new(vec![Plugin {
        name: "Invert".to_owned(),
    }]);

    let _ = vec![Effect {
        plugin_idx: PlugIdx::new(0),
    }];
    let image = Image {
        effects: vec![],
        layer: Layer::from_file("img/mc-skin.png").unwrap(),
    };

    dbg!(ctx, image);
}
