use cgmath::Vector2;
use image_ed::{
    context::Context,
    effects::{self, built_ins},
    image::{EffectInstance, Image, Layer},
};

fn main() {
    let mut ctx = Context::new();
    let value_inv_name = ctx.load_effect(built_ins::value_invert());
    let transform_name = ctx.load_effect(effects::Transform::new(100.0, 100.0));
    let image = Image {
        size: Vector2::new(512, 512),
        layers: vec![Layer {
            effects: vec![
                EffectInstance {
                    effect_name: value_inv_name,
                },
                EffectInstance {
                    effect_name: transform_name,
                },
            ],
            source: ctx.load_layer_from_file("img/logo.png").unwrap(),
        }],
    };
    ctx.render_to_image(&image).save("out.png").unwrap();
}
