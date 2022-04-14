use std::collections::HashMap;

use cgmath::Vector2;
use image_ed::{
    context::Context,
    effects::{self, built_ins},
    image::{EffectInstance, Image, Layer},
    types::Value,
};

fn main() {
    let mut ctx = Context::new();
    let value_inv_name = ctx.load_effect(built_ins::value_invert(ctx.device()));
    let brightness_contrast_name = ctx.load_effect(built_ins::brightness_contrast(ctx.device()));
    let transform_name_1 = ctx.load_effect(effects::Transform::new(-800.0, -800.0));
    let transform_name_2 = ctx.load_effect(effects::Transform::new(100.0, 100.0));
    let image = Image {
        size: Vector2::new(512, 512),
        layers: vec![
            Layer {
                effects: vec![],
                source_id: ctx.load_layer_from_file("img/mc-skin.png").unwrap(),
            },
            Layer {
                effects: vec![EffectInstance {
                    effect_name: transform_name_1,
                    params: HashMap::new(),
                }],
                source_id: ctx.load_layer_from_file("img/logo.png").unwrap(),
            },
            Layer {
                effects: vec![
                    EffectInstance {
                        effect_name: brightness_contrast_name.clone(),
                        params: hmap::hmap! {
                            "brightness".to_owned() => Value::F32(-0.4),
                            "contrast".to_owned() => Value::F32(3.0)
                        },
                    },
                    EffectInstance {
                        effect_name: value_inv_name,
                        params: HashMap::new(),
                    },
                    EffectInstance {
                        effect_name: transform_name_2,
                        params: HashMap::new(),
                    },
                ],
                source_id: ctx.load_layer_from_file("img/logo.png").unwrap(),
            },
        ],
    };
    ctx.render_to_image(&image).save("out.png").unwrap();
}
