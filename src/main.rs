use std::{collections::HashMap, time::Instant};

use cgmath::Vector2;
use hmap::hmap;
use image_ed::{
    context::Context,
    effects::{self, built_ins},
    image::{EffectInstance, Image, Layer},
    types::Value,
};

fn main() {
    let mut ctx = Context::new();
    let invert_name = ctx.load_effect(built_ins::invert(ctx.device()));
    let brightness_contrast_name = ctx.load_effect(built_ins::brightness_contrast(ctx.device()));
    let transform_name = ctx.load_effect(effects::Transform());

    let image = Image {
        size: Vector2::new(512, 512),
        layers: vec![
            Layer {
                effects: vec![],
                source_id: ctx.load_layer_from_file("img/mc-skin.png").unwrap(),
            },
            Layer {
                effects: vec![EffectInstance {
                    effect_name: transform_name.clone(),
                    params: hmap! {
                        "x".to_owned() => Value::I32(-800),
                        "y".to_owned() => Value::I32(-800)
                    },
                }],
                source_id: ctx.load_layer_from_file("img/logo.png").unwrap(),
            },
            Layer {
                effects: vec![
                    EffectInstance {
                        effect_name: brightness_contrast_name,
                        params: hmap! {
                            "contrast".to_owned() => Value::F32(-0.4),
                            "brightness".to_owned() => Value::F32(0.7)
                        },
                    },
                    EffectInstance {
                        effect_name: invert_name,
                        params: HashMap::new(),
                    },
                    EffectInstance {
                        effect_name: transform_name,
                        params: hmap! {
                            "x".to_owned() => Value::I32(100),
                            "y".to_owned() => Value::I32(100)
                        },
                    },
                ],
                source_id: ctx.load_layer_from_file("img/logo.png").unwrap(),
            },
        ],
    };

    let start = Instant::now();
    let img = ctx.render_to_image(&image);
    println!("Rendered in {:?}", start.elapsed());
    img.save("out.png").unwrap();
}
