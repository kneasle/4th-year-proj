use std::collections::HashMap;

use cgmath::Vector2;

use crate::{
    types::Value,
    utils::{round_up_to_extent, Rect},
};

use super::{EffectType, TextureRegion};

/// An effect which applies a translate-rotate-scale linear transformation to the source image.
// TODO: Implement rotation and scaling
#[derive(Debug)]
pub struct Transform();

impl EffectType for Transform {
    fn name(&self) -> String {
        "Transform".to_owned()
    }

    fn encode_commands(
        &self,
        _params: &HashMap<String, Value>,
        source: TextureRegion,
        out: TextureRegion,
        encoder: &mut wgpu::CommandEncoder,
        _device: &wgpu::Device,
    ) {
        // TODO: Be able to exploit the fact that this is a no-op

        // Regions should be the same size because we only are only translating the image
        assert_eq!(source.region.size(), out.region.size());
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTextureBase {
                texture: &source.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTextureBase {
                texture: &out.texture,
                mip_level: 0,
                // Note that we don't actually perform the translation because we always use the
                // top-left corner of the cache texture, regardless of where that corresponds in
                // 'virtual' texture space
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            round_up_to_extent(out.region.size()),
        );
    }

    fn transform_bbox(&self, params: &HashMap<String, Value>, r: Rect<f32>) -> Rect<f32> {
        r.translate(get_position(params))
    }

    fn inv_transform_bbox(&self, params: &HashMap<String, Value>, r: Rect<f32>) -> Rect<f32> {
        r.translate(-get_position(params))
    }
}

fn get_position(params: &HashMap<String, Value>) -> Vector2<f32> {
    Vector2 {
        x: params.get("x").unwrap().get_i32().unwrap() as f32,
        y: params.get("y").unwrap().get_i32().unwrap() as f32,
    }
}
