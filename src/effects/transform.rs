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
pub struct Transform {
    position: Vector2<f32>,
}

impl Transform {
    pub fn new(pos_x: f32, pos_y: f32) -> Self {
        // TODO: Have these determined by each instance
        Self {
            position: Vector2::new(pos_x, pos_y),
        }
    }
}

impl EffectType for Transform {
    fn name(&self) -> String {
        format!("Transform {},{}", self.position.x, self.position.y)
    }

    fn encode_commands(
        &self,
        params: &HashMap<String, Value>,
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

    fn transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r.translate(self.position)
    }

    fn inv_transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r.translate(-self.position)
    }
}
