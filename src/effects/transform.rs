use cgmath::Vector2;

use crate::utils::Rect;

use super::EffectType;

/// An effect which applies a translate-rotate-scale linear transformation to the source image.
// TODO: Implement rotation and scaling
#[derive(Debug)]
pub struct Transform {
    position: Vector2<f32>,
}

impl Transform {
    pub fn new(pos_x: f32, pos_y: f32) -> Self {
        Self {
            position: Vector2::new(pos_x, pos_y),
        }
    }
}

impl EffectType for Transform {
    fn name(&self) -> String {
        "Transform".to_owned()
    }

    fn transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r.translate(self.position)
    }

    fn inv_transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r.translate(-self.position)
    }
}
