use crate::utils::Rect;

use super::EffectType;

/// An effect who's effect can be computed independently for each pixel.
#[derive(Debug)]
pub struct PerPixel {
    pub name: String,
}

impl PerPixel {
    /// Creates a new `PerPixel` [`EffectType`]
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl EffectType for PerPixel {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r // Per-pixel effects don't change the bbox; they transform the pixels individually
    }

    fn inv_transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r // Per-pixel effects don't change the bbox; they transform the pixels individually
    }
}
