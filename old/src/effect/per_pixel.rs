use super::Rect;

/// An effect who's effect can be computed independently for each pixel.
#[derive(Debug)]
pub struct PerPixel {}

impl super::Effect for PerPixel {
    fn transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r
    }

    fn inv_transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r
    }
}
