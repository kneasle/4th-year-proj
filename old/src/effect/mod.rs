//! Module for storing image effects

mod per_pixel;

pub use per_pixel::PerPixel;

use cgmath::Point2;

/// Trait implemented by all effect types
pub trait Effect {
    /// Given a [`Rect`] `r` in _input space_, return the smallest [`Rect`] in _output space_ which
    /// is affected by the pixels in `r`.
    fn transform_bbox(&self, rect: Rect<f32>) -> Rect<f32>;

    /// Given a [`Rect`] `r` in _output space_, return the smallest [`Rect`] in _input space_ which
    /// covers the pre-image of every point within `r`.
    fn inv_transform_bbox(&self, rect: Rect<f32>) -> Rect<f32>;
}

/// A rectangular region in 2D space
// Invariant: max.x >= min.x && max.y >= min.y
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect<S> {
    min: Point2<S>,
    max: Point2<S>,
}

impl<S: Ord> Rect<S> {
    pub fn intersection(self, other: Self) -> Self {
        let min_x = self.min.x.max(other.min.x);
        let min_y = self.min.y.max(other.min.y);
        let max_x = self.max.x.min(other.max.x);
        let max_y = self.max.y.min(other.max.y);
        Self {
            min: Point2::new(min_x, min_y),
            max: Point2::new(max_x, max_y),
        }
    }

    /// Computes the smallest `Rect` to contain both `self` and `other`
    pub fn union(self, other: Self) -> Self {
        let min_x = self.min.x.min(other.min.x);
        let min_y = self.min.y.min(other.min.y);
        let max_x = self.max.x.max(other.max.x);
        let max_y = self.max.y.max(other.max.y);
        Self {
            min: Point2::new(min_x, min_y),
            max: Point2::new(max_x, max_y),
        }
    }
}
