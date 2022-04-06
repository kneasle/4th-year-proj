//! Image effects (i.e. any transformation)

mod per_pixel;
mod transform;

use std::{
    fmt::{Debug, Formatter},
    ops::Deref,
};

pub use per_pixel::PerPixel;
pub use transform::Transform;

use crate::utils::{Rect, SizedTexture};

#[derive(Debug)]
pub struct Effect {
    name: EffectName,
    ty: Box<dyn EffectType>,
}

impl Effect {
    pub fn new(ty: impl EffectType + 'static) -> Self {
        let ty = Box::new(ty);
        Self {
            name: EffectName(ty.name()),
            ty,
        }
    }

    #[must_use]
    pub fn name(&self) -> &EffectName {
        &self.name
    }
}

impl Deref for Effect {
    type Target = dyn EffectType;

    fn deref(&self) -> &Self::Target {
        self.ty.as_ref()
    }
}

/// Trait implemented by all effect types
pub trait EffectType: Debug {
    fn name(&self) -> String;

    fn encode_commands(
        &self,
        source: TextureRegion,
        out: TextureRegion,
        encoder: &mut wgpu::CommandEncoder,
    );

    /// Given a [`Rect`] `r` in _input space_, return the smallest [`Rect`] in _output space_ which
    /// is affected by the pixels in `r`.
    fn transform_bbox(&self, rect: Rect<f32>) -> Rect<f32>;

    /// Given a [`Rect`] `r` in _output space_, return the smallest [`Rect`] in _input space_ which
    /// covers the pre-image of every point within `r`.
    fn inv_transform_bbox(&self, rect: Rect<f32>) -> Rect<f32>;
}

/// A unique identifier for effects
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EffectName(String);

impl EffectName {
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<&str> for EffectName {
    fn from(s: &str) -> Self {
        EffectName(s.to_owned())
    }
}

impl Debug for EffectName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "EffectName({:?})", self.0)
    }
}

/// Built-in effects
pub mod built_ins {
    use super::PerPixel;

    pub fn value_invert() -> PerPixel {
        PerPixel::new("Value Invert".to_owned())
    }
}

/// A region of a texture which this [`Effect`] interacts with (by either reading or writing to it)
#[derive(Debug)]
pub struct TextureRegion<'tex> {
    /// The region (in virtual space) which is of interest to the [`Effect`]
    pub region: Rect<f32>,
    /// The cache texture which we're interacting with
    pub texture: &'tex SizedTexture,
}
