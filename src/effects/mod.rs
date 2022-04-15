//! Image effects (i.e. any transformation)

mod per_pixel;
mod transform;

use std::{
    collections::HashMap,
    fmt::{Debug, Formatter},
    ops::Deref,
};

pub use per_pixel::PerPixel;
pub use transform::Transform;

use crate::{
    types::Value,
    utils::{Rect, TextureRegion},
};

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
pub trait EffectType {
    fn name(&self) -> String;

    fn encode_commands(
        &self,
        params: &HashMap<String, Value>,
        source: TextureRegion,
        out: TextureRegion,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
    );

    /// Given a [`Rect`] `r` in _input space_, return the smallest [`Rect`] in _output space_ which
    /// is affected by the pixels in `r`.
    fn transform_bbox(&self, params: &HashMap<String, Value>, rect: Rect<f32>) -> Rect<f32>;

    /// Given a [`Rect`] `r` in _output space_, return the smallest [`Rect`] in _input space_ which
    /// covers the pre-image of every point within `r`.
    fn inv_transform_bbox(&self, params: &HashMap<String, Value>, rect: Rect<f32>) -> Rect<f32>;
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
    use std::collections::HashMap;

    use crate::types::{self, Type, Value};

    use super::{per_pixel::CustomUniforms, PerPixel};

    pub fn invert(device: &wgpu::Device) -> PerPixel {
        PerPixel::new(
            "Invert".to_owned(),
            "return vec4<f32>(vec3<f32>(1.0) - col.rgb, col.a);",
            vec![], // No parameters
            None,   // No custom uniforms
            device,
        )
    }

    // Based on GIMP's `Brightness/Contrast` operation (source here: https://gitlab.gnome.org/GNOME/gimp/-/blob/a6d59a9b688331085327f968b8bb061f5d7a42c2/app/operations/gimpoperationbrightnesscontrast.c#L119).
    // This implementation has a few key differences:
    // 1. Contrast is a 'difference', and ranges between -1 (remove all contrast) through 0 (no
    //    change) to +1 (maximum contrast).  GIMP's contrast ranges from 0 to 2
    // 2. Contrast is applied before brightness, so brightness still has an effect when saturation
    //    is very low (if contrast goes after, then low contrasts force everything to be grey).
    pub fn brightness_contrast(device: &wgpu::Device) -> PerPixel {
        PerPixel::new(
            "Brightness/Contrast".to_owned(),
            "
var contrasted: vec3<f32> = (col.rgb - 0.5) * params.slant + 0.5;
var darkened: vec3<f32> = contrasted * params.darken;
var lightened: vec3<f32> = darkened * (1.0 - params.lighten) + params.lighten;
return vec4<f32>(lightened, col.a);",
            vec![
                // Must be in range [-1, +1].  0 is no change
                ("contrast".to_owned(), Type::F32),
                // Must be in range [-1, +1].  0 is no change
                ("brightness".to_owned(), Type::F32),
            ],
            Some(CustomUniforms {
                types: vec![
                    // 'Exponential' version of `contrast`
                    ("slant".to_owned(), Type::F32),
                    // factor in 0..=1 by which the pixels are multiplied.  Smaller is darker
                    ("darken".to_owned(), Type::F32),
                    // factor in 0..=1 by which the pixels are lerped towards 1.  Larger is
                    // brighter
                    ("lighten".to_owned(), Type::F32),
                ],
                buffer_from_params: Box::new(|params: &HashMap<String, Value>| {
                    let mut builder = types::BufferBuilder::default();

                    let contrast = params["contrast"].get_f32().unwrap();
                    let brightness = params["brightness"].get_f32().unwrap();

                    // contrast -> slant (the factor of `0.99999` makes sure that `saturation = +1`
                    // doesn't give us tan(PI/2), which is undefined)
                    builder.add(f32::tan(
                        (contrast + 1.0) * 0.99999 * std::f32::consts::FRAC_PI_4,
                    ));
                    // brightness -> darken, lighten
                    let (darken, lighten) = if brightness < 0.0 {
                        (1.0 + brightness, 0.0) // -ve brightness only uses `darken`
                    } else {
                        (1.0, brightness) // +ve brightness only uses `lighten`
                    };
                    builder.add(darken);
                    builder.add(lighten);

                    builder.into()
                }),
            }),
            device,
        )
    }
}
