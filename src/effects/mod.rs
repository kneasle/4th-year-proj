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
        params: &HashMap<String, Value>,
        source: TextureRegion,
        out: TextureRegion,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
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
    use crate::types::Type;

    use super::PerPixel;

    pub fn value_invert(device: &wgpu::Device) -> PerPixel {
        PerPixel::new(
            "Value Invert".to_owned(),
            "return vec4<f32>(vec3<f32>(1.0) - col.rgb, col.a);",
            vec![], // No uniforms
            device,
        )
    }

    pub fn brightness_contrast(device: &wgpu::Device) -> PerPixel {
        PerPixel::new(
            "Brightness/Contrast".to_owned(),
            "
var brightened: vec3<f32> = vec3<f32>(params.brightness) + col.rgb;
var contrasted: vec3<f32> = (brightened - 0.5) * params.contrast + 0.5;
return vec4<f32>(contrasted, col.a);",
            vec![
                ("brightness".to_owned(), Type::F32),
                ("contrast".to_owned(), Type::F32),
            ],
            device,
        )
    }
}

///////////
// UTILS //
///////////

/// A wrapper around [`wgpu::BindGroupLayout`] which generates the correct settings for image
/// effects' source textures
#[derive(Debug)]
struct SourceTexBindGroupLayout {
    fx_name: String,
    layout: wgpu::BindGroupLayout,
}

impl SourceTexBindGroupLayout {
    fn new(fx_name: &str, device: &wgpu::Device) -> Self {
        Self {
            fx_name: fx_name.to_owned(),
            layout: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} source tex bind group layout", fx_name)),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            // TODO: We potentially don't need to do a ton of u8 -> f32 -> u8
                            // conversions.  I'm not sure if they actually slow things down; they're so
                            // widespread in games that GPUs almost certainly have custom hardware for
                            // it.
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            }),
        }
    }

    fn bind_group(&self, texture: &wgpu::Texture, device: &wgpu::Device) -> wgpu::BindGroup {
        // Source texture bind group
        let tex_in_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let tex_in_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{} tex input sampler", self.fx_name)),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} source tex bind group", self.fx_name)),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_in_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&tex_in_sampler),
                },
            ],
        })
    }
}
