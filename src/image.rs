use cgmath::Vector2;

use crate::{context::LayerId, effects::EffectName};

/// The specification for a full image
#[derive(Debug)]
pub struct Image {
    pub size: Vector2<u32>,
    pub layers: Vec<Layer>,
}

#[derive(Debug)]
pub struct Layer {
    pub effects: Vec<EffectInstance>,
    pub source: LayerId,
}

#[derive(Debug, Clone)]
pub struct EffectInstance {
    /// The name of the [`Effect`] of which this is an instance
    pub effect_name: EffectName,
}
