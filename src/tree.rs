use std::fmt::Debug;

use cgmath::Vector2;

use crate::context::{Context, LayerId};

/// An abstract tree view of the image being edited
#[derive(Debug, Clone)]
pub struct Tree {
    /// The stack of [`Effect`]s, in the order that they are applied
    pub effects: Vec<Effect>,
    /// The base image to which the [`Effect`]s are applied
    pub layer_id: LayerId,
}

impl Tree {
    pub fn dimensions(&self, ctx: &Context) -> Vector2<u32> {
        let extent = ctx.layer_dimensions(self.layer_id);
        Vector2::new(extent.width, extent.height)
    }
}

/// An instance of an image effect.  This can be viewed as a function which transforms images.
#[derive(Debug, Clone)]
pub struct Effect {
    /// The ID of the [`EffectType`] of which this is an instance
    pub id: String,
}
