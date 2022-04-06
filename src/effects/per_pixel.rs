use crate::utils::Rect;

use super::{EffectType, TextureRegion};

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

    fn encode_commands(
        &self,
        source: TextureRegion,
        out: TextureRegion,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Input/output regions should always be the same because per-pixel effects don't modify
        // the bboxes at all (i.e. `source.affected_region` and `out.affected_region` must both be
        // computed from the intersection of the same two bboxes)
        assert_eq!(source.region, out.region);
        // Create a render pass, which will end up rendering our single quad into the required
        // region of the image
        let tex_view = out
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Per-Pixel Effect"), // TODO: Better name
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &tex_view,
                resolve_target: None, // No multi-sampling
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None, // Not using depth or stencil
        });
    }

    fn transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r // Per-pixel effects don't change the bbox; they transform the pixels individually
    }

    fn inv_transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r // Per-pixel effects don't change the bbox; they transform the pixels individually
    }
}
