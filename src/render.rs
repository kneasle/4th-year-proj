//! Code for rendering images

use itertools::Itertools;

use crate::{
    context::Context,
    image::{EffectInstance, Image, Layer},
    utils::Rect,
};

pub(crate) fn render(ctx: &mut Context, image: &Image) {
    let annotated_image = AnnotatedImage::new(ctx, image);
    dbg!(annotated_image);
}

/////////////////////
// ANNOTATED IMAGE //
/////////////////////

#[derive(Debug)]
struct AnnotatedImage<'img> {
    layers: Vec<AnnotatedLayer<'img>>,
    source: &'img Image,
}

#[derive(Debug)]
struct AnnotatedLayer<'img> {
    /// The bounding box of the source region of this layer that actually needs to be computed
    effects: Vec<AnnotatedEffect<'img>>,
    source_bbox: Rect<f32>,
    source: &'img Layer,
}

#[derive(Debug)]
struct AnnotatedEffect<'img> {
    /// The bounding box required of the output region
    out_bbox: Rect<f32>,
    source: &'img EffectInstance,
}

impl<'img> AnnotatedImage<'img> {
    /// Take an [`Image`] and annotate every layer with the bounding box of the region that has to be
    /// computed.
    fn new(ctx: &Context, image: &'img Image) -> Self {
        let img_bbox = Rect::from_origin(image.size.x as f32, image.size.y as f32);
        AnnotatedImage {
            layers: image
                .layers
                .iter()
                .map(|layer| AnnotatedLayer::new(ctx, layer, img_bbox))
                .collect_vec(),
            source: image,
        }
    }
}

impl<'img> AnnotatedLayer<'img> {
    fn new(ctx: &Context, layer: &'img Layer, bbox_from_above: Rect<f32>) -> Self {
        // Propagate bboxes from below (i.e. compute the regions which are affected by the layer's
        // source)
        let mut bboxes_from_below = Vec::new();
        let layer_size = ctx.get_layer(layer.source).unwrap().size();
        let bbox_of_source_from_below =
            Rect::from_origin(layer_size.width as f32, layer_size.height as f32);
        let mut curr_bbox_from_below = bbox_of_source_from_below;
        for effect in layer.effects.iter().rev() {
            let effect_type = ctx.get_effect(&effect.effect_name).unwrap();
            curr_bbox_from_below = effect_type.transform_bbox(curr_bbox_from_below);
            // Push the bbox **after** the effect has been applied
            bboxes_from_below.push(curr_bbox_from_below);
        }
        bboxes_from_below.reverse();

        // Propagate bboxes downward, computing the true bboxes (i.e. the intersection of the
        // bboxes from above and below)
        let mut effects = Vec::new();
        let mut curr_bbox_from_above = bbox_from_above;
        for (effect, bbox_from_below) in layer.effects.iter().zip_eq(bboxes_from_below) {
            let effect_type = ctx.get_effect(&effect.effect_name).unwrap();
            curr_bbox_from_above = effect_type.inv_transform_bbox(curr_bbox_from_above);
            let combined_bbox = bbox_from_above.intersection(bbox_from_below);
            effects.push(AnnotatedEffect {
                out_bbox: combined_bbox,
                source: effect,
            });
        }

        // The bbox required by the lowest effect is the bbox of the source layer from above
        let bbox_of_source_from_above = curr_bbox_from_above;
        let bbox_of_source = bbox_of_source_from_above.intersection(bbox_of_source_from_below);
        AnnotatedLayer {
            source_bbox: bbox_of_source,
            effects,
            source: layer,
        }
    }
}
