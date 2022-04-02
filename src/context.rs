use std::{collections::HashMap, num::NonZeroU32, ops::Deref, path::Path};

use cgmath::Vector2;
use index_vec::IndexVec;
use itertools::Itertools;
use wgpu::util::DeviceExt;

use crate::{
    effects::{Effect, EffectName, EffectType},
    image::{EffectInstance, Image, Layer},
    utils::{Rect, SizedTexture},
};

/// Persistent state used for processing images.  Only one `Context` is required per instance of
/// the image editor.
pub struct Context {
    /* Loaded effect classes */
    effect_types: HashMap<EffectName, Effect>,

    /* wgpu essentials */
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    /* misc wgpu resources */
    /// A buffer containing vertices for a single quad with position2/uv data.  Rendering a texture
    /// to this mesh fills the screen completely
    quad_buffer: wgpu::Buffer,

    /* textures */
    output_texture: CacheTexture,
    intermediate_textures: [CacheTexture; 2],
    layers: IndexVec<LayerId, SizedTexture>,
}

index_vec::define_index_type! {
    /// Unique identifier for a layer source textures
    pub struct LayerId = usize;
}

impl Context {
    /// Creates a new `Context` with handles to GPU resources but no loaded [`EffectType`]s.
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();

        let quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad vertex buffer"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            effect_types: HashMap::new(),

            output_texture: CacheTexture::small(&device),
            intermediate_textures: [CacheTexture::small(&device), CacheTexture::small(&device)],

            instance,
            adapter,
            device,
            queue,

            quad_buffer,
            layers: IndexVec::new(),
        }
    }

    /// Loads a new [`EffectType`], returning its unique [`EffectName`]
    pub fn load_effect(&mut self, ty: impl EffectType + 'static) -> EffectName {
        let effect = Effect::new(ty);
        let name = effect.name().to_owned();
        self.effect_types.insert(name.clone(), effect);
        name
    }

    /// Given an [`EffectName`], return the corresponding [`Effect`] (or `None` if no [`Effect`]s
    /// have that name).
    pub fn get_effect(&self, name: &EffectName) -> Option<&Effect> {
        self.effect_types.get(name)
    }

    ////////////
    // LAYERS //
    ////////////

    /// Given a [`LayerId`], get the corresponding [`SizedTexture`] or `None` if no layer with that
    /// ID is found.
    pub fn get_layer(&self, id: LayerId) -> Option<&SizedTexture> {
        self.layers.get(id)
    }

    /// Load a new layer from a file and into GPU memory, returning its [`LayerId`].
    pub fn load_layer_from_file(&mut self, path: impl AsRef<Path>) -> image::ImageResult<LayerId> {
        let dyn_image = image::io::Reader::open(path)?.decode()?;
        let rgba_image = match dyn_image {
            image::DynamicImage::ImageRgba8(i) => i,
            image::DynamicImage::ImageRgb8(rgb_img) => {
                // Copy the RGB image into an RGBA image with the same colours but full opacity
                // PERF: This would be more performant if we could write the buffers directly.  It
                // would be even more performant if we could write the new buffer using SIMD
                // permutation instructions, but I really doubt this will ever become enough of a
                // bottleneck to justify that.
                let mut new_img = image::RgbaImage::new(rgb_img.width(), rgb_img.height());
                for y in 0..rgb_img.height() {
                    for x in 0..rgb_img.width() {
                        let image::Rgb([r, g, b]) = *rgb_img.get_pixel(x, y);
                        new_img.put_pixel(x, y, image::Rgba([r, g, b, u8::MAX]));
                    }
                }
                new_img
            }
            _ => panic!("Please load an RGBA8 or RGB8 file."),
        };
        Ok(self.load_layer(&rgba_image))
    }

    /// Load a new image into GPU memory and return its [`LayerId`].
    pub fn load_layer(&mut self, source: &image::RgbaImage) -> LayerId {
        let id = self.layers.next_idx();

        // Create an empty texture of the correct size on the GPU
        let dimensions = Vector2::from(source.dimensions());
        let tex_size = wgpu::Extent3d {
            width: dimensions.x,
            height: dimensions.y,
            depth_or_array_layers: 1, // Not using 3D textures
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Layer {:?} texture", id)),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // We need to copy to the texture and use it as a source for shaders
            //
            // TODO: Once we implement texture chunking, I don't think we'll need `COPY_SRC` or
            // `COPY_DST`
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        // Write to the GPU memory to store our texture
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            source,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * dimensions.x),
                rows_per_image: NonZeroU32::new(dimensions.y),
            },
            tex_size,
        );
        // Add the new layer, and return its ID
        let sized_tex = SizedTexture::new(texture, tex_size);
        let new_id = self.layers.push(sized_tex);
        assert_eq!(new_id, id);
        new_id
    }

    pub fn layer_dimensions(&self, id: LayerId) -> wgpu::Extent3d {
        self.layers[id].size()
    }
}

///////////////
// RENDERING //
///////////////

impl Context {
    /// Render a given [`Image`] to an CPU-memory [`image::RgbaImage`] buffer, ready to be written
    /// to a file
    pub fn render_to_image(&mut self, image: &Image) -> image::RgbaImage {
        let pixel_size = std::mem::size_of::<u32>() as u32;
        let img_dims = image.size;

        // Render the image into `self.output_texture` (resizing it if necessary)
        self.render(image);

        // Create a buffer into which we can copy our texture
        let output_buffer_size = (pixel_size * img_dims.x * img_dims.y) as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = self.device.create_buffer(&output_buffer_desc);
        // Copy the texture into the buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(pixel_size * img_dims.x),
                    rows_per_image: NonZeroU32::new(img_dims.y),
                },
            },
            self.output_texture.size(),
        );
        self.queue.submit(Some(encoder.finish()));

        // 'Map' the buffer (i.e. copying from GPU mem into main RAM)
        let buffer_data = {
            let buffer_slice = output_buffer.slice(..);

            // NOTE: We have to create the mapping THEN device.poll() before await
            // the future. Otherwise the application will freeze.
            let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
            self.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(mapping).unwrap();

            buffer_slice.get_mapped_range().to_vec()
        };
        output_buffer.unmap();

        // Create an RGB image from the data
        image::RgbaImage::from_raw(img_dims.x, img_dims.y, buffer_data).unwrap()
    }

    /// Render an image to the current [`output_texture`](Self::output_texture)
    pub fn render(&mut self, image: &Image) {
        // Annotate the image with the `Rect`s covered by the intermediate textures
        let annotated_image = AnnotatedImage::new(self, image);
        dbg!(&annotated_image);
        // Resize textures
        self.output_texture.resize(
            &self.device,
            wgpu::Extent3d {
                width: image.size.x,
                height: image.size.y,
                depth_or_array_layers: 1,
            },
        );
        let tex_size = annotated_image.max_intermediate_texture_size();
        let max_texture_extent = wgpu::Extent3d {
            width: tex_size.x.ceil() as u32,
            height: tex_size.y.ceil() as u32,
            depth_or_array_layers: 1,
        };
        for tex in &mut self.intermediate_textures {
            tex.resize(&self.device, max_texture_extent);
        }

        // Command encoder for all the effects' render passes
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Image rendering"),
            });
        self.clear_output_texture(&mut encoder);
        for layer in annotated_image.layers {}
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Clear the output buffer.  Layers will be rendered on top of this without clearing.
    fn clear_output_texture(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&format!("Clear output texture")),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &self
                    .output_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                resolve_target: None, // No multisampling
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                    store: true,
                },
            }],
            depth_stencil_attachment: None, // Not using depth or stencil
        });
    }
}

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

    /// Return the size of intermediate texture needed to store all the intermediate layers
    fn max_intermediate_texture_size(&self) -> Vector2<f32> {
        let mut max_width = 0.0;
        let mut max_height = 0.0;
        let mut resize_for_rect = |rect: Rect<f32>| {
            if rect.width() > max_width {
                max_width = rect.width();
            }
            if rect.height() > max_height {
                max_height = rect.height();
            }
        };
        for layer in &self.layers {
            resize_for_rect(layer.source_bbox);
            for effect in &layer.effects {
                resize_for_rect(effect.out_bbox);
            }
        }
        Vector2::new(max_width, max_height)
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

//////////////
// TEXTURES //
//////////////

#[derive(Debug)]
struct CacheTexture {
    tex: SizedTexture,
}

impl Deref for CacheTexture {
    type Target = SizedTexture;

    fn deref(&self) -> &Self::Target {
        &self.tex
    }
}

impl CacheTexture {
    /// Create a place-holder [`CacheTexture`] with the smallest size possible
    fn small(device: &wgpu::Device) -> Self {
        Self::new(device, wgpu::Extent3d::default())
    }

    /// Create a new [`CacheTexture`] with a given size
    fn new(device: &wgpu::Device, size: wgpu::Extent3d) -> Self {
        Self {
            tex: Self::new_texture(device, size),
        }
    }

    /// Resize `self` to make sure there's space for `required_size`
    fn resize(&mut self, device: &wgpu::Device, required_size: wgpu::Extent3d) {
        if required_size.width > self.tex.size().width
            || required_size.height > self.tex.size().height
            || required_size.depth_or_array_layers > self.tex.size().depth_or_array_layers
        {
            self.tex = Self::new_texture(device, required_size);
        }
    }

    /// Create a texture that can be used for storing intermediate values between effect chains
    fn new_texture(device: &wgpu::Device, size: wgpu::Extent3d) -> SizedTexture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        SizedTexture::new(texture, size)
    }
}

//////////
// QUAD //
//////////

/// The vertices to describe a single quad which maps one set of texture coordinates to the entire
/// screen.
// NOTE: The v-coordinates are inverted because wgpu uses y-up for world-space coordinates, but
// y-down for textures.  Note also that the positions are in clip coordinates - i.e. `(-1, -1)` is
// the top-left corner and `(1, 1)` is the bottom-right.
#[rustfmt::skip]
const QUAD_VERTICES: [Vertex; 4] = [
    Vertex { position: [-1.0, -1.0], uv: [0.0, 1.0], },
    Vertex { position: [ 1.0, -1.0], uv: [1.0, 1.0], },
    Vertex { position: [-1.0,  1.0], uv: [0.0, 0.0], },
    Vertex { position: [ 1.0,  1.0], uv: [1.0, 0.0], },
];

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2], // z coordinates set to 0 by the vertex shader
    uv: [f32; 2],
}

impl Vertex {
    fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            // TODO: Replace this with a `const` call to `wgpu::vertex_attr_array!`?
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0, // TODO: Is this what [[location(0)]] does?
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                },
            ],
        }
    }
}
