use std::{collections::HashMap, num::NonZeroU32, path::Path};

use cgmath::Vector2;
use image::RgbaImage;
use index_vec::IndexVec;
use wgpu::util::DeviceExt;

use crate::{
    texture::SizedTexture,
    tree::{Effect, Tree},
};

/// Persistent state used for processing images.  Only one `Context` is required per instance of
/// the image editor.
pub struct Context {
    /* Loaded effect classes */
    effect_types: HashMap<String, EffectType>,

    /* wgpu essentials */
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    /* shared wgpu resources */
    /// A buffer containing vertices for a single quad with position2/uv data.  Rendering a texture
    /// to this mesh fills the screen completely
    quad_buffer: wgpu::Buffer,
    output_texture: Option<SizedTexture>,
    layers: IndexVec<LayerId, SizedTexture>,
}

impl Context {
    pub const ID_VALUE_INVERT: &'static str = "value-invert";

    /// Creates a new `Context` with handles to GPU resources but no loaded [`EffectType`]s.
    pub fn empty() -> Self {
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

            instance,
            adapter,
            device,
            queue,

            quad_buffer,
            output_texture: None,
            layers: IndexVec::new(),
        }
    }

    /// Creates a new `Context`, preloaded with all builtin effects.
    pub fn with_builtin_effects() -> Self {
        let mut ctx = Self::empty();
        ctx.load_wgsl_effect(
            Self::ID_VALUE_INVERT,
            "Value Invert",
            include_str!("../shader/invert.wgsl"),
        );
        ctx
    }

    /// Load a image effect from its WGSL source code, returning its [`EffectTypeId`].
    pub fn load_wgsl_effect(&mut self, id: &str, name: &str, wgsl_source: &str) {
        let old_fx = self.effect_types.insert(
            id.to_owned(),
            EffectType::from_wgsl_source(name, wgsl_source, &self.device),
        );
        assert!(old_fx.is_none(), "shouldn't set the same effect ID twice");
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

///////////////////
// IMAGE EFFECTS //
///////////////////

/// A runtime-loaded image effect.  This specifies a 'class' of image effects with different
/// parameters.  Any number of instances of the resulting `EffectType` can then be instantiated
/// into the image trees.
pub struct EffectType {
    name: String,
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl EffectType {
    /// Load a new [`EffectType`] given a name and shader source.  The vertex and fragment
    /// shaders are expected to be named `vs_main` and `fs_main`, respectively.
    pub fn from_wgsl_source(name: &str, wgsl_source: &str, device: &wgpu::Device) -> Self {
        // Load the WGSL shader code we've been given
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });
        // Create a bind group for the texture that we'll be modifying
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} bind group layout", name)),
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
                    count: None, // We're not using texture arrays (yet?)
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        filtering: true,
                        comparison: false, // TODO: Should we change this?
                    },
                    count: None, // No texture arrays (yet?)
                },
            ],
        });
        // Create the render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} render layout", name)),
                bind_group_layouts: &[&bind_group_layout], // TODO: Add FX args here?
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{} render pipeline", name)),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",       // TODO: Make this configurable?
                buffers: &[Vertex::layout()], // TODO: Add FX args here
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main", // TODO: Make this configurable?
                // Write to all channels of an RGBA texture
                targets: &[wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 'front' is defined as anticlockwise verts
                cull_mode: Some(wgpu::Face::Back), // Cull 'back' faces
                polygon_mode: wgpu::PolygonMode::Fill,
                // We don't need these, and both require GPU features.  Therefore, we just disable
                // them to reduce the limits on the GPUs we can use.
                clamp_depth: false,
                conservative: false,
            },
            depth_stencil: None, // Don't use depth or stencil buffer
            multisample: wgpu::MultisampleState::default(), // Also don't use multi-sampling
        });
        Self {
            name: name.to_owned(),
            render_pipeline,
            bind_group_layout,
        }
    }
}

////////////////////////////////
// IMAGE PROCESSING/RENDERING //
////////////////////////////////

impl Context {
    pub fn render_to_image(&mut self, image_tree: &Tree) -> image::RgbaImage {
        let pixel_size = std::mem::size_of::<u32>() as u32;
        let img_dims = image_tree.dimensions(self);

        // Render the image into `self.output_texture` (resizing it if necessary)
        self.render_to_texture(image_tree);
        let output_texture = self.output_texture.as_ref().unwrap();

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
                texture: &output_texture,
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
            output_texture.size(),
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
        RgbaImage::from_raw(img_dims.x, img_dims.y, buffer_data).unwrap()
    }

    pub fn render_to_texture(&mut self, image_tree: &Tree) {
        /* We render effect chains by alternating between two textures (one of which is
         * output_texture).  This way, each effect never writes to its own input, but we only need
         * two textures for the whole chain.  The first effect reads directly from the source
         * layer.
         *
         * Example chains:
         *  0 effects: layer ---[copy]---> output_texture
         *  1 effect : layer -[effect 0]-> output_texture
         *  2 effects: layer -[effect 0]->  temp_texture  -[effect 1]-> output_texture
         *  3 effects: layer -[effect 0]-> output_texture -[effect 1]-> temp_texture
         *                                                -[effect 2]-> output_texture
         */

        // If the output texture is too small (or non-existent), create a new one big enough to
        // store the result of `image_tree`.
        let img_dims = image_tree.dimensions(self);
        let out_tex_needs_resize = match &self.output_texture {
            Some(tex) => tex.width() < img_dims.x || tex.height() < img_dims.y,
            None => true,
        };
        if out_tex_needs_resize {
            self.output_texture = Some(self.create_swap_texture(wgpu::Extent3d {
                width: img_dims.x,
                height: img_dims.y,
                depth_or_array_layers: 1,
            }));
        }
        let output_texture = self.output_texture.as_ref().unwrap();

        let layer = &self.layers[image_tree.layer_id];
        assert_eq!(output_texture.size(), layer.size()); // TODO: Handle layer sizes differently later

        // Encoder for the GPU commands which make up this effects chain
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FX chain"),
            });
        // Helper function to add a render pass
        let mut add_pass = |effect: &Effect, tex_in: &wgpu::Texture, tex_out: &wgpu::Texture| {
            self.effect_types[&effect.id].add_pass(self, &mut encoder, tex_in, tex_out)
        };

        match image_tree.effects.as_slice() {
            // Special case: if there aren't any effects, we just perform a copy without allocating any
            // temporary textures.
            [] => encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: layer,
                    mip_level: 0, // We're not doing any mipmapping
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTextureBase {
                    texture: output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                layer.size(),
            ),
            // If there's only one effect, write directly to the output without allocating a
            // temporary texture
            [only_effect] => add_pass(only_effect, layer, output_texture),
            // If there are at least two textures, then create a temp texture and continue with the
            // effect chains
            [first_effect, other_effects @ ..] => {
                assert!(!other_effects.is_empty());
                // Create a temporary texture, so we can swap between it and `output_texture`
                // TODO: Cache this
                let temp_texture = self.create_swap_texture(output_texture.size());
                // Every effect from `other_effects` will run `tex_in -> tex_out`, swapping each
                // time.  We want the last effect to have `tex_out == output_texture`, which
                // determines which way round `output_texture` and `temp_texture` are.  The
                // `first_effect` writes into `tex_in`.
                let (mut tex_in, mut tex_out) = match other_effects.len() % 2 {
                    0 => (output_texture, &temp_texture),
                    1 => (&temp_texture, output_texture),
                    _ => unreachable!("`x % 2` can't be anything other than 0 or 1"),
                };
                // Add effects, swapping `tex_{in,out}` each time so that each effect's output
                // becomes the next effect's input
                add_pass(first_effect, layer, tex_in);
                for effect in other_effects {
                    add_pass(effect, tex_in, tex_out);
                    std::mem::swap(&mut tex_in, &mut tex_out);
                }
                // `output_texture` would be the input to a hypothetical next effect
                assert!(std::ptr::eq(tex_in, output_texture));
            }
        }
        // Run the FX chain on the GPU
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Create a texture that can be used to alternate between images
    fn create_swap_texture(&self, size: wgpu::Extent3d) -> SizedTexture {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
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

impl EffectType {
    /// Creates a render/compute pass which applies `self` to `tex_in`, placing the result in
    /// `tex_out`
    fn add_pass(
        &self,
        context: &Context,
        encoder: &mut wgpu::CommandEncoder,
        tex_in: &wgpu::Texture,
        tex_out: &wgpu::Texture,
    ) {
        // Create a sampler & bind group for the input texture
        let tex_in_view = tex_in.create_view(&wgpu::TextureViewDescriptor::default());
        let tex_in_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{} input sampler", self.name)),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest, // Note: Interpolation method for layer stacking
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let tex_in_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} bind group", self.name)),
                layout: &self.bind_group_layout,
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
            });
        // Create the render pass
        let label = format!("{} render pass", self.name);
        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some(&label),
            // This `RenderPassColorAttachment` is targeted by `[[location(0)]]` in the WGSL shader
            color_attachments: &[wgpu::RenderPassColorAttachment {
                // Write to the whole texture
                view: &tex_out.create_view(&wgpu::TextureViewDescriptor::default()),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: true, // duh.  Why would we not want to keep what we're rendering?
                },
            }],
            depth_stencil_attachment: None, // Not using depth or stencil
        };
        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, context.quad_buffer.slice(..));
        render_pass.set_bind_group(0, &tex_in_bind_group, &[]); // Bind the input textures
        render_pass.draw(0..QUAD_VERTICES.len() as u32, 0..1); // Only use one instance
    }
}

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

////////////////
// MISC/UTILS //
////////////////

index_vec::define_index_type! {
    /// Numerical IDs for each layer loaded by a [`Context`]
    pub struct LayerId = usize;
}
