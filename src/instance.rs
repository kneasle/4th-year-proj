use std::num::NonZeroU32;

use cgmath::Vector2;
use image::RgbaImage;

use crate::tree::Tree;

/// A singleton for every editor, which holds data used by the rest of the library.
pub struct Instance {
    effect_types: EffectTypeVec<EffectType>,

    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Instance {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();

        Self {
            effect_types: EffectTypeVec::new(),

            instance,
            adapter,
            device,
            queue,
        }
    }

    /// Load a image effect from its WGSL source code, returning its [`EffectTypeId`].
    pub fn load_wgsl_effect(&mut self, name: String, wgsl_source: String) -> EffectTypeId {
        self.effect_types
            .push(EffectType::new(name, wgsl_source, &self.device))
    }

    pub fn render_to_image(&mut self, image_tree: &Tree) -> image::RgbaImage {
        let u32_size = std::mem::size_of::<u32>() as u32;
        let img_dims = image_tree.dimensions();

        // Create texture to which the image tree will be rendered
        let texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: img_dims.x,
                height: img_dims.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: Self::target_texture_usages(),
            label: None,
        };
        let texture = self.device.create_texture(&texture_desc);
        // Render the image into the texture
        self.render_to_texture(image_tree, &texture, img_dims);

        // Create a buffer into which we can copy our texture
        let output_buffer_size = (u32_size * img_dims.x * img_dims.y) as wgpu::BufferAddress;
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
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(u32_size * img_dims.x),
                    rows_per_image: NonZeroU32::new(img_dims.y),
                },
            },
            texture_desc.size,
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

    pub fn render_to_texture(
        &mut self,
        image_tree: &Tree,
        output_texture: &wgpu::Texture,
        dimensions: Vector2<u32>,
    ) {
        /* We render effect chains by alternating between two textures (the output texture and the
         * texture's source texture).  This way, each effect never writes to its own input, but we
         * only need two textures for the whole chain. */

        let layer_dims = Vector2::from(image_tree.layer.image.dimensions());
        assert_eq!(dimensions, layer_dims); // TODO: Handle layer sizes differently later

        // Create a new texture for the source image
        let layer_texture_size = wgpu::Extent3d {
            width: layer_dims.x,
            height: layer_dims.y,
            depth_or_array_layers: 1,
        };
        let layer_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Layer tex"),
            size: layer_texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        // Copy the source texture to the first texture in the chain.  Since textures alternate
        // between Effects, we write to `layer_texture` for odd numbers of FX and `output_texture`
        // for even numbers of FX.
        let (mut tex_in, mut tex_out) = if image_tree.effects.len() % 2 == 0 {
            (output_texture, &layer_texture)
        } else {
            (&layer_texture, output_texture)
        };
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: tex_in,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &image_tree.layer.image,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * layer_dims.x),
                rows_per_image: NonZeroU32::new(layer_dims.y),
            },
            layer_texture_size,
        );

        // Encode commands to process the image effects
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FX chain"),
            });
        for effect in &image_tree.effects {
            self.effect_types[effect.id].add_pass(
                &self,
                &mut encoder,
                tex_in,
                tex_out,
                layer_texture_size,
            );
            // Swap the textures so that this layer's output becomes the next layer's input
            std::mem::swap(&mut tex_in, &mut tex_out);
        }
        // `output_texture` would be the input to a hypothetical next effect
        assert!(std::ptr::eq(tex_in, output_texture));
        // Run the FX chain on the GPU
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// The [`TextureUsages`](wgpu::TextureUsages) needed for the texture into which the final
    /// composite image is rendered.
    pub fn target_texture_usages() -> wgpu::TextureUsages {
        wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT
    }
}

/// A runtime-loaded image effect.  This specifies a 'class' of image effects with different
/// parameters.  Any number of instances of the resulting `EffectType` can then be instantiated
/// into the image trees.
pub struct EffectType {
    name: String,
    render_pipeline: wgpu::RenderPipeline,
}

impl EffectType {
    /// Load a new [`EffectType`] given a name and shader source.  The vertex and fragment
    /// shaders are expected to be named `vs_main` and `fs_main`, respectively.
    pub fn new(name: String, wgsl_source: String, device: &wgpu::Device) -> Self {
        // Load the WGSL shader code we've been given
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some(&name),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} render layout", name)),
                // TODO: Add FX args here?
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{} render pipeline", name)),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // TODO: Make this configurable?
                buffers: &[],           // TODO: Add FX args here
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
                topology: wgpu::PrimitiveTopology::TriangleList,
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
            name,
            render_pipeline,
        }
    }

    /// Creates a render/compute pass which applies `self` to `tex_in`, placing the result in
    /// `tex_out`
    pub fn add_pass(
        &self,
        instance: &Instance,
        encoder: &mut wgpu::CommandEncoder,
        tex_in: &wgpu::Texture,
        tex_out: &wgpu::Texture,
        tex_size: wgpu::Extent3d,
    ) {
        let label = format!("{} render pass", self.name);
        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some(&label),
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
        render_pass.draw(0..3, 0..1); // First 3 vertices, first (and only instance)
    }
}

index_vec::define_index_type! { pub struct EffectTypeId = usize; }
pub type EffectTypeVec<T> = index_vec::IndexVec<EffectTypeId, T>;
