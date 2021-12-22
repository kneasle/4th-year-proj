use std::num::NonZeroU32;

use cgmath::Vector2;
use image::RgbaImage;

use crate::tree::Tree;

/// A singleton for every editor, which holds data used by the rest of the library.
#[derive(Debug)]
pub struct Instance {
    pub plugins: PlugVec<Plugin>,

    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Instance {
    pub fn new(plugins: impl IntoIterator<Item = Plugin>) -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();

        Self {
            plugins: plugins.into_iter().collect(),

            instance,
            adapter,
            device,
            queue,
        }
    }

    pub fn render_image(&mut self, image_tree: &Tree) -> image::RgbaImage {
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
        // Load the base layer texture to the GPU
        let layer_dims = Vector2::from(image_tree.layer.image.dimensions());
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
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &layer_texture,
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
        if image_tree.effects.is_empty() {
            // If the FX stack is empty, then copy the layer texture directly to the output texture
            assert_eq!(dimensions, layer_dims);
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &layer_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                layer_texture_size,
            );
        } else {
            todo!("Image effects not implemented yet");
        }
        // Run the FX chain on the GPU
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn target_texture_usages() -> wgpu::TextureUsages {
        wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST
    }
}

/// A runtime-loaded image effect plugin.  This specifies a 'class' of image effects with different
/// parameters.  Any number of instances of the resulting `Plugin` can then be instantiated into
/// the image trees.
#[derive(Debug, Clone)]
pub struct Plugin {
    pub name: String,
    pub wgsl_source: String,
}

index_vec::define_index_type! { pub struct PlugIdx = usize; }
pub type PlugVec<T> = index_vec::IndexVec<PlugIdx, T>;
