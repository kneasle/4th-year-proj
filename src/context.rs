use std::{collections::HashMap, num::NonZeroU32, path::Path};

use cgmath::Vector2;
use index_vec::IndexVec;
use wgpu::util::DeviceExt;

use crate::{
    effects::{Effect, EffectName, EffectType},
    image::Image,
    utils::SizedTexture,
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
    output_texture: Option<SizedTexture>,
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

            instance,
            adapter,
            device,
            queue,

            quad_buffer,
            output_texture: None,
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

    pub fn render(&mut self, image: &Image) {
        crate::render::render(self, image);
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
