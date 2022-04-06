use std::{
    fmt::{Debug, Formatter},
    ops::Deref,
};

use cgmath::{BaseNum, Point2, Vector2};

//////////////
// TEXTURES //
//////////////

/// A cached texture on the GPU, with convenience methods for e.g. resizing
#[derive(Debug)]
pub(crate) struct CacheTexture {
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
    pub fn small(device: &wgpu::Device) -> Self {
        Self::new(device, wgpu::Extent3d::default())
    }

    /// Create a new [`CacheTexture`] with a given size
    pub fn new(device: &wgpu::Device, size: wgpu::Extent3d) -> Self {
        Self {
            tex: Self::new_texture(device, size),
        }
    }

    /// Resize `self` to make sure there's space for `required_size`
    pub fn resize(&mut self, device: &wgpu::Device, required_size: wgpu::Extent3d) {
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

/// A [`wgpu::Texture`] which keeps a record of its dimensions (I'm not really sure why wgpu
/// doesn't give us access to this.  Maybe there's a way that I don't know about - @kneasle).
#[derive(Debug)]
pub struct SizedTexture {
    tex: wgpu::Texture,
    size: wgpu::Extent3d,
}

impl SizedTexture {
    pub fn new(tex: wgpu::Texture, size: wgpu::Extent3d) -> Self {
        Self { tex, size }
    }

    pub fn width(&self) -> u32 {
        self.size.width
    }

    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }

    pub fn tex(&self) -> &wgpu::Texture {
        &self.tex
    }
}

impl std::ops::Deref for SizedTexture {
    type Target = wgpu::Texture;

    fn deref(&self) -> &Self::Target {
        &self.tex
    }
}

//////////
// RECT //
//////////

/// An axis-aligned rectangular region in 2D space
// Invariant: max.x >= min.x && max.y >= min.y
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect<S> {
    min: Point2<S>,
    max: Point2<S>,
}

impl<S: BaseNum> Rect<S> {
    //////////////////
    // CONSTRUCTORS //
    //////////////////

    /// Creates a [`Rect`] with a given size and where the minimum corner is the origin (i.e.
    /// `(0, 0)`)
    pub fn from_origin(width: S, height: S) -> Self {
        Self {
            min: Point2::new(S::zero(), S::zero()),
            max: Point2::new(width, height),
        }
    }

    pub fn from_min_size(min: Point2<S>, size: Vector2<S>) -> Self {
        Self {
            min,
            max: min + size,
        }
    }

    /////////////
    // GETTERS //
    /////////////

    pub fn min(self) -> Point2<S> {
        self.min
    }

    pub fn max(self) -> Point2<S> {
        self.max
    }

    pub fn size(self) -> Vector2<S> {
        self.max - self.min
    }

    pub fn width(self) -> S {
        self.size().x
    }

    pub fn height(self) -> S {
        self.size().y
    }

    ////////////////
    // OPERATIONS //
    ////////////////

    /// Translates a [`Rect`] by some amount, preserving the size
    pub fn translate(self, by: Vector2<S>) -> Self {
        Self {
            min: self.min + by,
            max: self.max + by,
        }
    }
}

impl<S: PartialOrd> Rect<S> {
    pub fn intersection(self, other: Self) -> Self {
        let min_x = partial_max(self.min.x, other.min.x);
        let min_y = partial_max(self.min.y, other.min.y);
        let max_x = partial_min(self.max.x, other.max.x);
        let max_y = partial_min(self.max.y, other.max.y);
        Self {
            min: Point2::new(min_x, min_y),
            max: Point2::new(max_x, max_y),
        }
    }

    /// Computes the smallest `Rect` to contain both `self` and `other`
    pub fn union(self, other: Self) -> Self {
        let min_x = partial_min(self.min.x, other.min.x);
        let min_y = partial_min(self.min.y, other.min.y);
        let max_x = partial_max(self.max.x, other.max.x);
        let max_y = partial_max(self.max.y, other.max.y);
        Self {
            min: Point2::new(min_x, min_y),
            max: Point2::new(max_x, max_y),
        }
    }
}

impl<S: Debug> Debug for Rect<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rect(({:?}, {:?}) - ({:?}, {:?}))",
            self.min.x, self.min.y, self.max.x, self.max.y
        )
    }
}

//////////
// QUAD //
//////////

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 2], // z coordinates set to 0 by the vertex shader
    uv: [f32; 2],
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
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

//////////
// MISC //
//////////

/// Given a [`Vector2`] of `f32`, round this up to the nearest pixel and create a
/// [`wgpu::Extent3d`]
pub fn round_up_to_extent(tex_size: Vector2<f32>) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: tex_size.x.ceil() as u32,
        height: tex_size.y.ceil() as u32,
        depth_or_array_layers: 1,
    }
}

/// Given a floating-point [`Point2`], round this down to the nearest pixel and return that as a
/// [`wgpu::Origin3d`]
pub fn round_down_to_origin(point: Point2<f32>) -> wgpu::Origin3d {
    wgpu::Origin3d {
        x: point.x.floor() as u32,
        y: point.y.floor() as u32,
        z: 0,
    }
}

fn partial_max<S: PartialOrd>(x: S, y: S) -> S {
    if x < y {
        y
    } else {
        x
    }
}

fn partial_min<S: PartialOrd>(x: S, y: S) -> S {
    if x < y {
        x
    } else {
        y
    }
}
