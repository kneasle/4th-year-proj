//! Utilities for handling textures in GPU memory

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
