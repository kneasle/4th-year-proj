use std::{
    fmt::{Debug, Formatter},
    ops::{Deref, DerefMut},
    path::Path,
    rc::Rc,
};

use cgmath::Vector2;
use image::{DynamicImage, ImageError};

/// An abstract tree view of the image being edited
#[derive(Debug, Clone)]
pub struct Tree {
    /// The stack of [`Effect`]s, in the order that they are applied
    pub effects: Vec<Effect>,
    /// The base image to which the [`Effect`]s are applied
    pub layer: Rc<Layer>,
}

impl Tree {
    pub fn dimensions(&self) -> Vector2<u32> {
        self.layer.image.dimensions().into()
    }
}

/// An instance of an image effect.  This can be viewed as a function which transforms images.
#[derive(Debug, Clone)]
pub struct Effect {
    /// The ID of the [`EffectType`] of which this is an instance
    pub id: String,
}

/// A source image which forms the leaves of the image tree.
#[derive(Debug, Clone)]
pub struct Layer {
    pub image: DebuggableImage,
}

impl Layer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ImageError> {
        let dyn_image = image::io::Reader::open(path)?.decode()?;
        let rgba_image = match dyn_image {
            DynamicImage::ImageRgba8(i) => i,
            DynamicImage::ImageRgb8(rgb_img) => {
                // Copy the RGB image into an RGBA image with the same colours but full opacity
                // PERF: This would be more performant if we modify the buffers directly
                let mut new_img = image::RgbaImage::new(rgb_img.width(), rgb_img.height());
                for y in 0..rgb_img.height() {
                    for x in 0..rgb_img.width() {
                        let image::Rgb([r, g, b]) = *rgb_img.get_pixel(x, y);
                        new_img.put_pixel(x, y, image::Rgba([r, g, b, u8::MAX]))
                    }
                }
                new_img
            }
            _ => panic!("Please load an RGBA8 or RGB8 file."),
        };
        Ok(Self {
            image: DebuggableImage(rgba_image),
        })
    }
}

/// Wrapper of [`image::RgbaImage`] with a human-friendly [`Debug`] impl.
#[derive(Clone)]
#[repr(transparent)]
pub struct DebuggableImage(image::RgbaImage);

impl Deref for DebuggableImage {
    type Target = image::RgbaImage;

    fn deref(&self) -> &image::RgbaImage {
        &self.0
    }
}

impl DerefMut for DebuggableImage {
    fn deref_mut(&mut self) -> &mut image::RgbaImage {
        &mut self.0
    }
}

impl Debug for DebuggableImage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (w, h) = self.0.dimensions();
        write!(f, "RgbaImage({}x{})", w, h)
    }
}
