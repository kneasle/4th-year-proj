use std::fmt::{Debug, Formatter};

use cgmath::{BaseNum, Point2, Vector2};

////////////////////
// SIZED TEXTURES //
////////////////////

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
