//! Custom types used for storing plugin settings and sending dynamically typed data to the GPU.

use std::{alloc::Layout, collections::HashMap};

/// Enum of the different types which can be sent to the GPU
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    /// 32-bit floating point number
    F32,
    I32,
}

impl Type {
    pub fn wgsl_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
        }
    }

    pub fn layout(self) -> Layout {
        match self {
            Self::F32 => Layout::new::<f32>(),
            Self::I32 => Layout::new::<i32>(),
        }
    }
}

/// Possible values of a dynamic [`Type`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    F32(f32),
    I32(i32),
}

impl Value {
    pub fn type_(self) -> Type {
        match self {
            Self::F32(_) => Type::F32,
            Self::I32(_) => Type::I32,
        }
    }

    pub fn get_f32(self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn get_i32(self) -> Option<i32> {
        match self {
            Self::I32(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the raw bytes covering the value in `self`
    fn bytes(&self) -> &[u8] {
        match self {
            Self::F32(v) => bytemuck::bytes_of(v),
            Self::I32(v) => bytemuck::bytes_of(v),
        }
    }
}

#[derive(Debug, Default)]
pub struct BufferBuilder {
    inner: Vec<u8>,
}

impl BufferBuilder {
    pub fn add_value(&mut self, value: Value) {
        self.add_bytes(value.type_().layout(), value.bytes());
    }

    pub fn add<T: bytemuck::Pod>(&mut self, value: T) {
        self.add_bytes(Layout::new::<T>(), bytemuck::bytes_of(&value));
    }

    pub fn add_bytes(&mut self, layout: Layout, bytes: &[u8]) {
        assert_eq!(bytes.len(), layout.size());
        // Add (zero) padding until we reach a multiple of `alignment`
        while self.inner.len() % layout.align() != 0 {
            self.inner.push(0);
        }
        // Write the bytes of this type into the buffer
        self.inner.extend_from_slice(bytes);
    }

    pub fn finish(self) -> Vec<u8> {
        self.into()
    }
}

impl From<BufferBuilder> for Vec<u8> {
    fn from(b: BufferBuilder) -> Self {
        b.inner
    }
}

pub fn make_buffer(fields: &[(String, Type)], values: &HashMap<String, Value>) -> Vec<u8> {
    let mut buffer = BufferBuilder::default();
    for (field_name, field_type) in fields {
        let field_value = values[field_name];
        assert_eq!(field_value.type_(), *field_type);
        buffer.add_value(field_value);
    }
    buffer.into()
}
