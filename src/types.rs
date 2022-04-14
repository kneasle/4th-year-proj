//! Custom types used for storing plugin settings and sending dynamically typed data to the GPU.

use std::{alloc::Layout, collections::HashMap};

/// Enum of the different types which can be sent to the GPU
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    /// 32-bit floating point number
    F32,
}

impl Type {
    pub fn wgsl_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
        }
    }

    pub fn layout(self) -> Layout {
        match self {
            Self::F32 => Layout::new::<f32>(),
        }
    }
}

/// Possible values of a dynamic [`Type`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    F32(f32),
}

impl Value {
    pub fn type_(self) -> Type {
        match self {
            Self::F32(_) => Type::F32,
        }
    }

    /// Returns the raw bytes covering the value in `self`
    fn bytes(&self) -> &[u8] {
        match self {
            Self::F32(v) => bytemuck::bytes_of(v),
        }
    }

    fn write(self, expected_type: Type, buffer: &mut Vec<u8>) {
        assert_eq!(self.type_(), expected_type);
        let layout = expected_type.layout();
        // Add (zero) padding until we reach a multiple of `alignment`
        while buffer.len() % layout.align() != 0 {
            buffer.push(0);
        }
        // Write the bytes of this type into the buffer (zeroing them first)
        buffer.extend_from_slice(self.bytes());
    }
}

pub fn make_buffer(fields: &[(String, Type)], values: &HashMap<String, Value>) -> Vec<u8> {
    let mut buffer = Vec::new();
    for (field_name, field_type) in fields {
        values[field_name].write(*field_type, &mut buffer);
    }
    buffer
}
