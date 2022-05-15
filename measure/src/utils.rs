use crate::Context;

/// Copy contents of a GPU buffer back into CPU memory
pub fn map_buffer(ctx: &Context, buffer: &wgpu::Buffer) -> Vec<u8> {
    let buffer_data = {
        let buffer_slice = buffer.slice(..);

        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
        ctx.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(mapping).unwrap();

        buffer_slice.get_mapped_range().to_vec()
    };
    buffer.unmap();
    buffer_data
}

//////////
// QUAD //
//////////

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadVertex {
    position: [f32; 2], // z coordinates set to 0 by the vertex shader
    uv: [f32; 2],
}

impl QuadVertex {
    // Ideally we'd just put this on the final array, but attributes on expressions are apparently
    // experimental so for now we'll just stop formatting on the whole function
    #[rustfmt::skip]
    pub fn quad() -> [Self; 4] {
        // Texture coordinates:
        //
        //                    u
        //                0       1
        //     ^     +------------>
        //     |     |
        //    1|    0|    2 ----- 3
        //     |     |    | \     |
        //  y  |   v |    |   \   |
        //     |     |    |     \ |
        //   -1|    1v    0 ----- 1
        //     |
        //     +------------------>
        //               -1       1
        //                    x
        //
        // Note that we invert the y-coordinate of `o` because (-1, 1) is the top-left corner of
        // clip-space.
        [
            QuadVertex { uv: [0.0, 1.0], position: [-1.0, -1.0] },
            QuadVertex { uv: [1.0, 1.0], position: [ 1.0, -1.0] },
            QuadVertex { uv: [0.0, 0.0], position: [-1.0,  1.0] },
            QuadVertex { uv: [1.0, 0.0], position: [ 1.0,  1.0] },
        ]
    }

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
