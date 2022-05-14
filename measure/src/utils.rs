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
