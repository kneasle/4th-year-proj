use wgpu::util::DeviceExt;

use crate::utils::{QuadVertex, Rect, TextureRegion};

use super::EffectType;

/// An effect who's effect can be computed independently for each pixel.
#[derive(Debug)]
pub struct PerPixel {
    name: String,
    source_tex_layout: super::SourceTexBindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl PerPixel {
    /// Creates a new `PerPixel` [`EffectType`]
    pub fn new(name: String, per_pixel_code: &str, device: &wgpu::Device) -> Self {
        // Load shader code
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} shader module", name)),
            source: wgpu::ShaderSource::Wgsl(
                format!("{SHADER_HEADER}{per_pixel_code}{SHADER_FOOTER}").into(),
            ),
        });

        let source_tex_layout = super::SourceTexBindGroupLayout::new(&name, device);
        // Create render pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} render layout", name)),
            bind_group_layouts: &[&source_tex_layout.layout], // TODO: Add FX args here
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{} render pipeline", name)),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[QuadVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // verts go anti-clockwise
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                // Both of these require GPU features to use.  So we keep them disabled wherever
                // possible to increase the range of GPUs we can run on.
                clamp_depth: false,
                conservative: false,
            },
            depth_stencil: None, // Not using any depth testing
            multisample: wgpu::MultisampleState::default(), // Not using multi-sampling
        });

        Self {
            source_tex_layout,
            pipeline,

            name,
        }
    }
}

impl EffectType for PerPixel {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn encode_commands(
        &self,
        source: TextureRegion,
        out: TextureRegion,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
    ) {
        // Input/output regions should always be the same because per-pixel effects don't modify
        // the bboxes at all (i.e. `source.affected_region` and `out.affected_region` must both be
        // computed from the intersection of the same two bboxes)
        assert_eq!(source.region, out.region);

        let source_tex_bind_group = self.source_tex_layout.bind_group(&source.texture, device);
        // TODO: Store all the quads in one large buffer and send them all to the GPU in one go
        let quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} quad buffer", self.name)),
            contents: bytemuck::cast_slice(&QuadVertex::quad(&source, &out)),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create a render pass, which will end up rendering our single quad into the required
        // region of the image
        let render_pass_label = format!("{} render pass", self.name);
        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some(&render_pass_label), // TODO: Better name
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &out
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                resolve_target: None, // No multi-sampling
                ops: wgpu::Operations {
                    // For per-pixel effects, we don't need to clear anything because we're
                    // guaranteed to overwrite the entire region that will get used by the next
                    // effect.  For other effects (like rotations) we should clear the texture to
                    // transparent.
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None, // Not using depth or stencil
        };
        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
        render_pass.set_bind_group(0, &source_tex_bind_group, &[]);
        render_pass.draw(0..4, 0..1); // Quad always has 4 vertices
    }

    fn transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r // Per-pixel effects don't change the bbox; they transform the pixels individually
    }

    fn inv_transform_bbox(&self, r: Rect<f32>) -> Rect<f32> {
        r // Per-pixel effects don't change the bbox; they transform the pixels individually
    }
}

/// WGSL code which goes above the per-pixel code
const SHADER_HEADER: &str = "
struct VertexInput {
    [[location(0)]] position: vec2<f32>;
    [[location(1)]] tex_coords: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    return out;
}



[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;

fn modify_color(col: vec4<f32>) -> vec4<f32> {
";
/// WGSL code that goes after the per-pixel code
const SHADER_FOOTER: &str = "
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return modify_color(textureSample(t_diffuse, s_diffuse, in.tex_coords));
}";
