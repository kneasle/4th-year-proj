use std::{collections::HashMap, fmt::Write};

use wgpu::util::DeviceExt;

use crate::{
    types::{self, Type, Value},
    utils::{QuadVertex, Rect, TextureRegion},
};

use super::EffectType;

/// An effect who's effect can be computed independently for each pixel.
#[derive(Debug)]
pub struct PerPixel {
    name: String,
    param_types: Vec<(String, Type)>,
    params_layout: wgpu::BindGroupLayout, // Unused if `param_types.is_empty()`
    source_tex_layout: super::SourceTexBindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl PerPixel {
    /// Creates a new `PerPixel` [`EffectType`]
    pub fn new(
        name: String,
        per_pixel_code: &str,
        param_types: Vec<(String, Type)>,
        device: &wgpu::Device,
    ) -> Self {
        // Load shader code
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} shader module", name)),
            source: wgpu::ShaderSource::Wgsl(
                generate_wgsl_code(per_pixel_code, &param_types).into(),
            ),
        });

        // Bind group layouts (we only create a bind group layout for the parameters if there
        // actually are any, because `wgpu` doesn't allow 0-sized uniforms).
        let source_tex_layout = super::SourceTexBindGroupLayout::new(&name, device);
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} params bind group", name)),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let mut bind_group_layouts = vec![&source_tex_layout.layout];
        if !param_types.is_empty() {
            bind_group_layouts.push(&params_layout);
        }
        // Render pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} render layout", name)),
            bind_group_layouts: &bind_group_layouts,
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
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // Not using any depth testing
            multisample: wgpu::MultisampleState::default(), // Not using multi-sampling
            multiview: None,
        });

        Self {
            name,
            param_types,

            params_layout,
            source_tex_layout,
            pipeline,
        }
    }
}

impl EffectType for PerPixel {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn encode_commands(
        &self,
        params: &HashMap<String, Value>,
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
            label: Some(&render_pass_label),
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
        let params_bind_group; // Bind group here so it gets dropped after `render_pass`
        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
        render_pass.set_bind_group(0, &source_tex_bind_group, &[]);
        if !self.param_types.is_empty() {
            // TODO: Store all the params in a single buffer
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} params buffer", self.name)),
                contents: &types::make_buffer(&self.param_types, params),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            });
            params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} params bind group", self.name)),
                layout: &self.params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                }],
            });
            render_pass.set_bind_group(1, &params_bind_group, &[]);
        }
        render_pass.draw(0..4, 0..1); // Quad always has 4 vertices
    }

    fn transform_bbox(&self, _params: &HashMap<String, Value>, r: Rect<f32>) -> Rect<f32> {
        r // Per-pixel effects don't change the bbox; they transform the pixels individually
    }

    fn inv_transform_bbox(&self, _params: &HashMap<String, Value>, r: Rect<f32>) -> Rect<f32> {
        r // Per-pixel effects don't change the bbox; they transform the pixels individually
    }
}

/// Given information about a [`PerPixel`] effect, generate full WGSL code for the
/// corresponding shader.
// NOTE: wgpu doesn't allow us to create 0-sized uniforms.  None of our [`Type`]s have 0-size, so a
// uniform is 0-sized iff there are no uniforms.  Therefore, in the case of `uniforms = []` we
// don't emit any WGSL code for the uniforms at all.
fn generate_wgsl_code(per_pixel_code: &str, params: &[(String, Type)]) -> String {
    // TODO: Make sure that the uniforms are actually valid identifiers

    let mut code = String::new();
    // Add struct definition for the `Uniforms`.  These will always be attached to binding 0 of
    // bind group 1 (i.e. `[[group(1), binding(0)]]` in WGSL).
    if !params.is_empty() {
        code.push_str("struct Params {\n");
        for (var_name, type_) in params {
            write!(code, "    {}: {};\n", var_name, type_.wgsl_name()).unwrap();
        }
        code.push_str("};\n\n");
    }
    // Add bind group definitions
    code.push_str(
        "// Group 0 is always the input texture
[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;",
    );
    if !params.is_empty() {
        code.push_str(
            "
// Group 1 is the effect parameters
[[group(1), binding(0)]]
var<uniform> params: Params;",
        );
    }
    // Vertex shader code
    code.push_str(
        "

// VERTEX SHADER

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
",
    );
    // Add fragment code
    code.push_str(
        "
// FRAGMENT SHADER

fn modify_color(col: vec4<f32>) -> vec4<f32> {",
    );
    for line in per_pixel_code.lines() {
        code.push_str("\n    ");
        code.push_str(line);
    }
    code.push_str(
        "
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return modify_color(textureSample(t_diffuse, s_diffuse, in.tex_coords));
}",
    );

    code
}
