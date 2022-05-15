use std::{
    io::Write,
    time::{Duration, Instant},
};

use kneasle_ringing_utils::BigNumInt;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::Serialize;
use wgpu::util::DeviceExt;

use crate::{utils::QuadVertex, Context};

const ITERS: u32 = 100;

pub trait TestCase {
    fn name() -> &'static str;

    /// Construct a new instance of the `TestCase`, which can be used to run as many tests as
    /// needed
    fn new(ctx: &Context, size: u64) -> Self;

    fn run(&mut self, ctx: &Context, is_baseline: bool);
}

/// Generate LaTeX code that creates a graph plotting the results of running some [`TestCase`] with
/// some specific sizes
pub fn take_measurements<T: TestCase>(
    ctx: &Context,
    size_step: u64,
    size_max: u64,
) -> Vec<Measurement> {
    let mut results = Vec::new();
    // Don't have a 0-sized iteration, because almost every GPU storage type can't be 0-sized
    let mut size = size_step;
    while size <= size_max {
        results.push(Measurement::new::<T>(size, ctx));
        size += size_step;
    }
    results
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Measurement {
    size: u64,
    duration_secs: f64,
}

impl Measurement {
    /// Take a single measurement of some [`TestCase`], given data of a given `size`
    pub fn new<T: TestCase>(size: u64, ctx: &Context) -> Self {
        let mut case = T::new(ctx, size);

        print!(
            "{} @ {:>7} ",
            T::name(),
            BigNumInt(size as usize).to_string()
        );
        let baseline_duration = run_partial_case(&mut case, ctx, true);
        let full_duration = run_partial_case(&mut case, ctx, false);

        let duration = full_duration.saturating_sub(baseline_duration);
        println!(" {:?}", duration);
        Self {
            size,
            duration_secs: duration.as_secs_f64(),
        }
    }
}

/// Run either a baseline or full test
fn run_partial_case(case: &mut impl TestCase, ctx: &Context, is_baseline: bool) -> Duration {
    let start = Instant::now();
    for i in 0..ITERS {
        case.run(ctx, is_baseline);
        if i % 20 == 0 {
            print!(".");
            std::io::stdout().lock().flush().unwrap();
        }
    }
    start.elapsed() / ITERS
}

/// A single byte that we'll fill all the buffers with, set to something that the GPU is very
/// unlikely to have the buffer initialised to (i.e. not 0).  We use this to perform a sanity check
/// that the GPU is actually doing something.
const MAGIC_NUMBER: u8 = 140;

//////////////////////////
// COPY BUFFER GPU->CPU //
//////////////////////////

/// Measure latency/throughput of copying data from GPU->CPU.
pub struct BufGpuToCpu {
    buf_1: wgpu::Buffer,
    buf_2: wgpu::Buffer,
}

impl TestCase for BufGpuToCpu {
    fn name() -> &'static str {
        "GPU->CPU buffer copy"
    }

    fn new(ctx: &crate::Context, size: u64) -> Self {
        let create_buf = |n: usize| {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("GPU->CPU copy test :: input buffer #{}", n)),
                    contents: &vec![MAGIC_NUMBER; size as usize],
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
                })
        };

        Self {
            buf_1: create_buf(1),
            buf_2: create_buf(2),
        }
    }

    fn run(&mut self, ctx: &crate::Context, is_baseline: bool) {
        // Copy data from CPU->GPU.  Note how we only want to measure the time for a single buffer
        // copy, so we include the time taken to submit a queue in each part
        let buf_1 = crate::utils::map_buffer(ctx, &self.buf_1);
        let buf_2;
        let slice_to_check = if is_baseline {
            &buf_1
        } else {
            buf_2 = crate::utils::map_buffer(ctx, &self.buf_2);
            &buf_2
        };
        // Check the results
        for b in &buf_1 {
            assert_eq!(*b, MAGIC_NUMBER);
        }
        for b in slice_to_check {
            assert_eq!(*b, MAGIC_NUMBER);
        }
    }
}

//////////////////////////
// COPY BUFFER CPU->GPU //
//////////////////////////

/// Measure latency/throughput of copying data from CPU->GPU.
///
/// We do this by copying two buffers CPU->GPU, copying the first byte of each into another, then
/// copying this 2-byte wide buffer back to the CPU.  For a baseline, we only copy one buffer
/// CPU->GPU and copy the first byte twice.
pub struct BufCpuToGpu {
    input_buf_1: wgpu::Buffer,
    input_buf_2: wgpu::Buffer,
    source_data: Vec<u8>,
}

impl TestCase for BufCpuToGpu {
    fn name() -> &'static str {
        "CPU->GPU buffer copy"
    }

    fn new(ctx: &crate::Context, size: u64) -> Self {
        let create_input_buf = |n: usize| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("CPU->GPU copy test :: input buffer #{}", n)),
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        Self {
            source_data: vec![MAGIC_NUMBER; size as usize],
            input_buf_1: create_input_buf(1),
            input_buf_2: create_input_buf(2),
        }
    }

    fn run(&mut self, ctx: &crate::Context, is_baseline: bool) {
        // Copy data from CPU->GPU.  Note how we only want to measure the time for a single buffer
        // copy, so we include the time taken to submit a queue in each part
        ctx.queue
            .write_buffer(&self.input_buf_1, 0, &self.source_data);
        if !is_baseline {
            ctx.queue
                .write_buffer(&self.input_buf_2, 0, &self.source_data);
        }
        // Block this thread until the GPU operations are finished
        ctx.queue.submit(None); // Make sure that wgpu copies the data
        ctx.device.poll(wgpu::Maintain::Wait);
    }
}

///////////////////////
// CPU-BASED EFFECTS //
///////////////////////

pub struct CpuInvert {
    input_buffer: Vec<[u8; 4]>,
    output_buffer: Vec<[u8; 4]>,
}

impl TestCase for CpuInvert {
    fn name() -> &'static str {
        "CPU invert"
    }

    fn new(_ctx: &Context, size: u64) -> Self {
        Self {
            input_buffer: vec![[0; 4]; size as usize],
            output_buffer: vec![[0; 4]; size as usize],
        }
    }

    fn run(&mut self, _ctx: &Context, is_baseline: bool) {
        if is_baseline {
            return;
        }
        self.input_buffer
            .par_iter_mut()
            .zip(self.output_buffer.par_iter_mut())
            .for_each(|([r_in, g_in, b_in, a_in], [r_out, g_out, b_out, a_out])| {
                *r_out = 255 - *r_in;
                *g_out = 255 - *g_in;
                *b_out = 255 - *b_in;
                *a_out = *a_in;
            });
    }
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct BrightnessContrastParams {
    slant: f32,
    lighten: f32,
    darken: f32,
}

impl Default for BrightnessContrastParams {
    fn default() -> Self {
        Self {
            slant: 1.5,
            darken: 1.0,
            lighten: 0.5,
        }
    }
}

pub struct CpuBrightnessContrast {
    input_buffer: Vec<[f32; 4]>,
    output_buffer: Vec<[f32; 4]>,

    params: BrightnessContrastParams,
}

impl TestCase for CpuBrightnessContrast {
    fn name() -> &'static str {
        "CPU brightness/contrast (f32)"
    }

    fn new(_ctx: &Context, size: u64) -> Self {
        Self {
            input_buffer: vec![[0.0; 4]; size as usize],
            output_buffer: vec![[0.0; 4]; size as usize],

            params: BrightnessContrastParams::default(),
        }
    }

    fn run(&mut self, _ctx: &Context, is_baseline: bool) {
        if is_baseline {
            return;
        }

        let params = self.params;
        self.input_buffer
            .par_iter_mut()
            .zip(self.output_buffer.par_iter_mut())
            .for_each(|([r_in, g_in, b_in, a_in], [r_out, g_out, b_out, a_out])| {
                let contrasted_r = (*r_in + 0.5) * params.slant + 0.5;
                let contrasted_g = (*g_in + 0.5) * params.slant + 0.5;
                let contrasted_b = (*b_in + 0.5) * params.slant + 0.5;

                *r_out = (contrasted_r * params.darken) * (1.0 - params.lighten) + params.lighten;
                *g_out = (contrasted_g * params.darken) * (1.0 - params.lighten) + params.lighten;
                *b_out = (contrasted_b * params.darken) * (1.0 - params.lighten) + params.lighten;
                *a_out = *a_in;
            });
    }
}

pub struct CpuBrightnessContrastBytes {
    input_buffer: Vec<[u8; 4]>,
    output_buffer: Vec<[u8; 4]>,
    params: BrightnessContrastParams,
}

impl TestCase for CpuBrightnessContrastBytes {
    fn name() -> &'static str {
        "CPU brightness/contrast (u8)"
    }

    fn new(_ctx: &Context, size: u64) -> Self {
        Self {
            input_buffer: vec![[0; 4]; size as usize],
            output_buffer: vec![[0; 4]; size as usize],
            params: BrightnessContrastParams::default(),
        }
    }

    fn run(&mut self, _ctx: &Context, is_baseline: bool) {
        if is_baseline {
            return;
        }

        let params = self.params;
        self.input_buffer
            .par_iter_mut()
            .zip(self.output_buffer.par_iter_mut())
            .for_each(|([r_in, g_in, b_in, a_in], [r_out, g_out, b_out, a_out])| {
                let contrasted_r = ((*r_in as f32) / 255.0 + 0.5) * params.slant + 0.5;
                let contrasted_g = ((*g_in as f32) / 255.0 + 0.5) * params.slant + 0.5;
                let contrasted_b = ((*b_in as f32) / 255.0 + 0.5) * params.slant + 0.5;

                let bc_r = (contrasted_r * params.darken) * (1.0 - params.lighten) + params.lighten;
                let bc_g = (contrasted_g * params.darken) * (1.0 - params.lighten) + params.lighten;
                let bc_b = (contrasted_b * params.darken) * (1.0 - params.lighten) + params.lighten;

                *r_out = (bc_r * 255.0) as u8;
                *g_out = (bc_g * 255.0) as u8;
                *b_out = (bc_b * 255.0) as u8;
                *a_out = *a_in;
            });
    }
}

///////////////////////
// GPU-BASED FILTERS //
///////////////////////

pub struct RenderPassBrightnessContrast {
    render_pipeline: wgpu::RenderPipeline,
    tex_bind_group_layout: wgpu::BindGroupLayout,
    uniforms_bind_group_layout: wgpu::BindGroupLayout,

    input_texture: wgpu::Texture,
    output_texture: wgpu::Texture,
    params: BrightnessContrastParams,
}

impl TestCase for RenderPassBrightnessContrast {
    fn name() -> &'static str {
        "GPU brightness/contrast (render pass)"
    }

    fn new(ctx: &Context, size: u64) -> Self {
        assert_eq!(size % 1000, 0);

        // Load shader code
        let shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some("Render pass brightness/contrast shader module"),
                source: wgpu::ShaderSource::Wgsl(RENDER_PASS_BC_SHADER.into()),
            });

        // Bind group layouts (we only create a bind group layout for the parameters if there
        // actually are any, because `wgpu` doesn't allow 0-sized uniforms).
        let source_tex_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RP B/C source tex bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                // TODO: We potentially don't need to do a ton of u8 -> f32 -> u8
                                // conversions.  I'm not sure if they actually slow things down; they're so
                                // widespread in games that GPUs almost certainly have custom hardware for
                                // it.
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let uniforms_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RP B/C params bind group"),
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
        // Render pipeline
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RP B/C render layout"),
                bind_group_layouts: &[&source_tex_layout, &uniforms_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("RP B/C render pipeline"),
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

        // Create textures
        let tex_descriptor = wgpu::TextureDescriptor {
            label: None,
            // Texture is always 1000 pixels tall
            size: wgpu::Extent3d {
                width: size as u32 / 2000,
                height: 2000,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        Self {
            render_pipeline,
            uniforms_bind_group_layout: uniforms_layout,
            tex_bind_group_layout: source_tex_layout,

            input_texture: ctx.device.create_texture(&tex_descriptor),
            output_texture: ctx.device.create_texture(&tex_descriptor),
            params: BrightnessContrastParams::default(),
        }
    }

    fn run(&mut self, ctx: &Context, is_baseline: bool) {
        // Source texture bind group
        let tex_in_view = self
            .input_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let tex_in_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("RP B/C tex input sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let source_tex_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RP B/C source tex bind group"),
            layout: &self.tex_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_in_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&tex_in_sampler),
                },
            ],
        });
        // TODO: Store all the quads in one large buffer and send them all to the GPU in one go
        let quad_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RP B/C quad buffer"),
                contents: bytemuck::cast_slice(&QuadVertex::quad()),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RP B/C params buffer"),
                contents: &bytemuck::cast::<_, [u8; 12]>(self.params),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            });
        let params_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RP B/C uniforms bind group"),
            layout: &self.uniforms_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        // Create a render pass, which will end up rendering our single quad into the required
        // region of the image
        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some("RP B/C render pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &self
                    .output_texture
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

        // Create the render pipeline itself
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RP B/C cmd encoder"),
            });
        if !is_baseline {
            let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, quad_buffer.slice(..));
            render_pass.set_bind_group(0, &source_tex_bind_group, &[]);
            render_pass.set_bind_group(1, &params_bind_group, &[]);
            render_pass.draw(0..4, 0..1); // Quad always has 4 vertices
        }

        // Send the render pass to the GPU, and wait for it to run
        ctx.queue.submit(Some(encoder.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }
}

const RENDER_PASS_BC_SHADER: &'static str = "
struct Params {
    slant: f32;
    darken: f32;
    lighten: f32;
};

// Group 0 is always the input texture
[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;
// Group 1 is the effect parameters
[[group(1), binding(0)]]
var<uniform> params: Params;


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


// FRAGMENT SHADER

fn modify_color(col: vec4<f32>) -> vec4<f32> {
    var contrasted: vec3<f32> = (col.rgb - 0.5) * params.slant + 0.5;
    var darkened: vec3<f32> = contrasted * params.darken;
    var lightened: vec3<f32> = darkened * (1.0 - params.lighten) + params.lighten;
    return vec4<f32>(lightened, col.a);
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return modify_color(textureSample(t_diffuse, s_diffuse, in.tex_coords));
}";

//////////////////////////////////////
// BRIGHTNESS/CONTRAST COMPUTE PASS //
//////////////////////////////////////

pub struct ComputePassBrightnessContrast {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    workgroup_count: u32,

    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
}

impl TestCase for ComputePassBrightnessContrast {
    fn name() -> &'static str {
        "GPU brightness/contrast (compute pass)"
    }

    fn new(ctx: &Context, size: u64) -> Self {
        let shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(COMPUTE_PASS_BC_SHADER.into()),
            });

        // Create pipeline
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("CP B/C bind group layout"),
                    entries: &[
                        // input_buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    size * std::mem::size_of::<[f32; 4]>() as u64,
                                ),
                            },
                            count: None,
                        },
                        // output_buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(size * 16),
                            },
                            count: None,
                        },
                        // Params
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                // TODO: Report RustFmt bug:
                                min_binding_size: wgpu::BufferSize::new(
                                    // comment to force nice formatting
                                    std::mem::size_of::<BrightnessContrastParams>() as _,
                                ),
                            },
                            count: None,
                        },
                    ],
                });
        let compute_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("CP B/C compute pipeline"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("CP B/C compute pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &shader,
                    entry_point: "main",
                });

        // Create buffers
        let params_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CP B/C params buffer"),
                contents: bytemuck::bytes_of(&BrightnessContrastParams::default()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let input_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CP B/C input buffer"),
            size: size * std::mem::size_of::<[f32; 4]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CP B/C output buffer"),
            size: size * std::mem::size_of::<[f32; 4]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create bind group for the buffers
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CP B/C bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            compute_pipeline,
            bind_group,
            workgroup_count: ((size as f32) / 100.0).ceil() as u32,

            input_buffer,
            output_buffer,
            params_buffer,
        }
    }

    fn run(&mut self, ctx: &Context, is_baseline: bool) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        if !is_baseline {
            assert_eq!(self.workgroup_count % 1000, 0);
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CP B/C compute pass"),
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch(1000, self.workgroup_count / 1000, 1);
        }

        // Submit work to the GPU and wait for it to run
        ctx.queue.submit(Some(encoder.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }
}

const COMPUTE_PASS_BC_SHADER: &'static str = "
struct Params {
    slant: f32;
    darken: f32;
    lighten: f32;
};

struct Pixel {
    col: vec4<f32>;
};

struct Image {
    pixels: [[stride(16)]] array<Pixel>;
};

// Group 0 is always the input texture
[[group(0), binding(0)]] var<storage, read>       input_buffer : Image;
[[group(0), binding(1)]] var<storage, read_write> output_buffer: Image;
[[group(0), binding(2)]] var<uniform> params: Params;


fn modify_color(col: vec4<f32>) -> vec4<f32> {
    var contrasted: vec3<f32> = (col.rgb - 0.5) * params.slant + 0.5;
    var darkened: vec3<f32> = contrasted * params.darken;
    var lightened: vec3<f32> = darkened * (1.0 - params.lighten) + params.lighten;
    return vec4<f32>(lightened, col.a);
}

[[stage(compute), workgroup_size(100)]]
fn main([[builtin(global_invocation_id)]] invok_id: vec3<u32>) {
    var pix_index = invok_id.x + invok_id.y * u32(1000);
    if (pix_index >= arrayLength(&input_buffer.pixels)) {
        return;
    }

    output_buffer.pixels[pix_index].col = modify_color(input_buffer.pixels[pix_index].col);
}";
