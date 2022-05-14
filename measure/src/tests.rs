use std::{
    io::Write,
    time::{Duration, Instant},
};

use kneasle_ringing_utils::BigNumInt;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::Serialize;
use wgpu::util::DeviceExt;

use crate::Context;

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

pub struct CpuBrightnessContrast {
    input_buffer: Vec<[f32; 4]>,
    output_buffer: Vec<[f32; 4]>,

    slant: f32,
    lighten: f32,
    darken: f32,
}

impl TestCase for CpuBrightnessContrast {
    fn name() -> &'static str {
        "CPU brightness/contrast (f32)"
    }

    fn new(_ctx: &Context, size: u64) -> Self {
        Self {
            input_buffer: vec![[0.0; 4]; size as usize],
            output_buffer: vec![[0.0; 4]; size as usize],

            slant: 1.5,
            darken: 1.0,
            lighten: 0.5,
        }
    }

    fn run(&mut self, _ctx: &Context, is_baseline: bool) {
        if is_baseline {
            return;
        }

        let slant = self.slant;
        let darken = self.darken;
        let lighten = self.lighten;
        self.input_buffer
            .par_iter_mut()
            .zip(self.output_buffer.par_iter_mut())
            .for_each(|([r_in, g_in, b_in, a_in], [r_out, g_out, b_out, a_out])| {
                let contrasted_r = (*r_in + 0.5) * slant + 0.5;
                let contrasted_g = (*g_in + 0.5) * slant + 0.5;
                let contrasted_b = (*b_in + 0.5) * slant + 0.5;

                *r_out = (contrasted_r * darken) * (1.0 - lighten) + lighten;
                *g_out = (contrasted_g * darken) * (1.0 - lighten) + lighten;
                *b_out = (contrasted_b * darken) * (1.0 - lighten) + lighten;
                *a_out = *a_in;
            });
    }
}

pub struct CpuBrightnessContrastBytes {
    input_buffer: Vec<[u8; 4]>,
    output_buffer: Vec<[u8; 4]>,

    slant: f32,
    lighten: f32,
    darken: f32,
}

impl TestCase for CpuBrightnessContrastBytes {
    fn name() -> &'static str {
        "CPU brightness/contrast (u8)"
    }

    fn new(_ctx: &Context, size: u64) -> Self {
        Self {
            input_buffer: vec![[0; 4]; size as usize],
            output_buffer: vec![[0; 4]; size as usize],

            slant: 1.5,
            darken: 1.0,
            lighten: 0.5,
        }
    }

    fn run(&mut self, _ctx: &Context, is_baseline: bool) {
        if is_baseline {
            return;
        }

        let slant = self.slant;
        let darken = self.darken;
        let lighten = self.lighten;
        self.input_buffer
            .par_iter_mut()
            .zip(self.output_buffer.par_iter_mut())
            .for_each(|([r_in, g_in, b_in, a_in], [r_out, g_out, b_out, a_out])| {
                let contrasted_r = ((*r_in as f32) / 255.0 + 0.5) * slant + 0.5;
                let contrasted_g = ((*g_in as f32) / 255.0 + 0.5) * slant + 0.5;
                let contrasted_b = ((*b_in as f32) / 255.0 + 0.5) * slant + 0.5;

                *r_out = (((contrasted_r * darken) * (1.0 - lighten) + lighten) * 255.0) as u8;
                *g_out = (((contrasted_g * darken) * (1.0 - lighten) + lighten) * 255.0) as u8;
                *b_out = (((contrasted_b * darken) * (1.0 - lighten) + lighten) * 255.0) as u8;
                *a_out = *a_in;
            });
    }
}
