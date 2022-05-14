use std::time::{Duration, Instant};

use kneasle_ringing_utils::BigNumInt;
use serde::Serialize;
use wgpu::util::DeviceExt;

use crate::Context;

const DURATION_PER_TEST: Duration = Duration::from_millis(100);

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

        let (baseline_duration, baseline_iters) = run_partial_case(&mut case, ctx, true);
        let (full_duration, full_iters) = run_partial_case(&mut case, ctx, false);

        let duration = full_duration.saturating_sub(baseline_duration);
        println!(
            "{} @ {}: {:?} ({}+{} iters)",
            T::name(),
            BigNumInt(size as usize),
            duration,
            baseline_iters,
            full_iters
        );
        Self {
            size,
            duration_secs: duration.as_secs_f64(),
        }
    }
}

/// Run either a baseline or full test
fn run_partial_case(case: &mut impl TestCase, ctx: &Context, is_baseline: bool) -> (Duration, u32) {
    let start = Instant::now();
    let mut iters = 0u32;
    while start.elapsed() < DURATION_PER_TEST {
        for _ in 0..10 {
            case.run(ctx, is_baseline);
            iters += 1;
        }
    }
    (start.elapsed() / iters, iters)
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
