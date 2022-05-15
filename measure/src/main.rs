use std::collections::BTreeMap;

use serde::Serialize;
use tests::{Measurement, TestCase};

mod tests;
mod utils;

fn main() {
    // Initialise everything
    env_logger::init();
    let ctx = Context::new();
    println!("Testing on {}", ctx.adapter.get_info().name);
    // Run tests
    let mut results = Results::new(&ctx.adapter.get_info());
    results.take_measurements::<tests::RenderPassBrightnessContrast>(&ctx, 1_000_000, 10_000_000);
    results.take_measurements::<tests::ComputePassBrightnessContrast>(&ctx, 1_000_000, 8_000_000);
    results.take_measurements::<tests::CpuBrightnessContrastBytes>(&ctx, 1_000_000, 10_000_000);
    results.take_measurements::<tests::CpuBrightnessContrast>(&ctx, 1_000_000, 10_000_000);
    // results.take_measurements::<tests::CpuInvert>(&ctx, 1_000_000, 10_000_000);
    // results.take_measurements::<tests::BufGpuToCpu>(&ctx, 10_000_000, 100_000_000);
    // results.take_measurements::<tests::BufCpuToGpu>(&ctx, 10_000_000, 100_000_000);

    // Save tests to JSON
    let json = serde_json::to_string(&results).unwrap();
    let json = goldilocks_json_fmt::format(&json).unwrap();
    println!("{}", json);
    std::fs::write("results.json", json.as_bytes()).unwrap();
}

/// The results to be serialized to a JSON file
#[derive(Debug, Serialize)]
struct Results {
    gpu_name: String,
    backend: String,

    measurements: BTreeMap<String, Vec<Measurement>>,
}

impl Results {
    fn new(adapter_info: &wgpu::AdapterInfo) -> Self {
        Self {
            gpu_name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            measurements: BTreeMap::new(),
        }
    }

    fn take_measurements<T: TestCase>(&mut self, ctx: &Context, size_step: u64, size_max: u64) {
        self.measurements.insert(
            T::name().to_owned(),
            tests::take_measurements::<T>(ctx, size_step, size_max),
        );
    }
}

/// Immutable context for all the test cases
pub struct Context {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Context {
    fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        Self {
            device,
            queue,
            adapter,
        }
    }
}
