/// A singleton for every editor, which holds data used by the rest of the library.
#[derive(Debug)]
pub struct Ctx {
    pub plugins: PlugVec<Plugin>,

    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Ctx {
    pub fn new(plugins: impl IntoIterator<Item = Plugin>) -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();

        Self {
            plugins: plugins.into_iter().collect(),

            instance,
            adapter,
            device,
            queue,
        }
    }
}

/// A runtime-loaded (image effect) `Plugin`
#[derive(Debug, Clone)]
pub struct Plugin {
    pub name: String,
}

index_vec::define_index_type! { pub struct PlugIdx = usize; }
pub type PlugVec<T> = index_vec::IndexVec<PlugIdx, T>;
