//! A Vulkan backend for emu-gpu. Guest surfaces become Vulkan images:
//! uploaded from guest memory on first use in a submission, rendered to
//! on the host GPU, and written back to guest memory at `flush`, so the
//! guest sees the same memory as with the software backend.
//!
//! For now every flush writes back and drops all images, so guest CPU
//! writes between submissions are always picked up. Keeping images alive
//! across submissions needs tracking of guest writes, which comes later.
//!
//! Draws translate the guest's shaders to SPIR-V and run as Vulkan
//! pipelines (cached by their shaders and state); anything not
//! translated yet falls back to the software renderer.
//!
//! Requires Vulkan 1.3 (dynamic rendering).

use ash::vk;
use emu_gpu::backend::{Backend, GuestMemory, Rect, SoftwareBackend, Surface};
use emu_gpu::draw::{topology, Draw};
use emu_gpu::shader::spirv;
use emu_gpu::surface::ColorFormat;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::CStr;
use std::hash::{Hash, Hasher};

fn vk_format(f: ColorFormat) -> vk::Format {
    match f {
        ColorFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        ColorFormat::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        ColorFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        ColorFormat::A2B10G10R10Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        ColorFormat::Rgba8Unorm | ColorFormat::Rgbx8Unorm => vk::Format::R8G8B8A8_UNORM,
        ColorFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        ColorFormat::B5G6R5Unorm => vk::Format::R5G6B5_UNORM_PACK16,
    }
}

fn err(what: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |e| format!("{what}: {e}")
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    size: usize,
}

/// A guest surface on the host GPU.
struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// Staging memory for upload and download.
    staging: Buffer,
}

pub struct VulkanBackend {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    /// Name of the physical device, for logs.
    pub device_name: String,
    /// Is `cmd` recording?
    recording: bool,
    images: HashMap<(usize, Surface), Image>,
    software: SoftwareBackend,
    /// Descriptor set and pipeline layouts by the constant buffers the
    /// vertex and pixel shaders read (bit masks).
    layouts: HashMap<(u32, u32), (vk::DescriptorSetLayout, vk::PipelineLayout)>,
    /// Descriptor sets of the current batch come from here; reset at flush.
    descriptor_pool: vk::DescriptorPool,
    /// Pipelines by a hash of their shaders and fixed state.
    pipelines: HashMap<u64, (vk::Pipeline, [vk::ShaderModule; 2])>,
    /// Vertex buffers of the current batch, freed at flush.
    transient: Vec<Buffer>,
    /// Draws done on the GPU / in the software fallback (for tests).
    pub gpu_draws: u64,
    pub fallback_draws: u64,
}

impl VulkanBackend {
    /// Opens the first Vulkan 1.3 device with a graphics queue,
    /// preferring a discrete GPU.
    pub fn new() -> Result<VulkanBackend, String> {
        // SAFETY: loads the system's Vulkan loader; the function pointers
        // it returns are used according to the Vulkan specification.
        unsafe { Self::create() }
    }

    unsafe fn create() -> Result<VulkanBackend, String> {
        let entry = ash::Entry::load().map_err(|e| format!("no Vulkan loader: {e}"))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"switch2-emu")
            .api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .map_err(err("vkCreateInstance"))?;
        let mut best: Option<(u32, vk::PhysicalDevice, u32, String)> = None;
        for pd in instance
            .enumerate_physical_devices()
            .map_err(err("devices"))?
        {
            let props = instance.get_physical_device_properties(pd);
            if props.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            let Some(qf) = instance
                .get_physical_device_queue_family_properties(pd)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            else {
                continue;
            };
            let rank = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 2,
            };
            if best.as_ref().is_none_or(|(r, ..)| rank < *r) {
                let name = CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                best = Some((rank, pd, qf as u32, name));
            }
        }
        let Some((_, pd, qf, device_name)) = best else {
            instance.destroy_instance(None);
            return Err("no Vulkan 1.3 device with a graphics queue".into());
        };
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf)
            .queue_priorities(&priorities)];
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let device = instance
            .create_device(
                pd,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .push_next(&mut v13),
                None,
            )
            .map_err(err("vkCreateDevice"))?;
        let queue = device.get_device_queue(qf, 0);
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qf)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(err("command pool"))?;
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )
            .map_err(err("command buffer"))?[0];
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(err("fence"))?;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 8192,
        }];
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1024)
                    .pool_sizes(&pool_sizes),
                None,
            )
            .map_err(err("descriptor pool"))?;
        Ok(VulkanBackend {
            memory_props: instance.get_physical_device_memory_properties(pd),
            _entry: entry,
            instance,
            device,
            queue,
            pool,
            cmd,
            fence,
            device_name,
            recording: false,
            images: HashMap::new(),
            software: SoftwareBackend,
            layouts: HashMap::new(),
            descriptor_pool,
            pipelines: HashMap::new(),
            transient: Vec::new(),
            gpu_draws: 0,
            fallback_draws: 0,
        })
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Result<u32, String> {
        (0..self.memory_props.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && self.memory_props.memory_types[i as usize]
                        .property_flags
                        .contains(want)
            })
            .ok_or_else(|| format!("no memory type with {want:?}"))
    }

    unsafe fn begin(&mut self) -> Result<(), String> {
        if !self.recording {
            self.device
                .begin_command_buffer(
                    self.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(err("begin"))?;
            self.recording = true;
        }
        Ok(())
    }

    unsafe fn barrier(&self, image: vk::Image, from: vk::ImageLayout, to: vk::ImageLayout) {
        let b = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(from)
            .new_layout(to)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        let barriers = [b];
        self.device.cmd_pipeline_barrier2(
            self.cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&barriers),
        );
    }

    /// Orders color writes of earlier rendering before the next one.
    unsafe fn color_barrier(&self) {
        let mb = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )];
        self.device.cmd_pipeline_barrier2(
            self.cmd,
            &vk::DependencyInfo::default().memory_barriers(&mb),
        );
    }

    unsafe fn create_buffer(&self, size: usize) -> Result<Buffer, String> {
        self.create_buffer_with(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        )
    }

    unsafe fn create_buffer_with(
        &self,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer, String> {
        let buffer = self
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(usage),
                None,
            )
            .map_err(err("buffer"))?;
        let req = self.device.get_buffer_memory_requirements(buffer);
        let ty = self.memory_type(
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            )
            .map_err(err("buffer memory"))?;
        self.device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(err("bind buffer"))?;
        let ptr = self
            .device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .map_err(err("map"))? as *mut u8;
        Ok(Buffer {
            buffer,
            memory,
            ptr,
            size,
        })
    }

    /// The image for a surface, created and filled from guest memory if
    /// it is not cached. Left in COLOR_ATTACHMENT_OPTIMAL.
    unsafe fn image(&mut self, g: &GuestMemory, s: &Surface) -> Result<vk::ImageView, String> {
        if let Some(img) = self.images.get(&(g.asid, *s)) {
            return Ok(img.view);
        }
        let format = vk_format(s.format);
        let image = self
            .device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: s.width,
                        height: s.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST,
                    ),
                None,
            )
            .map_err(err("image"))?;
        let req = self.device.get_image_memory_requirements(image);
        let ty = self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            )
            .map_err(err("image memory"))?;
        self.device
            .bind_image_memory(image, memory, 0)
            .map_err(err("bind image"))?;
        let view = self
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
            .map_err(err("image view"))?;

        // Upload the current guest contents.
        let linear = s
            .read_linear(g)
            .map_err(|f| format!("reading surface: {f:?}"))?;
        let staging = self.create_buffer(linear.len())?;
        std::ptr::copy_nonoverlapping(linear.as_ptr(), staging.ptr, linear.len());
        self.begin()?;
        self.barrier(
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        self.device.cmd_copy_buffer_to_image(
            self.cmd,
            staging.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[self.copy_region(s)],
        );
        self.barrier(
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        self.images.insert(
            (g.asid, *s),
            Image {
                image,
                memory,
                view,
                staging,
            },
        );
        Ok(view)
    }

    fn copy_region(&self, s: &Surface) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: s.width,
                height: s.height,
                depth: 1,
            })
    }

    unsafe fn destroy_image(&self, img: Image) {
        self.device.destroy_image_view(img.view, None);
        self.device.destroy_image(img.image, None);
        self.device.free_memory(img.memory, None);
        self.device.destroy_buffer(img.staging.buffer, None);
        self.device.free_memory(img.staging.memory, None);
    }

    unsafe fn flush_images(&mut self, g: &mut GuestMemory) -> Result<(), String> {
        if !self.recording {
            return Ok(());
        }
        let keys: Vec<(usize, Surface)> = self.images.keys().copied().collect();
        for key in &keys {
            let img = &self.images[key];
            self.barrier(
                img.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            self.device.cmd_copy_image_to_buffer(
                self.cmd,
                img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                img.staging.buffer,
                &[self.copy_region(&key.1)],
            );
        }
        let mb = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        let mbs = [mb];
        self.device.cmd_pipeline_barrier2(
            self.cmd,
            &vk::DependencyInfo::default().memory_barriers(&mbs),
        );
        self.device
            .end_command_buffer(self.cmd)
            .map_err(err("end"))?;
        self.recording = false;
        let cmds = [self.cmd];
        self.device
            .queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                self.fence,
            )
            .map_err(err("submit"))?;
        self.device
            .wait_for_fences(&[self.fence], true, u64::MAX)
            .map_err(err("wait"))?;
        self.device
            .reset_fences(&[self.fence])
            .map_err(err("reset fence"))?;
        self.device
            .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())
            .map_err(err("reset"))?;
        for b in std::mem::take(&mut self.transient) {
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
        self.device
            .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .map_err(err("reset descriptor pool"))?;
        let mut result = Ok(());
        for key in keys {
            let img = self.images.remove(&key).unwrap();
            let data = std::slice::from_raw_parts(img.staging.ptr, img.staging.size);
            if key.0 == g.asid {
                if let Err(f) = key.1.write_linear(g, data) {
                    result = Err(format!("writing surface back: {f:?}"));
                }
            }
            self.destroy_image(img);
        }
        result
    }
}

/// What a draw needs on Vulkan, worked out before touching the device.
struct Plan {
    vs: spirv::Translated,
    ps: spirv::Translated,
    topology: vk::PrimitiveTopology,
    /// (location, stream, offset, format)
    attributes: Vec<(u32, u32, u32, vk::Format)>,
    /// (stream, stride)
    bindings: Vec<(u32, u32)>,
    viewport: vk::Viewport,
}

fn vertex_format(widths: u32, kind: u32) -> Option<vk::Format> {
    Some(match (widths, kind) {
        (0x01, 7) => vk::Format::R32G32B32A32_SFLOAT,
        (0x02, 7) => vk::Format::R32G32B32_SFLOAT,
        (0x04, 7) => vk::Format::R32G32_SFLOAT,
        (0x12, 7) => vk::Format::R32_SFLOAT,
        (0x03, 7) => vk::Format::R16G16B16A16_SFLOAT,
        (0x0A, 2) => vk::Format::R8G8B8A8_UNORM,
        (0x03, 2) => vk::Format::R16G16B16A16_UNORM,
        _ => return None,
    })
}

fn plan(d: &Draw) -> Result<Plan, String> {
    let header = |stage| emu_gpu::shader::Header {
        stage,
        omap_targets: [0; 8],
        omap_depth: false,
    };
    let vs = spirv::translate(&d.vs, &header(emu_gpu::shader::Stage::Vertex))?;
    let ps = spirv::translate(&d.ps, &d.ps_header)?;
    let topology = match d.topology {
        topology::TRIANGLES => vk::PrimitiveTopology::TRIANGLE_LIST,
        topology::TRIANGLE_STRIP => vk::PrimitiveTopology::TRIANGLE_STRIP,
        topology::TRIANGLE_FAN => vk::PrimitiveTopology::TRIANGLE_FAN,
        t => return Err(format!("topology {t}")),
    };
    // Attributes the vertex shader reads: generic i is location i.
    let mut read = [false; 32];
    for i in &d.vs {
        if let emu_gpu::shader::decode::Inst::Ald { addr, count, .. } = *i {
            for k in 0..count {
                let a = addr + 4 * k;
                if (0x80..0x280).contains(&a) {
                    read[((a - 0x80) / 16) as usize] = true;
                }
            }
        }
    }
    let mut attributes = Vec::new();
    let mut bindings: Vec<(u32, u32)> = Vec::new();
    for (i, a) in d.attributes.iter().enumerate() {
        if !read[i] {
            continue;
        }
        let s = d
            .streams
            .get(a.stream as usize)
            .copied()
            .unwrap_or_default();
        if a.constant || !s.enabled || a.swap_rb {
            return Err(format!("attribute {i} not fetched from a buffer"));
        }
        let f = vertex_format(a.widths, a.kind)
            .ok_or_else(|| format!("attribute format {:#x}/{}", a.widths, a.kind))?;
        attributes.push((i as u32, a.stream, a.offset, f));
        if !bindings.iter().any(|b| b.0 == a.stream) {
            bindings.push((a.stream, s.stride));
        }
    }
    let v = &d.viewport;
    if v.scale[0] <= 0.0 {
        return Err("viewport with negative width".into());
    }
    let (near, far) = (v.offset[2], v.offset[2] + v.scale[2]);
    if !(0.0..=1.0).contains(&near) || !(0.0..=1.0).contains(&far) {
        return Err("viewport depth range outside 0..1".into());
    }
    let viewport = vk::Viewport {
        x: v.offset[0] - v.scale[0],
        y: v.offset[1] - v.scale[1],
        width: 2.0 * v.scale[0],
        height: 2.0 * v.scale[1],
        min_depth: near,
        max_depth: far,
    };
    Ok(Plan {
        vs,
        ps,
        topology,
        attributes,
        bindings,
        viewport,
    })
}

impl VulkanBackend {
    unsafe fn shader_module(&self, words: &[u32]) -> Result<vk::ShaderModule, String> {
        self.device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
            .map_err(err("shader module"))
    }

    /// Layouts for shaders reading the constant buffers in the masks.
    unsafe fn layout(
        &mut self,
        vs: u32,
        ps: u32,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout), String> {
        if let Some(&l) = self.layouts.get(&(vs, ps)) {
            return Ok(l);
        }
        let mut bindings = Vec::new();
        for (mask, stage, flags) in [
            (
                vs,
                emu_gpu::shader::Stage::Vertex,
                vk::ShaderStageFlags::VERTEX,
            ),
            (
                ps,
                emu_gpu::shader::Stage::Pixel,
                vk::ShaderStageFlags::FRAGMENT,
            ),
        ] {
            for i in (0..32).filter(|i| mask & (1 << i) != 0) {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(spirv::cbuf_binding(stage, i))
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(flags),
                );
            }
        }
        let dsl = self
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(err("descriptor set layout"))?;
        let sets = [dsl];
        let layout = self
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets),
                None,
            )
            .map_err(err("pipeline layout"))?;
        self.layouts.insert((vs, ps), (dsl, layout));
        Ok((dsl, layout))
    }

    /// A host-visible buffer with `data`, freed at the next flush.
    unsafe fn transient_buffer(
        &mut self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<vk::Buffer, String> {
        let b = self.create_buffer_with(data.len().max(16), usage)?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), b.ptr, data.len());
        let buffer = b.buffer;
        self.transient.push(b);
        Ok(buffer)
    }

    unsafe fn pipeline(&mut self, p: &Plan, format: vk::Format) -> Result<vk::Pipeline, String> {
        let mut h = DefaultHasher::new();
        (
            &p.vs.words,
            &p.ps.words,
            p.topology.as_raw(),
            format.as_raw(),
        )
            .hash(&mut h);
        for a in &p.attributes {
            (a.0, a.1, a.2, a.3.as_raw()).hash(&mut h);
        }
        p.bindings.hash(&mut h);
        let key = h.finish();
        let (_, layout) = self.layout(p.vs.cbufs, p.ps.cbufs)?;
        if let Some((pipe, _)) = self.pipelines.get(&key) {
            return Ok(*pipe);
        }
        let vs = self.shader_module(&p.vs.words)?;
        let ps = self.shader_module(&p.ps.words)?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(ps)
                .name(c"main"),
        ];
        let bindings: Vec<_> = p
            .bindings
            .iter()
            .map(|&(stream, stride)| {
                vk::VertexInputBindingDescription::default()
                    .binding(stream)
                    .stride(stride)
                    .input_rate(vk::VertexInputRate::VERTEX)
            })
            .collect();
        let attributes: Vec<_> = p
            .attributes
            .iter()
            .map(|&(location, stream, offset, format)| {
                vk::VertexInputAttributeDescription::default()
                    .location(location)
                    .binding(stream)
                    .format(format)
                    .offset(offset)
            })
            .collect();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(p.topology);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [format];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        let pipe = self
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, e)| format!("pipeline: {e}"))?[0];
        self.pipelines.insert(key, (pipe, [vs, ps]));
        Ok(pipe)
    }

    unsafe fn draw_gpu(
        &mut self,
        g: &mut GuestMemory,
        d: &Draw,
        target: &Surface,
        p: Plan,
    ) -> Result<(), String> {
        let view = self.image(g, target)?;
        let pipe = self.pipeline(&p, vk_format(target.format))?;
        // Indices (as u32) of all runs, and how many vertices are used.
        let mut indices = Vec::new();
        let mut runs = Vec::new(); // (first vertex or index, count)
        let last = match &d.index {
            Some(ib) => {
                let mut max = 0i64;
                for &(first, count) in &d.ranges {
                    let ix = emu_gpu::draw::read_indices(g, ib, first, count)?;
                    max = ix
                        .iter()
                        .map(|&i| i as i64 + i64::from(ib.base_vertex) + 1)
                        .fold(max, i64::max);
                    runs.push((indices.len() as u32, count));
                    indices.extend(ix);
                }
                max.max(0) as u32
            }
            None => {
                runs.extend(&d.ranges);
                d.ranges.iter().map(|&(f, c)| f + c).max().unwrap_or(0)
            }
        };
        // Vertex data: each stream from its start to the last vertex used.
        let mut buffers = Vec::new();
        for &(stream, stride) in &p.bindings {
            let s = d.streams[stream as usize];
            let size = (u64::from(last) * u64::from(stride)).max(16) as usize;
            let mut data = vec![0u8; size];
            g.read(s.addr, &mut data)
                .map_err(|f| format!("vertex buffer: {f:?}"))?;
            buffers.push((
                stream,
                self.transient_buffer(&data, vk::BufferUsageFlags::VERTEX_BUFFER)?,
            ));
        }
        let index_buffer = if d.index.is_some() {
            let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
            Some(self.transient_buffer(&bytes, vk::BufferUsageFlags::INDEX_BUFFER)?)
        } else {
            None
        };
        // Constant buffers: uploaded and bound in one descriptor set.
        let (dsl, layout) = self.layout(p.vs.cbufs, p.ps.cbufs)?;
        let mut writes = Vec::new();
        for (mask, bound, stage) in [
            (p.vs.cbufs, &d.vs_cbufs, emu_gpu::shader::Stage::Vertex),
            (p.ps.cbufs, &d.ps_cbufs, emu_gpu::shader::Stage::Pixel),
        ] {
            for i in (0..32u32).filter(|i| mask & (1 << i) != 0) {
                let data = match bound.get(i as usize).copied().flatten() {
                    Some(cb) => {
                        let mut data = vec![0u8; (cb.size as usize).next_multiple_of(16).max(16)];
                        g.read(cb.addr, &mut data[..cb.size as usize])
                            .map_err(|f| format!("constant buffer: {f:?}"))?;
                        data
                    }
                    None => vec![0u8; 16],
                };
                let buffer = self.transient_buffer(&data, vk::BufferUsageFlags::UNIFORM_BUFFER)?;
                writes.push((spirv::cbuf_binding(stage, i), buffer, data.len() as u64));
            }
        }
        let set = if writes.is_empty() {
            None
        } else {
            let layouts = [dsl];
            let set = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&layouts),
                )
                .map_err(err("descriptor set"))?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = writes
                .iter()
                .map(|&(_, buffer, range)| {
                    [vk::DescriptorBufferInfo {
                        buffer,
                        offset: 0,
                        range,
                    }]
                })
                .collect();
            let w: Vec<_> = writes
                .iter()
                .zip(&infos)
                .map(|(&(binding, ..), info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            self.device.update_descriptor_sets(&w, &[]);
            Some(set)
        };
        self.begin()?;
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: target.width,
                height: target.height,
            },
        };
        self.color_barrier();
        self.device.cmd_begin_rendering(
            self.cmd,
            &vk::RenderingInfo::default()
                .render_area(area)
                .layer_count(1)
                .color_attachments(&attachments),
        );
        let (x0, y0) = (d.clip.x0.min(target.width), d.clip.y0.min(target.height));
        let (x1, y1) = (
            d.clip.x1.min(target.width).max(x0),
            d.clip.y1.min(target.height).max(y0),
        );
        let scissor = vk::Rect2D {
            offset: vk::Offset2D {
                x: x0 as i32,
                y: y0 as i32,
            },
            extent: vk::Extent2D {
                width: x1 - x0,
                height: y1 - y0,
            },
        };
        self.device.cmd_set_viewport(self.cmd, 0, &[p.viewport]);
        self.device.cmd_set_scissor(self.cmd, 0, &[scissor]);
        self.device
            .cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::GRAPHICS, pipe);
        for (stream, buffer) in buffers {
            self.device
                .cmd_bind_vertex_buffers(self.cmd, stream, &[buffer], &[0]);
        }
        if let Some(set) = set {
            self.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[set],
                &[],
            );
        }
        match (index_buffer, &d.index) {
            (Some(ib), Some(desc)) => {
                self.device
                    .cmd_bind_index_buffer(self.cmd, ib, 0, vk::IndexType::UINT32);
                for &(first, count) in &runs {
                    self.device
                        .cmd_draw_indexed(self.cmd, count, 1, first, desc.base_vertex, 0);
                }
            }
            _ => {
                for &(first, count) in &runs {
                    self.device.cmd_draw(self.cmd, count, 1, first, 0);
                }
            }
        }
        self.device.cmd_end_rendering(self.cmd);
        Ok(())
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clear(
        &mut self,
        g: &mut GuestMemory,
        s: &Surface,
        r: Rect,
        mut color: [f32; 4],
        mask: [bool; 4],
    ) -> Result<(), String> {
        if !mask.iter().all(|&m| m) {
            // vkCmdClearAttachments ignores write masks: do it in
            // software on up-to-date guest memory.
            self.flush(g)?;
            return self.software.clear(g, s, r, color, mask);
        }
        if s.format == ColorFormat::Rgbx8Unorm {
            color[3] = 1.0; // the X byte is written as 0xFF
        }
        // SAFETY: Vulkan calls on objects this backend owns, recorded into
        // its single command buffer and submitted in `flush`.
        unsafe {
            let view = self.image(g, s)?;
            self.begin()?;
            let attachments = [vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)];
            let area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: s.width,
                    height: s.height,
                },
            };
            self.color_barrier();
            self.device.cmd_begin_rendering(
                self.cmd,
                &vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .color_attachments(&attachments),
            );
            let clear = vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: vk::ClearValue {
                    color: vk::ClearColorValue { float32: color },
                },
            };
            let rect = vk::ClearRect {
                rect: vk::Rect2D {
                    offset: vk::Offset2D {
                        x: r.x0 as i32,
                        y: r.y0 as i32,
                    },
                    extent: vk::Extent2D {
                        width: r.x1 - r.x0,
                        height: r.y1 - r.y0,
                    },
                },
                base_array_layer: 0,
                layer_count: 1,
            };
            self.device
                .cmd_clear_attachments(self.cmd, &[clear], &[rect]);
            self.device.cmd_end_rendering(self.cmd);
        }
        Ok(())
    }

    fn draw(&mut self, g: &mut GuestMemory, d: &Draw) -> Result<(), String> {
        let Some(target) = d.target else {
            return Ok(());
        };
        let plan = match plan(d) {
            Ok(p) => p,
            Err(why) => {
                // Not translatable yet: draw in software on up-to-date
                // guest memory.
                let _ = why;
                self.fallback_draws += 1;
                self.flush(g)?;
                return self.software.draw(g, d);
            }
        };
        // SAFETY: see `clear`.
        unsafe { self.draw_gpu(g, d, &target, plan) }?;
        self.gpu_draws += 1;
        Ok(())
    }

    fn flush(&mut self, g: &mut GuestMemory) -> Result<(), String> {
        // SAFETY: see `clear`.
        unsafe { self.flush_images(g) }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        // SAFETY: waits for the device, then destroys what it created.
        unsafe {
            let _ = self.device.device_wait_idle();
            for (_, img) in std::mem::take(&mut self.images) {
                self.destroy_image(img);
            }
            for b in std::mem::take(&mut self.transient) {
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
            for (_, (pipe, modules)) in std::mem::take(&mut self.pipelines) {
                self.device.destroy_pipeline(pipe, None);
                for m in modules {
                    self.device.destroy_shader_module(m, None);
                }
            }
            for (_, (dsl, layout)) in std::mem::take(&mut self.layouts) {
                self.device.destroy_pipeline_layout(layout, None);
                self.device.destroy_descriptor_set_layout(dsl, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
