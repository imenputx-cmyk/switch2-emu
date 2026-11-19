//! A Vulkan backend for emu-gpu. Guest surfaces become Vulkan images:
//! uploaded from guest memory on first use in a submission, rendered to
//! on the host GPU, and written back to guest memory at `flush`, so the
//! guest sees the same memory as with the software backend.
//!
//! For now every flush writes back and drops all images, so guest CPU
//! writes between submissions are always picked up. Keeping images alive
//! across submissions needs tracking of guest writes, which comes later.
//!
//! Requires Vulkan 1.3 (dynamic rendering).

use ash::vk;
use emu_gpu::backend::{Backend, GuestMemory, Rect, SoftwareBackend, Surface};
use emu_gpu::surface::ColorFormat;
use std::collections::HashMap;
use std::ffi::CStr;

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

    unsafe fn create_buffer(&self, size: usize) -> Result<Buffer, String> {
        let buffer = self
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
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

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
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

    fn draw(&mut self, g: &mut GuestMemory, d: &emu_gpu::draw::Draw) -> Result<(), String> {
        // Shaders are not translated to SPIR-V yet: draw in software on
        // up-to-date guest memory.
        self.flush(g)?;
        self.software.draw(g, d)
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
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
