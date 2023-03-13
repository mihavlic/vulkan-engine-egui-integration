use crate::utils::EguiDescriptorSetAllocator;
use egui::{Color32, TexturesDelta};
use graph::{
    device::{staging::ImageWrite, submission, Device},
    object,
    smallvec::smallvec,
    storage::DefaultAhashRandomstate,
    util::ffi_ptr::AsFFiPtr,
};
use pumice::{util::ObjectHandle, vk};
use pumice_vma as vma;
use slice_group_by::GroupBy;
use std::{collections::HashMap, ffi::c_void};

const VERTEX_BUFFER_COUNT: usize = 1024 * 1024 * 4;
const INDEX_BUFFER_COUNT: usize = 1024 * 1024 * 2;

const VERTEX_BUFFER_BYTES: u64 =
    (VERTEX_BUFFER_COUNT * std::mem::size_of::<egui::epaint::Vertex>()) as u64;
const INDEX_BUFFER_BYTES: u64 = (INDEX_BUFFER_COUNT * 4) as u64;

#[repr(C)]
pub struct PushConstants {
    screen_size: [f32; 2],
    need_srgb_conv: i32,
}

struct TextureImage {
    image: object::Image,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    options: egui::TextureOptions,
}

enum RendererEvent {
    SetUnused(vk::DescriptorSet),
}

pub struct Renderer {
    need_srgb_conv: bool,
    resolve_attachment: bool,

    queue: submission::Queue,
    render_pass: object::RenderPass,
    framebuffer: Option<object::Framebuffer>,

    set_allocator: EguiDescriptorSetAllocator,
    desc_layout: object::DescriptorSetLayout,
    pipeline: object::ConcreteGraphicsPipeline,

    vertex_buffer: (object::Buffer, *mut egui::epaint::Vertex),
    index_buffer: (object::Buffer, *mut u32),

    // sender: mpsc::Sender<RendererEvent>,
    // receiver: mpsc::Receiver<RendererEvent>,
    pending_retired_sets: Vec<vk::DescriptorSet>,
    pending_retired_textures: Vec<egui::TextureId>,

    samplers: HashMap<egui::TextureOptions, object::Sampler, DefaultAhashRandomstate>,
    texture_images: HashMap<egui::TextureId, TextureImage, DefaultAhashRandomstate>,
    next_native_tex_id: u64,
}

impl Renderer {
    pub unsafe fn new_with_render_pass(
        queue: submission::Queue,
        format_is_srgb: bool,
        format: vk::Format,

        samples: vk::SampleCountFlags,

        color_load_op: vk::AttachmentLoadOp,
        color_store_op: vk::AttachmentStoreOp,

        color_src_layout: vk::ImageLayout,
        color_src_stages: vk::PipelineStageFlags,
        color_src_access: vk::AccessFlags,
        color_final_layout: vk::ImageLayout,

        resolve_enable: bool,

        resolve_load_op: vk::AttachmentLoadOp,
        resolve_store_op: vk::AttachmentStoreOp,

        resolve_src_layout: vk::ImageLayout,
        resolve_src_stages: vk::PipelineStageFlags,
        resolve_src_access: vk::AccessFlags,
        resolve_final_layout: vk::ImageLayout,

        device: &Device,
    ) -> Renderer {
        if resolve_enable {
            assert!(samples != vk::SampleCountFlags::C1);
        }

        let render_pass = {
            let mut attachments = Vec::new();
            let mut dependencies = Vec::new();

            // color
            let color_ref = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };
            attachments.push(vk::AttachmentDescription {
                format,
                samples,
                load_op: color_load_op,
                store_op: color_store_op,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: color_src_layout,
                final_layout: color_final_layout,
                ..Default::default()
            });
            dependencies.push(vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: color_src_stages,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: color_src_access,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ..Default::default()
            });

            // color resolve
            let mut resolve_ref = None;
            if resolve_enable {
                resolve_ref = Some(vk::AttachmentReference {
                    attachment: 1,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                });
                attachments.push(vk::AttachmentDescription {
                    format,
                    samples: vk::SampleCountFlags::C1,
                    load_op: resolve_load_op,
                    store_op: resolve_store_op,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: resolve_src_layout,
                    final_layout: resolve_final_layout,
                    ..Default::default()
                });
                dependencies.push(vk::SubpassDependency {
                    src_subpass: vk::SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: resolve_src_stages,
                    dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access_mask: resolve_src_access,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    ..Default::default()
                });
            }

            let subpass = object::SubpassDescription {
                color_attachments: vec![color_ref],
                resolve_attachments: resolve_ref.into_iter().collect(),
                ..Default::default()
            };

            device
                .create_render_pass(object::RenderPassCreateInfo {
                    attachments,
                    subpasses: vec![subpass],
                    dependencies,
                    ..Default::default()
                })
                .unwrap()
        };

        let (pipeline, desc_layout) = {
            const VS_BYTES: &[u8] = include_bytes!("../shaders/vertex.spv");
            const FS_BYTES: &[u8] = include_bytes!("../shaders/fragment.spv");

            let vs = device
                .create_shader_module_spirv_unaligned(VS_BYTES)
                .unwrap();
            let fs = device
                .create_shader_module_spirv_unaligned(FS_BYTES)
                .unwrap();

            let set_layout = device
                .create_descriptor_set_layout(object::DescriptorSetLayoutCreateInfo {
                    flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                    bindings: vec![object::DescriptorBinding {
                        binding: 0,
                        count: 1,
                        kind: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        stages: vk::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: smallvec![],
                    }],
                })
                .unwrap();

            let layout = device
                .create_pipeline_layout(object::PipelineLayoutCreateInfo {
                    set_layouts: vec![set_layout.clone()],
                    push_constants: vec![vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        offset: 0,
                        size: std::mem::size_of::<PushConstants>() as u32,
                    }],
                })
                .unwrap();

            let info = object::GraphicsPipelineCreateInfo::builder()
                .stages([
                    object::PipelineStage {
                        flags: vk::PipelineShaderStageCreateFlags::empty(),
                        stage: vk::ShaderStageFlags::VERTEX,
                        module: vs,
                        name: "main".into(),
                        specialization_info: None,
                    },
                    object::PipelineStage {
                        flags: vk::PipelineShaderStageCreateFlags::empty(),
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        module: fs,
                        name: "main".into(),
                        specialization_info: None,
                    },
                ])
                .input_assembly(object::state::InputAssembly {
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    primitive_restart_enable: false,
                })
                .viewport(object::state::Viewport {
                    // the actual contents are ignored, it is just important to have one for each
                    viewports: smallvec![Default::default()],
                    scissors: smallvec![Default::default()],
                })
                .vertex_input(object::state::VertexInput {
                    vertex_bindings: [object::state::InputBinding {
                        binding: 0,
                        stride: std::mem::size_of::<egui::epaint::Vertex>() as u32,
                        input_rate: vk::VertexInputRate::VERTEX,
                    }]
                    .to_vec(),
                    vertex_attributes: [
                        object::state::InputAttribute {
                            location: 0,
                            binding: 0,
                            format: vk::Format::R32G32_SFLOAT,
                            offset: 0,
                        },
                        object::state::InputAttribute {
                            location: 0,
                            binding: 1,
                            format: vk::Format::R32G32_SFLOAT,
                            offset: 8,
                        },
                        object::state::InputAttribute {
                            location: 0,
                            binding: 2,
                            format: vk::Format::R8G8B8A8_UNORM,
                            offset: 16,
                        },
                    ]
                    .to_vec(),
                })
                .rasterization(object::state::Rasterization {
                    depth_clamp_enable: false,
                    rasterizer_discard_enable: false,
                    polygon_mode: vk::PolygonMode::FILL,
                    cull_mode: vk::CullModeFlags::NONE,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    line_width: 1.0,
                    ..Default::default()
                })
                .multisample(object::state::Multisample {
                    rasterization_samples: samples,
                    sample_shading_enable: false,
                    ..Default::default()
                })
                .depth_stencil(object::state::DepthStencil::default())
                .color_blend(object::state::ColorBlend {
                    attachments: vec![object::state::Attachment {
                        color_write_mask: vk::ColorComponentFlags::all(),
                        src_color_blend_factor: vk::BlendFactor::ONE,
                        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        ..Default::default()
                    }],
                    ..Default::default()
                })
                .dynamic_state([vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT])
                .render_pass(object::RenderPassMode::Normal {
                    subpass: 0,
                    render_pass: render_pass.clone(),
                })
                .layout(layout)
                .finish();

            (device.create_graphics_pipeline(info).unwrap(), set_layout)
        };

        let need_srgb_conv = !format_is_srgb;
        let vertex_buffer = Self::create_buffer(
            VERTEX_BUFFER_BYTES,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            device,
        );

        let index_buffer = Self::create_buffer(
            INDEX_BUFFER_BYTES,
            vk::BufferUsageFlags::INDEX_BUFFER,
            device,
        );

        Renderer {
            need_srgb_conv,
            resolve_attachment: resolve_enable,
            queue,
            render_pass,
            framebuffer: None,
            set_allocator: EguiDescriptorSetAllocator::new(),
            desc_layout,
            pipeline,
            vertex_buffer,
            index_buffer,
            pending_retired_sets: Vec::new(),
            pending_retired_textures: Vec::new(),
            samplers: Default::default(),
            texture_images: Default::default(),
            next_native_tex_id: 0,
        }
    }

    unsafe fn create_buffer<T>(
        size: u64,
        usage: vk::BufferUsageFlags,
        device: &Device,
    ) -> (object::Buffer, *mut T) {
        let buffer = device
            .create_buffer(
                object::BufferCreateInfo {
                    flags: vk::BufferCreateFlags::empty(),
                    size,
                    usage,
                    sharing_mode_concurrent: false,
                    label: None,
                },
                vma::AllocationCreateInfo {
                    flags: vma::AllocationCreateFlags::MAPPED
                        | vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
                    usage: vma::MemoryUsage::AutoPreferDevice,
                    ..Default::default()
                },
            )
            .unwrap();

        let info = device
            .allocator()
            .get_allocation_info(buffer.get_allocation());

        let align = std::mem::align_of::<T>();
        assert!(
            (info.mapped_data as usize & (align - 1)) == 0,
            "Pointer is not properly aligned"
        );

        (buffer, info.mapped_data.cast())
    }

    unsafe fn get_sampler(
        samplers: &mut HashMap<egui::TextureOptions, object::Sampler, DefaultAhashRandomstate>,
        options: egui::TextureOptions,
        device: &Device,
    ) -> vk::Sampler {
        let to_vk = |filter: egui::TextureFilter| match filter {
            egui::TextureFilter::Nearest => vk::Filter::NEAREST,
            egui::TextureFilter::Linear => vk::Filter::LINEAR,
        };

        samplers
            .entry(options)
            .or_insert_with(|| {
                device
                    .create_descriptor_sampler(object::SamplerCreateInfo {
                        mag_filter: to_vk(options.magnification),
                        min_filter: to_vk(options.minification),
                        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        ..Default::default()
                    })
                    .unwrap()
            })
            .get_handle()
    }
    unsafe fn create_image([width, height]: [u32; 2], device: &Device) -> TextureImage {}
    unsafe fn update_texture(
        &mut self,
        texture_id: egui::TextureId,
        deltas: impl Iterator<Item = egui::epaint::ImageDelta>,
        device: &Device,
    ) {
        let srgb_data = deltas
            .map(|delta| match &delta.image {
                egui::ImageData::Color(image) => {
                    assert_eq!(
                        image.width() * image.height(),
                        image.pixels.len(),
                        "Mismatch between texture size and texel count"
                    );

                    let slice = unsafe {
                        // Color32 is repr(C) [u8; 4]
                        // this makes it castable as a &[u8]
                        std::slice::from_raw_parts(
                            image.pixels.as_ptr().cast::<u8>(),
                            image.width() * image.height() * 4,
                        )
                    };

                    slice.to_vec()
                }
                egui::ImageData::Font(image) => {
                    // hope that the codegen is decent
                    image
                        .srgba_pixels(None)
                        .flat_map(|c| c.to_array())
                        .collect()
                }
            })
            .collect::<Vec<_>>();

        let image = self
            .texture_images
            .entry(texture_id)
            .or_insert_with(default);

        let offset = delta
            .pos
            .map(|[x, y]| vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: 0,
            })
            .unwrap_or_default();

        let (width, height) = image.image.get_create_info().size.get_2d().unwrap();
        let extent = vk::Extent3D {
            width: width - offset.x as u32,
            height: height - offset.y as u32,
            depth: 1,
        };

        device.write_image(
            &image.image,
            std::alloc::Layout::new::<Color32>(),
            vec![ImageWrite {
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: offset,
                image_extent: extent,
            }],
            move |ptr, _, _| {
                ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
            },
        );

        if image.options != delta.options {
            image.options = delta.options;
            let sampler = Self::get_sampler(&mut self.samplers, delta.options, device);
            let new_set =
                self.set_allocator
                    .allocate(image.view, sampler, &self.desc_layout, device);

            self.pending_retired_sets.push(image.set);
            image.set = new_set;
        }
    }

    /// Record paint commands.
    pub unsafe fn paint(
        &mut self,
        command_buffer: vk::CommandBuffer,

        color_view: vk::ImageView,
        color_flags: vk::ImageCreateFlags,
        color_usage: vk::ImageUsageFlags,
        color_view_formats: &[vk::Format],

        resolve_view: vk::ImageView,
        resolve_flags: vk::ImageCreateFlags,
        resolve_usage: vk::ImageUsageFlags,
        resolve_view_formats: &[vk::Format],

        clear: Option<vk::ClearColorValue>,
        pixels_per_point: f32,

        clipped_meshes: &[egui::ClippedPrimitive],
        textures_delta: &TexturesDelta,
        [width, height]: [u32; 2],

        device: &Device,
    ) {
        if self.resolve_attachment {
            assert!(resolve_view != vk::ImageView::null());
        }

        for set in self.pending_retired_sets.drain(..) {
            self.set_allocator.free_set(set);
        }

        for tex in self.pending_retired_textures.drain(..) {
            let entry = self.texture_images.remove(&tex).unwrap();
            self.set_allocator.free_set(entry.set);
        }

        self.pending_retired_textures
            .extend(textures_delta.free.iter().copied());

        let mut texture_deltas: Vec<_> = textures_delta.set.iter().cloned().collect();
        texture_deltas.sort_by_key(|d| d.0);

        for deltas in texture_deltas.binary_group_by_key(|d| d.0) {
            self.update_texture(*id, &image_delta, device);
        }

        let framebuffer_size_differs = || {
            let info = self.framebuffer.as_ref().unwrap().get_create_info();
            width != info.width || height != info.height
        };
        if self.framebuffer.is_none() || framebuffer_size_differs() {
            drop(self.framebuffer.take());

            let mut attachments = Vec::new();
            attachments.push(object::ImagelessAttachment {
                flags: color_flags,
                usage: color_usage,
                width,
                height,
                layer_count: 1,
                view_formats: color_view_formats.into(),
            });
            if self.resolve_attachment {
                attachments.push(object::ImagelessAttachment {
                    flags: resolve_flags,
                    usage: resolve_usage,
                    width,
                    height,
                    layer_count: 1,
                    view_formats: resolve_view_formats.into(),
                });
            }

            self.framebuffer = Some(
                device
                    .create_framebuffer(object::FramebufferCreateInfo {
                        flags: vk::FramebufferCreateFlags::IMAGELESS,
                        render_pass: self.render_pass.clone(),
                        attachments: object::AttachmentMode::Imageless(attachments),
                        width,
                        height,
                        layers: 1,
                    })
                    .unwrap(),
            );
        }

        let framebuffer = self.framebuffer.as_deref().unwrap();
        let d = device.device();

        {
            let mut clear_value = None;
            let mut clear_value_count = 0;
            if let Some(color) = clear {
                clear_value = Some(vk::ClearValue { color });
                clear_value_count = 1;
            }

            let attachment_info = vk::RenderPassAttachmentBeginInfoKHR {
                attachment_count: 1 + self.resolve_attachment as u32,
                p_attachments: [color_view, resolve_view].as_ptr(),
                ..Default::default()
            };

            let begin = vk::RenderPassBeginInfo {
                p_next: &attachment_info as *const _ as *const _,
                render_pass: self.render_pass.get_handle(),
                framebuffer: framebuffer.get_handle(),
                render_area: vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D { width, height },
                },
                clear_value_count,
                p_clear_values: clear_value.as_ffi_ptr(),
                ..Default::default()
            };
            d.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
        }

        let pipeline_layout = &*self.pipeline.get_object().get_create_info().layout;

        // prepare pipeline state
        {
            d.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.get_handle(),
            );
            d.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.vertex_buffer.0.get_handle()],
                &[0],
            );
            d.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.0.get_handle(),
                0,
                vk::IndexType::UINT32,
            );
            d.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: width as f32,
                    height: height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            let width_in_points = width as f32 / pixels_per_point;
            let height_in_points = height as f32 / pixels_per_point;
            d.cmd_push_constants(
                command_buffer,
                pipeline_layout.get_handle(),
                vk::ShaderStageFlags::VERTEX,
                0,
                std::mem::size_of::<PushConstants>() as u32,
                &PushConstants {
                    screen_size: [width_in_points, height_in_points],
                    need_srgb_conv: self.need_srgb_conv as i32,
                } as *const _ as *const c_void,
            );
        }

        let mut vertex_buffer_ptr = self.vertex_buffer.1;
        let vertex_buffer_ptr_end = vertex_buffer_ptr.add(VERTEX_BUFFER_COUNT);

        let mut index_buffer_ptr = self.index_buffer.1;
        let index_buffer_ptr_end = index_buffer_ptr.add(INDEX_BUFFER_COUNT);

        let mut vertex_offset = 0;
        let mut index_offset = 0;

        let mut current_texture = None;

        // draw meshes
        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_meshes
        {
            let egui::epaint::Primitive::Mesh(mesh) = primitive else {
                panic!("Callback primitive is Unsupported");
            };

            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            if let Some(texture) = self.texture_images.get(&mesh.texture_id) {
                if current_texture != Some(mesh.texture_id) {
                    d.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout.get_handle(),
                        0,
                        &[texture.set],
                        &[],
                    );
                    current_texture = Some(mesh.texture_id);
                }
            } else {
                eprintln!(
                    "Attempt to draw mesh with invalid TextureID: {:?}",
                    mesh.texture_id
                );
                // we continue with drawing with the previous texture perhaps the obviously wrong
                // output will make developers fix stuff quicker
            };

            let v_end = vertex_buffer_ptr.add(mesh.vertices.len());
            let i_end = index_buffer_ptr.add(mesh.vertices.len());

            if v_end > vertex_buffer_ptr_end || i_end > index_buffer_ptr_end {
                panic!("Ran out of memory for mesh buffers");
            }

            vertex_buffer_ptr.copy_from_nonoverlapping(mesh.vertices.as_ptr(), mesh.vertices.len());
            index_buffer_ptr.copy_from_nonoverlapping(mesh.indices.as_ptr(), mesh.indices.len());

            // Transform clip rect to physical pixels:
            let clip_min_x = pixels_per_point * clip_rect.min.x;
            let clip_min_y = pixels_per_point * clip_rect.min.y;
            let clip_max_x = pixels_per_point * clip_rect.max.x;
            let clip_max_y = pixels_per_point * clip_rect.max.y;

            // Round to integer:
            let clip_min_x = clip_min_x.round() as i32;
            let clip_min_y = clip_min_y.round() as i32;
            let clip_max_x = clip_max_x.round() as i32;
            let clip_max_y = clip_max_y.round() as i32;

            // Clamp:
            let clip_min_x = clip_min_x.clamp(0, width as i32);
            let clip_min_y = clip_min_y.clamp(0, height as i32);
            let clip_max_x = clip_max_x.clamp(clip_min_x, width as i32);
            let clip_max_y = clip_max_y.clamp(clip_min_y, height as i32);

            d.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: clip_min_x,
                        y: clip_min_y,
                    },
                    extent: vk::Extent2D {
                        width: (clip_min_x - clip_max_x) as u32,
                        height: (clip_min_y - clip_max_y) as u32,
                    },
                }],
            );
            d.cmd_draw_indexed(
                command_buffer,
                mesh.indices.len() as u32,
                1,
                index_offset,
                vertex_offset,
                0,
            );

            vertex_buffer_ptr = v_end;
            index_buffer_ptr = i_end;

            vertex_offset += mesh.vertices.len() as i32;
            index_offset += mesh.indices.len() as u32;
        }

        d.cmd_end_render_pass(command_buffer);
    }
}
