// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::{convert::TryFrom, hash::BuildHasher, sync::Arc};

use ahash::AHashMap;
use bytemuck::{Pod, Zeroable};
use egui::{
    epaint::{Mesh, Primitive},
    ClippedPrimitive, PaintCallbackInfo, Rect, TexturesDelta,
};
use graph::{
    device::{submission, Device},
    object,
    smallvec::{smallvec, SmallVec},
    storage::DefaultAhashRandomstate,
    util::ffi_ptr::AsFFiPtr,
};
use pumice::vk;
// use vk
//     buffer::{
//         cpu_pool::CpuBufferPoolChunk, BufferUsage, CpuAccessibleBuffer, CpuBufferPool,
//         TypedBufferAccess,
//     },
//     command_buffer::{
//         allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo,
//         CommandBufferInheritanceInfo, CommandBufferUsage, CopyBufferToImageInfo, ImageBlit,
//         PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
//         SecondaryAutoCommandBuffer, SubpassContents,
//     },
//     descriptor_set::{
//         allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout,
//         PersistentDescriptorSet, WriteDescriptorSet,
//     },
//     device::Queue,
//     format::{Format, NumericType},
//     image::{
//         view::ImageView, ImageAccess, ImageLayout, ImageUsage, ImageViewAbstract, ImmutableImage,
//         SampleCount,
//     },
//     memory::allocator::{MemoryUsage, StandardMemoryAllocator},
//     pipeline::{
//         graphics::{
//             color_blend::{AttachmentBlend, BlendFactor, ColorBlendState},
//             input_assembly::InputAssemblyState,
//             multisample::MultisampleState,
//             rasterization::{CullMode as CullModeEnum, RasterizationState},
//             vertex_input::BuffersDefinition,
//             viewport::{Scissor, Viewport, ViewportState},
//         },
//         GraphicsPipeline, Pipeline, PipelineBindPoint,
//     },
//     render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
//     sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode},
//     sync::GpuFuture,
//     u64,
// };

use crate::utils::Allocators;

use pumice_vma as vma;

const VERTEX_BUFFER_SIZE: u64 = 1024 * 1024 * 4 * (std::mem::size_of::<EguiVertex>() as u64);
const INDEX_BUFFER_SIZE: u64 = 1024 * 1024 * 2 * 4;

/// Should match vertex definition of egui (except color is `[f32; 4]`)
#[repr(C)]
pub struct EguiVertex {
    pub position: [f32; 2],
    pub tex_coords: [f32; 2],
    pub color: [f32; 4],
}

#[repr(C)]
pub struct PushConstants {
    screen_size: [f32; 2],
    need_srgb_conv: i32,
}

pub struct Renderer {
    gfx_queue: submission::Queue,
    renderpass: Option<object::RenderPass>,
    is_overlay: bool,
    need_srgb_conv: bool,

    #[allow(unused)]
    format: vk::Format,
    sampler: object::Sampler,

    allocators: Allocators,
    vertex_buffer: object::Buffer,
    index_buffer: object::Buffer,
    pipeline: object::ConcreteGraphicsPipeline,

    texture_desc_sets: AHashMap<egui::TextureId, vk::DescriptorSet>,
    texture_images: AHashMap<egui::TextureId, vk::ImageView>,
    next_native_tex_id: u64,
}

impl Renderer {
    // pub fn new_with_subpass(
    //     gfx_queue : Arc<Queue>,
    //     final_output_format : Format,
    //     subpass : Subpass,
    // ) -> Renderer {
    //     let need_srgb_conv = final_output_format.type_color().unwrap() == NumericType::UNORM;
    //     let allocators = Allocators::new_default(gfx_queue.device());
    //     let (vertex_buffer_pool, index_buffer_pool) = Self::create_buffers(&allocators.memory);
    //     let pipeline = Self::create_pipeline(gfx_queue.clone(), subpass.clone());
    //     let sampler = Sampler::new(
    //         gfx_queue.device().clone(),
    //         SamplerCreateInfo {
    //             mag_filter : Filter::Linear,
    //             min_filter : Filter::Linear,
    //             address_mode : [SamplerAddressMode::ClampToEdge; 3],
    //             mipmap_mode : SamplerMipmapMode::Linear,
    //             ..Default::default()
    //         },
    //     )
    //     .unwrap();
    //     Renderer {
    //         gfx_queue,
    //         format : final_output_format,
    //         render_pass : None,
    //         vertex_buffer_pool,
    //         index_buffer_pool,
    //         pipeline,
    //         subpass,
    //         texture_desc_sets : AHashMap::default(),
    //         texture_images : AHashMap::default(),
    //         next_native_tex_id : 0,
    //         is_overlay : false,
    //         need_srgb_conv,
    //         sampler,
    //         allocators,
    //     }
    // }

    /// Creates a new [Renderer] which is responsible for rendering egui with its own renderpass
    /// See examples
    pub unsafe fn new_with_render_pass(
        gfx_queue: submission::Queue,
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

        meta_device: &Device,
    ) -> Renderer {
        if resolve_enable {
            assert!(samples != vk::SampleCountFlags::C1);
        }

        let device = meta_device.device();
        let callbacks = meta_device.allocator_callbacks();

        let render_pass = {
            let mut attachments = SmallVec::<[_; 2]>::new();
            let mut dependencies = SmallVec::<[_; 2]>::new();

            // color
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

            let color_ref = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };

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

            let subpass = vk::SubpassDescription {
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                color_attachment_count: 1,
                p_color_attachments: &color_ref,
                p_resolve_attachments: resolve_ref.as_ffi_ptr(),
                ..Default::default()
            };

            let render_pass_info = vk::RenderPassCreateInfo {
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                subpass_count: 1,
                p_subpasses: &subpass,
                dependency_count: dependencies.len() as u32,
                p_dependencies: dependencies.as_ptr(),
                ..Default::default()
            };

            device
                .create_render_pass(&render_pass_info, callbacks)
                .unwrap()
        };

        let pipeline = {
            const VS_BYTES: &[u8] = include_bytes!("../shaders/vertex.spv");
            const FS_BYTES: &[u8] = include_bytes!("../shaders/fragment.spv");

            let mut vs_cursor = std::io::Cursor::new(VS_BYTES);
            let mut fs_cursor = std::io::Cursor::new(VS_BYTES);

            let vs = meta_device
                .create_shader_module_read(&mut vs_cursor)
                .unwrap();
            let fs = meta_device
                .create_shader_module_read(&mut fs_cursor)
                .unwrap();

            let sampler = meta_device
                .create_descriptor_sampler(object::SamplerCreateInfo {
                    mag_filter: vk::Filter::LINEAR,
                    min_filter: vk::Filter::LINEAR,
                    mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                    address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    ..Default::default()
                })
                .unwrap();

            let set_layout = meta_device
                .create_descriptor_set_layout(object::DescriptorSetLayoutCreateInfo {
                    flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                    bindings: vec![object::DescriptorBinding {
                        binding: 0,
                        count: 1,
                        kind: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        stages: vk::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: smallvec![sampler],
                    }],
                })
                .unwrap();

            let layout = meta_device
                .create_pipeline_layout(object::PipelineLayoutCreateInfo {
                    set_layouts: vec![set_layout],
                    push_constants: vec![vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        offset: 0,
                        size: std::mem::size_of::<PushConstants>(),
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
                        // 3 floats for position
                        stride: std::mem::size_of::<EguiVertex>(),
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
                    render_pass,
                })
                .layout(layout.clone())
                .finish();

            meta_device.create_graphics_pipeline(info).unwrap()
        };

        let need_srgb_conv = !format_is_srgb;
        let vertex_buffer = Self::create_buffer(
            VERTEX_BUFFER_SIZE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            meta_device,
        );
        let index_buffer = Self::create_buffer(
            INDEX_BUFFER_SIZE,
            vk::BufferUsageFlags::INDEX_BUFFER,
            meta_device,
        );

        Renderer {
            gfx_queue,
            format,
            renderpass: Some(renderpass),
            vertex_buffer,
            index_buffer,
            pipeline,
            texture_desc_sets: AHashMap::with_hasher((DefaultAhashRandomstate).build_hasher()),
            texture_images: AHashMap::with_hasher((DefaultAhashRandomstate).build_hasher()),
            allocators: Allocators::new_default(todo!()),
            next_native_tex_id: 0,
            is_overlay,
            need_srgb_conv,
            sampler,
        }
    }

    pub fn has_renderpass(&self) -> bool {
        self.renderpass.is_some()
    }

    unsafe fn create_buffer(
        size: u64,
        usage: vk::BufferUsageFlags,
        device: &Device,
    ) -> object::Buffer {
        device
            .create_buffer(
                object::BufferCreateInfo {
                    flags: vk::BufferCreateFlags::empty(),
                    size,
                    usage,
                    sharing_mode_concurrent: false,
                    label: None,
                },
                vma::AllocationCreateInfo {
                    flags: vma::AllocationCreateFlags::empty(),
                    usage: vma::MemoryUsage::AutoPreferDevice,
                    ..Default::default()
                },
            )
            .unwrap()
    }
    /// Creates a descriptor set for images
    fn sampled_image_desc_set(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        image: Arc<dyn ImageViewAbstract + 'static>,
    ) -> Arc<PersistentDescriptorSet> {
        PersistentDescriptorSet::new(
            &self.allocators.descriptor_set,
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                image,
                self.sampler.clone(),
            )],
        )
        .unwrap()
    }

    /// Registers a user texture. User texture needs to be unregistered when it is no longer needed
    pub fn register_image(
        &mut self,
        image: Arc<dyn ImageViewAbstract + Send + Sync>,
    ) -> egui::TextureId {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let desc_set = self.sampled_image_desc_set(layout, image.clone());
        let id = egui::TextureId::User(self.next_native_tex_id);
        self.next_native_tex_id += 1;
        self.texture_desc_sets.insert(id, desc_set);
        self.texture_images.insert(id, image);
        id
    }

    /// Unregister user texture.
    pub fn unregister_image(&mut self, texture_id: egui::TextureId) {
        self.texture_desc_sets.remove(&texture_id);
        self.texture_images.remove(&texture_id);
    }

    fn update_texture(&mut self, texture_id: egui::TextureId, delta: &egui::epaint::ImageDelta) {
        // Extract pixel data from egui
        let data: Vec<u8> = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                image
                    .pixels
                    .iter()
                    .flat_map(|color| color.to_array())
                    .collect()
            }
            egui::ImageData::Font(image) => image
                .srgba_pixels(None)
                .flat_map(|color| color.to_array())
                .collect(),
        };
        // Create buffer to be copied to the image
        let texture_data_buffer = CpuAccessibleBuffer::from_iter(
            &self.allocators.memory,
            BufferUsage {
                transfer_src: true,
                ..BufferUsage::empty()
            },
            false,
            data,
        )
        .unwrap();
        // Create image
        let (img, init) = ImmutableImage::uninitialized(
            &self.allocators.memory,
            vk::ImageDimensions::Dim2d {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                array_layers: 1,
            },
            Format::R8G8B8A8_SRGB,
            vk::MipmapsCount::One,
            ImageUsage {
                transfer_dst: true,
                transfer_src: true,
                sampled: true,
                ..ImageUsage::empty()
            },
            Default::default(),
            ImageLayout::ShaderReadOnlyOptimal,
            Some(self.gfx_queue.queue_family_index()),
        )
        .unwrap();
        let font_image = ImageView::new_default(img).unwrap();

        // Create command buffer builder
        let mut cbb = AutoCommandBufferBuilder::primary(
            &self.allocators.command_buffer,
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Copy buffer to image
        cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            texture_data_buffer,
            init,
        ))
        .unwrap();

        // Blit texture data to existing image if delta pos exists (e.g. font changed)
        if let Some(pos) = delta.pos {
            if let Some(existing_image) = self.texture_images.get(&texture_id) {
                let src_dims = font_image.image().dimensions();
                let top_left = [pos[0] as u32, pos[1] as u32, 0];
                let bottom_right = [
                    pos[0] as u32 + src_dims.width(),
                    pos[1] as u32 + src_dims.height(),
                    1,
                ];

                cbb.blit_image(BlitImageInfo {
                    src_image_layout: ImageLayout::General,
                    dst_image_layout: ImageLayout::General,
                    regions: [ImageBlit {
                        src_subresource: font_image.image().subresource_layers(),
                        src_offsets: [[0, 0, 0], [src_dims.width(), src_dims.height(), 1]],
                        dst_subresource: existing_image.image().subresource_layers(),
                        dst_offsets: [top_left, bottom_right],
                        ..Default::default()
                    }]
                    .into(),
                    filter: Filter::Nearest,
                    ..BlitImageInfo::images(font_image.image().clone(), existing_image.image())
                })
                .unwrap();
            }
            // Otherwise save the newly created image
        } else {
            let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
            let font_desc_set = self.sampled_image_desc_set(layout, font_image.clone());
            self.texture_desc_sets.insert(texture_id, font_desc_set);
            self.texture_images.insert(texture_id, font_image);
        }
        // Execute command buffer
        let command_buffer = cbb.build().unwrap();
        let finished = command_buffer.execute(self.gfx_queue.clone()).unwrap();
        let _fut = finished.then_signal_fence_and_flush().unwrap();
    }

    fn get_rect_scissor(
        &self,
        scale_factor: f32,
        framebuffer_dimensions: [u32; 2],
        rect: Rect,
    ) -> Scissor {
        let min = rect.min;
        let min = egui::Pos2 {
            x: min.x * scale_factor,
            y: min.y * scale_factor,
        };
        let min = egui::Pos2 {
            x: min.x.clamp(0.0, framebuffer_dimensions[0] as f32),
            y: min.y.clamp(0.0, framebuffer_dimensions[1] as f32),
        };
        let max = rect.max;
        let max = egui::Pos2 {
            x: max.x * scale_factor,
            y: max.y * scale_factor,
        };
        let max = egui::Pos2 {
            x: max.x.clamp(min.x, framebuffer_dimensions[0] as f32),
            y: max.y.clamp(min.y, framebuffer_dimensions[1] as f32),
        };
        Scissor {
            origin: [min.x.round() as u32, min.y.round() as u32],
            dimensions: [
                (max.x.round() - min.x) as u32,
                (max.y.round() - min.y) as u32,
            ],
        }
    }

    fn create_subbuffers(
        &self,
        mesh: &Mesh,
    ) -> (
        Arc<CpuBufferPoolChunk<EguiVertex>>,
        Arc<CpuBufferPoolChunk<u32>>,
    ) {
        // Copy vertices to buffer
        let v_slice = &mesh.vertices;

        let vertex_chunk = self
            .vertex_buffer_pool
            .from_iter(v_slice.iter().map(|v| EguiVertex {
                position: [v.pos.x, v.pos.y],
                tex_coords: [v.uv.x, v.uv.y],
                color: [
                    v.color.r() as f32 / 255.0,
                    v.color.g() as f32 / 255.0,
                    v.color.b() as f32 / 255.0,
                    v.color.a() as f32 / 255.0,
                ],
            }))
            .unwrap();

        // Copy indices to buffer
        let i_slice = &mesh.indices;
        let index_chunk = self.index_buffer_pool.from_iter(i_slice.clone()).unwrap();

        (vertex_chunk, index_chunk)
    }

    fn create_secondary_command_buffer_builder(
        &self,
    ) -> AutoCommandBufferBuilder<SecondaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::secondary(
            &self.allocators.command_buffer,
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap()
    }

    // Starts the rendering pipeline and returns [`AutoCommandBufferBuilder`] for drawing
    fn start(
        &mut self,
        final_image: Arc<dyn ImageViewAbstract + 'static>,
    ) -> (AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, [u32; 2]) {
        // Get dimensions
        let img_dims = final_image.image().dimensions().width_height();
        // Create framebuffer (must be in same order as render pass description in `new`
        let framebuffer = Framebuffer::new(
            self.renderpass
                .as_ref()
                .expect(
                    "No renderpass on this renderer (created with subpass), use 'draw_subpass' \
                     instead",
                )
                .clone(),
            FramebufferCreateInfo {
                attachments: vec![final_image],
                ..Default::default()
            },
        )
        .unwrap();
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.allocators.command_buffer,
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        // Add clear values here for attachments and begin render pass
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![if !self.is_overlay {
                        Some([0.0; 4].into())
                    } else {
                        None
                    }],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap();
        (command_buffer_builder, img_dims)
    }

    /// Executes our draw commands on the final image and returns a `GpuFuture` to wait on
    pub fn draw_on_image<F>(
        &mut self,
        clipped_meshes: &[ClippedPrimitive],
        textures_delta: &TexturesDelta,
        scale_factor: f32,
        before_future: F,
        final_image: Arc<dyn ImageViewAbstract + 'static>,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        for (id, image_delta) in &textures_delta.set {
            self.update_texture(*id, image_delta);
        }

        let (mut command_buffer_builder, framebuffer_dimensions) = self.start(final_image);
        let mut builder = self.create_secondary_command_buffer_builder();
        self.draw_egui(
            scale_factor,
            clipped_meshes,
            framebuffer_dimensions,
            &mut builder,
        );
        // Execute draw commands
        let command_buffer = builder.build().unwrap();
        command_buffer_builder
            .execute_commands(command_buffer)
            .unwrap();
        let done_future = self.finish(command_buffer_builder, Box::new(before_future));

        for &id in &textures_delta.free {
            self.unregister_image(id);
        }

        done_future
    }

    // Finishes the rendering pipeline
    fn finish(
        &self,
        mut command_buffer_builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        before_main_cb_future: Box<dyn GpuFuture>,
    ) -> Box<dyn GpuFuture> {
        // We end render pass
        command_buffer_builder.end_render_pass().unwrap();
        // Then execute our whole command buffer
        let command_buffer = command_buffer_builder.build().unwrap();
        let after_main_cb = before_main_cb_future
            .then_execute(self.gfx_queue.clone(), command_buffer)
            .unwrap();
        let future = after_main_cb
            .then_signal_fence_and_flush()
            .expect("Failed to signal fence and flush");
        // Return our future
        Box::new(future)
    }

    pub fn draw_on_subpass_image(
        &mut self,
        clipped_meshes: &[ClippedPrimitive],
        textures_delta: &TexturesDelta,
        scale_factor: f32,
        framebuffer_dimensions: [u32; 2],
    ) -> SecondaryAutoCommandBuffer {
        for (id, image_delta) in &textures_delta.set {
            self.update_texture(*id, image_delta);
        }
        let mut builder = self.create_secondary_command_buffer_builder();
        self.draw_egui(
            scale_factor,
            clipped_meshes,
            framebuffer_dimensions,
            &mut builder,
        );
        let buffer = builder.build().unwrap();
        for &id in &textures_delta.free {
            self.unregister_image(id);
        }
        buffer
    }

    fn draw_egui(
        &mut self,
        scale_factor: f32,
        clipped_meshes: &[ClippedPrimitive],
        framebuffer_dimensions: [u32; 2],
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
    ) {
        let push_constants = vs::ty::PushConstants {
            screen_size: [
                framebuffer_dimensions[0] as f32 / scale_factor,
                framebuffer_dimensions[1] as f32 / scale_factor,
            ],
            need_srgb_conv: self.need_srgb_conv.into(),
        };

        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_meshes
        {
            match primitive {
                Primitive::Mesh(mesh) => {
                    // Nothing to draw if we don't have vertices & indices
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }
                    if self.texture_desc_sets.get(&mesh.texture_id).is_none() {
                        eprintln!("This texture no longer exists {:?}", mesh.texture_id);
                        continue;
                    }

                    let scissors = vec![self.get_rect_scissor(
                        scale_factor,
                        framebuffer_dimensions,
                        *clip_rect,
                    )];

                    let (vertices, indices) = self.create_subbuffers(mesh);

                    let desc_set = self
                        .texture_desc_sets
                        .get(&mesh.texture_id)
                        .unwrap()
                        .clone();
                    builder
                        .bind_pipeline_graphics(self.pipeline.clone())
                        .set_viewport(
                            0,
                            vec![Viewport {
                                origin: [0.0, 0.0],
                                dimensions: [
                                    framebuffer_dimensions[0] as f32,
                                    framebuffer_dimensions[1] as f32,
                                ],
                                depth_range: 0.0..1.0,
                            }],
                        )
                        .set_scissor(0, scissors)
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            self.pipeline.layout().clone(),
                            0,
                            desc_set.clone(),
                        )
                        .push_constants(self.pipeline.layout().clone(), 0, push_constants)
                        .bind_vertex_buffers(0, vertices.clone())
                        .bind_index_buffer(indices.clone())
                        .draw_indexed(indices.len() as u32, 1, 0, 0, 0)
                        .unwrap();
                }
                Primitive::Callback(callback) => {
                    if callback.rect.is_positive() {
                        let rect_min_x = scale_factor * callback.rect.min.x;
                        let rect_min_y = scale_factor * callback.rect.min.y;
                        let rect_max_x = scale_factor * callback.rect.max.x;
                        let rect_max_y = scale_factor * callback.rect.max.y;

                        let rect_min_x = rect_min_x.round();
                        let rect_min_y = rect_min_y.round();
                        let rect_max_x = rect_max_x.round();
                        let rect_max_y = rect_max_y.round();

                        let scissors = vec![self.get_rect_scissor(
                            scale_factor,
                            framebuffer_dimensions,
                            *clip_rect,
                        )];

                        builder
                            .set_viewport(
                                0,
                                vec![Viewport {
                                    origin: [rect_min_x, rect_min_y],
                                    dimensions: [rect_max_x - rect_min_x, rect_max_y - rect_min_y],
                                    depth_range: 0.0..1.0,
                                }],
                            )
                            .set_scissor(0, scissors);

                        let info = egui::PaintCallbackInfo {
                            viewport: callback.rect,
                            clip_rect: *clip_rect,
                            pixels_per_point: scale_factor,
                            screen_size_px: framebuffer_dimensions,
                        };

                        if let Some(callback) = callback.callback.downcast_ref::<CallbackFn>() {
                            (callback.f)(
                                info,
                                &mut CallbackContext {
                                    builder,
                                    resources: self.render_resources(),
                                },
                            );
                        } else {
                            println!(
                                "Warning : Unsupported render callback. Expected \
                                 egui_winit_vk"
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn render_resources(&self) -> RenderResources {
        RenderResources {
            queue: self.queue(),
            subpass: self.subpass.clone(),
            memory_allocator: self.allocators.memory.clone(),
            descriptor_set_allocator: &self.allocators.descriptor_set,
            command_buffer_allocator: &self.allocators.command_buffer,
        }
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.gfx_queue.clone()
    }

    pub fn allocators(&self) -> &Allocators {
        &self.allocators
    }
}

/// A set of objects used to perform custom rendering in a `PaintCallback`. It
/// includes [`RenderResources`] for constructing a subpass pipeline and a secondary
/// command buffer for pushing render commands onto it.
///
/// # Example
///
/// See the `triangle` demo source for a detailed usage example.
pub struct CallbackContext<'a> {
    pub builder: &'a mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
    pub resources: RenderResources<'a>,
}

/// A set of resources used to construct the render pipeline. These can be reused
/// to create additional pipelines and buffers to be rendered in a `PaintCallback`.
///
/// # Example
///
/// See the `triangle` demo source for a detailed usage example.
#[derive(Clone)]
pub struct RenderResources<'a> {
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub descriptor_set_allocator: &'a StandardDescriptorSetAllocator,
    pub command_buffer_allocator: &'a StandardCommandBufferAllocator,
    pub queue: Arc<Queue>,
    pub subpass: Subpass,
}

pub type CallbackFnDef = dyn Fn(PaintCallbackInfo, &mut CallbackContext) + Sync + Send;

/// A callback function that can be used to compose an [`epaint::PaintCallback`] for
/// custom rendering with [`pumice`].
///
/// The callback is passed an [`egui::PaintCallbackInfo`] and a [`CallbackContext`] which
/// can be used to construct Vulkano graphics pipelines and buffers.
///
/// # Example
///
/// See the `triangle` demo source for a detailed usage example.
pub struct CallbackFn {
    pub(crate) f: Box<CallbackFnDef>,
}
impl CallbackFn {
    pub fn new<F: Fn(PaintCallbackInfo, &mut CallbackContext) + Sync + Send + 'static>(
        callback: F,
    ) -> Self {
        let f = Box::new(callback);
        CallbackFn { f }
    }
}
