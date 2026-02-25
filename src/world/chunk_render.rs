use std::collections::HashMap;

use super::chunk::ChunkCoord;
use super::mesher::{ChunkMesh, ChunkVertex};

/// GPU-side data for a single chunk mesh.
pub struct ChunkGpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

/// Manages the chunk render pipeline and all loaded chunk GPU meshes.
pub struct ChunkRenderer {
    pipeline: wgpu::RenderPipeline,
    meshes: HashMap<ChunkCoord, ChunkGpuMesh>,
}

impl ChunkRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chunk_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/chunk.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("chunk_pipeline_layout"),
            bind_group_layouts: &[global_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("chunk_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[ChunkVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            meshes: HashMap::new(),
        }
    }

    /// Upload a chunk mesh to the GPU. Replaces any existing mesh for this coord.
    pub fn upload_chunk(
        &mut self,
        device: &wgpu::Device,
        coord: ChunkCoord,
        mesh: &ChunkMesh,
    ) {
        if mesh.is_empty() {
            self.meshes.remove(&coord);
            return;
        }

        use wgpu::util::DeviceExt;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_vertex_buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_index_buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.meshes.insert(
            coord,
            ChunkGpuMesh {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
            },
        );
    }

    /// Remove a chunk mesh from the GPU.
    pub fn remove_chunk(&mut self, coord: &ChunkCoord) {
        self.meshes.remove(coord);
    }

    /// Draw all loaded chunk meshes.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        global_bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, global_bind_group, &[]);

        for gpu_mesh in self.meshes.values() {
            render_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
            render_pass.set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
        }
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.meshes.len()
    }
}
