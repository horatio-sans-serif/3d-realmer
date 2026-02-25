//! Renders imported GLTF meshes with per-object model matrices.
//!
//! Unlike the instanced ObjectRenderer (same cube geometry everywhere),
//! MeshRenderer handles meshes with unique geometry. Each mesh gets its
//! own vertex/index buffers. Per-object transforms and colors are passed
//! via a uniform bind group (group 1).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bitparty::bridge::WorldObjectId;
use glam::{Mat4, Quat, Vec3};

use super::gltf_loader::{LoadedMesh, MeshVertex};

/// Per-object uniform data: model matrix + color.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniforms {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

/// GPU-side data for a loaded mesh.
struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

/// A mesh object placed in the world.
pub struct MeshInstance {
    pub world_object_id: WorldObjectId,
    pub mesh_path: PathBuf,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub color: [f32; 4],
}

/// Renders GLTF mesh objects with per-object transforms.
pub struct MeshRenderer {
    pipeline: wgpu::RenderPipeline,
    object_bind_group_layout: wgpu::BindGroupLayout,
    /// GPU meshes keyed by file path.
    gpu_meshes: HashMap<PathBuf, GpuMesh>,
    /// Per-object uniform buffer and bind group (reused each frame).
    object_uniform_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
    /// Current mesh instances to render.
    instances: Vec<MeshInstance>,
}

impl MeshRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh.wgsl").into()),
        });

        let object_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh_object_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[global_bind_group_layout, &object_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[MeshVertex::desc()],
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

        let object_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_object_uniform_buffer"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_object_bind_group"),
            layout: &object_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: object_uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            object_bind_group_layout,
            gpu_meshes: HashMap::new(),
            object_uniform_buffer,
            object_bind_group,
            instances: Vec::new(),
        }
    }

    /// Upload mesh geometry to the GPU.
    pub fn upload_mesh(&mut self, device: &wgpu::Device, path: &Path, mesh: &LoadedMesh) {
        use wgpu::util::DeviceExt;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vertex_buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_index_buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.gpu_meshes.insert(
            path.to_path_buf(),
            GpuMesh {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
            },
        );
    }

    /// Set the list of mesh instances to render this frame.
    pub fn set_instances(&mut self, instances: Vec<MeshInstance>) {
        self.instances = instances;
    }

    /// Draw all mesh instances. Each instance is drawn as a separate draw call
    /// with its own model matrix, since each mesh has different geometry.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        global_bind_group: &'a wgpu::BindGroup,
        queue: &wgpu::Queue,
    ) {
        if self.instances.is_empty() {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, global_bind_group, &[]);

        for instance in &self.instances {
            let gpu_mesh = match self.gpu_meshes.get(&instance.mesh_path) {
                Some(m) => m,
                None => continue,
            };

            // Build model matrix from position/rotation/scale.
            let position = Vec3::from(instance.position);
            let rotation = Quat::from_array(instance.rotation);
            let scale = Vec3::from(instance.scale);
            let model = Mat4::from_scale_rotation_translation(scale, rotation, position);

            let uniforms = ObjectUniforms {
                model: model.to_cols_array_2d(),
                color: instance.color,
            };

            queue.write_buffer(
                &self.object_uniform_buffer,
                0,
                bytemuck::cast_slice(&[uniforms]),
            );

            render_pass.set_bind_group(1, &self.object_bind_group, &[]);
            render_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                gpu_mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
        }
    }

    pub fn has_mesh(&self, path: &Path) -> bool {
        self.gpu_meshes.contains_key(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_uniforms_layout() {
        assert_eq!(std::mem::size_of::<ObjectUniforms>(), 80);
    }
}
