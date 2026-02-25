//! Renders world objects as instanced colored cubes.
//!
//! Each world object is drawn as a 0.5m cube at its position.
//! Uses the same global uniforms (bind group 0) as the chunk shader.
//! Per-instance data (position offset + color) is passed via a vertex
//! buffer with Instance step mode.

const MAX_INSTANCES: usize = 1024;

/// Per-vertex data for the unit cube.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl CubeVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        static ATTRS: &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
        ];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CubeVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}

/// Per-instance data: world position offset and RGBA color.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectInstance {
    pub offset: [f32; 3],
    pub _pad: f32,
    pub color: [f32; 4],
}

impl ObjectInstance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        static ATTRS: &[wgpu::VertexAttribute] = &[
            // offset: vec3<f32> at location 2
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
            // color: vec4<f32> at location 3, offset 16 (skip 4 bytes padding)
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            },
        ];

        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ObjectInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }
}

/// Renders world objects as instanced cubes.
pub struct ObjectRenderer {
    pipeline: wgpu::RenderPipeline,
    cube_vertex_buffer: wgpu::Buffer,
    cube_index_buffer: wgpu::Buffer,
    cube_index_count: u32,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
}

impl ObjectRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("object_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/object.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("object_pipeline_layout"),
            bind_group_layouts: &[global_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("object_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[CubeVertex::desc(), ObjectInstance::desc()],
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

        let (vertices, indices) = unit_cube();
        let cube_index_count = indices.len() as u32;

        use wgpu::util::DeviceExt;

        let cube_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube_vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let cube_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube_index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_instance_buffer"),
            size: (MAX_INSTANCES * std::mem::size_of::<ObjectInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            cube_vertex_buffer,
            cube_index_buffer,
            cube_index_count,
            instance_buffer,
            instance_count: 0,
        }
    }

    /// Update the instance buffer with new positions and colors.
    pub fn update_instances(&mut self, queue: &wgpu::Queue, instances: &[ObjectInstance]) {
        self.instance_count = instances.len().min(MAX_INSTANCES) as u32;
        if self.instance_count > 0 {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instances[..self.instance_count as usize]),
            );
        }
    }

    /// Draw all object instances in the current render pass.
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        global_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.instance_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, global_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.cube_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(
            self.cube_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.cube_index_count, 0, 0..self.instance_count);
    }
}

/// Generate a unit cube (0.5m, matching physics collider half-extent of 0.25).
/// 24 vertices (4 per face with normals), 36 indices.
fn unit_cube() -> (Vec<CubeVertex>, Vec<u32>) {
    let h = 0.25; // half-size

    #[rustfmt::skip]
    let vertices = vec![
        // Front face (+Z)
        CubeVertex { position: [-h, -h,  h], normal: [ 0.0,  0.0,  1.0] },
        CubeVertex { position: [ h, -h,  h], normal: [ 0.0,  0.0,  1.0] },
        CubeVertex { position: [ h,  h,  h], normal: [ 0.0,  0.0,  1.0] },
        CubeVertex { position: [-h,  h,  h], normal: [ 0.0,  0.0,  1.0] },
        // Back face (-Z)
        CubeVertex { position: [ h, -h, -h], normal: [ 0.0,  0.0, -1.0] },
        CubeVertex { position: [-h, -h, -h], normal: [ 0.0,  0.0, -1.0] },
        CubeVertex { position: [-h,  h, -h], normal: [ 0.0,  0.0, -1.0] },
        CubeVertex { position: [ h,  h, -h], normal: [ 0.0,  0.0, -1.0] },
        // Top face (+Y)
        CubeVertex { position: [-h,  h,  h], normal: [ 0.0,  1.0,  0.0] },
        CubeVertex { position: [ h,  h,  h], normal: [ 0.0,  1.0,  0.0] },
        CubeVertex { position: [ h,  h, -h], normal: [ 0.0,  1.0,  0.0] },
        CubeVertex { position: [-h,  h, -h], normal: [ 0.0,  1.0,  0.0] },
        // Bottom face (-Y)
        CubeVertex { position: [-h, -h, -h], normal: [ 0.0, -1.0,  0.0] },
        CubeVertex { position: [ h, -h, -h], normal: [ 0.0, -1.0,  0.0] },
        CubeVertex { position: [ h, -h,  h], normal: [ 0.0, -1.0,  0.0] },
        CubeVertex { position: [-h, -h,  h], normal: [ 0.0, -1.0,  0.0] },
        // Right face (+X)
        CubeVertex { position: [ h, -h,  h], normal: [ 1.0,  0.0,  0.0] },
        CubeVertex { position: [ h, -h, -h], normal: [ 1.0,  0.0,  0.0] },
        CubeVertex { position: [ h,  h, -h], normal: [ 1.0,  0.0,  0.0] },
        CubeVertex { position: [ h,  h,  h], normal: [ 1.0,  0.0,  0.0] },
        // Left face (-X)
        CubeVertex { position: [-h, -h, -h], normal: [-1.0,  0.0,  0.0] },
        CubeVertex { position: [-h, -h,  h], normal: [-1.0,  0.0,  0.0] },
        CubeVertex { position: [-h,  h,  h], normal: [-1.0,  0.0,  0.0] },
        CubeVertex { position: [-h,  h, -h], normal: [-1.0,  0.0,  0.0] },
    ];

    #[rustfmt::skip]
    let indices: Vec<u32> = (0..6u32)
        .flat_map(|face| {
            let base = face * 4;
            [base, base + 1, base + 2, base, base + 2, base + 3]
        })
        .collect();

    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_cube_geometry() {
        let (vertices, indices) = unit_cube();
        assert_eq!(vertices.len(), 24);
        assert_eq!(indices.len(), 36);

        // All indices should be in range.
        for &idx in &indices {
            assert!(idx < vertices.len() as u32);
        }
    }

    #[test]
    fn object_instance_layout() {
        assert_eq!(std::mem::size_of::<ObjectInstance>(), 32);
        let inst = ObjectInstance {
            offset: [1.0, 2.0, 3.0],
            _pad: 0.0,
            color: [1.0, 0.0, 0.0, 1.0],
        };
        let bytes = bytemuck::bytes_of(&inst);
        assert_eq!(bytes.len(), 32);
    }
}
