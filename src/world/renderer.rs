use std::sync::Arc;

use anyhow::Result;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use super::camera::Camera;
use super::chunk_render::ChunkRenderer;
use super::line_render::LineRenderer;
use super::mesh_render::MeshRenderer;
use super::object_render::ObjectRenderer;
use super::weather::WeatherSystem;

/// Shared global uniforms passed to all render pipelines (bind group 0).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
    pub sun_direction: [f32; 3],
    pub _pad1: f32,
    pub sun_color: [f32; 3],
    pub _pad2: f32,
    pub ambient_color: [f32; 3],
    pub fog_start: f32,
    pub fog_end: f32,
    pub _pad3: [f32; 3],
    pub fog_color: [f32; 3],
    pub time: f32,
    pub wind_direction: [f32; 2],
    pub wind_strength: f32,
    pub _pad4: f32,
}

impl Default for GlobalUniforms {
    fn default() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 3],
            _pad0: 0.0,
            sun_direction: [0.3, -1.0, 0.5],
            _pad1: 0.0,
            sun_color: [1.0, 0.95, 0.8],
            _pad2: 0.0,
            ambient_color: [0.15, 0.18, 0.25],
            fog_start: 64.0,
            fog_end: 128.0,
            _pad3: [0.0; 3],
            fog_color: [0.7, 0.8, 0.95],
            time: 0.0,
            wind_direction: [1.0, 0.0],
            wind_strength: 0.3,
            _pad4: 0.0,
        }
    }
}

/// Sky-driven lighting and fog values, updated each frame from the Sky module.
pub struct SkyUniforms {
    pub sun_direction: [f32; 3],
    pub sun_color: [f32; 3],
    pub ambient_color: [f32; 3],
    pub fog_color: [f32; 3],
    pub fog_start: f32,
    pub fog_end: f32,
    pub wind_direction: [f32; 2],
    pub wind_strength: f32,
}

impl Default for SkyUniforms {
    fn default() -> Self {
        Self {
            sun_direction: [0.3, -1.0, 0.5],
            sun_color: [1.0, 0.95, 0.8],
            ambient_color: [0.15, 0.18, 0.25],
            fog_color: [0.7, 0.8, 0.95],
            fog_start: 64.0,
            fog_end: 128.0,
            wind_direction: [1.0, 0.0],
            wind_strength: 0.3,
        }
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    global_uniform_buffer: wgpu::Buffer,
    global_bind_group_layout: wgpu::BindGroupLayout,
    global_bind_group: wgpu::BindGroup,
    clear_color: wgpu::Color,
    start_time: std::time::Instant,
    sky_uniforms: SkyUniforms,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("bitparty-world"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (depth_texture, depth_view) = Self::create_depth_texture(&device, &config);

        let global_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global_uniforms"),
            size: std::mem::size_of::<GlobalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("global_bind_group_layout"),
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

        let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("global_bind_group"),
            layout: &global_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_uniform_buffer.as_entire_binding(),
            }],
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
            depth_view,
            global_uniform_buffer,
            global_bind_group_layout,
            global_bind_group,
            clear_color: wgpu::Color {
                r: 0.4,
                g: 0.6,
                b: 0.9,
                a: 1.0,
            },
            start_time: std::time::Instant::now(),
            sky_uniforms: SkyUniforms::default(),
        })
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            let (depth_texture, depth_view) =
                Self::create_depth_texture(&self.device, &self.config);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
        }
    }

    pub fn render(
        &mut self,
        camera: &Camera,
        chunk_renderer: &ChunkRenderer,
        object_renderer: &ObjectRenderer,
        line_renderer: &LineRenderer,
        mesh_renderer: &MeshRenderer,
        weather: &WeatherSystem,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let uniforms = GlobalUniforms {
            view_proj: camera.view_projection_matrix().to_cols_array_2d(),
            camera_pos: camera.position().to_array(),
            _pad0: 0.0,
            sun_direction: self.sky_uniforms.sun_direction,
            _pad1: 0.0,
            sun_color: self.sky_uniforms.sun_color,
            _pad2: 0.0,
            ambient_color: self.sky_uniforms.ambient_color,
            fog_start: self.sky_uniforms.fog_start,
            fog_end: self.sky_uniforms.fog_end,
            _pad3: [0.0; 3],
            fog_color: self.sky_uniforms.fog_color,
            time: self.start_time.elapsed().as_secs_f32(),
            wind_direction: self.sky_uniforms.wind_direction,
            wind_strength: self.sky_uniforms.wind_strength,
            _pad4: 0.0,
        };

        self.queue.write_buffer(
            &self.global_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // 1. Voxel chunk pass
            chunk_renderer.render(&mut render_pass, &self.global_bind_group);

            // 2. World objects (instanced cubes)
            object_renderer.render(&mut render_pass, &self.global_bind_group);

            // 3. GLTF mesh objects
            mesh_renderer.render(&mut render_pass, &self.global_bind_group, &self.queue);

            // 4. Connection lines (additive blend)
            line_renderer.render(&mut render_pass, &self.global_bind_group);

            // 5. Weather particles (alpha blend, drawn last)
            weather.render(&mut render_pass, &self.global_bind_group);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn set_sky_uniforms(&mut self, sky_uniforms: SkyUniforms) {
        self.sky_uniforms = sky_uniforms;
    }

    pub fn set_clear_color(&mut self, color: [f64; 4]) {
        self.clear_color = wgpu::Color {
            r: color[0],
            g: color[1],
            b: color[2],
            a: color[3],
        };
    }

    pub fn global_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.global_bind_group_layout
    }
}
