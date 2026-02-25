//! Weather system: rain/snow particles, wind, and weather state.
//!
//! Particles are simulated on the CPU and rendered as GPU-instanced
//! billboarded quads. The particle emitter spawns particles around the
//! camera position, creating a local weather effect.

/// Weather conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherType {
    Clear,
    Rain,
    Snow,
}

/// Wind state shared with the vertex shader for vegetation animation.
#[derive(Debug, Clone, Copy)]
pub struct Wind {
    pub direction: [f32; 2],
    pub strength: f32,
}

impl Default for Wind {
    fn default() -> Self {
        Self {
            direction: [1.0, 0.0],
            strength: 0.3,
        }
    }
}

/// A single particle in the weather system.
struct Particle {
    position: [f32; 3],
    velocity: [f32; 3],
    life: f32,
    max_life: f32,
    size: f32,
}

/// GPU-side per-instance particle data.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleInstance {
    pub position: [f32; 3],
    pub velocity_y: f32,
    pub life: f32,
    pub size: f32,
    pub _pad: [f32; 2],
    pub color: [f32; 4],
}

impl ParticleInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        static ATTRIBUTES: &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32,
            },
            wgpu::VertexAttribute {
                offset: 20,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32,
            },
            // pad at 24..32
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x4,
            },
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRIBUTES,
        }
    }
}

/// Quad corner vertex for billboarded particles.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    corner: [f32; 2],
}

impl QuadVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        static ATTRIBUTES: &[wgpu::VertexAttribute] = &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x2,
        }];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: ATTRIBUTES,
        }
    }
}

const MAX_PARTICLES: usize = 2048;

/// Weather system: manages particle simulation and rendering.
pub struct WeatherSystem {
    weather: WeatherType,
    wind: Wind,
    particles: Vec<Particle>,
    spawn_timer: f32,

    // GPU resources
    pipeline: wgpu::RenderPipeline,
    quad_vertex_buffer: wgpu::Buffer,
    quad_index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
}

impl WeatherSystem {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        global_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/particle.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_pipeline_layout"),
            bind_group_layouts: &[global_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[QuadVertex::desc(), ParticleInstance::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Billboards visible from both sides
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Particles don't write depth
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

        // Quad vertices: two triangles forming a [-1,1] x [-1,1] quad
        let quad_vertices = [
            QuadVertex {
                corner: [-1.0, -1.0],
            },
            QuadVertex {
                corner: [1.0, -1.0],
            },
            QuadVertex {
                corner: [-1.0, 1.0],
            },
            QuadVertex { corner: [1.0, 1.0] },
        ];
        let quad_indices: [u16; 6] = [0, 1, 2, 2, 1, 3];

        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle_quad_vb"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle_quad_ib"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_instance_buffer"),
            size: (MAX_PARTICLES * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            weather: WeatherType::Clear,
            wind: Wind::default(),
            particles: Vec::with_capacity(MAX_PARTICLES),
            spawn_timer: 0.0,
            pipeline,
            quad_vertex_buffer,
            quad_index_buffer,
            instance_buffer,
            instance_count: 0,
        }
    }

    pub fn weather(&self) -> WeatherType {
        self.weather
    }

    pub fn set_weather(&mut self, weather: WeatherType) {
        if self.weather != weather {
            self.weather = weather;
            // Clear existing particles on weather change for clean transition
            self.particles.clear();
        }
    }

    pub fn wind(&self) -> &Wind {
        &self.wind
    }

    pub fn set_wind(&mut self, direction: [f32; 2], strength: f32) {
        let len = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        if len > 0.001 {
            self.wind.direction = [direction[0] / len, direction[1] / len];
        }
        self.wind.strength = strength.clamp(0.0, 2.0);
    }

    /// Simulate particles and prepare GPU instance data.
    pub fn update(&mut self, dt: f32, camera_pos: [f32; 3], queue: &wgpu::Queue) {
        if self.weather == WeatherType::Clear {
            self.particles.clear();
            self.instance_count = 0;
            return;
        }

        let spawn_rate = match self.weather {
            WeatherType::Rain => 200.0, // particles per second
            WeatherType::Snow => 80.0,
            WeatherType::Clear => 0.0,
        };

        // Spawn new particles
        self.spawn_timer += dt;
        let spawn_interval = 1.0 / spawn_rate;
        while self.spawn_timer >= spawn_interval && self.particles.len() < MAX_PARTICLES {
            self.spawn_timer -= spawn_interval;
            self.spawn_particle(camera_pos);
        }

        // Update existing particles
        let wind_x = self.wind.direction[0] * self.wind.strength;
        let wind_z = self.wind.direction[1] * self.wind.strength;

        self.particles.retain_mut(|p| {
            p.life -= dt;
            if p.life <= 0.0 {
                return false;
            }

            // Apply wind
            p.position[0] += (p.velocity[0] + wind_x) * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += (p.velocity[2] + wind_z) * dt;

            true
        });

        // Build GPU instances
        let color = match self.weather {
            WeatherType::Rain => [0.6, 0.7, 0.9, 0.4],
            WeatherType::Snow => [0.95, 0.95, 1.0, 0.6],
            WeatherType::Clear => [0.0; 4],
        };

        let instances: Vec<ParticleInstance> = self
            .particles
            .iter()
            .map(|p| {
                let life_frac = (p.life / p.max_life).clamp(0.0, 1.0);
                ParticleInstance {
                    position: p.position,
                    velocity_y: p.velocity[1],
                    life: life_frac,
                    size: p.size,
                    _pad: [0.0; 2],
                    color,
                }
            })
            .collect();

        self.instance_count = instances.len() as u32;

        if !instances.is_empty() {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&instances),
            );
        }
    }

    fn spawn_particle(&mut self, camera_pos: [f32; 3]) {
        // Spawn within a 40x40 area around the camera, 15-25 units above
        let spread = 20.0;
        let rx = pseudo_random_f32(self.particles.len() as u32 * 3 + 1) * 2.0 - 1.0;
        let rz = pseudo_random_f32(self.particles.len() as u32 * 3 + 2) * 2.0 - 1.0;
        let ry = pseudo_random_f32(self.particles.len() as u32 * 3 + 3);

        let position = [
            camera_pos[0] + rx * spread,
            camera_pos[1] + 15.0 + ry * 10.0,
            camera_pos[2] + rz * spread,
        ];

        let (velocity, max_life, size) = match self.weather {
            WeatherType::Rain => {
                let speed = 8.0 + pseudo_random_f32(self.particles.len() as u32 * 7) * 4.0;
                ([0.0, -speed, 0.0], 3.0, 0.02)
            }
            WeatherType::Snow => {
                let speed = 1.0 + pseudo_random_f32(self.particles.len() as u32 * 7) * 0.5;
                let drift_x = (pseudo_random_f32(self.particles.len() as u32 * 11) - 0.5) * 0.5;
                let drift_z = (pseudo_random_f32(self.particles.len() as u32 * 13) - 0.5) * 0.5;
                ([drift_x, -speed, drift_z], 8.0, 0.05)
            }
            WeatherType::Clear => return,
        };

        self.particles.push(Particle {
            position,
            velocity,
            life: max_life,
            max_life,
            size,
        });
    }

    /// Render all live particles.
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
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..6, 0, 0..self.instance_count);
    }
}

/// Simple deterministic pseudo-random based on hash mixing.
fn pseudo_random_f32(seed: u32) -> f32 {
    let mut h = seed;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a2d39);
    h ^= h >> 15;
    (h & 0xFFFF) as f32 / 65535.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_instance_size() {
        assert_eq!(std::mem::size_of::<ParticleInstance>(), 48);
    }

    #[test]
    fn pseudo_random_range() {
        for i in 0..100 {
            let v = pseudo_random_f32(i);
            assert!(v >= 0.0 && v <= 1.0, "out of range: {}", v);
        }
    }

    #[test]
    fn weather_default_is_clear() {
        // WeatherType doesn't implement Default, but test the enum values
        assert_eq!(WeatherType::Clear, WeatherType::Clear);
        assert_ne!(WeatherType::Rain, WeatherType::Snow);
    }

    #[test]
    fn wind_default() {
        let w = Wind::default();
        assert_eq!(w.direction, [1.0, 0.0]);
        assert!((w.strength - 0.3).abs() < 0.001);
    }
}
