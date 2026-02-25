// Particle rendering shader.
// Renders billboarded particles (rain drops, snow flakes) as stretched quads.
// Bind group 0: global uniforms (camera, lighting, fog)
// Vertex slot 0: quad corner vertex
// Vertex slot 1: per-instance particle data (position, velocity, life, size)

struct GlobalUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    sun_direction: vec3<f32>,
    _pad1: f32,
    sun_color: vec3<f32>,
    _pad2: f32,
    ambient_color: vec3<f32>,
    fog_start: f32,
    fog_end: f32,
    _pad3: vec3<f32>,
    fog_color: vec3<f32>,
    time: f32,
    wind_direction: vec2<f32>,
    wind_strength: f32,
    _pad4: f32,
};

@group(0) @binding(0)
var<uniform> globals: GlobalUniforms;

struct VertexInput {
    // Quad corner: (-1,-1), (1,-1), (-1,1), (1,1)
    @location(0) corner: vec2<f32>,
};

struct InstanceInput {
    @location(1) position: vec3<f32>,
    @location(2) velocity_y: f32,
    @location(3) life: f32,
    @location(4) size: f32,
    @location(5) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
    @location(2) particle_color: vec4<f32>,
    @location(3) world_pos: vec3<f32>,
};

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // Billboard: stretch the quad in the velocity direction (rain streaks)
    let stretch = abs(inst.velocity_y) * 0.02;
    let width = inst.size;
    let height = inst.size + stretch;

    // Camera-facing billboard: extract right and up from view-projection
    // For rain, we want velocity-aligned streaks, so use world up + right
    let right = vec3<f32>(1.0, 0.0, 0.0) * width;
    let up = vec3<f32>(0.0, 1.0, 0.0) * height;

    let world_pos = inst.position
        + right * vert.corner.x
        + up * vert.corner.y;

    out.clip_position = globals.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = vert.corner * 0.5 + 0.5;
    out.alpha = inst.life;
    out.particle_color = inst.color;
    out.world_pos = world_pos;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Soft circle falloff
    let dist = distance(in.uv, vec2<f32>(0.5, 0.5));
    let circle = 1.0 - smoothstep(0.3, 0.5, dist);

    // Distance fog
    let cam_dist = distance(in.world_pos, globals.camera_pos);
    let fog_factor = clamp((cam_dist - globals.fog_start) / (globals.fog_end - globals.fog_start), 0.0, 1.0);

    let base_color = in.particle_color.rgb;
    let fogged = mix(base_color, globals.fog_color, fog_factor);

    let alpha = circle * in.alpha * in.particle_color.a;
    return vec4<f32>(fogged, alpha);
}
