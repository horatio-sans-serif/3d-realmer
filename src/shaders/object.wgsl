// World object rendering shader.
// Bind group 0: global uniforms (shared with chunk shader).
// Vertex slot 0: cube geometry (position, normal).
// Vertex slot 1: per-instance data (offset, color) with Instance step mode.

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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) offset: vec3<f32>,
    @location(3) color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = position + offset;
    out.clip_position = globals.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.normal = normal;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.normal);
    let sun_dir = normalize(-globals.sun_direction);

    // Diffuse lighting
    let ndotl = max(dot(normal, sun_dir), 0.0);
    let diffuse = globals.sun_color * ndotl;

    // Ambient
    let ambient = globals.ambient_color;

    // Combined lighting
    let lit_color = in.color.rgb * (ambient + diffuse);

    // Distance fog
    let dist = distance(in.world_pos, globals.camera_pos);
    let fog_factor = clamp((dist - globals.fog_start) / (globals.fog_end - globals.fog_start), 0.0, 1.0);
    let final_color = mix(lit_color, globals.fog_color, fog_factor);

    return vec4<f32>(final_color, in.color.a);
}
