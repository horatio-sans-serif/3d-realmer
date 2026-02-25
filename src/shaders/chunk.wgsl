// Chunk voxel rendering shader.
// Bind group 0: global uniforms (camera, lighting, fog)
// Vertex input: position, normal, color

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
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position;

    // Wind animation for vegetation (leaf and grass blocks).
    // Detect vegetation by color: green channel > 0.35 and green > red * 1.5.
    let is_vegetation = in.color.g > 0.35 && in.color.g > in.color.r * 1.5;
    if is_vegetation && globals.wind_strength > 0.01 {
        let freq = 2.0;
        let phase = pos.x * 0.7 + pos.z * 0.5;
        let sway = sin(globals.time * freq + phase) * globals.wind_strength * 0.15;
        pos.x += globals.wind_direction.x * sway;
        pos.z += globals.wind_direction.y * sway;
    }

    out.clip_position = globals.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    out.normal = in.normal;
    out.color = in.color;
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
    let lit_color = in.color * (ambient + diffuse);

    // Distance fog
    let dist = distance(in.world_pos, globals.camera_pos);
    let fog_factor = clamp((dist - globals.fog_start) / (globals.fog_end - globals.fog_start), 0.0, 1.0);
    let final_color = mix(lit_color, globals.fog_color, fog_factor);

    return vec4<f32>(final_color, 1.0);
}
