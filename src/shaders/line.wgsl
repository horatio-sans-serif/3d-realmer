// Connection line rendering shader.
// Drawn with LineList topology and additive blending for a glow effect.
// Bind group 0: global uniforms (shared with chunk shader).

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
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = globals.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    out.world_pos = position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Pulsating glow
    let pulse = 0.7 + 0.3 * sin(globals.time * 3.0);
    let glowing = in.color * pulse;

    // Distance fog (softer fade for lines)
    let dist = distance(in.world_pos, globals.camera_pos);
    let fog_factor = clamp((dist - globals.fog_start) / (globals.fog_end - globals.fog_start), 0.0, 1.0);
    let final_color = mix(glowing.rgb, globals.fog_color * 0.1, fog_factor);

    return vec4<f32>(final_color, glowing.a * (1.0 - fog_factor));
}
