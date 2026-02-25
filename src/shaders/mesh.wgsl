// GLTF mesh rendering shader.
// Bind group 0: global uniforms (shared).
// Bind group 1: per-object model matrix + color.

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

struct ObjectUniforms {
    model: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> globals: GlobalUniforms;

@group(1) @binding(0)
var<uniform> object: ObjectUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = (object.model * vec4<f32>(position, 1.0)).xyz;
    out.clip_position = globals.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    // Transform normal by the inverse transpose of the model matrix upper-3x3.
    // For uniform scale, model matrix upper-3x3 works directly.
    let normal_matrix = mat3x3<f32>(
        object.model[0].xyz,
        object.model[1].xyz,
        object.model[2].xyz,
    );
    out.world_normal = normalize(normal_matrix * normal);
    out.color = object.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let sun_dir = normalize(-globals.sun_direction);

    let ndotl = max(dot(normal, sun_dir), 0.0);
    let diffuse = globals.sun_color * ndotl;
    let ambient = globals.ambient_color;
    let lit_color = in.color.rgb * (ambient + diffuse);

    let dist = distance(in.world_pos, globals.camera_pos);
    let fog_factor = clamp((dist - globals.fog_start) / (globals.fog_end - globals.fog_start), 0.0, 1.0);
    let final_color = mix(lit_color, globals.fog_color, fog_factor);

    return vec4<f32>(final_color, in.color.a);
}
