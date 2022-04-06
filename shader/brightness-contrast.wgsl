// Uniforms:

// Group 0 is always the input texture
[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;
// Group 1 is the effect parameters
struct Uniforms {
    brightness: f32,
    contrast: f32,
}
[[group(1), binding(0)]]
var<uniform> uniforms: Uniforms;

struct VertexInput {
    [[location(0)]] position: vec2<f32>;
    [[location(1)]] tex_coords: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    return out;
}



fn modify_color(col: vec4<f32>) -> vec4<f32> {
    // TODO: Implement contrast
    return vec4<f32>(vec3<f32>(uniforms.brightness) + col.rgb, col.a);
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return modify_color(textureSample(t_diffuse, s_diffuse, in.tex_coords));
}
