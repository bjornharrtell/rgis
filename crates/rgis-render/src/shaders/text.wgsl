struct ScreenUniform {
    size: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> screen: ScreenUniform;
@group(1) @binding(0) var atlas_texture: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) halo_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) halo_color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let ndc_x = (in.position.x / screen.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (in.position.y / screen.size.y) * 2.0;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.halo_color = in.halo_color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = textureSample(atlas_texture, atlas_sampler, in.uv).r;
    // Clamp the antialiasing width: `fwidth` grows unbounded as a label
    // is minified far below its natural glyph resolution (e.g. zoomed
    // out a lot), which otherwise blurs small text into an illegible grey
    // smudge instead of just aliasing a bit.
    let w = clamp(fwidth(dist), 0.02, 0.15);
    let fill_alpha = smoothstep(0.5 - w, 0.5 + w, dist);
    // Halo ring sits just outside the fill edge (rather than far below
    // it) so it reads as a thin outline instead of a thick blurry glow.
    let outline_alpha = smoothstep(0.5 - w * 3.0, 0.5 - w, dist) * (1.0 - fill_alpha)
        + fill_alpha;
    let color = mix(in.halo_color, in.color, fill_alpha);
    return vec4<f32>(color.rgb, color.a * outline_alpha);
}
