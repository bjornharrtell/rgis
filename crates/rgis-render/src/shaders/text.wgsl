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

// Glyph-PBF SDF bitmaps (the `node-fontnik`/Mapbox `tiny-sdf` wire format
// this app's glyph server -- and every MapLibre-compatible one -- uses)
// place the true ink boundary at raw value ~192/255, not the naive halfway
// point 128/255: the encoder reserves extra distance-field headroom on the
// *outside* of each glyph for a halo to sample into, so 0.5 sits deep
// inside the ink rather than right at its edge. Using 0.5 as "inside"
// (as this shader originally did) makes every glyph render bloated/blobby
// with jagged edges, since that part of the gradient is steep and close
// to the buffer's noise floor at these small on-screen sizes.
const FILL_EDGE: f32 = 0.75;
// The halo is a second, larger shape at a lower threshold that the fill
// draws on top of -- i.e. it must extend *outward* from the fill edge
// (toward lower `dist`), not inward, or it can never show as an outline
// around the letterforms the way MapLibre's own text rendering does.
const HALO_EDGE: f32 = 0.5;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = textureSample(atlas_texture, atlas_sampler, in.uv).r;
    // Clamp the antialiasing width: `fwidth` grows unbounded as a label
    // is minified far below its natural glyph resolution (e.g. zoomed
    // out a lot), which otherwise blurs small text into an illegible grey
    // smudge instead of just aliasing a bit.
    let w = clamp(fwidth(dist), 0.02, 0.15);
    let fill_alpha = smoothstep(FILL_EDGE - w, FILL_EDGE + w, dist);
    let halo_alpha = smoothstep(HALO_EDGE - w, HALO_EDGE + w, dist);
    let color = mix(in.halo_color, in.color, fill_alpha);
    return vec4<f32>(color.rgb, color.a * halo_alpha);
}
