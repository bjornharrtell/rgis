// Renders a single basemap tile's pre-tessellated mesh. Unlike vector.wgsl,
// vertex positions are in tile-local mercator METRES (not screen space) —
// the small per-tile `tile` uniform (offset + scale, updated every frame,
// cheap) does the screen positioning on the GPU, so the same persistent
// per-tile vertex/index buffers can be reused unchanged across every
// pan/zoom frame instead of re-tessellating or re-uploading geometry.

struct ScreenUniform {
    size: vec2<f32>,
    _padding: vec2<f32>,
};

struct TileTransform {
    offset: vec2<f32>,
    scale: f32,
    // Unused here (only `basemap_line.wgsl` reads it) — kept so both
    // shaders agree on the uniform buffer's byte layout.
    width_scale: f32,
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;
@group(1) @binding(0)
var<uniform> tile: TileTransform;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let screen_pos = tile.offset + in.position * tile.scale;
    let ndc_x = (screen_pos.x / screen.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / screen.size.y) * 2.0;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
