// Renders a single basemap tile's pre-tessellated LINE mesh (roads,
// waterways, boundaries, casings, polygon outlines). Unlike fill geometry
// (basemap.wgsl), each vertex has a `center` (tile-local metres, scaled
// exactly like fill positions) plus an `extrude` offset that's applied in
// SCREEN PIXELS, scaled only by `tile.width_scale` (derived from the
// *current* viewport zoom, not the tile's own position scale). This keeps
// line width constant in device pixels instead of stretching/snapping with
// the tile's zoom, matching how MapLibre GL JS decouples line width from
// tile-position scaling.

struct ScreenUniform {
    size: vec2<f32>,
    _padding: vec2<f32>,
};

struct TileTransform {
    offset: vec2<f32>,
    scale: f32,
    width_scale: f32,
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;
@group(1) @binding(0)
var<uniform> tile: TileTransform;

struct VertexInput {
    @location(0) center: vec2<f32>,
    @location(1) extrude: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let screen_pos = tile.offset + in.center * tile.scale + in.extrude * tile.width_scale;
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
