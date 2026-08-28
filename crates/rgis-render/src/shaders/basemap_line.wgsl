// Renders a single basemap tile's pre-tessellated LINE mesh (roads,
// waterways, boundaries, casings, polygon outlines). Unlike fill geometry
// (basemap.wgsl), each vertex has a `center` (tile-local metres, scaled
// exactly like fill positions) plus an `extrude` offset that's applied in
// SCREEN PIXELS, scaled only by `tile.width_scale` (derived from the
// *current* viewport zoom, not the tile's own position scale). This keeps
// line width constant in device pixels instead of stretching/snapping with
// the tile's zoom, matching how MapLibre GL JS decouples line width from
// tile-position scaling.
//
// Since neither wgpu's WebGPU nor WebGL2 web backends give this app control
// over MSAA sample count (always 1 sample on web), edges are instead
// antialiased analytically: the vertex shader pushes each vertex a small
// constant number of DEVICE PIXELS past its true edge (`AA_MARGIN_PX`) and
// outputs a matching `ratio` (the vertex's distance from the centerline,
// normalized so `ratio = ±1` exactly at the true, unmargined edge); the
// fragment shader then fades alpha to 0 across that margin using
// `fwidth(ratio)` so the feather is ~1 screen pixel wide regardless of the
// line's width or the tile's current scale. This is the same general
// technique MapLibre GL JS itself uses (it renders with `antialias: false`
// and does edge AA in its line/fill shaders rather than relying on the
// canvas context or MSAA).

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
    @location(2) half_width: f32,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) ratio: f32,
};

// Antialiasing feather width, in device/logical pixels (same units as
// `extrude`/`width_scale`'s output). Also caps how far a vertex can be
// pushed past its true edge, so a near-zero-width line (e.g. at very low
// zoom, see `zoom_scale`) can't blow up into a huge quad.
const AA_MARGIN_PX: f32 = 1.0;
const MAX_MARGIN_SCALE: f32 = 8.0;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let hw_screen = abs(in.half_width) * tile.width_scale;
    var pushed_extrude = in.extrude * tile.width_scale;
    var ratio = 0.0;
    if hw_screen > 0.0001 {
        let margin_scale = min((hw_screen + AA_MARGIN_PX) / hw_screen, MAX_MARGIN_SCALE);
        pushed_extrude = pushed_extrude * margin_scale;
        ratio = sign(in.half_width) * margin_scale;
    }
    let screen_pos = tile.offset + in.center * tile.scale + pushed_extrude;
    let ndc_x = (screen_pos.x / screen.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / screen.size.y) * 2.0;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.color = in.color;
    out.ratio = ratio;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = abs(in.ratio);
    let feather = max(fwidth(dist), 0.0001);
    let coverage = 1.0 - smoothstep(1.0 - feather, 1.0 + feather, dist);
    var out_color = in.color;
    out_color.a = out_color.a * coverage;
    return out_color;
}

