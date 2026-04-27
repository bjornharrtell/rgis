pub mod tess;

use std::collections::{HashMap, HashSet};

use glow::HasContext;
use rgis_core::{Layer, LayerId, Viewport};

use crate::{MapRenderer, TileImage};
use tess::GpuMesh;

// ── Compiled shader programs ──────────────────────────────────────────────────

struct Programs {
    vector: glow::Program,
    tile: glow::Program,
}

// ── Per-layer GPU buffers ─────────────────────────────────────────────────────

struct LayerBuffers {
    fill_vao: glow::VertexArray,
    fill_vbo: glow::Buffer,
    fill_ebo: glow::Buffer,
    fill_index_count: i32,

    stroke_vao: glow::VertexArray,
    stroke_vbo: glow::Buffer,
    stroke_ebo: glow::Buffer,
    stroke_index_count: i32,

    points_vao: glow::VertexArray,
    points_vbo: glow::Buffer,
    points_ebo: glow::Buffer,
    points_index_count: i32,

    /// The zoom level at which strokes/points were tessellated.
    /// Fill buffers are viewport-independent; strokes/points are rebuilt when zoom changes.
    tessellated_zoom: f64,
}

pub struct GlRenderer {
    gl: glow::Context,
    programs: Programs,
    layer_cache: HashMap<LayerId, LayerBuffers>,
    viewport_width: u32,
    viewport_height: u32,

    tile_vao: glow::VertexArray,
    tile_vbo: glow::Buffer,
    /// GL textures keyed by tile coord (z, x, y). Uploaded once, reused every frame.
    /// RefCell allows &self draw_tiles to populate the cache without conflicting with gl borrow.
    tile_textures: std::cell::RefCell<HashMap<(u8, u32, u32), glow::Texture>>,
    /// Tracks how many consecutive frames each texture has been unused.
    /// Textures are only deleted once this exceeds EVICT_GRACE_FRAMES, giving
    /// the fallback-pyramid time to stop referencing a texture before it is freed.
    tile_age: std::cell::RefCell<HashMap<(u8, u32, u32), u8>>,
}

impl GlRenderer {
    /// # Safety
    /// The caller must ensure a valid OpenGL context is current.
    pub unsafe fn new(gl: glow::Context) -> Self {
        let programs = compile_programs(&gl);

        let (tile_vao, tile_vbo) = create_tile_quad_buffers(&gl);

        Self {
            gl,
            programs,
            layer_cache: HashMap::new(),
            viewport_width: 1,
            viewport_height: 1,
            tile_vao,
            tile_vbo,
            tile_textures: std::cell::RefCell::new(HashMap::new()),
            tile_age: std::cell::RefCell::new(HashMap::new()),
        }
    }
}

impl MapRenderer for GlRenderer {
    fn resize(&mut self, width: u32, height: u32) {
        self.viewport_width = width;
        self.viewport_height = height;
        unsafe {
            self.gl.viewport(0, 0, width as i32, height as i32);
        }
    }

    fn render(&mut self, viewport: &Viewport, layers: &[Layer], tiles: &[TileImage], allow_retessellate: bool) {
        unsafe {
            let gl = &self.gl;
            gl.clear_color(0.15, 0.15, 0.15, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            // Screen-space ortho for tiles (tile screen rects are computed in pixels).
            let tile_proj = ortho_matrix(
                self.viewport_width as f32,
                self.viewport_height as f32,
            );

            // World→NDC matrix for vector layers (vertices are in Web Mercator metres).
            let vec_proj = world_to_ndc_matrix(viewport);

            // Draw basemap tiles
            self.draw_tiles(tiles, &tile_proj);

            // Draw vector layers bottom→top
            let mut sorted: Vec<&Layer> = layers.iter().filter(|l| l.visible).collect();
            sorted.sort_by_key(|l| l.z_order);

            for layer in sorted {
                // Rebuild stroke/point geometry when zoom has drifted far enough
                // — but only when the caller permits it (i.e. no zoom animation
                // is in flight).  Fill buffers are viewport-independent and
                // never need rebuilding here.
                let needs_rebuild = allow_retessellate && self.layer_cache.get(&layer.id)
                    .map(|b| (b.tessellated_zoom - viewport.zoom).abs() > 0.5)
                    .unwrap_or(true);
                if needs_rebuild {
                    // Build new buffers BEFORE freeing old ones so the layer is
                    // always present on the GPU (true double-buffer swap).
                    let new_bufs = build_layer_buffers(gl, layer, viewport);
                    if let Some(old) = self.layer_cache.insert(layer.id, new_bufs) {
                        free_layer_buffers(gl, old);
                    }
                } else if !self.layer_cache.contains_key(&layer.id) {
                    // First upload (no old buffers to double-buffer against).
                    let bufs = build_layer_buffers(gl, layer, viewport);
                    self.layer_cache.insert(layer.id, bufs);
                }

                let bufs = self.layer_cache.get(&layer.id).unwrap();
                let style = &layer.style;

                // Fill
                if bufs.fill_index_count > 0 {
                    gl.use_program(Some(self.programs.vector));
                    set_uniform_mat4(gl, self.programs.vector, "u_transform", &vec_proj);
                    set_uniform_color(gl, self.programs.vector, "u_color", style.fill);
                    gl.bind_vertex_array(Some(bufs.fill_vao));
                    gl.draw_elements(glow::TRIANGLES, bufs.fill_index_count, glow::UNSIGNED_INT, 0);
                }

                // Stroke
                if bufs.stroke_index_count > 0 {
                    gl.use_program(Some(self.programs.vector));
                    set_uniform_mat4(gl, self.programs.vector, "u_transform", &vec_proj);
                    set_uniform_color(gl, self.programs.vector, "u_color", style.stroke);
                    gl.bind_vertex_array(Some(bufs.stroke_vao));
                    gl.draw_elements(glow::TRIANGLES, bufs.stroke_index_count, glow::UNSIGNED_INT, 0);
                }

                // Points
                if bufs.points_index_count > 0 {
                    gl.use_program(Some(self.programs.vector));
                    set_uniform_mat4(gl, self.programs.vector, "u_transform", &vec_proj);
                    set_uniform_color(gl, self.programs.vector, "u_color", style.fill);
                    gl.bind_vertex_array(Some(bufs.points_vao));
                    gl.draw_elements(glow::TRIANGLES, bufs.points_index_count, glow::UNSIGNED_INT, 0);
                }
            }

            gl.bind_vertex_array(None);
        }
    }

    fn invalidate_layer(&mut self, layer_id: LayerId) {
        if let Some(bufs) = self.layer_cache.remove(&layer_id) {
            unsafe { free_layer_buffers(&self.gl, bufs) };
        }
    }
}

// ── Tile rendering ────────────────────────────────────────────────────────────

impl GlRenderer {
    unsafe fn draw_tiles(&self, tiles: &[TileImage], proj: &[f32; 16]) {
        let gl = &self.gl;
        let mut tile_textures = self.tile_textures.borrow_mut();
        gl.use_program(Some(self.programs.tile));
        set_uniform_mat4(gl, self.programs.tile, "u_transform", proj);

        gl.bind_vertex_array(Some(self.tile_vao));

        for tile in tiles {
            // Retrieve cached GL texture or upload it once.
            let tex = if let Some(&cached) = tile_textures.get(&tile.coord) {
                cached
            } else {
                let tex = gl.create_texture().unwrap();
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                gl.tex_image_2d(
                    glow::TEXTURE_2D, 0, glow::RGBA as i32,
                    tile.width as i32, tile.height as i32, 0,
                    glow::RGBA, glow::UNSIGNED_BYTE,
                    Some(tile.rgba.as_slice()),
                );
                tile_textures.insert(tile.coord, tex);
                tex
            };

            gl.bind_texture(glow::TEXTURE_2D, Some(tex));

            // Update the quad VBO for this tile's current screen position and UV sub-rect.
            let [x, y, w, h] = tile.screen_rect;
            let [u0, v0, du, dv] = tile.src_rect;
            #[rustfmt::skip]
            let quad: [f32; 16] = [
                x,     y,     u0,      v0,
                x + w, y,     u0 + du, v0,
                x + w, y + h, u0 + du, v0 + dv,
                x,     y + h, u0,      v0 + dv,
            ];
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.tile_vbo));
            gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytemuck_cast_slice(&quad));

            gl.draw_elements(glow::TRIANGLES, 6, glow::UNSIGNED_INT, 0);
        }

        // Evict GPU textures that have not been referenced for EVICT_GRACE_FRAMES
        // consecutive frames.  Using a grace window means tiles being used as
        // fallbacks for their children (or parents) remain alive across the few
        // frames it takes the correct-zoom tiles to arrive, eliminating flicker.
        // The grace counter also keeps VRAM bounded: any tile unused for
        // EVICT_GRACE_FRAMES frames is guaranteed to be deleted.
        const EVICT_GRACE_FRAMES: u8 = 8;
        const MAX_GPU_TILES: usize = 512;

        let referenced: HashSet<(u8, u32, u32)> = tiles.iter().map(|t| t.coord).collect();

        let mut ages = self.tile_age.borrow_mut();
        // Reset age for every referenced coord; increment all others.
        for coord in tile_textures.keys() {
            if referenced.contains(coord) {
                ages.insert(*coord, 0);
            } else {
                *ages.entry(*coord).or_insert(0) += 1;
            }
        }
        let over_cap = tile_textures.len() > MAX_GPU_TILES;
        tile_textures.retain(|coord, tex| {
            let age = ages.get(coord).copied().unwrap_or(0);
            let evict = age > EVICT_GRACE_FRAMES || over_cap;
            if evict {
                gl.delete_texture(*tex);
                ages.remove(coord);
                false
            } else {
                true
            }
        });
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Screen-space orthographic projection (column-major).
/// Maps [0, width] × [0, height] pixel space to NDC. Used for tile quads.
fn ortho_matrix(width: f32, height: f32) -> [f32; 16] {
    let l = 0.0_f32;
    let r = width;
    let b = height;
    let t = 0.0_f32;
    let n = -1.0_f32;
    let f = 1.0_f32;
    [
        2.0 / (r - l), 0.0,           0.0,            0.0,
        0.0,           2.0 / (t - b), 0.0,            0.0,
        0.0,           0.0,           -2.0 / (f - n), 0.0,
        -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n), 1.0,
    ]
}

/// World→NDC matrix (column-major) mapping Web Mercator metres to clip space.
/// Derivation:
///   x_ndc = 2*(wx - cx) / (width_px * res)
///   y_ndc = 2*(wy - cy) / (height_px * res)   ← Mercator y+ = north = NDC y+
fn world_to_ndc_matrix(vp: &Viewport) -> [f32; 16] {
    let res = vp.resolution();
    let sx = (2.0 / (vp.width_px as f64 * res)) as f32;
    let sy = (2.0 / (vp.height_px as f64 * res)) as f32;
    let tx = (-2.0 * vp.center.x / (vp.width_px as f64 * res)) as f32;
    let ty = (-2.0 * vp.center.y / (vp.height_px as f64 * res)) as f32;
    [
        sx,  0.0, 0.0, 0.0,  // col 0
        0.0, sy,  0.0, 0.0,  // col 1
        0.0, 0.0, 1.0, 0.0,  // col 2
        tx,  ty,  0.0, 1.0,  // col 3  (translation)
    ]
}

unsafe fn set_uniform_mat4(gl: &glow::Context, prog: glow::Program, name: &str, mat: &[f32; 16]) {
    if let Some(loc) = gl.get_uniform_location(prog, name) {
        gl.uniform_matrix_4_f32_slice(Some(&loc), false, mat);
    }
}

unsafe fn set_uniform_color(
    gl: &glow::Context,
    prog: glow::Program,
    name: &str,
    color: rgis_core::Color,
) {
    if let Some(loc) = gl.get_uniform_location(prog, name) {
        gl.uniform_4_f32(Some(&loc), color.r, color.g, color.b, color.a);
    }
}

unsafe fn upload_mesh(gl: &glow::Context, mesh: &GpuMesh) -> (glow::VertexArray, glow::Buffer, glow::Buffer, i32) {
    let vao = gl.create_vertex_array().unwrap();
    let vbo = gl.create_buffer().unwrap();
    let ebo = gl.create_buffer().unwrap();

    gl.bind_vertex_array(Some(vao));

    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    gl.buffer_data_u8_slice(
        glow::ARRAY_BUFFER,
        bytemuck_cast_slice(&mesh.vertices),
        glow::STATIC_DRAW,
    );

    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
    gl.buffer_data_u8_slice(
        glow::ELEMENT_ARRAY_BUFFER,
        bytemuck_cast_slice(&mesh.indices),
        glow::STATIC_DRAW,
    );

    // Attrib 0: vec2 position
    gl.enable_vertex_attrib_array(0);
    gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);

    gl.bind_vertex_array(None);

    (vao, vbo, ebo, mesh.indices.len() as i32)
}

unsafe fn build_layer_buffers(gl: &glow::Context, layer: &Layer, viewport: &Viewport) -> LayerBuffers {
    let res = viewport.resolution() as f32;

    // Fills are viewport-independent — world-space coordinates never change.
    let fill_mesh = tess::tessellate_fills(&layer.features);
    let (fill_vao, fill_vbo, fill_ebo, fill_index_count) = upload_mesh(gl, &fill_mesh);

    // Strokes/points use world-space width scaled to current zoom so they appear
    // approximately `stroke_width` screen pixels thick.
    let stroke_width_world = layer.style.stroke_width * res;
    let stroke_mesh = tess::tessellate_strokes(&layer.features, stroke_width_world);
    let (stroke_vao, stroke_vbo, stroke_ebo, stroke_index_count) = upload_mesh(gl, &stroke_mesh);

    let point_radius_world = layer.style.point_radius * res;
    let points_mesh = tess::build_points_mesh(&layer.features, point_radius_world);
    let (points_vao, points_vbo, points_ebo, points_index_count) = upload_mesh(gl, &points_mesh);

    LayerBuffers {
        fill_vao, fill_vbo, fill_ebo, fill_index_count,
        stroke_vao, stroke_vbo, stroke_ebo, stroke_index_count,
        points_vao, points_vbo, points_ebo, points_index_count,
        tessellated_zoom: viewport.zoom,
    }
}

unsafe fn free_layer_buffers(gl: &glow::Context, bufs: LayerBuffers) {
    gl.delete_vertex_array(bufs.fill_vao);
    gl.delete_buffer(bufs.fill_vbo);
    gl.delete_buffer(bufs.fill_ebo);
    gl.delete_vertex_array(bufs.stroke_vao);
    gl.delete_buffer(bufs.stroke_vbo);
    gl.delete_buffer(bufs.stroke_ebo);
    gl.delete_vertex_array(bufs.points_vao);
    gl.delete_buffer(bufs.points_vbo);
    gl.delete_buffer(bufs.points_ebo);
}

unsafe fn create_tile_quad_buffers(gl: &glow::Context) -> (glow::VertexArray, glow::Buffer) {
    let vao = gl.create_vertex_array().unwrap();
    let vbo = gl.create_buffer().unwrap();
    let ebo = gl.create_buffer().unwrap();

    gl.bind_vertex_array(Some(vao));

    // Placeholder 16 floats (pos + uv per vertex, 4 vertices)
    let data = [0.0_f32; 16];
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
    gl.buffer_data_u8_slice(
        glow::ARRAY_BUFFER,
        bytemuck_cast_slice(&data),
        glow::DYNAMIC_DRAW,
    );

    let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
    gl.buffer_data_u8_slice(
        glow::ELEMENT_ARRAY_BUFFER,
        bytemuck_cast_slice(&indices),
        glow::STATIC_DRAW,
    );

    // Attrib 0: vec2 pos (stride 16, offset 0)
    gl.enable_vertex_attrib_array(0);
    gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 16, 0);
    // Attrib 1: vec2 uv (stride 16, offset 8)
    gl.enable_vertex_attrib_array(1);
    gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 16, 8);

    gl.bind_vertex_array(None);

    (vao, vbo)
}

unsafe fn compile_programs(gl: &glow::Context) -> Programs {
    let vector = compile_program(
        gl,
        include_str!("shaders/vector.vert"),
        include_str!("shaders/vector.frag"),
    );
    let tile = compile_program(
        gl,
        include_str!("shaders/tile.vert"),
        include_str!("shaders/tile.frag"),
    );
    Programs { vector, tile }
}

unsafe fn compile_program(gl: &glow::Context, vert_src: &str, frag_src: &str) -> glow::Program {
    let vert = compile_shader(gl, glow::VERTEX_SHADER, vert_src);
    let frag = compile_shader(gl, glow::FRAGMENT_SHADER, frag_src);

    let prog = gl.create_program().unwrap();
    gl.attach_shader(prog, vert);
    gl.attach_shader(prog, frag);
    gl.link_program(prog);

    if !gl.get_program_link_status(prog) {
        panic!("Program link error: {}", gl.get_program_info_log(prog));
    }

    gl.delete_shader(vert);
    gl.delete_shader(frag);

    prog
}

unsafe fn compile_shader(gl: &glow::Context, ty: u32, src: &str) -> glow::Shader {
    let shader = gl.create_shader(ty).unwrap();
    gl.shader_source(shader, src);
    gl.compile_shader(shader);
    if !gl.get_shader_compile_status(shader) {
        panic!("Shader compile error: {}", gl.get_shader_info_log(shader));
    }
    shader
}

/// Reinterpret a slice of plain-old-data as bytes without the `bytemuck` crate.
fn bytemuck_cast_slice<T: Copy>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    }
}
