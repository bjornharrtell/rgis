//! wgpu render resources for drawing the tessellated vector mesh, raster
//! tile images, and SDF text labels inside an `egui_wgpu::Callback`, so the
//! map renders on the same wgpu device/surface as the rest of the (egui) UI
//! — on native and in the browser (WebGPU/WebGL2) alike.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lru::LruCache;
use rgis_tiles::{GLYPH_BUFFER, Glyph, TileCoord, glyph_range_start};
use rustc_hash::FxHashMap;
use wgpu::util::DeviceExt;

use crate::basemap::{LineVertex, TileMesh};
use crate::mesh::{SceneMesh, Vertex};
use crate::text::{GlyphBitmapRanges, LabelGlyphInstance};

/// MSAA sample count for the map's wgpu pipelines. Must match whatever
/// sample count eframe's shared renderer was created with (native: set via
/// `NativeOptions::multisampling`; the web/WebOptions API has no equivalent
/// knob, so it always renders at 1 sample there).
pub const MSAA_SAMPLES: u32 = if cfg!(target_arch = "wasm32") { 1 } else { 4 };

const VERTEX_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

const LINE_VERTEX_ATTRS: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Float32x4];

const TILE_VERTEX_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32];

const TEXT_VERTEX_ATTRS: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Float32x4];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TileVertex {
    position: [f32; 2],
    uv: [f32; 2],
    /// `raster-opacity` (style paint property), constant across a tile's
    /// four corners -- see `TileDraw::opacity`.
    opacity: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TextVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
    halo_color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ScreenUniform {
    size: [f32; 2],
    _padding: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TileTransformUniform {
    offset: [f32; 2],
    scale: f32,
    /// See `basemap::TileTransform::width_scale`.
    width_scale: f32,
}

/// A basemap tile's tile-local-metres mesh (see `basemap::build_tile_mesh`)
/// plus the small per-frame screen transform needed to position it. The
/// mesh itself is only uploaded to the GPU once per tile (see
/// `MapRenderResources::ensure_basemap_tile_buffer`) and reused unchanged
/// across every subsequent frame/pan/zoom -- only the transform changes.
pub struct BasemapTileDraw {
    pub coord: TileCoord,
    pub mesh: Arc<TileMesh>,
    pub offset: [f32; 2],
    pub scale: f32,
    pub width_scale: f32,
    /// See `basemap::TileTransform::size`.
    pub size: f32,
}

/// Persistent GPU vertex/index buffers for one sub-mesh (fill or lines) of
/// a basemap tile.
struct SubMeshBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

/// Persistent GPU buffers for one basemap tile, uploaded once when the tile
/// first appears and reused for every future draw of that tile (including
/// during pan/zoom) until it's evicted. Either sub-mesh may be absent (e.g.
/// a tile with no roads has no `lines`). The per-tile screen transform is
/// NOT stored here -- see `MapRenderResources::tile_transform_pool_buffer`,
/// which holds every visible tile's transform in one buffer so drawing many
/// tiles doesn't require a bind-group switch per tile.
struct TileGpuMesh {
    fill: Option<SubMeshBuffers>,
    lines: Option<SubMeshBuffers>,
}

/// A single textured screen-space quad to draw this frame: a raster tile
/// or a sprite icon, sharing one GPU pipeline since both are just "sample
/// an RGBA texture through a UV sub-rect and blend onto a screen-pixel
/// quad". `key` identifies the *source texture* (not the individual draw):
/// raster tiles use one key per tile (a distinct image each), while every
/// icon sprite from the same style shares a single key (they all sample
/// the one fetched sprite atlas image, uploaded once and cached -- see
/// `ensure_tile_texture`).
pub struct TileDraw {
    pub key: u64,
    pub rect: [f32; 4],
    pub rgba: Arc<image::RgbaImage>,
    /// Normalized `[u0, v0, u1, v1]` sub-rect of `rgba` to sample --
    /// `[0.0, 0.0, 1.0, 1.0]` (the whole image) for a raster tile, or a
    /// sprite's packed atlas sub-rect for an icon.
    pub uv_rect: [f32; 4],
    /// `raster-opacity`/`icon-opacity` evaluated for the layer this draw
    /// belongs to (`1.0` if the style doesn't specify one).
    pub opacity: f32,
}

struct TileGpuTexture {
    bind_group: wgpu::BindGroup,
}

#[derive(Debug, Clone, Copy)]
struct AtlasRect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// Extra gap (px) left between adjacent glyphs' reserved rects in
/// [`GlyphAtlas`]'s shelf packing, on top of each glyph's own
/// `GLYPH_BUFFER` -- prevents bilinear sampling at one glyph's edge from
/// picking up a neighboring glyph's texels.
const ATLAS_PADDING: u32 = 2;

/// Shelf-packed R8 atlas shared across all map-label draws.
struct GlyphAtlas {
    _texture: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: u32,
    cursor_x: u32,
    cursor_y: u32,
    shelf_height: u32,
    packed: FxHashMap<(String, u32), AtlasRect>,
    /// Whether the whole texture has had a single initializing zero-fill
    /// write yet (see `ensure`'s doc comment for why).
    initialized: bool,
}

impl GlyphAtlas {
    fn new(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        size: u32,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgis-glyph-atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgis-glyph-atlas-bind-group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Self {
            _texture: texture,
            _view: view,
            bind_group,
            size,
            cursor_x: 0,
            cursor_y: 0,
            shelf_height: 0,
            packed: FxHashMap::default(),
            initialized: false,
        }
    }

    fn ensure(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        fontstack: &str,
        codepoint: u32,
        glyph: &Glyph,
    ) -> Option<AtlasRect> {
        let key = (fontstack.to_string(), codepoint);
        if let Some(rect) = self.packed.get(&key).copied() {
            return Some(rect);
        }

        // WebGL2 (notably Firefox) tracks a texture's initialization state
        // at the whole-texture level: the very first write to it -- even
        // one covering the whole surface -- is otherwise a `texSubImage`
        // into storage the browser still considers uninitialized, which it
        // silently (and slowly) full-clears itself first, logging "Texture
        // has not been initialized prior to a partial upload"/"is incurring
        // lazy initialization". Since every glyph after the very first one
        // packed here is written as a small sub-rectangle of this shared
        // atlas, without this explicit upfront zero-fill *every single
        // glyph* would otherwise be the "first write" from the browser's
        // point of view for that region and could still trigger it; doing
        // one full-surface write covering the whole atlas immediately marks
        // the entire texture initialized so subsequent per-glyph partial
        // uploads don't re-trigger the browser's implicit clear.
        if !self.initialized {
            let zeros = vec![0u8; (self.size * self.size) as usize];
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self._texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &zeros,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.size),
                    rows_per_image: Some(self.size),
                },
                wgpu::Extent3d {
                    width: self.size,
                    height: self.size,
                    depth_or_array_layers: 1,
                },
            );
            self.initialized = true;
        }

        let w = glyph.width + 2 * GLYPH_BUFFER;
        let h = glyph.height + 2 * GLYPH_BUFFER;
        if w == 0
            || h == 0
            || w > self.size
            || h > self.size
            || glyph.bitmap.len() != (w * h) as usize
        {
            return None;
        }

        // Reserve `w + ATLAS_PADDING`/`h + ATLAS_PADDING` in the shelf
        // layout (not just `w`/`h`) so adjacent glyphs' texels are never
        // direct neighbors in the atlas -- bilinear filtering (needed for
        // smooth SDF edges) otherwise blends a glyph's edge pixels with
        // whatever unrelated glyph happens to be packed right next to it,
        // which showed up as faint ghosting/smearing artifacts around
        // glyphs sharing a shelf.
        if self.cursor_x + w + ATLAS_PADDING > self.size {
            self.cursor_x = 0;
            self.cursor_y += self.shelf_height + ATLAS_PADDING;
            self.shelf_height = 0;
        }
        if self.cursor_y + h > self.size {
            return None;
        }

        let rect = AtlasRect {
            x: self.cursor_x,
            y: self.cursor_y,
            w,
            h,
        };
        self.cursor_x += w + ATLAS_PADDING;
        self.shelf_height = self.shelf_height.max(h);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self._texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x,
                    y: rect.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &glyph.bitmap,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        self.packed.insert(key, rect);
        Some(rect)
    }
}

/// Persistent GPU resources for the map. Created once and stored in eframe's
/// `egui_wgpu::CallbackResources`, then reused every frame by [`MapCallback`].
pub struct MapRenderResources {
    vector_pipeline: wgpu::RenderPipeline,
    screen_uniform_buffer: wgpu::Buffer,
    screen_bind_group: wgpu::BindGroup,

    basemap_pipeline: wgpu::RenderPipeline,
    basemap_line_pipeline: wgpu::RenderPipeline,
    /// Bounded LRU rather than an exact-per-frame-visible set: tiles that
    /// scroll just off screen keep their GPU buffers for a while, so
    /// panning back over recently-visited ground reuses them instead of
    /// re-uploading (which real pan gestures do constantly, since tiles
    /// leave visibility one at a time along the trailing edge).
    tile_gpu_meshes: LruCache<TileCoord, TileGpuMesh>,
    /// One shared uniform buffer holding every currently-visible basemap
    /// tile's [`TileTransformUniform`], each at a `tile_transform_stride`-
    /// aligned offset, bound via `tile_transform_bind_group` with a
    /// per-draw *dynamic offset* instead of a per-tile bind group -- with
    /// potentially dozens of tiles visible at once (e.g. mid zoom over a
    /// dense road network), switching an actual bind group per tile was a
    /// major per-frame cost (especially on WebGL2, which emulates bind
    /// groups as a sequence of individual GL calls); a dynamic offset into
    /// one already-bound buffer is far cheaper.
    tile_transform_pool_buffer: wgpu::Buffer,
    tile_transform_bind_group: wgpu::BindGroup,
    tile_transform_bind_group_layout: wgpu::BindGroupLayout,
    /// Byte stride between consecutive tiles' slots in
    /// `tile_transform_pool_buffer`, rounded up to the device's required
    /// dynamic-uniform-offset alignment.
    tile_transform_stride: u64,
    /// Number of tile slots currently allocated in
    /// `tile_transform_pool_buffer`; grown (never shrunk) as needed.
    tile_transform_capacity: u64,

    tile_pipeline: wgpu::RenderPipeline,
    tile_bind_group_layout: wgpu::BindGroupLayout,
    tile_sampler: wgpu::Sampler,
    /// Bounded LRU cache (see `TILE_TEXTURE_CACHE_SIZE`), *not* dropped
    /// wholesale each frame for tiles that happen to be off-screen: raster
    /// tile textures used to be pruned down to exactly the current frame's
    /// visible set, so a tile scrolling briefly out of view and back (very
    /// common while panning) forced its GPU texture to be recreated and
    /// re-uploaded from scratch. Besides the redundant upload cost, WebGL2
    /// (Firefox especially) treats every brand-new texture's first write as
    /// uninitialized and forces a full-texture clear before the partial
    /// `texSubImage`, logging "Texture has not been initialized prior to a
    /// partial upload" -- so constant recreation during pan/zoom meant
    /// constant forced clears, worsening exactly the perf/memory symptoms
    /// panning was showing. Keeping recently-used-but-currently-offscreen
    /// tiles cached (evicted only once the LRU actually fills up) avoids
    /// that churn.
    tile_textures: LruCache<u64, TileGpuTexture>,
    /// Raster tiles' quad vertices, rewritten (not reallocated, unless the
    /// visible tile count grows) every frame.
    tile_vertex_buffer: GrowableBuffer,
    /// Per-tile absolute (non-shared) quad indices, rewritten every frame
    /// alongside `tile_vertex_buffer`. Each tile's 6 indices point at its
    /// own 4 vertices directly (`i*4 + 0/1/2/3`) rather than reusing a
    /// single shared unit-quad index pattern via `draw_indexed`'s
    /// `base_vertex` parameter: WebGL2 only supports a nonzero
    /// `base_vertex` through the `ANGLE_base_vertex_base_instance`
    /// extension, which isn't reliably available (e.g. on Firefox), and
    /// `wgpu`'s GL backend panics ("Draw elements instanced base vertex is
    /// not supported") without it. Baking absolute indices in here keeps
    /// every tile's draw call to a plain `base_vertex = 0` while still only
    /// binding the vertex/index buffers once per pass (not once per tile).
    tile_index_buffer: GrowableBuffer,
    /// Background quad + user vector layers' geometry, rewritten every
    /// frame.
    scene_vertex_buffer: GrowableBuffer,
    scene_index_buffer: GrowableBuffer,
    glyph_atlas: GlyphAtlas,
    text_pipeline: wgpu::RenderPipeline,
    _text_atlas_bind_group_layout: wgpu::BindGroupLayout,
    _text_atlas_sampler: wgpu::Sampler,
    text_vertex_buffer: GrowableBuffer,
    text_index_buffer: GrowableBuffer,
}

impl MapRenderResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let screen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rgis-screen-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let screen_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rgis-screen-uniform"),
            size: std::mem::size_of::<ScreenUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let screen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgis-screen-bind-group"),
            layout: &screen_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen_uniform_buffer.as_entire_binding(),
            }],
        });

        let vector_pipeline =
            create_vector_pipeline(device, target_format, &screen_bind_group_layout);

        let tile_transform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rgis-basemap-tile-transform-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            TileTransformUniform,
                        >() as u64),
                    },
                    count: None,
                }],
            });

        let basemap_pipeline = create_basemap_pipeline(
            device,
            target_format,
            &screen_bind_group_layout,
            &tile_transform_bind_group_layout,
        );

        let basemap_line_pipeline = create_basemap_line_pipeline(
            device,
            target_format,
            &screen_bind_group_layout,
            &tile_transform_bind_group_layout,
        );

        let tile_transform_stride = align_up(
            std::mem::size_of::<TileTransformUniform>() as u64,
            device.limits().min_uniform_buffer_offset_alignment as u64,
        );
        let tile_transform_capacity = INITIAL_TILE_TRANSFORM_CAPACITY;
        let tile_transform_pool_buffer = create_tile_transform_pool_buffer(
            device,
            tile_transform_stride,
            tile_transform_capacity,
        );
        let tile_transform_bind_group = create_tile_transform_bind_group(
            device,
            &tile_transform_bind_group_layout,
            &tile_transform_pool_buffer,
        );

        let tile_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rgis-tile-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let tile_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rgis-tile-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let tile_pipeline = create_tile_pipeline(
            device,
            target_format,
            &screen_bind_group_layout,
            &tile_bind_group_layout,
        );

        let text_atlas_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rgis-text-atlas-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let text_atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rgis-text-atlas-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let text_pipeline = create_text_pipeline(
            device,
            target_format,
            &screen_bind_group_layout,
            &text_atlas_bind_group_layout,
        );

        let glyph_atlas = GlyphAtlas::new(
            device,
            &text_atlas_bind_group_layout,
            &text_atlas_sampler,
            GLYPH_ATLAS_SIZE,
        );

        Self {
            vector_pipeline,
            screen_uniform_buffer,
            screen_bind_group,
            basemap_pipeline,
            basemap_line_pipeline,
            tile_gpu_meshes: LruCache::new(
                std::num::NonZeroUsize::new(TILE_GPU_MESH_CACHE_SIZE).unwrap(),
            ),
            tile_transform_pool_buffer,
            tile_transform_bind_group,
            tile_transform_bind_group_layout,
            tile_transform_stride,
            tile_transform_capacity,
            tile_pipeline,
            tile_bind_group_layout,
            tile_sampler,
            tile_textures: LruCache::new(
                std::num::NonZeroUsize::new(TILE_TEXTURE_CACHE_SIZE).unwrap(),
            ),
            tile_vertex_buffer: GrowableBuffer::new(
                wgpu::BufferUsages::VERTEX,
                "rgis-tile-vertices",
            ),
            tile_index_buffer: GrowableBuffer::new(wgpu::BufferUsages::INDEX, "rgis-tile-indices"),
            scene_vertex_buffer: GrowableBuffer::new(
                wgpu::BufferUsages::VERTEX,
                "rgis-vector-vertices",
            ),
            scene_index_buffer: GrowableBuffer::new(
                wgpu::BufferUsages::INDEX,
                "rgis-vector-indices",
            ),
            glyph_atlas,
            text_pipeline,
            _text_atlas_bind_group_layout: text_atlas_bind_group_layout,
            _text_atlas_sampler: text_atlas_sampler,
            text_vertex_buffer: GrowableBuffer::new(
                wgpu::BufferUsages::VERTEX,
                "rgis-text-vertices",
            ),
            text_index_buffer: GrowableBuffer::new(wgpu::BufferUsages::INDEX, "rgis-text-indices"),
        }
    }

    fn update_screen_size(&self, queue: &wgpu::Queue, width: f32, height: f32) {
        let uniform = ScreenUniform {
            size: [width.max(1.0), height.max(1.0)],
            _padding: [0.0, 0.0],
        };
        queue.write_buffer(&self.screen_uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn ensure_tile_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, tile: &TileDraw) {
        if self.tile_textures.get(&tile.key).is_some() {
            // Already cached; `get` (rather than `peek`) promotes it to
            // most-recently-used so it isn't the next thing evicted.
            return;
        }

        let size = wgpu::Extent3d {
            width: tile.rgba.width(),
            height: tile.rgba.height(),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgis-tile-texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            tile.rgba.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * tile.rgba.width()),
                rows_per_image: Some(tile.rgba.height()),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgis-tile-bind-group"),
            layout: &self.tile_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.tile_sampler),
                },
            ],
        });

        self.tile_textures
            .put(tile.key, TileGpuTexture { bind_group });
    }

    /// Uploads a basemap tile's fill + line meshes to the GPU once; a no-op
    /// if that tile's buffers already exist (the whole point -- pan/zoom
    /// never re-uploads or reallocates). Touches the LRU entry so tiles
    /// still being drawn stay recently-used and aren't the first evicted.
    fn ensure_basemap_tile_buffer(
        &mut self,
        device: &wgpu::Device,
        coord: TileCoord,
        mesh: &TileMesh,
    ) {
        if self.tile_gpu_meshes.get(&coord).is_some() {
            return;
        }
        let fill = (!mesh.fill.indices.is_empty()).then(|| SubMeshBuffers {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rgis-basemap-tile-fill-vertices"),
                contents: bytemuck::cast_slice(&mesh.fill.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rgis-basemap-tile-fill-indices"),
                contents: bytemuck::cast_slice(&mesh.fill.indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: mesh.fill.indices.len() as u32,
        });
        let lines = (!mesh.lines.indices.is_empty()).then(|| SubMeshBuffers {
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rgis-basemap-tile-line-vertices"),
                contents: bytemuck::cast_slice(&mesh.lines.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rgis-basemap-tile-line-indices"),
                contents: bytemuck::cast_slice(&mesh.lines.indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: mesh.lines.indices.len() as u32,
        });
        self.tile_gpu_meshes.put(coord, TileGpuMesh { fill, lines });
    }

    /// Grows `tile_transform_pool_buffer` (and recreates the bind group
    /// bound to it) if fewer than `needed` tile slots are currently
    /// allocated. Never shrinks, so a brief spike in visible tile count
    /// doesn't cause repeated reallocation on every subsequent frame.
    fn ensure_tile_transform_capacity(&mut self, device: &wgpu::Device, needed: u64) {
        if needed <= self.tile_transform_capacity {
            return;
        }
        let capacity = needed
            .next_power_of_two()
            .max(INITIAL_TILE_TRANSFORM_CAPACITY);
        self.tile_transform_pool_buffer =
            create_tile_transform_pool_buffer(device, self.tile_transform_stride, capacity);
        self.tile_transform_bind_group = create_tile_transform_bind_group(
            device,
            &self.tile_transform_bind_group_layout,
            &self.tile_transform_pool_buffer,
        );
        self.tile_transform_capacity = capacity;
    }
}

/// Initial number of tile slots in `MapRenderResources::tile_transform_pool_buffer`.
const INITIAL_TILE_TRANSFORM_CAPACITY: u64 = 64;

/// Fixed-size shared glyph atlas; new glyphs are shelf-packed until full.
const GLYPH_ATLAS_SIZE: u32 = 2048;

/// Number of basemap tiles' GPU mesh buffers kept alive at once. Larger than
/// a typical viewport's visible-tile count so tiles that scroll just off
/// screen (the common case during panning) don't need re-uploading if the
/// user pans back before they'd be evicted.
const TILE_GPU_MESH_CACHE_SIZE: usize = 512;

/// Number of raster-tile (and sprite icon) GPU textures kept alive at once
/// -- see `MapRenderResources::tile_textures`'s docs for why this needs to
/// outlive a single frame's visible set rather than being pruned down to it
/// every frame.
const TILE_TEXTURE_CACHE_SIZE: usize = 256;

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment.max(1)) * alignment.max(1)
}

/// Indices for a single axis-aligned quad drawn as two triangles, shared by
/// every raster tile (each tile only differs in its vertex positions/UVs).
const UNIT_QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

/// A GPU buffer for content that's rebuilt from scratch every frame (the
/// scene mesh and raster-tile quads, whose vertex/index counts change with
/// the current viewport/layers) -- as opposed to a basemap tile's mesh,
/// which is uploaded once and reused. Grows (reallocating) when new data no
/// longer fits, but never shrinks, so the common case (data fits in the
/// existing buffer) is just a `queue.write_buffer` instead of creating and
/// immediately discarding a brand new buffer every frame. WebGL2 drivers
/// can lag in reclaiming deleted buffers' memory, so constant per-frame
/// reallocation was ratcheting up GPU memory use over a long session even
/// though each individual buffer was short-lived.
struct GrowableBuffer {
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
    len: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl GrowableBuffer {
    fn new(usage: wgpu::BufferUsages, label: &'static str) -> Self {
        Self {
            buffer: None,
            capacity: 0,
            len: 0,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            label,
        }
    }

    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        self.len = data.len() as u64;
        if self.len == 0 {
            return;
        }
        if self.len > self.capacity {
            let capacity = self.len.next_power_of_two();
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: capacity,
                usage: self.usage,
                mapped_at_creation: false,
            }));
            self.capacity = capacity;
        }
        queue.write_buffer(self.buffer.as_ref().expect("just ensured"), 0, data);
    }

    fn slice(&self) -> Option<wgpu::BufferSlice<'_>> {
        (self.len > 0).then(|| {
            self.buffer
                .as_ref()
                .expect("len > 0 implies a buffer was allocated")
                .slice(..self.len)
        })
    }
}

fn create_tile_transform_pool_buffer(
    device: &wgpu::Device,
    stride: u64,
    capacity: u64,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rgis-basemap-tile-transform-pool"),
        size: stride * capacity,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_tile_transform_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    pool_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rgis-basemap-tile-transform-pool-bind-group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: pool_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<TileTransformUniform>() as u64),
            }),
        }],
    })
}

/// Per-frame data handed to the egui paint callback. Built fresh every frame
/// from the current viewport/project state.
pub struct MapCallback {
    /// Background quad (indices `0..background_index_count`) followed by
    /// user layer geometry, sharing one vertex/index buffer so they can
    /// still be drawn as two `draw_indexed` sub-ranges around the basemap
    /// tile draws below.
    pub mesh: SceneMesh,
    pub background_index_count: u32,
    /// Basemap tiles, drawn between the background and the user layers.
    /// Each tile's mesh is only uploaded to the GPU once (see
    /// `MapRenderResources::ensure_basemap_tile_buffer`); only the small
    /// `offset`/`scale` transform is refreshed every frame.
    pub basemap_tiles: Vec<BasemapTileDraw>,
    /// Raster background tiles (drawn first, at index
    /// `0..raster_tile_count`) followed by sprite icon quads for symbol
    /// labels (drawn later, above every other layer, at index
    /// `raster_tile_count..tiles.len()`) -- sharing one vertex/index
    /// buffer/pipeline (see `TileDraw`'s docs) the same way `mesh` splits
    /// background vs. user-layer geometry around `background_index_count`.
    pub tiles: Vec<TileDraw>,
    pub raster_tile_count: u32,
    pub labels: Vec<LabelGlyphInstance>,
    pub glyph_bitmaps: GlyphBitmapRanges,
    pub width: f32,
    pub height: f32,
}

impl egui_wgpu::CallbackTrait for MapCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(resources) = callback_resources.get_mut::<MapRenderResources>() else {
            return Vec::new();
        };

        resources.update_screen_size(queue, self.width, self.height);

        for tile in &self.tiles {
            resources.ensure_tile_texture(device, queue, tile);
        }

        for draw in &self.basemap_tiles {
            resources.ensure_basemap_tile_buffer(device, draw.coord, &draw.mesh);
        }

        resources.ensure_tile_transform_capacity(device, self.basemap_tiles.len() as u64);
        let stride = resources.tile_transform_stride;
        let basemap_draws = self
            .basemap_tiles
            .iter()
            .enumerate()
            .filter_map(|(slot, draw)| {
                if !resources.tile_gpu_meshes.contains(&draw.coord) {
                    return None;
                }
                let uniform = TileTransformUniform {
                    offset: draw.offset,
                    scale: draw.scale,
                    width_scale: draw.width_scale,
                };
                let transform_offset = slot as u64 * stride;
                queue.write_buffer(
                    &resources.tile_transform_pool_buffer,
                    transform_offset,
                    bytemuck::bytes_of(&uniform),
                );
                Some(BasemapDrawPrepared {
                    coord: draw.coord,
                    offset: draw.offset,
                    size: draw.size,
                    transform_offset: transform_offset as u32,
                })
            })
            .collect();

        resources.scene_vertex_buffer.write(
            device,
            queue,
            bytemuck::cast_slice(&self.mesh.vertices),
        );
        resources
            .scene_index_buffer
            .write(device, queue, bytemuck::cast_slice(&self.mesh.indices));

        let mut tile_vertices = Vec::with_capacity(self.tiles.len() * 4);
        let mut tile_indices = Vec::with_capacity(self.tiles.len() * 6);
        for (i, tile) in self.tiles.iter().enumerate() {
            let [x, y, w, h] = tile.rect;
            let [u0, v0, u1, v1] = tile.uv_rect;
            let opacity = tile.opacity;
            tile_vertices.push(TileVertex {
                position: [x, y],
                uv: [u0, v0],
                opacity,
            });
            tile_vertices.push(TileVertex {
                position: [x + w, y],
                uv: [u1, v0],
                opacity,
            });
            tile_vertices.push(TileVertex {
                position: [x + w, y + h],
                uv: [u1, v1],
                opacity,
            });
            tile_vertices.push(TileVertex {
                position: [x, y + h],
                uv: [u0, v1],
                opacity,
            });
            let base = (i * 4) as u32;
            tile_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        resources
            .tile_vertex_buffer
            .write(device, queue, bytemuck::cast_slice(&tile_vertices));
        resources
            .tile_index_buffer
            .write(device, queue, bytemuck::cast_slice(&tile_indices));

        let mut text_vertices = Vec::with_capacity(self.labels.len() * 4);
        let mut text_indices = Vec::with_capacity(self.labels.len() * 6);
        for label in &self.labels {
            let range_start = glyph_range_start(label.codepoint);
            let Some(range) = self
                .glyph_bitmaps
                .get(&(label.fontstack.clone(), range_start))
            else {
                continue;
            };
            let Some(glyph) = range.get(&label.codepoint) else {
                continue;
            };
            let Some(atlas_rect) = resources.glyph_atlas.ensure(
                device,
                queue,
                &label.fontstack,
                label.codepoint,
                glyph,
            ) else {
                continue;
            };

            let [x, y, w, h] = label.rect;
            let atlas_size = resources.glyph_atlas.size as f32;
            let u0 = atlas_rect.x as f32 / atlas_size;
            let v0 = atlas_rect.y as f32 / atlas_size;
            let u1 = (atlas_rect.x + atlas_rect.w) as f32 / atlas_size;
            let v1 = (atlas_rect.y + atlas_rect.h) as f32 / atlas_size;
            let (sin_a, cos_a) = label.angle.sin_cos();
            let [ax, ay] = label.anchor;
            let rotate = |px: f32, py: f32| -> [f32; 2] {
                let dx = px - ax;
                let dy = py - ay;
                [ax + dx * cos_a - dy * sin_a, ay + dx * sin_a + dy * cos_a]
            };
            let base = text_vertices.len() as u32;
            text_vertices.extend_from_slice(&[
                TextVertex {
                    position: rotate(x, y),
                    uv: [u0, v0],
                    color: label.color,
                    halo_color: label.halo_color,
                },
                TextVertex {
                    position: rotate(x + w, y),
                    uv: [u1, v0],
                    color: label.color,
                    halo_color: label.halo_color,
                },
                TextVertex {
                    position: rotate(x + w, y + h),
                    uv: [u1, v1],
                    color: label.color,
                    halo_color: label.halo_color,
                },
                TextVertex {
                    position: rotate(x, y + h),
                    uv: [u0, v1],
                    color: label.color,
                    halo_color: label.halo_color,
                },
            ]);
            text_indices.extend(
                UNIT_QUAD_INDICES
                    .iter()
                    .map(|&index| base + u32::from(index)),
            );
        }
        resources
            .text_vertex_buffer
            .write(device, queue, bytemuck::cast_slice(&text_vertices));
        resources
            .text_index_buffer
            .write(device, queue, bytemuck::cast_slice(&text_indices));

        callback_resources.insert(FramePrepared {
            index_count: self.mesh.indices.len() as u32,
            background_index_count: self.background_index_count,
            basemap_draws,
            tile_keys: self.tiles.iter().map(|tile| tile.key).collect(),
            raster_tile_count: self.raster_tile_count,
            text_index_count: text_indices.len() as u32,
        });

        Vec::new()
    }

    fn paint(
        &self,
        info: epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(resources) = callback_resources.get::<MapRenderResources>() else {
            return;
        };
        let Some(frame) = callback_resources.get::<FramePrepared>() else {
            return;
        };
        // The vector `background` layer (an opaque full-viewport quad, e.g.
        // liberty's `#f8f4f0`) must be the very first thing drawn: it has
        // alpha 1, so anything drawn *before* it would be fully erased, not
        // blended. Raster style tiles (e.g. a `raster` source like
        // `natural_earth`, which sits directly above `background` in the
        // style's layer order) are drawn next, then basemap tiles and
        // vector layers on top of that. Sprite icon quads share the same
        // pipeline/buffers as the raster tiles but are drawn in a second
        // pass further below (after basemap/vector layers), since symbol
        // icons must render above everything else -- see
        // `frame.raster_tile_count`.
        if let (Some(vertex_slice), Some(index_slice)) = (
            resources.scene_vertex_buffer.slice(),
            resources.scene_index_buffer.slice(),
        ) {
            render_pass.set_pipeline(&resources.vector_pipeline);
            render_pass.set_bind_group(0, &resources.screen_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_slice);
            render_pass.set_index_buffer(index_slice, wgpu::IndexFormat::Uint32);
            if frame.background_index_count > 0 {
                render_pass.draw_indexed(0..frame.background_index_count, 0, 0..1);
            }
        }

        if let (Some(tile_vertex_slice), Some(tile_index_slice)) = (
            resources.tile_vertex_buffer.slice(),
            resources.tile_index_buffer.slice(),
        ) {
            render_pass.set_pipeline(&resources.tile_pipeline);
            render_pass.set_bind_group(0, &resources.screen_bind_group, &[]);
            render_pass.set_vertex_buffer(0, tile_vertex_slice);
            render_pass.set_index_buffer(tile_index_slice, wgpu::IndexFormat::Uint32);
            for (i, key) in frame.tile_keys[..frame.raster_tile_count as usize]
                .iter()
                .enumerate()
            {
                if let Some(texture) = resources.tile_textures.peek(key) {
                    render_pass.set_bind_group(1, &texture.bind_group, &[]);
                    let index_start = (i * 6) as u32;
                    render_pass.draw_indexed(index_start..index_start + 6, 0, 0..1);
                }
            }
        }

        // Basemap tiles: persistent per-tile GPU buffers uploaded once (see
        // `MapRenderResources::ensure_basemap_tile_buffer`), positioned via a
        // tiny per-tile transform uniform recomputed every frame instead of
        // re-tessellating or re-uploading geometry -- drawn between the
        // background and the user layers so layers stay on top. Fills are
        // drawn before lines (roads/casings/outlines on top of polygons).
        //
        // Known parity gap: within each category (fills among themselves,
        // lines among themselves) draw order matches the style document's
        // own layer order (see `build_tile_mesh`'s doc comment), but the two
        // categories are two separate passes/draw calls, so a `line` layer
        // that comes *before* a later `fill` layer in the style (e.g.
        // liberty's `park`/`park_outline`, index 2-3, vs. `water`, index
        // 17) always renders on top of it here, whereas MapLibre would
        // paint the fill over the line per the style's real order. This
        // shows up as e.g. a park/reserve outline that dips into a lake or
        // the sea still being visible there, when it should be hidden
        // under the water fill painted after it. Fixing this needs
        // per-layer-interleaved draw calls (or a depth/stencil trick)
        // instead of one fills-then-lines pass per tile; not attempted
        // here as it's a much larger rendering-architecture change than a
        // style-evaluation fix.
        //
        // Each tile is scissor-clipped to its own screen-space square: MVT
        // tiles include a small "buffer" zone of geometry duplicated a bit
        // past the tile edge (so wide strokes aren't cut off mid-width at
        // the seam) -- without clipping, adjacent tiles' buffer zones
        // overlap and double-draw the same (semi-transparent) geometry,
        // visibly darkening/duplicating it right at tile boundaries.
        if !frame.basemap_draws.is_empty() {
            let vp = info.viewport_in_pixels();
            let ppp = info.pixels_per_point;
            let scissor_for = |offset: [f32; 2], size: f32| -> Option<(u32, u32, u32, u32)> {
                let left = vp.left_px as f32 + offset[0] * ppp;
                let top = vp.top_px as f32 + offset[1] * ppp;
                // Floor the leading edge and ceil the trailing edge instead
                // of rounding to nearest, so adjacent tiles' rects always
                // overlap by up to 1px rather than occasionally leaving a
                // 1px gap when a shared tile edge falls near a pixel's
                // rounding threshold (visible as thin white seams that only
                // appear at some fractional zoom levels).
                let clip_left = left.floor().max(vp.left_px as f32);
                let clip_top = top.floor().max(vp.top_px as f32);
                let clip_right = (left + size * ppp)
                    .ceil()
                    .min((vp.left_px + vp.width_px) as f32);
                let clip_bottom = (top + size * ppp)
                    .ceil()
                    .min((vp.top_px + vp.height_px) as f32);
                let width = clip_right - clip_left;
                let height = clip_bottom - clip_top;
                (width >= 1.0 && height >= 1.0).then_some((
                    clip_left as u32,
                    clip_top as u32,
                    width as u32,
                    height as u32,
                ))
            };

            render_pass.set_pipeline(&resources.basemap_pipeline);
            render_pass.set_bind_group(0, &resources.screen_bind_group, &[]);
            for draw in &frame.basemap_draws {
                if let Some(tile_mesh) = resources.tile_gpu_meshes.peek(&draw.coord)
                    && let Some(fill) = &tile_mesh.fill
                    && let Some((x, y, w, h)) = scissor_for(draw.offset, draw.size)
                {
                    render_pass.set_scissor_rect(x, y, w, h);
                    render_pass.set_bind_group(
                        1,
                        &resources.tile_transform_bind_group,
                        &[draw.transform_offset],
                    );
                    render_pass.set_vertex_buffer(0, fill.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(fill.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..fill.index_count, 0, 0..1);
                }
            }

            render_pass.set_pipeline(&resources.basemap_line_pipeline);
            for draw in &frame.basemap_draws {
                if let Some(tile_mesh) = resources.tile_gpu_meshes.peek(&draw.coord)
                    && let Some(lines) = &tile_mesh.lines
                    && let Some((x, y, w, h)) = scissor_for(draw.offset, draw.size)
                {
                    render_pass.set_scissor_rect(x, y, w, h);
                    render_pass.set_bind_group(
                        1,
                        &resources.tile_transform_bind_group,
                        &[draw.transform_offset],
                    );
                    render_pass.set_vertex_buffer(0, lines.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(lines.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..lines.index_count, 0, 0..1);
                }
            }

            // Restore the full callback viewport as the scissor rect so the
            // user-layer draws below aren't clipped to the last tile drawn.
            if vp.width_px > 0 && vp.height_px > 0 {
                render_pass.set_scissor_rect(
                    vp.left_px as u32,
                    vp.top_px as u32,
                    vp.width_px as u32,
                    vp.height_px as u32,
                );
            }
        }

        if let (Some(vertex_buffer), Some(index_buffer)) = (
            resources.scene_vertex_buffer.slice(),
            resources.scene_index_buffer.slice(),
        ) && frame.background_index_count < frame.index_count
        {
            render_pass.set_pipeline(&resources.vector_pipeline);
            render_pass.set_bind_group(0, &resources.screen_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer);
            render_pass.set_index_buffer(index_buffer, wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(frame.background_index_count..frame.index_count, 0, 0..1);
        }

        // Sprite icon quads (see `TileDraw`'s docs): drawn here, above the
        // background/basemap/vector layers above, using the same
        // vertex/index buffers and pipeline as the raster tiles drawn at
        // the very top of this function -- just a later, disjoint slice of
        // `frame.tile_keys`/the vertex buffer (indices
        // `raster_tile_count..tile_keys.len()`).
        if let (Some(tile_vertex_slice), Some(tile_index_slice)) = (
            resources.tile_vertex_buffer.slice(),
            resources.tile_index_buffer.slice(),
        ) && (frame.raster_tile_count as usize) < frame.tile_keys.len()
        {
            render_pass.set_pipeline(&resources.tile_pipeline);
            render_pass.set_bind_group(0, &resources.screen_bind_group, &[]);
            render_pass.set_vertex_buffer(0, tile_vertex_slice);
            render_pass.set_index_buffer(tile_index_slice, wgpu::IndexFormat::Uint32);
            for (i, key) in frame.tile_keys[frame.raster_tile_count as usize..]
                .iter()
                .enumerate()
            {
                if let Some(texture) = resources.tile_textures.peek(key) {
                    render_pass.set_bind_group(1, &texture.bind_group, &[]);
                    let index_start = ((frame.raster_tile_count as usize + i) * 6) as u32;
                    render_pass.draw_indexed(index_start..index_start + 6, 0, 0..1);
                }
            }
        }

        if frame.text_index_count > 0
            && let (Some(vertex_buffer), Some(index_buffer)) = (
                resources.text_vertex_buffer.slice(),
                resources.text_index_buffer.slice(),
            )
        {
            render_pass.set_pipeline(&resources.text_pipeline);
            render_pass.set_bind_group(0, &resources.screen_bind_group, &[]);
            render_pass.set_bind_group(1, &resources.glyph_atlas.bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer);
            render_pass.set_index_buffer(index_buffer, wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..frame.text_index_count, 0, 0..1);
        }
    }
}

/// Per-frame draw metadata computed in `prepare` and consumed in `paint`.
/// The actual GPU buffers live as persistent fields on
/// `MapRenderResources` (`scene_vertex_buffer`/`scene_index_buffer`/
/// `tile_vertex_buffer`/`tile_index_buffer`) instead of here, since they're
/// reused (grown, never reallocated from scratch) across frames rather
/// than rebuilt every time.
struct FramePrepared {
    index_count: u32,
    background_index_count: u32,
    basemap_draws: Vec<BasemapDrawPrepared>,
    tile_keys: Vec<u64>,
    raster_tile_count: u32,
    text_index_count: u32,
}

/// A basemap tile draw ready for `paint`: the persistent GPU mesh buffers
/// live in `MapRenderResources::tile_gpu_meshes` (looked up by `coord`);
/// `transform_offset` is this tile's dynamic-offset slot in
/// `MapRenderResources::tile_transform_pool_buffer` -- nothing tile-specific
/// is allocated per frame.
struct BasemapDrawPrepared {
    coord: TileCoord,
    offset: [f32; 2],
    size: f32,
    transform_offset: u32,
}

fn create_vector_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    screen_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rgis-vector-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vector.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rgis-vector-pipeline-layout"),
        bind_group_layouts: &[Some(screen_bind_group_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("rgis-vector-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &VERTEX_ATTRS,
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}

fn create_basemap_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    screen_bind_group_layout: &wgpu::BindGroupLayout,
    tile_transform_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rgis-basemap-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/basemap.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rgis-basemap-pipeline-layout"),
        bind_group_layouts: &[
            Some(screen_bind_group_layout),
            Some(tile_transform_bind_group_layout),
        ],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("rgis-basemap-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &VERTEX_ATTRS,
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}

fn create_basemap_line_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    screen_bind_group_layout: &wgpu::BindGroupLayout,
    tile_transform_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rgis-basemap-line-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/basemap_line.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rgis-basemap-line-pipeline-layout"),
        bind_group_layouts: &[
            Some(screen_bind_group_layout),
            Some(tile_transform_bind_group_layout),
        ],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("rgis-basemap-line-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &LINE_VERTEX_ATTRS,
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}

fn create_tile_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    screen_bind_group_layout: &wgpu::BindGroupLayout,
    tile_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rgis-tile-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tile.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rgis-tile-pipeline-layout"),
        bind_group_layouts: &[Some(screen_bind_group_layout), Some(tile_bind_group_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("rgis-tile-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TileVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &TILE_VERTEX_ATTRS,
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}

fn create_text_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    screen_bind_group_layout: &wgpu::BindGroupLayout,
    atlas_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rgis-text-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/text.wgsl").into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rgis-text-pipeline-layout"),
        bind_group_layouts: &[
            Some(screen_bind_group_layout),
            Some(atlas_bind_group_layout),
        ],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("rgis-text-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TextVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &TEXT_VERTEX_ATTRS,
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    })
}
