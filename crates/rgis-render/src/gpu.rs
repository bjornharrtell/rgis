//! wgpu render resources for drawing the tessellated vector mesh and raster
//! tile images inside an `egui_wgpu::Callback`, so the map renders on the
//! same wgpu device/surface as the rest of the (egui) UI — on native and in
//! the browser (WebGPU/WebGL2) alike.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lru::LruCache;
use rgis_tiles::TileCoord;
use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::util::DeviceExt;

use crate::basemap::{LineVertex, TileMesh};
use crate::mesh::{SceneMesh, Vertex};

/// MSAA sample count for the map's wgpu pipelines. Must match whatever
/// sample count eframe's shared renderer was created with (native: set via
/// `NativeOptions::multisampling`; the web/WebOptions API has no equivalent
/// knob, so it always renders at 1 sample there).
pub const MSAA_SAMPLES: u32 = if cfg!(target_arch = "wasm32") { 1 } else { 4 };

const VERTEX_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

const LINE_VERTEX_ATTRS: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Float32x4];

const TILE_VERTEX_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TileVertex {
    position: [f32; 2],
    uv: [f32; 2],
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

/// A single raster tile to draw this frame: its screen-pixel rectangle
/// (`[x, y, width, height]`), a stable cache key (used to avoid re-uploading
/// the same tile texture every frame), and its RGBA pixels.
pub struct TileDraw {
    pub key: u64,
    pub rect: [f32; 4],
    pub rgba: Arc<image::RgbaImage>,
}

struct TileGpuTexture {
    bind_group: wgpu::BindGroup,
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
    tile_textures: FxHashMap<u64, TileGpuTexture>,
    /// Raster tiles' quad vertices, rewritten (not reallocated, unless the
    /// visible tile count grows) every frame.
    tile_vertex_buffer: GrowableBuffer,
    /// `UNIT_QUAD_INDICES` uploaded once -- every raster tile quad shares
    /// the same index pattern, so this never changes after `new`.
    tile_index_buffer: wgpu::Buffer,
    /// Background quad + user vector layers' geometry, rewritten every
    /// frame.
    scene_vertex_buffer: GrowableBuffer,
    scene_index_buffer: GrowableBuffer,
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

        let tile_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rgis-tile-indices"),
            contents: bytemuck::cast_slice(&UNIT_QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

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
            tile_textures: FxHashMap::default(),
            tile_vertex_buffer: GrowableBuffer::new(
                wgpu::BufferUsages::VERTEX,
                "rgis-tile-vertices",
            ),
            tile_index_buffer,
            scene_vertex_buffer: GrowableBuffer::new(
                wgpu::BufferUsages::VERTEX,
                "rgis-vector-vertices",
            ),
            scene_index_buffer: GrowableBuffer::new(
                wgpu::BufferUsages::INDEX,
                "rgis-vector-indices",
            ),
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
        if self.tile_textures.contains_key(&tile.key) {
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
            .insert(tile.key, TileGpuTexture { bind_group });
    }

    /// Drops cached tile textures whose key wasn't requested this frame, so
    /// the GPU-side cache tracks whatever `rgis-tiles`'s CPU-side LRU cache
    /// currently holds instead of growing without bound.
    fn evict_stale_tiles(&mut self, live_keys: &FxHashSet<u64>) {
        self.tile_textures.retain(|key, _| live_keys.contains(key));
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

/// Number of basemap tiles' GPU mesh buffers kept alive at once. Larger than
/// a typical viewport's visible-tile count so tiles that scroll just off
/// screen (the common case during panning) don't need re-uploading if the
/// user pans back before they'd be evicted.
const TILE_GPU_MESH_CACHE_SIZE: usize = 512;

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
    pub tiles: Vec<TileDraw>,
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

        let live_keys: FxHashSet<u64> = self.tiles.iter().map(|tile| tile.key).collect();
        for tile in &self.tiles {
            resources.ensure_tile_texture(device, queue, tile);
        }
        resources.evict_stale_tiles(&live_keys);

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
        for tile in &self.tiles {
            let [x, y, w, h] = tile.rect;
            tile_vertices.push(TileVertex {
                position: [x, y],
                uv: [0.0, 0.0],
            });
            tile_vertices.push(TileVertex {
                position: [x + w, y],
                uv: [1.0, 0.0],
            });
            tile_vertices.push(TileVertex {
                position: [x + w, y + h],
                uv: [1.0, 1.0],
            });
            tile_vertices.push(TileVertex {
                position: [x, y + h],
                uv: [0.0, 1.0],
            });
        }
        resources
            .tile_vertex_buffer
            .write(device, queue, bytemuck::cast_slice(&tile_vertices));

        callback_resources.insert(FramePrepared {
            index_count: self.mesh.indices.len() as u32,
            background_index_count: self.background_index_count,
            basemap_draws,
            tile_keys: self.tiles.iter().map(|tile| tile.key).collect(),
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
        // OSM tiles are the map background, so draw them first; the vector
        // background quad, basemap tiles, and vector layers are then drawn
        // on top so layers stay visible above the basemap.
        if let Some(tile_vertex_slice) = resources.tile_vertex_buffer.slice() {
            render_pass.set_pipeline(&resources.tile_pipeline);
            render_pass.set_bind_group(0, &resources.screen_bind_group, &[]);
            render_pass.set_vertex_buffer(0, tile_vertex_slice);
            render_pass.set_index_buffer(
                resources.tile_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            for (i, key) in frame.tile_keys.iter().enumerate() {
                if let Some(texture) = resources.tile_textures.get(key) {
                    render_pass.set_bind_group(1, &texture.bind_group, &[]);
                    render_pass.draw_indexed(0..6, (i * 4) as i32, 0..1);
                }
            }
        }

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

        // Basemap tiles: persistent per-tile GPU buffers uploaded once (see
        // `MapRenderResources::ensure_basemap_tile_buffer`), positioned via a
        // tiny per-tile transform uniform recomputed every frame instead of
        // re-tessellating or re-uploading geometry -- drawn between the
        // background and the user layers so layers stay on top. Fills are
        // drawn before lines (roads/casings/outlines on top of polygons).
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
