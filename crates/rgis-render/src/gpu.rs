//! wgpu render resources for drawing the tessellated vector mesh and raster
//! tile images inside an `egui_wgpu::Callback`, so the map renders on the
//! same wgpu device/surface as the rest of the (egui) UI — on native and in
//! the browser (WebGPU/WebGL2) alike.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::mesh::{SceneMesh, Vertex};

const VERTEX_ATTRS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4];

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

    tile_pipeline: wgpu::RenderPipeline,
    tile_bind_group_layout: wgpu::BindGroupLayout,
    tile_sampler: wgpu::Sampler,
    tile_textures: HashMap<u64, TileGpuTexture>,
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

        Self {
            vector_pipeline,
            screen_uniform_buffer,
            screen_bind_group,
            tile_pipeline,
            tile_bind_group_layout,
            tile_sampler,
            tile_textures: HashMap::new(),
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
    fn evict_stale_tiles(&mut self, live_keys: &std::collections::HashSet<u64>) {
        self.tile_textures.retain(|key, _| live_keys.contains(key));
    }
}

/// Per-frame data handed to the egui paint callback. Built fresh every frame
/// from the current viewport/project state.
pub struct MapCallback {
    pub mesh: SceneMesh,
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

        let live_keys: std::collections::HashSet<u64> =
            self.tiles.iter().map(|tile| tile.key).collect();
        for tile in &self.tiles {
            resources.ensure_tile_texture(device, queue, tile);
        }
        resources.evict_stale_tiles(&live_keys);

        let vertex_buffer = if self.mesh.vertices.is_empty() {
            None
        } else {
            Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rgis-vector-vertices"),
                    contents: bytemuck::cast_slice(&self.mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            )
        };
        let index_buffer = if self.mesh.indices.is_empty() {
            None
        } else {
            Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rgis-vector-indices"),
                    contents: bytemuck::cast_slice(&self.mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            )
        };

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
        let tile_vertex_buffer = if tile_vertices.is_empty() {
            None
        } else {
            Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rgis-tile-vertices"),
                    contents: bytemuck::cast_slice(&tile_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            )
        };
        const UNIT_QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let tile_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rgis-tile-indices"),
            contents: bytemuck::cast_slice(&UNIT_QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        callback_resources.insert(FramePrepared {
            vertex_buffer,
            index_buffer,
            index_count: self.mesh.indices.len() as u32,
            tile_vertex_buffer,
            tile_index_buffer,
            tile_keys: self.tiles.iter().map(|tile| tile.key).collect(),
        });

        Vec::new()
    }

    fn paint(
        &self,
        _info: epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(resources) = callback_resources.get::<MapRenderResources>() else {
            return;
        };
        let Some(frame) = callback_resources.get::<FramePrepared>() else {
            return;
        };

        if let (Some(vertex_buffer), Some(index_buffer)) =
            (&frame.vertex_buffer, &frame.index_buffer)
        {
            render_pass.set_pipeline(&resources.vector_pipeline);
            render_pass.set_bind_group(0, &resources.screen_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..frame.index_count, 0, 0..1);
        }

        if let Some(tile_vertex_buffer) = &frame.tile_vertex_buffer {
            render_pass.set_pipeline(&resources.tile_pipeline);
            render_pass.set_bind_group(0, &resources.screen_bind_group, &[]);
            render_pass.set_vertex_buffer(0, tile_vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(frame.tile_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            for (i, key) in frame.tile_keys.iter().enumerate() {
                if let Some(texture) = resources.tile_textures.get(key) {
                    render_pass.set_bind_group(1, &texture.bind_group, &[]);
                    render_pass.draw_indexed(0..6, (i * 4) as i32, 0..1);
                }
            }
        }
    }
}

/// Scratch GPU buffers rebuilt every frame in `prepare` and consumed in
/// `paint`. Stored in `CallbackResources` since `paint` only has an immutable
/// `&CallbackResources` (the buffers themselves are still mutated via
/// interior mutability of the type map's insert-per-frame overwrite).
struct FramePrepared {
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    tile_vertex_buffer: Option<wgpu::Buffer>,
    tile_index_buffer: wgpu::Buffer,
    tile_keys: Vec<u64>,
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
        multisample: wgpu::MultisampleState::default(),
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
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
