//! Spike: can the rgis map render inside a gpui-kit (gpui-pre) window without a
//! custom GPU surface API? The map is drawn offscreen on our own wgpu device,
//! read back to the CPU and shown with `Window::paint_image`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui_kit::component::button::Button;
use gpui_kit::*;
use rgis_core::{Layer, Project};
use rgis_render::{GlyphBitmapRanges, MapCallback, MapRenderResources, build_scene_mesh};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

struct Offscreen {
    device: wgpu::Device,
    queue: wgpu::Queue,
    resources: Option<MapRenderResources>,
    size: (u32, u32),
    color: wgpu::Texture,
    msaa: Option<wgpu::TextureView>,
    readback: wgpu::Buffer,
    padded_row: u32,
}

#[derive(Default)]
struct Timings {
    render: Duration,
    readback: Duration,
    upload: Duration,
    frames: u32,
}

impl Offscreen {
    fn new(size: (u32, u32)) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("no wgpu adapter");
        eprintln!("adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("no wgpu device");
        let resources = Some(MapRenderResources::new(&device, FORMAT));
        let (color, msaa, readback, padded_row) = Self::targets(&device, size);
        Self {
            device,
            queue,
            resources,
            size,
            color,
            msaa,
            readback,
            padded_row,
        }
    }

    fn targets(
        device: &wgpu::Device,
        (w, h): (u32, u32),
    ) -> (wgpu::Texture, Option<wgpu::TextureView>, wgpu::Buffer, u32) {
        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("spike-color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let msaa = (rgis_render::MSAA_SAMPLES > 1).then(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("spike-msaa"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: rgis_render::MSAA_SAMPLES,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        let padded_row = (w * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spike-readback"),
            size: padded_row as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        (color, msaa, readback, padded_row)
    }

    fn resize(&mut self, size: (u32, u32)) {
        if size != self.size {
            let (color, msaa, readback, padded_row) = Self::targets(&self.device, size);
            self.color = color;
            self.msaa = msaa;
            self.readback = readback;
            self.padded_row = padded_row;
            self.size = size;
        }
    }

    /// Renders the map and returns tightly packed BGRA pixels.
    fn render(&mut self, callback: &MapCallback, t: &mut Timings) -> Vec<u8> {
        let (w, h) = self.size;
        let start = Instant::now();
        let color_view = self.color.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let resources = self.resources.take().expect("resources");
        self.resources = Some(match &self.msaa {
            Some(msaa) => resources.render_into(
                &self.device,
                &self.queue,
                &mut encoder,
                msaa,
                Some(&color_view),
                callback,
            ),
            None => resources.render_into(
                &self.device,
                &self.queue,
                &mut encoder,
                &color_view,
                None,
                callback,
            ),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        let submitted = Instant::now();
        t.render += submitted - start;
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        {
            let data = slice.get_mapped_range();
            for row in 0..h as usize {
                let s = row * self.padded_row as usize;
                out.extend_from_slice(&data[s..s + (w * 4) as usize]);
            }
        }
        self.readback.unmap();
        t.readback += submitted.elapsed();
        t.frames += 1;
        out
    }
}

fn sample_project() -> Project {
    let mut project = Project::default();
    let bytes = include_bytes!("../../rgis-app/assets/sample.geojson");
    let loaded = rgis_io::load_bytes_with_crs("sample.geojson", bytes)
        .and_then(|l| l.into_web_mercator())
        .expect("sample loads");
    let id = project.next_layer_id();
    let layer = Layer::new(id, loaded.name, loaded.features);
    let bounds = layer.bounds;
    project.add_layer(layer);
    if let Some(b) = bounds {
        project.viewport.fit_bounds(&b);
    }
    project
}

fn callback(project: &mut Project, w: u32, h: u32, frame: u32) -> MapCallback {
    project.viewport.width_px = w;
    project.viewport.height_px = h;
    // Slowly pan so every frame is different.
    let mut viewport = project.viewport.clone();
    viewport.center.x += frame as f64 * 500.0;
    let mesh = build_scene_mesh(&project.layers, &viewport);
    MapCallback {
        mesh,
        background_index_count: 0,
        basemap_tiles: Vec::new(),
        tiles: Vec::new(),
        raster_tile_count: 0,
        vector_tile_count: 0,
        labels: Vec::new(),
        glyph_bitmaps: GlyphBitmapRanges::default(),
        width: w as f32,
        height: h as f32,
    }
}

fn bench(w: u32, h: u32, frames: u32) {
    let mut project = sample_project();
    let mut gpu = Offscreen::new((w, h));
    let mut t = Timings::default();
    let mut upload = Duration::ZERO;
    for f in 0..frames {
        let cb = callback(&mut project, w, h, f);
        let px = gpu.render(&cb, &mut t);
        if f == 0
            && let Ok(path) = std::env::var("SPIKE_DUMP")
        {
            let mut rgba = px.clone();
            rgba.as_chunks_mut::<4>()
                .0
                .iter_mut()
                .for_each(|p| p.swap(0, 2));
            image::RgbaImage::from_raw(w, h, rgba)
                .unwrap()
                .save(path)
                .unwrap();
        }
        let s = Instant::now();
        let img = image::RgbaImage::from_raw(w, h, px).expect("size");
        let _ = gpui_kit::RenderImage::new(smallvec::smallvec![image::Frame::new(img)]);
        upload += s.elapsed();
    }
    let n = frames as f64;
    println!(
        "{w}x{h}: render+submit {:.2} ms, GPU wait+readback {:.2} ms, image wrap {:.2} ms (avg of {frames})",
        t.render.as_secs_f64() * 1e3 / n,
        t.readback.as_secs_f64() * 1e3 / n,
        upload.as_secs_f64() * 1e3 / n
    );
}

struct MapView {
    gpu: Offscreen,
    project: Project,
    frame: u32,
    timings: Timings,
    last_report: Instant,
    paused: bool,
}

impl Render for MapView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let scale = window.scale_factor();
        let vp = window.viewport_size();
        let w = ((f32::from(vp.width) - 220.0).max(1.0) * scale) as u32;
        let h = ((f32::from(vp.height) - 28.0).max(1.0) * scale) as u32;
        self.gpu.resize((w, h));
        let cb = callback(&mut self.project, w, h, self.frame);
        let pixels = self.gpu.render(&cb, &mut self.timings);
        let s = Instant::now();
        let img = image::RgbaImage::from_raw(w, h, pixels).expect("size");
        let image = Arc::new(RenderImage::new(smallvec::smallvec![image::Frame::new(
            img
        )]));
        self.timings.upload += s.elapsed();
        if !self.paused {
            self.frame += 1;
            window.request_animation_frame();
        }
        if self.last_report.elapsed() > Duration::from_secs(2) && self.timings.frames > 0 {
            let n = self.timings.frames as f64;
            eprintln!(
                "{w}x{h}: {:.1} fps | render {:.2} ms | readback {:.2} ms | wrap {:.2} ms",
                n / self.last_report.elapsed().as_secs_f64(),
                self.timings.render.as_secs_f64() * 1e3 / n,
                self.timings.readback.as_secs_f64() * 1e3 / n,
                self.timings.upload.as_secs_f64() * 1e3 / n
            );
            self.timings = Timings::default();
            self.last_report = Instant::now();
        }
        if let Some(limit) = std::env::var("SPIKE_FRAMES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            && self.frame >= limit
        {
            cx.quit();
        }
        let paused = self.paused;
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(
                div()
                    .h(px(28.0))
                    .px_2()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        Button::new("pause")
                            .label(if paused { "Resume" } else { "Pause" })
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.paused = !this.paused;
                                cx.notify();
                            })),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .flex()
                    .child(div().w(px(220.0)).child("Layers (gpui-kit sidebar)"))
                    .child(
                        canvas(
                            |_, _, _| (),
                            move |bounds, _, window, _| {
                                let _ = window.paint_image(
                                    bounds,
                                    bounds,
                                    Corners::default(),
                                    image.clone(),
                                    0,
                                    false,
                                );
                            },
                        )
                        .size_full(),
                    ),
            )
    }
}

fn main() {
    if let Ok(spec) = std::env::var("SPIKE_BENCH") {
        for part in spec.split(',') {
            if let Some((w, h)) = part.split_once('x') {
                bench(w.parse().unwrap(), h.parse().unwrap(), 120);
            }
        }
        return;
    }
    gpui_kit::application().run(|cx| {
        gpui_kit::init(cx);
        gpui_kit::open_window(WindowOptions::default(), cx, |_, cx| {
            cx.new(|_| MapView {
                gpu: Offscreen::new((1280, 720)),
                project: sample_project(),
                frame: 0,
                timings: Timings::default(),
                last_report: Instant::now(),
                paused: false,
            })
        })
        .expect("window");
        cx.activate(true);
    });
}
