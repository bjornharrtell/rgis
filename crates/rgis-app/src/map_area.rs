use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::Arc,
};

use glib::clone;
use gtk4::prelude::*;
use rgis_core::Project;
use rgis_render::{GlRenderer, MapRenderer, TileImage};
use rgis_tiles::{tile_screen_rect, visible_tiles, OsmTileSource, TileFetcher, TileCoord, TileReady};

/// Load a GL function pointer by searching all currently loaded shared libraries.
/// After GTK4 realizes a GLArea it has already loaded the platform GL library
/// (EGL on Wayland, GLX on X11), so core GL 3.3 symbols are always present.
unsafe fn gl_proc_addr(s: &std::ffi::CStr) -> *const std::ffi::c_void {
    libc::dlsym(libc::RTLD_DEFAULT, s.as_ptr() as *const _) as *const _
}

/// Wraps a `gtk4::GLArea` that renders the map.
#[derive(Clone)]
pub struct MapArea {
    gl_area: gtk4::GLArea,
    state: Rc<RefCell<MapState>>,
    /// Separated into its own RefCell so `render()` can borrow project + renderer
    /// simultaneously without cloning layer data.
    renderer: Rc<RefCell<Option<GlRenderer>>>,
}

struct MapState {
    project: Rc<RefCell<Project>>,
    tile_fetcher: Arc<TileFetcher>,
    /// Loaded tiles keyed by coord. Arc pixel data is shared; no per-frame memcopy.
    tile_cache: HashMap<TileCoord, (Arc<Vec<u8>>, u32, u32)>,
    drag_start: Option<(f64, f64)>,
    drag_last: Option<(f64, f64)>,
    /// Last known pointer position in widget-local pixels.
    cursor_px: [f32; 2],
}

impl MapArea {
    pub fn new(project: Rc<RefCell<Project>>, tile_fetcher: Arc<TileFetcher>) -> Self {
        let gl_area = gtk4::GLArea::new();
        gl_area.set_hexpand(true);
        gl_area.set_vexpand(true);
        gl_area.set_required_version(3, 3);
        gl_area.set_has_depth_buffer(false);

        let renderer: Rc<RefCell<Option<GlRenderer>>> = Rc::new(RefCell::new(None));

        let state = Rc::new(RefCell::new(MapState {
            project,
            tile_fetcher,
            tile_cache: HashMap::new(),
            drag_start: None,
            drag_last: None,
            cursor_px: [0.0, 0.0],
        }));

        // realize: create the GL renderer
        gl_area.connect_realize(clone!(#[strong] renderer, move |area| {
            area.make_current();
            if let Some(err) = area.error() {
                eprintln!("GLArea: context error on realize: {err}");
                return;
            }
            let gl = unsafe {
                glow::Context::from_loader_function_cstr(|s| gl_proc_addr(s))
            };
            *renderer.borrow_mut() = Some(unsafe { GlRenderer::new(gl) });
        }));

        // unrealize: drop renderer
        gl_area.connect_unrealize(clone!(#[strong] renderer, move |_| {
            *renderer.borrow_mut() = None;
        }));

        // resize
        gl_area.connect_resize(clone!(#[strong] state, #[strong] renderer, move |_, w, h| {
            {
                let s = state.borrow();
                let mut proj = s.project.borrow_mut();
                proj.viewport.width_px = w as u32;
                proj.viewport.height_px = h as u32;
            }
            if let Some(r) = renderer.borrow_mut().as_mut() {
                r.resize(w as u32, h as u32);
            }
        }));

        // render — renderer and project are in separate RefCells so no clone needed
        gl_area.connect_render(clone!(#[strong] state, #[strong] renderer, move |_area, _ctx| {
            let mut rend_cell = renderer.borrow_mut();
            let Some(rend) = rend_cell.as_mut() else {
                return glib::Propagation::Proceed;
            };

            let s = state.borrow();
            let proj = s.project.borrow();
            let viewport = proj.viewport.clone(); // Viewport is a handful of numbers, cheap

            // Build TileImages for visible tiles that are already loaded.
            // Arc::clone is a reference-count bump; no pixel data is copied.
            let visible = visible_tiles(&viewport, &OsmTileSource);
            let tile_images: Vec<TileImage> = visible.iter()
                .filter_map(|coord| {
                    s.tile_cache.get(coord).map(|(rgba, w, h)| TileImage {
                        coord: (coord.z, coord.x, coord.y),
                        rgba: Arc::clone(rgba),
                        width: *w,
                        height: *h,
                        screen_rect: tile_screen_rect(*coord, &viewport),
                    })
                })
                .collect();

            // Request any visible tiles not yet in the cache.
            for &coord in &visible {
                if !s.tile_cache.contains_key(&coord) {
                    s.tile_fetcher.request(coord);
                }
            }

            rend.render(&viewport, &proj.layers, &tile_images);
            glib::Propagation::Proceed
        }));

        // Pan gesture — positive drag delta → map follows finger (natural/intuitive).
        let drag = gtk4::GestureDrag::new();
        drag.set_button(gtk4::gdk::BUTTON_PRIMARY);
        drag.connect_drag_begin(clone!(#[strong] state, move |_, x, y| {
            let mut s = state.borrow_mut();
            s.drag_start = Some((x, y));
            s.drag_last = Some((x, y));
        }));
        drag.connect_drag_update(clone!(#[strong] state, #[weak] gl_area, move |_, dx, dy| {
            let mut s = state.borrow_mut();
            if let Some((lx, ly)) = s.drag_last {
                let start = s.drag_start.unwrap_or((0.0, 0.0));
                let cx = start.0 + dx;
                let cy = start.1 + dy;
                let ddx = cx - lx;
                let ddy = cy - ly;
                // Scale logical-pixel delta to device pixels so pan distance
                // matches the physical pointer movement on HiDPI displays.
                let scale = gl_area.scale_factor() as f32;
                s.project.borrow_mut().viewport.pan(ddx as f32 * scale, ddy as f32 * scale);
                s.drag_last = Some((cx, cy));
                drop(s);
                gl_area.queue_render();
            }
        }));
        drag.connect_drag_end(clone!(#[strong] state, move |_, _, _| {
            let mut s = state.borrow_mut();
            s.drag_start = None;
            s.drag_last = None;
        }));
        gl_area.add_controller(drag);

        // Track pointer position so scroll-to-zoom uses the cursor as anchor.
        // Motion events are in logical pixels; convert to device pixels so the
        // stored position is consistent with viewport.width_px / height_px.
        let motion = gtk4::EventControllerMotion::new();
        motion.connect_motion(clone!(#[strong] state, #[weak] gl_area, move |_, x, y| {
            let scale = gl_area.scale_factor() as f32;
            state.borrow_mut().cursor_px = [x as f32 * scale, y as f32 * scale];
        }));
        gl_area.add_controller(motion);

        // Scroll to zoom — zoom toward the pointer position.
        let scroll = gtk4::EventControllerScroll::new(
            gtk4::EventControllerScrollFlags::VERTICAL,
        );
        scroll.connect_scroll(clone!(#[strong] state, #[weak] gl_area, #[upgrade_or_else] || glib::Propagation::Proceed, move |_ctrl, _dx, dy| {
            let s = state.borrow_mut();
            let cursor = s.cursor_px;
            s.project.borrow_mut().viewport.zoom_toward(cursor, -dy * 0.25);
            drop(s);
            gl_area.queue_render();
            glib::Propagation::Proceed
        }));
        gl_area.add_controller(scroll);

        Self { gl_area, state, renderer }
    }

    pub fn widget(&self) -> &gtk4::GLArea {
        &self.gl_area
    }

    pub fn queue_render(&self) {
        self.gl_area.queue_render();
    }

    /// Called when a tile has been fetched. Flattens pixel data into an Arc once;
    /// subsequent render frames clone the Arc cheaply.
    pub fn on_tile_ready(&self, ready: TileReady) {
        let mut s = self.state.borrow_mut();
        s.tile_cache.entry(ready.coord).or_insert_with(|| {
            let img = ready.image;
            let w = img.width();
            let h = img.height();
            let rgba = Arc::new(img.as_ref().to_vec());
            (rgba, w, h)
        });
        self.gl_area.queue_render();
    }

    pub fn invalidate_layer(&self, id: rgis_core::LayerId) {
        if let Some(r) = self.renderer.borrow_mut().as_mut() {
            r.invalidate_layer(id);
        }
    }
}
