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

use crate::drag_zoom_box::Interactions;

/// Load a GL function pointer by searching all currently loaded shared libraries.
/// After GTK4 realizes a GLArea it has already loaded the platform GL library
/// (EGL on Wayland, GLX on X11), so core GL 3.3 symbols are always present.
unsafe fn gl_proc_addr(s: &std::ffi::CStr) -> *const std::ffi::c_void {
    libc::dlsym(libc::RTLD_DEFAULT, s.as_ptr() as *const _) as *const _
}

/// Wraps a `gtk4::GLArea` that renders the map, placed inside a
/// `gtk4::Overlay` so that interaction visuals (e.g. the zoom-box
/// rubber-band rectangle) can be drawn on top without touching the GL context.
#[derive(Clone)]
pub struct MapArea {
    gl_area: gtk4::GLArea,
    overlay: gtk4::Overlay,
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

    // ── Smooth zoom animation ─────────────────────────────────────────────
    /// Target zoom level for the ongoing animation.  Updated on every scroll
    /// event; the tick-callback eases `viewport.zoom` toward this value.
    zoom_target: f64,
    /// The Web-Mercator world position (metres) that should remain fixed
    /// under the cursor throughout the animation.
    zoom_anchor: [f64; 2],
    /// The screen-pixel position of the zoom anchor.
    zoom_anchor_screen: [f32; 2],
    /// When `Some`, the tick-callback eases both zoom and center toward the
    /// target values (used by the drag-zoom-box interaction).
    /// When `None`, anchor-based zoom is used (scroll wheel).
    zoom_center_target: Option<[f64; 2]>,
    /// Whether a zoom tick-callback is currently registered and running.
    zoom_animating: bool,
}

impl MapArea {
    pub fn new(
        project: Rc<RefCell<Project>>,
        tile_fetcher: Arc<TileFetcher>,
        interactions: Interactions,
    ) -> Self {
        let gl_area = gtk4::GLArea::new();
        gl_area.set_hexpand(true);
        gl_area.set_vexpand(true);
        gl_area.set_required_version(3, 3);
        gl_area.set_has_depth_buffer(false);
        // Render only when explicitly requested via queue_render(); this avoids
        // redundant frames and is prerequisite for flicker-free double buffering.
        gl_area.set_auto_render(false);

        // Transparent drawing area used by interaction overlays (e.g. zoom-box
        // rubber-band). It sits on top of the GL area but passes all input
        // events through so that gestures registered on gl_area still fire.
        let drawing_area = gtk4::DrawingArea::new();
        drawing_area.set_hexpand(true);
        drawing_area.set_vexpand(true);
        drawing_area.set_can_focus(false);
        // Ensure pointer events pass through to the GL area below.
        drawing_area.set_can_target(false);

        // Wrap GL area + drawing-area overlay into a single widget.
        let overlay = gtk4::Overlay::new();
        overlay.set_child(Some(&gl_area));
        overlay.add_overlay(&drawing_area);

        let renderer: Rc<RefCell<Option<GlRenderer>>> = Rc::new(RefCell::new(None));

        let initial_zoom = project.borrow().viewport.zoom;
        let state = Rc::new(RefCell::new(MapState {
            project,
            tile_fetcher,
            tile_cache: HashMap::new(),
            drag_start: None,
            drag_last: None,
            cursor_px: [0.0, 0.0],
            zoom_target: initial_zoom,
            zoom_anchor: [0.0, 0.0],
            zoom_anchor_screen: [0.0, 0.0],
            zoom_center_target: None,
            zoom_animating: false,
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
            let mut tile_images: Vec<TileImage> = Vec::with_capacity(visible.len());

            for coord in &visible {
                if let Some((rgba, w, h)) = s.tile_cache.get(coord) {
                    // Tile is ready at the correct zoom level.
                    tile_images.push(TileImage {
                        coord: (coord.z, coord.x, coord.y),
                        rgba: Arc::clone(rgba),
                        width: *w,
                        height: *h,
                        screen_rect: tile_screen_rect(*coord, &viewport),
                        src_rect: [0.0, 0.0, 1.0, 1.0],
                    });
                } else {
                    // Walk up the zoom pyramid looking for an ancestor tile.
                    // Each level up, this tile occupies a 2^k × 2^k sub-grid
                    // of the ancestor, so the UV sub-rect shrinks by 1/2 per level.
                    let mut ax = coord.x;
                    let mut ay = coord.y;
                    let mut found = false;
                    for k in 1u8..=4 {
                        if coord.z < k { break; }
                        ax >>= 1;
                        ay >>= 1;
                        let az = coord.z - k;
                        let ancestor = rgis_tiles::TileCoord { z: az, x: ax, y: ay };
                        if let Some((rgba, w, h)) = s.tile_cache.get(&ancestor) {
                            // Compute the UV sub-rect of the ancestor that
                            // corresponds to `coord`.
                            let scale = 1.0_f32 / (1u32 << k) as f32;
                            let sub_x = coord.x & ((1 << k) - 1);
                            let sub_y = coord.y & ((1 << k) - 1);
                            let u0 = sub_x as f32 * scale;
                            let v0 = sub_y as f32 * scale;
                            tile_images.push(TileImage {
                                coord: (ancestor.z, ancestor.x, ancestor.y),
                                rgba: Arc::clone(rgba),
                                width: *w,
                                height: *h,
                                screen_rect: tile_screen_rect(*coord, &viewport),
                                src_rect: [u0, v0, scale, scale],
                            });
                            found = true;
                            break;
                        }
                    }
                    // If no ancestor was found just leave a gap; the background
                    // colour will show until the tile arrives.
                    let _ = found;
                }
            }

            // Request any visible tiles not yet in the cache.
            for &coord in &visible {
                if !s.tile_cache.contains_key(&coord) {
                    s.tile_fetcher.request(coord);
                }
            }

            // Retessellate geometry only when no zoom animation is in progress.
            // During animation the old (slightly-misscaled) stroke geometry is
            // reused; on the settle frame allow_retessellate=true so it is
            // rebuilt once at the final zoom level.
            let allow_retessellate = !s.zoom_animating;
            rend.render(&viewport, &proj.layers, &tile_images, allow_retessellate);
            glib::Propagation::Proceed
        }));

        // Pan gesture — positive drag delta → map follows finger (natural/intuitive).
        // Denied when Shift is held so the DragZoomBox interaction takes over.
        let drag = gtk4::GestureDrag::new();
        drag.set_button(gtk4::gdk::BUTTON_PRIMARY);
        drag.connect_drag_begin(clone!(#[strong] state, move |gesture, x, y| {
            let shift = gesture
                .current_event()
                .map(|ev| ev.modifier_state().contains(gtk4::gdk::ModifierType::SHIFT_MASK))
                .unwrap_or(false);
            if shift {
                gesture.set_state(gtk4::EventSequenceState::Denied);
                return;
            }
            let mut s = state.borrow_mut();
            // Settle any in-flight zoom animation immediately so that the
            // tick callback stops overwriting viewport.center and the pan
            // gesture has full control from the very first drag event.
            if s.zoom_animating {
                let target = s.zoom_target;
                let mut proj = s.project.borrow_mut();
                proj.viewport.zoom = target;
                if let Some([tcx, tcy]) = s.zoom_center_target {
                    // Fit-zoom animation: settle at target center.
                    proj.viewport.center.x = tcx;
                    proj.viewport.center.y = tcy;
                } else {
                    // Anchor-based zoom: recompute center so anchor stays put.
                    let anchor = s.zoom_anchor;
                    let anchor_screen = s.zoom_anchor_screen;
                    let res = (2.0 * rgis_core::EARTH_HALF_CIRC) / (256.0 * 2_f64.powf(target));
                    proj.viewport.center.x =
                        anchor[0] - (anchor_screen[0] as f64 - proj.viewport.width_px as f64 * 0.5) * res;
                    proj.viewport.center.y =
                        anchor[1] + (anchor_screen[1] as f64 - proj.viewport.height_px as f64 * 0.5) * res;
                }
                drop(proj);
                s.zoom_animating = false;
                s.zoom_center_target = None;
            }
            s.drag_start = Some((x, y));
            s.drag_last = Some((x, y));
        }));
        drag.connect_drag_update(clone!(#[strong] state, #[weak] gl_area, move |_gesture, dx, dy| {
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

        // Scroll to zoom — smooth animated zoom toward the pointer position.
        let scroll = gtk4::EventControllerScroll::new(
            gtk4::EventControllerScrollFlags::VERTICAL,
        );
        scroll.connect_scroll(clone!(#[strong] state, #[weak] gl_area, #[upgrade_or_else] || glib::Propagation::Proceed, move |_ctrl, _dx, dy| {
            let mut s = state.borrow_mut();
            let cursor = s.cursor_px;
            let delta = -dy * 0.125;

            // World-space anchor: the map point under the cursor that must
            // remain fixed throughout the animation.
            let anchor_world = {
                let proj = s.project.borrow();
                let w = proj.viewport.screen_to_world(cursor);
                [w.x, w.y]
            };
            s.zoom_anchor = anchor_world;
            s.zoom_anchor_screen = cursor;
            // Switching back to anchor-based mode; cancel any fit-zoom animation.
            s.zoom_center_target = None;

            // If no animation is running, sync zoom_target with the actual
            // viewport zoom first (handles external zoom changes like fit_bounds).
            if !s.zoom_animating {
                let current_zoom = { let proj = s.project.borrow(); proj.viewport.zoom };
                s.zoom_target = current_zoom;
            }
            s.zoom_target = (s.zoom_target + delta).clamp(0.0, 22.0);

            if !s.zoom_animating {
                s.zoom_animating = true;
                drop(s);
                // Register a per-frame tick-callback that eases the viewport
                // zoom toward zoom_target with exponential decay (~20 % per
                // frame at 60 fps, giving a snappy ~150 ms feel).
                gl_area.add_tick_callback(clone!(#[strong] state, move |area, _clock| {
                    let animating = state.borrow().zoom_animating;
                    if !animating {
                        return glib::ControlFlow::Break;
                    }

                    let (target, anchor, anchor_screen, center_target) = {
                        let s = state.borrow();
                        (s.zoom_target, s.zoom_anchor, s.zoom_anchor_screen, s.zoom_center_target)
                    };

                    let (new_zoom, new_cx, new_cy, settled) = {
                        let s = state.borrow();
                        let proj = s.project.borrow();
                        let vp = &proj.viewport;
                        let z = vp.zoom + (target - vp.zoom) * 0.2;
                        if let Some([tcx, tcy]) = center_target {
                            // Fit-to-bounds: ease both zoom and center independently.
                            let settled_z = (z - target).abs() < 1e-4;
                            let z = if settled_z { target } else { z };
                            let cx = vp.center.x + (tcx - vp.center.x) * 0.2;
                            let cy = vp.center.y + (tcy - vp.center.y) * 0.2;
                            let settled_c = (cx - tcx).abs() < 1.0 && (cy - tcy).abs() < 1.0;
                            let (cx, cy) = if settled_c { (tcx, tcy) } else { (cx, cy) };
                            (z, cx, cy, settled_z && settled_c)
                        } else {
                            // Anchor-based: keep a world point fixed under the cursor.
                            let settled = (z - target).abs() < 1e-4;
                            let z = if settled { target } else { z };
                            let res = (2.0 * rgis_core::EARTH_HALF_CIRC)
                                / (256.0 * 2_f64.powf(z));
                            let cx = anchor[0]
                                - (anchor_screen[0] as f64 - vp.width_px as f64 * 0.5) * res;
                            let cy = anchor[1]
                                + (anchor_screen[1] as f64 - vp.height_px as f64 * 0.5) * res;
                            (z, cx, cy, settled)
                        }
                    };

                    {
                        let s = state.borrow();
                        let mut proj = s.project.borrow_mut();
                        proj.viewport.zoom = new_zoom;
                        proj.viewport.center.x = new_cx;
                        proj.viewport.center.y = new_cy;
                    }

                    if settled {
                        let mut s = state.borrow_mut();
                        s.zoom_animating = false;
                        s.zoom_center_target = None;
                    }

                    area.queue_render();

                    if settled {
                        glib::ControlFlow::Break
                    } else {
                        glib::ControlFlow::Continue
                    }
                }));
            }
            glib::Propagation::Proceed
        }));
        gl_area.add_controller(scroll);

        // Closure shared with interactions that want smooth animated zoom toward
        // a specific viewport (zoom + center), as opposed to the anchor-based
        // scroll zoom.  Sets targets and starts the tick-callback if not already
        // running; if a tick-callback is already running it picks up the new
        // targets automatically on its next iteration.
        let animate_to: Rc<dyn Fn(f64, [f64; 2])> = Rc::new({
            let state = Rc::clone(&state);
            let gl_area_weak = gl_area.downgrade();
            move |target_zoom: f64, target_center: [f64; 2]| {
                let Some(gl_area) = gl_area_weak.upgrade() else { return };
                let mut s = state.borrow_mut();
                s.zoom_target = target_zoom;
                s.zoom_center_target = Some(target_center);
                let was_animating = s.zoom_animating;
                if !was_animating {
                    s.zoom_animating = true;
                }
                drop(s);
                if !was_animating {
                    gl_area.add_tick_callback(clone!(#[strong] state, move |area, _clock| {
                        let animating = state.borrow().zoom_animating;
                        if !animating {
                            return glib::ControlFlow::Break;
                        }

                        let (target, anchor, anchor_screen, center_target) = {
                            let s = state.borrow();
                            (s.zoom_target, s.zoom_anchor, s.zoom_anchor_screen, s.zoom_center_target)
                        };

                        let (new_zoom, new_cx, new_cy, settled) = {
                            let s = state.borrow();
                            let proj = s.project.borrow();
                            let vp = &proj.viewport;
                            let z = vp.zoom + (target - vp.zoom) * 0.2;
                            if let Some([tcx, tcy]) = center_target {
                                let settled_z = (z - target).abs() < 1e-4;
                                let z = if settled_z { target } else { z };
                                let cx = vp.center.x + (tcx - vp.center.x) * 0.2;
                                let cy = vp.center.y + (tcy - vp.center.y) * 0.2;
                                let settled_c = (cx - tcx).abs() < 1.0 && (cy - tcy).abs() < 1.0;
                                let (cx, cy) = if settled_c { (tcx, tcy) } else { (cx, cy) };
                                (z, cx, cy, settled_z && settled_c)
                            } else {
                                let settled = (z - target).abs() < 1e-4;
                                let z = if settled { target } else { z };
                                let res = (2.0 * rgis_core::EARTH_HALF_CIRC)
                                    / (256.0 * 2_f64.powf(z));
                                let cx = anchor[0]
                                    - (anchor_screen[0] as f64 - vp.width_px as f64 * 0.5) * res;
                                let cy = anchor[1]
                                    + (anchor_screen[1] as f64 - vp.height_px as f64 * 0.5) * res;
                                (z, cx, cy, settled)
                            }
                        };

                        {
                            let s = state.borrow();
                            let mut proj = s.project.borrow_mut();
                            proj.viewport.zoom = new_zoom;
                            proj.viewport.center.x = new_cx;
                            proj.viewport.center.y = new_cy;
                        }

                        if settled {
                            let mut s = state.borrow_mut();
                            s.zoom_animating = false;
                            s.zoom_center_target = None;
                        }

                        area.queue_render();

                        if settled {
                            glib::ControlFlow::Break
                        } else {
                            glib::ControlFlow::Continue
                        }
                    }));
                }
            }
        });

        // Attach interaction controllers.
        let project_for_interactions = {
            let s = state.borrow();
            Rc::clone(&s.project)
        };
        interactions
            .drag_zoom_box
            .attach(&gl_area, &drawing_area, project_for_interactions, Rc::clone(&animate_to));

        Self { gl_area, overlay, state, renderer }
    }

    pub fn widget(&self) -> &gtk4::Overlay {
        &self.overlay
    }

    pub fn queue_render(&self) {
        self.gl_area.queue_render();
    }

    /// Called when a tile has been fetched. Flattens pixel data into an Arc once;
    /// subsequent render frames clone the Arc cheaply.
    pub fn on_tile_ready(&self, ready: TileReady) {
        let mut s = self.state.borrow_mut();
        // Cap the CPU tile cache.  Keep the 2 nearest zoom levels so ancestor
        // tiles remain available as fallback placeholders.  Only purge when we
        // significantly exceed the limit so we don't thrash on the boundary.
        const MAX_CPU_TILES: usize = 512;
        const PURGE_THRESHOLD: usize = 600;
        if s.tile_cache.len() >= PURGE_THRESHOLD {
            let z = ready.coord.z;
            // Retain tiles within 2 zoom levels of the arriving tile.
            s.tile_cache.retain(|coord, _| coord.z.abs_diff(z) <= 2);
            // If still too large (very large viewport / many levels), hard-cap by
            // dropping the farthest zoom level until we are under the limit.
            while s.tile_cache.len() > MAX_CPU_TILES {
                // Find the zoom level farthest from z among cached tiles.
                let worst_z = s.tile_cache.keys()
                    .map(|c| c.z)
                    .max_by_key(|&cz| cz.abs_diff(z))
                    .unwrap();
                s.tile_cache.retain(|coord, _| coord.z != worst_z);
            }
        }
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
