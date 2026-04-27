use std::{cell::RefCell, rc::Rc};

use glib::clone;
use gtk4::prelude::*;
use rgis_core::Project;

// ── Internal gesture state ────────────────────────────────────────────────────

struct State {
    active: bool,
    /// Drag-start position in logical (CSS) pixels.
    start: (f64, f64),
    /// Current end position in logical (CSS) pixels.
    current: (f64, f64),
}

// ── Public interaction type ───────────────────────────────────────────────────

/// Rubber-band zoom-box interaction.
///
/// Hold **Shift** and drag to draw a selection rectangle; releasing the mouse
/// fits the viewport to that rectangle.  Enabled by default.
///
/// # Example – disable at compile-time
/// ```rust,no_run
/// use crate::drag_zoom_box::DragZoomBox;
/// DragZoomBox::new().enabled(false)
/// ```
pub struct DragZoomBox {
    enabled: bool,
}

impl Default for DragZoomBox {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl DragZoomBox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable this interaction (default: `true`).
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Wire up gesture controllers on `gl_area` and the rubber-band
    /// `drawing_area` overlay.  Called once during `MapArea` construction.
    pub(crate) fn attach(
        self,
        gl_area: &gtk4::GLArea,
        drawing_area: &gtk4::DrawingArea,
        project: Rc<RefCell<Project>>,
    ) {
        if !self.enabled {
            return;
        }

        let state: Rc<RefCell<State>> = Rc::new(RefCell::new(State {
            active: false,
            start: (0.0, 0.0),
            current: (0.0, 0.0),
        }));

        // Draw the rubber-band rectangle on every expose of `drawing_area`.
        drawing_area.set_draw_func(clone!(
            #[strong]
            state,
            move |_area, cr, _w, _h| {
                let s = state.borrow();
                if !s.active {
                    return;
                }
                let (x0, y0) = s.start;
                let (x1, y1) = s.current;
                let rx = x0.min(x1);
                let ry = y0.min(y1);
                let rw = (x1 - x0).abs();
                let rh = (y1 - y0).abs();

                // Semi-transparent blue fill.
                cr.set_source_rgba(0.2, 0.4, 1.0, 0.15);
                cr.rectangle(rx, ry, rw, rh);
                let _ = cr.fill();

                // Solid blue border.
                cr.set_source_rgba(0.2, 0.4, 1.0, 0.85);
                cr.set_line_width(1.5);
                cr.rectangle(rx, ry, rw, rh);
                let _ = cr.stroke();
            }
        ));

        let drag = gtk4::GestureDrag::new();
        drag.set_button(gtk4::gdk::BUTTON_PRIMARY);
        // Claim before the pan gesture so we receive updates while Shift is held.
        drag.set_propagation_phase(gtk4::PropagationPhase::Capture);

        drag.connect_drag_begin(clone!(
            #[strong]
            state,
            move |gesture, x, y| {
                let shift = gesture
                    .current_event()
                    .map(|ev| {
                        ev.modifier_state()
                            .contains(gtk4::gdk::ModifierType::SHIFT_MASK)
                    })
                    .unwrap_or(false);

                if !shift {
                    // Let the pan gesture handle normal drags.
                    gesture.set_state(gtk4::EventSequenceState::Denied);
                    return;
                }

                gesture.set_state(gtk4::EventSequenceState::Claimed);
                let mut s = state.borrow_mut();
                s.active = true;
                s.start = (x, y);
                s.current = (x, y);
            }
        ));

        drag.connect_drag_update(clone!(
            #[strong]
            state,
            #[weak]
            drawing_area,
            move |_gesture, dx, dy| {
                let (sx, sy) = {
                    let s = state.borrow();
                    if !s.active {
                        return;
                    }
                    s.start
                };
                state.borrow_mut().current = (sx + dx, sy + dy);
                drawing_area.queue_draw();
            }
        ));

        drag.connect_drag_end(clone!(
            #[strong]
            state,
            #[weak]
            gl_area,
            #[weak]
            drawing_area,
            move |_gesture, dx, dy| {
                let (sx, sy, ex, ey) = {
                    let mut s = state.borrow_mut();
                    if !s.active {
                        return;
                    }
                    let (sx, sy) = s.start;
                    let (ex, ey) = (sx + dx, sy + dy);
                    s.active = false;
                    (sx, sy, ex, ey)
                };

                // Clear the overlay rectangle.
                drawing_area.queue_draw();

                // Scale logical pixels → device pixels so coordinates are
                // consistent with viewport.width_px / height_px.
                let scale = gl_area.scale_factor() as f32;
                let px0 = [sx as f32 * scale, sy as f32 * scale];
                let px1 = [ex as f32 * scale, ey as f32 * scale];

                // Ignore accidental tiny boxes (e.g. mis-clicks).
                if (px1[0] - px0[0]).abs() < 5.0 || (px1[1] - px0[1]).abs() < 5.0 {
                    return;
                }

                let mut proj = project.borrow_mut();
                let w0 = proj.viewport.screen_to_world(px0);
                let w1 = proj.viewport.screen_to_world(px1);
                proj.viewport.fit_bounds(&rgis_core::Bounds {
                    min_x: w0.x.min(w1.x),
                    min_y: w0.y.min(w1.y),
                    max_x: w0.x.max(w1.x),
                    max_y: w0.y.max(w1.y),
                });
                drop(proj);
                gl_area.queue_render();
            }
        ));

        gl_area.add_controller(drag);
    }
}

// ── Interactions bundle ───────────────────────────────────────────────────────

/// All optional map interactions in one place, similar to OpenLayers'
/// `defaultInteractions()`.  Every field is enabled by default.
///
/// # Example – opt out of drag-zoom-box
/// ```rust,no_run
/// Interactions {
///     drag_zoom_box: DragZoomBox::new().enabled(false),
/// }
/// ```
pub struct Interactions {
    pub drag_zoom_box: DragZoomBox,
}

impl Default for Interactions {
    fn default() -> Self {
        Self {
            drag_zoom_box: DragZoomBox::default(),
        }
    }
}
