//! Browser (wasm32) build of rgis.
//!
//! Boots the shared [`rgis_app::RgisApp`] (wgpu + egui) into the page's
//! `<canvas id="rgis-canvas">` via `eframe::WebRunner`, using the WebGPU/WebGL
//! backend selected by `wgpu` at runtime.

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

// dlmalloc (the default wasm32 allocator) fragments badly under this app's
// churn of variably-sized tile-mesh Vecs, permanently growing `WebAssembly
// .Memory` (which can only grow, never shrink) even while every logical
// tile cache stays within its configured bound -- see repo memory notes.
// talc is a drop-in replacement with less WASM memory overhead.
#[cfg(all(not(target_feature = "atomics"), target_family = "wasm"))]
#[global_allocator]
static TALC: talc::wasm::WasmDynamicTalc = talc::wasm::new_wasm_dynamic_allocator();

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();

    // Debug-only hook for automated testing: lets a test script jump the
    // viewport directly (`window.debugJumpViewport(lon, lat, zoom)`)
    // instead of simulating imprecise wheel/drag input, to reliably
    // stress-test tile loading over a specific area.
    let debug_jump_viewport =
        Closure::wrap(Box::new(rgis_app::debug_jump_viewport) as Box<dyn Fn(f64, f64, f64)>);
    js_sys::Reflect::set(
        &web_sys::window().expect("no global `window`"),
        &"debugJumpViewport".into(),
        debug_jump_viewport.as_ref().unchecked_ref(),
    )
    .expect("failed to install window.debugJumpViewport");
    debug_jump_viewport.forget();

    // Debug-only hook: read the main thread's current wasm memory usage
    // (bytes) on demand, via `window.debugMemBytes()`.
    let debug_mem_bytes =
        Closure::wrap(Box::new(rgis_app::debug_wasm_memory_bytes) as Box<dyn Fn() -> u32>);
    js_sys::Reflect::set(
        &web_sys::window().expect("no global `window`"),
        &"debugMemBytes".into(),
        debug_mem_bytes.as_ref().unchecked_ref(),
    )
    .expect("failed to install window.debugMemBytes");
    debug_mem_bytes.forget();

    // Debug-only hook: read the cumulative count of distinct tiles ever
    // tessellated this session, via `window.debugDistinctTileCount()` --
    // used to distinguish "memory grows because more distinct tiles were
    // visited" from an actual leak.
    let debug_distinct_tile_count =
        Closure::wrap(Box::new(rgis_app::debug_distinct_tile_count) as Box<dyn Fn() -> u32>);
    js_sys::Reflect::set(
        &web_sys::window().expect("no global `window`"),
        &"debugDistinctTileCount".into(),
        debug_distinct_tile_count.as_ref().unchecked_ref(),
    )
    .expect("failed to install window.debugDistinctTileCount");
    debug_distinct_tile_count.forget();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no global `window`")
            .document()
            .expect("no document on window");
        let canvas = document
            .get_element_by_id("rgis-canvas")
            .expect("missing #rgis-canvas element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("#rgis-canvas must be a <canvas>");

        let runner = eframe::WebRunner::new();
        runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|cc| {
                    let mut app = rgis_app::RgisApp::new(cc);
                    app.queue_load_sample();
                    Ok(Box::new(app))
                }),
            )
            .await
            .expect("failed to start eframe on rgis-canvas");
    });
}
