//! Browser bootstrap for the GPUI web platform.

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

#[cfg(all(not(target_feature = "atomics"), target_family = "wasm"))]
#[global_allocator]
static TALC: talc::wasm::WasmDynamicTalc = talc::wasm::new_wasm_dynamic_allocator();

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    install_debug_hooks();
    rgis_app::run();
}

fn install_debug_hooks() {
    let window = web_sys::window().expect("no global window");

    let debug_jump_viewport =
        Closure::wrap(Box::new(rgis_app::debug_jump_viewport) as Box<dyn Fn(f64, f64, f64)>);
    js_sys::Reflect::set(
        &window,
        &"debugJumpViewport".into(),
        debug_jump_viewport.as_ref().unchecked_ref(),
    )
    .expect("failed to install window.debugJumpViewport");
    debug_jump_viewport.forget();

    let debug_mem_bytes =
        Closure::wrap(Box::new(rgis_app::debug_wasm_memory_bytes) as Box<dyn Fn() -> u32>);
    js_sys::Reflect::set(
        &window,
        &"debugMemBytes".into(),
        debug_mem_bytes.as_ref().unchecked_ref(),
    )
    .expect("failed to install window.debugMemBytes");
    debug_mem_bytes.forget();

    let debug_distinct_tile_count =
        Closure::wrap(Box::new(rgis_app::debug_distinct_tile_count) as Box<dyn Fn() -> u32>);
    js_sys::Reflect::set(
        &window,
        &"debugDistinctTileCount".into(),
        debug_distinct_tile_count.as_ref().unchecked_ref(),
    )
    .expect("failed to install window.debugDistinctTileCount");
    debug_distinct_tile_count.forget();
}
