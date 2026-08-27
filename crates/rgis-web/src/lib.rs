//! Browser (wasm32) build of rgis.
//!
//! Boots the shared [`rgis_app::RgisApp`] (wgpu + egui) into the page's
//! `<canvas id="rgis-canvas">` via `eframe::WebRunner`, using the WebGPU/WebGL
//! backend selected by `wgpu` at runtime.

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

/// A demo dataset (world borders, simplified) bundled with the wasm binary so
/// the map has something to show without requiring a file picker.
const SAMPLE_GEOJSON: &[u8] = include_bytes!("../assets/sample.geojson");

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();

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
                    app.queue_load_bytes("sample.geojson".to_string(), SAMPLE_GEOJSON.to_vec());
                    Ok(Box::new(app))
                }),
            )
            .await
            .expect("failed to start eframe on rgis-canvas");
    });
}
