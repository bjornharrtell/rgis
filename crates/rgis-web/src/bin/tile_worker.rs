//! Web Worker entry point that decodes raw MVT tile bytes and tessellates
//! them into a [`rgis_render::TileMeshWire`], off the main thread.
//!
//! Built by Trunk as a separate `data-type="worker"` asset (see
//! `../../index.html`) and spawned from the main thread by
//! `rgis-app`'s wasm-only `tile_worker_pool` module, which also documents
//! the message protocol implemented here.

use js_sys::{Array, Float32Array, Uint8Array, Uint32Array};
use rgis_render::{TileMeshWire, build_tile_mesh};
use rgis_tiles::{TileCoord, decode_vector_tile};
use wasm_bindgen::{JsCast, JsValue, prelude::Closure};
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

// Same allocator swap as the main thread's module (`rgis-web/src/lib.rs`)
// -- this worker is a SEPARATE wasm instance with its own linear memory, so
// it needs its own `#[global_allocator]`.
#[cfg(all(not(target_feature = "atomics"), target_family = "wasm"))]
#[global_allocator]
static TALC: talc::wasm::WasmDynamicTalc = talc::wasm::new_wasm_dynamic_allocator();

fn main() {
    console_error_panic_hook::set_once();

    let scope = DedicatedWorkerGlobalScope::from(JsValue::from(js_sys::global()));
    let scope_clone = scope.clone();

    let onmessage = Closure::wrap(Box::new(move |msg: MessageEvent| {
        let data = Array::from(&msg.data());
        let z = data.get(0).as_f64().expect("z to be a number") as u8;
        let x = data.get(1).as_f64().expect("x to be a number") as u32;
        let y = data.get(2).as_f64().expect("y to be a number") as u32;
        let bytes = Uint8Array::from(data.get(3)).to_vec();
        let coord = TileCoord { z, x, y };

        let reply = Array::new();
        reply.push(&(z as f64).into());
        reply.push(&(x as f64).into());
        reply.push(&(y as f64).into());

        match decode_vector_tile(&bytes) {
            Ok(tile) => {
                let wire = TileMeshWire::from(&build_tile_mesh(&tile, coord));

                let fill_vertices = Float32Array::from(wire.fill_vertices.as_slice());
                let fill_indices = Uint32Array::from(wire.fill_indices.as_slice());
                let line_vertices = Float32Array::from(wire.line_vertices.as_slice());
                let line_indices = Uint32Array::from(wire.line_indices.as_slice());

                let transfer = Array::new();
                transfer.push(&fill_vertices.buffer());
                transfer.push(&fill_indices.buffer());
                transfer.push(&line_vertices.buffer());
                transfer.push(&line_indices.buffer());

                reply.push(&fill_vertices);
                reply.push(&fill_indices);
                reply.push(&line_vertices);
                reply.push(&line_indices);
                // Plain JSON string, not transferred (strings are
                // structured-cloned, not backed by a transferable
                // ArrayBuffer) -- see `TileMeshWire::labels_json`.
                reply.push(&JsValue::from_str(&wire.labels_json));

                scope_clone
                    .post_message_with_transfer(&reply.into(), &transfer.into())
                    .expect("posting tile mesh result succeeds");
            }
            Err(_) => {
                // Length-3 reply (no mesh arrays) signals decode failure to
                // `TileWorkerPool`.
                scope_clone
                    .post_message(&reply.into())
                    .expect("posting decode-failure result succeeds");
            }
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    scope.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();

    // A zero-length array signals readiness: messages sent to a worker
    // before its wasm module finishes loading (and this closure is
    // registered) are silently dropped, so `TileWorkerPool` waits for this
    // ping before dispatching any real job to this worker.
    scope
        .post_message(&Array::new().into())
        .expect("posting ready message succeeds");
}
