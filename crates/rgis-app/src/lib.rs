//! GPUI application entry points shared by the native and browser targets.

pub mod labels;
pub mod raster;
pub mod ui;

#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(target_arch = "wasm32")]
pub use web::{debug_distinct_tile_count, debug_jump_viewport, debug_wasm_memory_bytes, run};
