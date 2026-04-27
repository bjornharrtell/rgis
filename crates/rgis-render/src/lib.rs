mod gl;
pub use gl::GlRenderer;

use rgis_core::{Layer, Viewport};

/// RGBA tile image, width x height pixels.
pub struct TileImage {
    /// Tile coordinate (z, x, y) — used by the renderer to cache the GL texture.
    pub coord: (u8, u32, u32),
    /// Shared pixel data; cloning this is a reference-count bump, not a memcopy.
    pub rgba: std::sync::Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    /// Screen-pixel rectangle [x, y, w, h] where this tile should be drawn.
    pub screen_rect: [f32; 4],
}

/// Abstraction over rendering backends.
pub trait MapRenderer {
    fn resize(&mut self, width: u32, height: u32);
    fn render(
        &mut self,
        viewport: &Viewport,
        layers: &[Layer],
        tiles: &[TileImage],
    );
    fn invalidate_layer(&mut self, layer_id: rgis_core::LayerId);
}
