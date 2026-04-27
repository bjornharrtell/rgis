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
    /// UV sub-rectangle [u_min, v_min, u_size, v_size] within the texture.
    /// `[0.0, 0.0, 1.0, 1.0]` for a tile drawn at its own zoom level;
    /// a sub-rect when this is a parent tile standing in for a not-yet-loaded
    /// child (the relevant quadrant of the parent image fills the child slot).
    pub src_rect: [f32; 4],
}

/// Abstraction over rendering backends.
pub trait MapRenderer {
    fn resize(&mut self, width: u32, height: u32);
    /// Render a frame.
    ///
    /// `allow_retessellate` — when `false` the renderer must not rebuild any
    /// zoom-dependent geometry buffers.  Pass `false` while a zoom animation
    /// is in progress so that the old (slightly-wrong-width) tessellation is
    /// reused for every animation frame.  Pass `true` on the settle frame so
    /// the geometry is rebuilt once at the final zoom level.
    fn render(
        &mut self,
        viewport: &Viewport,
        layers: &[Layer],
        tiles: &[TileImage],
        allow_retessellate: bool,
    );
    fn invalidate_layer(&mut self, layer_id: rgis_core::LayerId);
}
