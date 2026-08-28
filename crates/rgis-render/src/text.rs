use std::collections::HashMap;
use std::sync::Arc;

use rgis_tiles::Glyph;

/// One glyph quad to draw this frame, already positioned in screen pixels.
pub struct LabelGlyphInstance {
    /// `[x, y, w, h]` in top-left-origin screen pixels, matching the full
    /// buffered glyph bitmap (ink plus `GLYPH_BUFFER` padding).
    pub rect: [f32; 4],
    pub fontstack: String,
    pub codepoint: u32,
    pub color: [f32; 4],
    pub halo_color: [f32; 4],
}

/// Shared glyph-range data needed by the GPU callback to lazily pack any
/// as-yet-unseen glyph bitmaps into the persistent atlas.
pub type GlyphBitmapRanges = HashMap<(String, u32), Arc<HashMap<u32, Glyph>>>;

/// Everything needed to draw one frame's worth of label glyphs.
pub struct LabelDraw {
    pub glyphs: Vec<LabelGlyphInstance>,
}
