use std::collections::HashMap;
use std::sync::Arc;

use rgis_render::{BasemapTileDraw, GlyphBitmapRanges, LabelGlyphInstance};
use rgis_tiles::{GLYPH_BUFFER, GLYPH_PIXELS_PER_EM, GlyphFetcher, glyph_range_start};

/// Projects the labels in visible basemap tiles into screen-space glyph quads.
///
/// Missing glyph ranges are requested asynchronously; the next animation frame
/// retries them and uploads the decoded ranges through `MapRenderResources`.
pub fn collect_label_draws(
    basemap_tiles: &[BasemapTileDraw],
    glyph_fetcher: &Arc<GlyphFetcher>,
) -> (Vec<LabelGlyphInstance>, GlyphBitmapRanges) {
    let mut glyphs = Vec::new();
    let mut glyph_bitmaps = GlyphBitmapRanges::default();

    for draw in basemap_tiles {
        for label in &draw.mesh.labels {
            if label.text.is_empty() {
                continue;
            }
            let codepoints: Vec<u32> = label
                .text
                .chars()
                .map(|character| character as u32)
                .collect();
            let mut ranges = HashMap::new();
            let mut missing = false;
            for &codepoint in &codepoints {
                let range_start = glyph_range_start(codepoint);
                let Some(range) = glyph_fetcher.get_cached(&label.fontstack, codepoint) else {
                    glyph_fetcher.request(&label.fontstack, codepoint);
                    missing = true;
                    continue;
                };
                if !range.contains_key(&codepoint) {
                    glyph_fetcher.request(&label.fontstack, codepoint);
                    missing = true;
                    continue;
                }
                ranges.entry(range_start).or_insert(range);
            }
            if missing {
                continue;
            }

            let scale = label.font_size / GLYPH_PIXELS_PER_EM;
            let total_advance = codepoints
                .iter()
                .filter_map(|codepoint| {
                    ranges
                        .get(&glyph_range_start(*codepoint))
                        .and_then(|range| range.get(codepoint))
                        .map(|glyph| glyph.advance as f32 * scale)
                })
                .sum::<f32>();
            let position = [
                label.position[0] * draw.scale + draw.offset[0],
                label.position[1] * draw.scale + draw.offset[1],
            ];
            let baseline_y = position[1] + label.font_size * 0.35;
            let mut pen_x = position[0] - total_advance * 0.5;

            for codepoint in codepoints {
                let range_start = glyph_range_start(codepoint);
                let Some(range) = ranges.get(&range_start) else {
                    continue;
                };
                let Some(glyph) = range.get(&codepoint) else {
                    continue;
                };
                let x = pen_x + (glyph.left - GLYPH_BUFFER as i32) as f32 * scale;
                let y = baseline_y - (glyph.top + GLYPH_BUFFER as i32) as f32 * scale;
                let w = (glyph.width + 2 * GLYPH_BUFFER) as f32 * scale;
                let h = (glyph.height + 2 * GLYPH_BUFFER) as f32 * scale;
                glyphs.push(LabelGlyphInstance {
                    rect: [x, y, w, h],
                    anchor: position,
                    angle: label.angle,
                    fontstack: label.fontstack.clone(),
                    codepoint,
                    color: label.color,
                    halo_color: label.halo_color,
                });
                glyph_bitmaps
                    .entry((label.fontstack.clone(), range_start))
                    .or_insert_with(|| Arc::clone(range));
                pen_x += glyph.advance as f32 * scale;
            }
        }
    }

    (glyphs, glyph_bitmaps)
}
