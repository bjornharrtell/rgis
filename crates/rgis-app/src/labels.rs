use std::collections::HashMap;
use std::sync::Arc;

use rgis_render::{
    BasemapTileDraw, GlyphBitmapRanges, LabelGlyphInstance, TileDraw, VECTOR_TEXTURE_KEY,
};
use rgis_tiles::{GLYPH_BUFFER, GLYPH_PIXELS_PER_EM, GlyphFetcher, SpriteAtlas, glyph_range_start};

// Keep this distinct from rgis-render's stable user-vector texture key.
const SPRITE_ATLAS_TILE_KEY: u64 = VECTOR_TEXTURE_KEY - 1;

#[derive(Clone, Copy)]
struct LabelRect {
    min: [f32; 2],
    max: [f32; 2],
}

impl LabelRect {
    fn from_glyph(rect: [f32; 4]) -> Self {
        Self {
            min: [rect[0], rect[1]],
            max: [rect[0] + rect[2], rect[1] + rect[3]],
        }
    }

    fn expand(self, amount: f32) -> Self {
        Self {
            min: [self.min[0] - amount, self.min[1] - amount],
            max: [self.max[0] + amount, self.max[1] + amount],
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: [self.min[0].min(other.min[0]), self.min[1].min(other.min[1])],
            max: [self.max[0].max(other.max[0]), self.max[1].max(other.max[1])],
        }
    }

    fn intersects(self, other: Self) -> bool {
        self.min[0] < other.max[0]
            && self.max[0] > other.min[0]
            && self.min[1] < other.max[1]
            && self.max[1] > other.min[1]
    }
}

struct ProjectedLabel {
    position: [f32; 2],
    text: String,
    font_size: f32,
    color: [f32; 4],
    halo_color: [f32; 4],
    halo_width: f32,
    halo_blur: f32,
    max_width: f32,
    fontstack: String,
    priority: i32,
    angle: f32,
    path: Option<Vec<[f32; 2]>>,
    text_anchor_center: bool,
    icon: Option<String>,
    icon_size: f32,
}

/// Projects visible basemap labels into screen-space glyph quads, preserving
/// priority order and avoiding overlapping labels on both rendering targets.
pub fn collect_label_draws(
    basemap_tiles: &[BasemapTileDraw],
    glyph_fetcher: &Arc<GlyphFetcher>,
    sprite_atlas: Option<&Arc<SpriteAtlas>>,
) -> (Vec<LabelGlyphInstance>, GlyphBitmapRanges, Vec<TileDraw>) {
    let mut projected = Vec::new();
    for draw in basemap_tiles {
        for label in &draw.mesh.labels {
            projected.push(ProjectedLabel {
                position: [
                    label.position[0] * draw.scale + draw.offset[0],
                    label.position[1] * draw.scale + draw.offset[1],
                ],
                text: label.text.clone(),
                font_size: label.font_size,
                color: label.color,
                halo_color: label.halo_color,
                halo_width: label.halo_width,
                halo_blur: label.halo_blur,
                max_width: label.max_width,
                fontstack: label.fontstack.clone(),
                priority: label.priority,
                angle: label.angle,
                path: label.path.as_ref().map(|path| {
                    path.iter()
                        .map(|point| {
                            [
                                point[0] * draw.scale + draw.offset[0],
                                point[1] * draw.scale + draw.offset[1],
                            ]
                        })
                        .collect()
                }),
                text_anchor_center: label.text_anchor_center,
                icon: label.icon.clone(),
                icon_size: label.icon_size,
            });
        }
    }
    projected.sort_by_key(|label| label.priority);

    let mut glyphs = Vec::new();
    let mut glyph_bitmaps = GlyphBitmapRanges::default();
    let mut icons = Vec::new();
    let mut placed = Vec::new();

    for label in projected {
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
        let (halo_color, halo) = halo_params(&label, scale);
        let total_advance = codepoints
            .iter()
            .filter_map(|codepoint| {
                ranges
                    .get(&glyph_range_start(*codepoint))
                    .and_then(|range| range.get(codepoint))
                    .map(|glyph| glyph.advance as f32 * scale)
            })
            .sum::<f32>();
        let baseline_offset =
            glyph_run_baseline_offset(codepoints.iter().filter_map(|codepoint| {
                ranges
                    .get(&glyph_range_start(*codepoint))
                    .and_then(|range| range.get(codepoint))
            })) * scale;
        let mut label_glyphs = Vec::with_capacity(codepoints.len());
        let mut bounds: Option<LabelRect> = None;

        if let Some(path) = &label.path {
            let Some(total_length) = path_length(path) else {
                continue;
            };
            let (_, probe_angle) = point_and_angle_at(path, total_length * 0.5);
            let forward = probe_angle.cos() >= 0.0;
            let mut path_position = total_length * 0.5 - total_advance * 0.5;
            for codepoint in codepoints {
                let range_start = glyph_range_start(codepoint);
                let Some(range) = ranges.get(&range_start) else {
                    continue;
                };
                let Some(glyph) = range.get(&codepoint) else {
                    continue;
                };
                let sample_position = if forward {
                    path_position
                } else {
                    total_length - path_position
                };
                let (anchor, mut angle) = point_and_angle_at(path, sample_position);
                if !forward {
                    angle += std::f32::consts::PI;
                }
                let x = anchor[0] + (glyph.left - GLYPH_BUFFER as i32) as f32 * scale;
                let y =
                    anchor[1] + baseline_offset - (glyph.top + GLYPH_BUFFER as i32) as f32 * scale;
                let w = (glyph.width + 2 * GLYPH_BUFFER) as f32 * scale;
                let h = (glyph.height + 2 * GLYPH_BUFFER) as f32 * scale;
                let glyph_rect = [x, y, w, h];
                bounds = Some(match bounds {
                    Some(existing) => existing.union(LabelRect::from_glyph(glyph_rect)),
                    None => LabelRect::from_glyph(glyph_rect),
                });
                label_glyphs.push(LabelGlyphInstance {
                    rect: glyph_rect,
                    anchor,
                    angle,
                    fontstack: label.fontstack.clone(),
                    codepoint,
                    color: label.color,
                    halo_color,
                    halo,
                });
                glyph_bitmaps
                    .entry((label.fontstack.clone(), range_start))
                    .or_insert_with(|| Arc::clone(range));
                path_position += glyph.advance as f32 * scale;
            }
        } else {
            let advance_of = |codepoint: u32| {
                ranges
                    .get(&glyph_range_start(codepoint))
                    .and_then(|range| range.get(&codepoint))
                    .map_or(0.0, |glyph| glyph.advance as f32 * scale)
            };
            let lines = wrap_lines(&codepoints, &advance_of, label.max_width * label.font_size);
            let line_height = label.font_size * LINE_HEIGHT_EM;
            let first_baseline = if label.text_anchor_center {
                label.position[1] + baseline_offset - (lines.len() as f32 - 1.0) * 0.5 * line_height
            } else {
                label.position[1] + label.font_size * 0.35
            };
            for (line_index, line) in lines.iter().enumerate() {
                let baseline_y = first_baseline + line_index as f32 * line_height;
                let line_advance: f32 = codepoints[line.clone()]
                    .iter()
                    .map(|&codepoint| advance_of(codepoint))
                    .sum();
                let mut pen_x = label.position[0] - line_advance * 0.5;
                for &codepoint in &codepoints[line.clone()] {
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
                    let glyph_rect = [x, y, w, h];
                    bounds = Some(match bounds {
                        Some(existing) => existing.union(LabelRect::from_glyph(glyph_rect)),
                        None => LabelRect::from_glyph(glyph_rect),
                    });
                    label_glyphs.push(LabelGlyphInstance {
                        rect: glyph_rect,
                        anchor: label.position,
                        angle: label.angle,
                        fontstack: label.fontstack.clone(),
                        codepoint,
                        color: label.color,
                        halo_color,
                        halo,
                    });
                    glyph_bitmaps
                        .entry((label.fontstack.clone(), range_start))
                        .or_insert_with(|| Arc::clone(range));
                    pen_x += glyph.advance as f32 * scale;
                }
            }
        }

        let icon_draw = label.icon.as_deref().and_then(|icon_name| {
            let atlas = sprite_atlas?;
            let sprite = atlas.rects.get(icon_name)?;
            let width = sprite.width as f32 * label.icon_size;
            let height = sprite.height as f32 * label.icon_size;
            let rect = [
                label.position[0] - width * 0.5,
                label.position[1] - height * 0.5,
                width,
                height,
            ];
            Some((
                LabelRect::from_glyph(rect),
                TileDraw {
                    key: SPRITE_ATLAS_TILE_KEY,
                    rect,
                    rgba: Arc::clone(&atlas.image),
                    uv_rect: [
                        sprite.x as f32 / atlas.image.width() as f32,
                        sprite.y as f32 / atlas.image.height() as f32,
                        (sprite.x + sprite.width) as f32 / atlas.image.width() as f32,
                        (sprite.y + sprite.height) as f32 / atlas.image.height() as f32,
                    ],
                    opacity: 1.0,
                },
            ))
        });
        let Some(label_bounds) = bounds
            .map(|bounds: LabelRect| bounds.expand(2.0))
            .or_else(|| icon_draw.as_ref().map(|(bounds, _)| bounds.expand(2.0)))
        else {
            continue;
        };
        let label_bounds = icon_draw
            .as_ref()
            .map(|(icon_bounds, _)| label_bounds.union(icon_bounds.expand(2.0)))
            .unwrap_or(label_bounds);
        if placed
            .iter()
            .any(|placed: &LabelRect| placed.intersects(label_bounds))
        {
            continue;
        }
        placed.push(label_bounds);
        glyphs.extend(label_glyphs);
        if let Some((_, icon_draw)) = icon_draw {
            icons.push(icon_draw);
        }
    }

    (glyphs, glyph_bitmaps, icons)
}

fn glyph_run_baseline_offset<'a>(glyphs: impl Iterator<Item = &'a rgis_tiles::Glyph>) -> f32 {
    let (mut max_ascent, mut max_descent) = (i32::MIN, i32::MIN);
    for glyph in glyphs {
        max_ascent = max_ascent.max(glyph.top);
        max_descent = max_descent.max(glyph.height as i32 - glyph.top);
    }
    if max_ascent == i32::MIN || max_descent == i32::MIN {
        return 0.0;
    }
    (max_ascent - max_descent) as f32 * 0.5
}

fn path_length(path: &[[f32; 2]]) -> Option<f32> {
    let length = path
        .windows(2)
        .map(|pair| {
            let dx = pair[1][0] - pair[0][0];
            let dy = pair[1][1] - pair[0][1];
            dx.hypot(dy)
        })
        .sum();
    (length > f32::EPSILON).then_some(length)
}

fn point_and_angle_at(path: &[[f32; 2]], target: f32) -> ([f32; 2], f32) {
    let mut walked = 0.0;
    for pair in path.windows(2) {
        let dx = pair[1][0] - pair[0][0];
        let dy = pair[1][1] - pair[0][1];
        let segment_length = dx.hypot(dy);
        if walked + segment_length >= target || segment_length <= f32::EPSILON {
            let t = if segment_length > f32::EPSILON {
                ((target - walked) / segment_length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            return ([pair[0][0] + dx * t, pair[0][1] + dy * t], dy.atan2(dx));
        }
        walked += segment_length;
    }
    let pair = path
        .windows(2)
        .last()
        .expect("path_length guarantees at least one segment");
    let dx = pair[1][0] - pair[0][0];
    let dy = pair[1][1] - pair[0][1];
    (*path.last().unwrap(), dy.atan2(dx))
}

/// `text-line-height` default, in ems.
const LINE_HEIGHT_EM: f32 = 1.2;

/// Greedily breaks `codepoints` into lines no wider than `max_advance`
/// (screen pixels), only at spaces, which are dropped at a break. A single
/// word wider than the limit stays on its own line.
fn wrap_lines(
    codepoints: &[u32],
    advance_of: &dyn Fn(u32) -> f32,
    max_advance: f32,
) -> Vec<std::ops::Range<usize>> {
    const SPACE: u32 = ' ' as u32;
    let mut lines = Vec::new();
    let mut start = 0;
    let mut width = 0.0;
    let mut last_space: Option<usize> = None;
    let mut index = 0;
    while index < codepoints.len() {
        let codepoint = codepoints[index];
        if codepoint == SPACE {
            last_space = Some(index);
        }
        width += advance_of(codepoint);
        if max_advance > 0.0
            && width > max_advance
            && let Some(space) = last_space.filter(|&space| space > start)
        {
            lines.push(start..space);
            start = space + 1;
            last_space = None;
            width = codepoints[start..=index]
                .iter()
                .map(|&codepoint| advance_of(codepoint))
                .sum();
        }
        index += 1;
    }
    lines.push(start..codepoints.len());
    lines
}

/// SDF encoding constants shared with MapLibre: the glyph edge sits at
/// `SDF_EDGE / SDF_SCALE` and a screen pixel of halo is `1 / (scale *
/// SDF_SCALE)` of distance, where `scale` is the on-screen glyph scale.
const SDF_SCALE: f32 = 8.0;
const SDF_EDGE: f32 = 6.0;

/// The halo colour and `[edge, blur]` shader parameters for `label`. Without
/// a halo the colour matches the text, so antialiased edges don't pick up a
/// fringe of it.
fn halo_params(label: &ProjectedLabel, scale: f32) -> ([f32; 4], [f32; 2]) {
    if label.halo_width <= 0.0 {
        return (label.color, [SDF_EDGE / SDF_SCALE, 0.0]);
    }
    let edge = ((SDF_EDGE - label.halo_width / scale) / SDF_SCALE).max(0.05);
    let blur = label.halo_blur / (scale * SDF_SCALE);
    (label.halo_color, [edge, blur])
}

#[cfg(test)]
mod halo_tests {
    use super::*;

    fn label(width: f32, blur: f32) -> ProjectedLabel {
        ProjectedLabel {
            position: [0.0, 0.0],
            text: String::new(),
            font_size: 24.0,
            color: [0.0, 0.0, 0.0, 1.0],
            halo_color: [1.0, 1.0, 1.0, 1.0],
            halo_width: width,
            halo_blur: blur,
            max_width: 10.0,
            fontstack: String::new(),
            priority: 0,
            angle: 0.0,
            path: None,
            text_anchor_center: true,
            icon: None,
            icon_size: 1.0,
        }
    }

    #[test]
    fn wraps_at_spaces_within_the_limit() {
        let text: Vec<u32> = "aa bb cc".chars().map(|c| c as u32).collect();
        let advance = |_: u32| 1.0;
        assert_eq!(wrap_lines(&text, &advance, 100.0), vec![0..8]);
        assert_eq!(wrap_lines(&text, &advance, 5.0), vec![0..5, 6..8]);
        assert_eq!(wrap_lines(&text, &advance, 2.0), vec![0..2, 3..5, 6..8]);
        // An over-long word stays whole.
        let word: Vec<u32> = "abcdefgh".chars().map(|c| c as u32).collect();
        assert_eq!(wrap_lines(&word, &advance, 3.0), vec![0..8]);
        // No limit means a single line.
        assert_eq!(wrap_lines(&text, &advance, 0.0), vec![0..8]);
    }

    #[test]
    fn no_halo_matches_the_text_colour_and_edge() {
        let (color, halo) = halo_params(&label(0.0, 0.0), 1.0);
        assert_eq!(color, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(halo, [0.75, 0.0]);
    }

    #[test]
    fn halo_widens_with_width_and_shrinks_with_scale() {
        let (_, one) = halo_params(&label(1.0, 0.0), 1.0);
        let (_, two) = halo_params(&label(2.0, 0.0), 1.0);
        assert!(two[0] < one[0] && one[0] < 0.75);
        let (_, small) = halo_params(&label(2.0, 0.0), 0.5);
        assert!(small[0] < two[0]);
        let (_, blurred) = halo_params(&label(1.0, 2.0), 1.0);
        assert!(blurred[1] > 0.0);
    }
}
