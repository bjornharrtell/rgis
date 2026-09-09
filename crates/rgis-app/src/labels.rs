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
                    halo_color: label.halo_color,
                });
                glyph_bitmaps
                    .entry((label.fontstack.clone(), range_start))
                    .or_insert_with(|| Arc::clone(range));
                path_position += glyph.advance as f32 * scale;
            }
        } else {
            let baseline_y = if label.text_anchor_center {
                label.position[1] + baseline_offset
            } else {
                label.position[1] + label.font_size * 0.35
            };
            let mut pen_x = label.position[0] - total_advance * 0.5;
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
                    halo_color: label.halo_color,
                });
                glyph_bitmaps
                    .entry((label.fontstack.clone(), range_start))
                    .or_insert_with(|| Arc::clone(range));
                pen_x += glyph.advance as f32 * scale;
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
