//! Fetching + decoding of MapLibre/Mapbox-style "glyph PBF" files: signed
//! distance field (SDF) bitmaps for a 256-codepoint range of one
//! "fontstack" (e.g. `"Noto Sans Regular"`), fetched on demand from the
//! same host serving OpenFreeMap's vector tiles, exactly like MapLibre GL
//! JS itself does for the same basemap style (see the style's own
//! `"glyphs": "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf"`
//! template) -- rather than bundling any font files locally. Since
//! OpenFreeMap's glyph server precomposes each fontstack from dozens of
//! per-script Noto faces, this gets full-script glyph coverage (Han,
//! Arabic, Devanagari, Thai, ...) for free, without shipping any of those
//! (often multi-megabyte) font files in this app's own binary/wasm bundle.
//!
//! The wire format is Mapbox's `glyphs.proto` (stable and widely
//! reimplemented -- martin, tileserver-gl, os2gearth, ... all use it):
//!
//! ```proto
//! message glyph {
//!     required uint32 id = 1;
//!     optional bytes bitmap = 2;  // SDF, (width+2*3) x (height+2*3) px
//!     required uint32 width = 3;  // ink-only, excludes the 3px buffer
//!     required uint32 height = 4;
//!     required sint32 left = 5;
//!     required sint32 top = 6;
//!     required uint32 advance = 7;
//! }
//! message fontstack {
//!     required string name = 1;
//!     required string range = 2;
//!     repeated glyph glyphs = 3;
//! }
//! message glyphs {
//!     repeated fontstack stacks = 1;
//! }
//! ```
//!
//! There's no general-purpose protobuf crate already in this workspace (MVT
//! decoding uses `fast-mvt`, which only exposes its own vector-tile-shaped
//! reader), and this schema is tiny and fixed, so it's decoded here with a
//! small hand-rolled varint/length-delimited reader instead of pulling in a
//! protoc-based codegen toolchain for one message.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_channel::{Receiver, Sender};

use crate::TileError;

/// Padding (px) added around every glyph's ink on all four sides in the
/// SDF bitmap, per the glyph PBF convention -- `bitmap` is always
/// `(width + 2*GLYPH_BUFFER) x (height + 2*GLYPH_BUFFER)` even though
/// `width`/`height` describe the glyph's own ink-only box.
pub const GLYPH_BUFFER: u32 = 3;

/// Fixed em-square size (px) glyph PBF servers rasterize SDF bitmaps at (a
/// stable cross-implementation convention, e.g. `node-fontnik`/`martin`) --
/// `Glyph`'s metrics (and the `advance` used to lay out following glyphs)
/// scale linearly against this to reach a caller's actual desired font
/// size.
pub const GLYPH_PIXELS_PER_EM: f32 = 24.0;

/// A single decoded glyph: an SDF bitmap plus the metrics needed to
/// position it relative to a text cursor/baseline.
#[derive(Debug, Clone, Default)]
pub struct Glyph {
    /// `(width + 2*GLYPH_BUFFER) * (height + 2*GLYPH_BUFFER)` single-channel
    /// (0-255) bytes, row-major.
    pub bitmap: Vec<u8>,
    /// Ink-only width/height, excluding `GLYPH_BUFFER` padding.
    pub width: u32,
    pub height: u32,
    /// Horizontal bearing: ink starts `left` px right of the pen position.
    pub left: i32,
    /// Vertical bearing: the *signed* Y offset (screen/bitmap convention --
    /// down is positive) from the baseline to the ink's top edge. Real
    /// glyph-PBF data makes this negative for ordinary upright glyphs
    /// (ink rises *above* the baseline), e.g. digits in "Noto Sans
    /// Regular" are `top: -9` -- it is NOT a positive "how many px above
    /// the baseline" magnitude, despite reading that way at a glance (a
    /// real bug in `rgis-app`'s label baseline centering once assumed
    /// exactly that; see `glyph_run_baseline_offset` there).
    pub top: i32,
    /// Pen advance to the next glyph.
    pub advance: u32,
}

/// The first codepoint of the 256-wide range containing `codepoint` --
/// glyph PBFs are always fetched in these fixed-size blocks.
pub fn glyph_range_start(codepoint: u32) -> u32 {
    (codepoint / 256) * 256
}

/// Decodes a `glyphs.proto` `glyphs` message (a whole `{range}.pbf`
/// response), returning every glyph it contains keyed by codepoint. A
/// response only ever contains one `fontstack` in practice (one was
/// requested), but all are merged here just in case a server ever combines
/// several.
pub fn decode_glyphs(bytes: &[u8]) -> Result<HashMap<u32, Glyph>, TileError> {
    let mut reader = PbfReader::new(bytes);
    let mut glyphs = HashMap::new();
    while let Some((field, wire)) = reader.read_tag()? {
        if field == 1 && wire == WIRE_LEN {
            parse_fontstack(reader.read_bytes()?, &mut glyphs)?;
        } else {
            reader.skip(wire)?;
        }
    }
    Ok(glyphs)
}

fn parse_fontstack(bytes: &[u8], out: &mut HashMap<u32, Glyph>) -> Result<(), TileError> {
    let mut reader = PbfReader::new(bytes);
    while let Some((field, wire)) = reader.read_tag()? {
        if field == 3 && wire == WIRE_LEN {
            if let Some((id, glyph)) = parse_glyph(reader.read_bytes()?)? {
                out.insert(id, glyph);
            }
        } else {
            reader.skip(wire)?;
        }
    }
    Ok(())
}

fn parse_glyph(bytes: &[u8]) -> Result<Option<(u32, Glyph)>, TileError> {
    let mut reader = PbfReader::new(bytes);
    let mut id = None;
    let mut glyph = Glyph::default();
    while let Some((field, wire)) = reader.read_tag()? {
        match (field, wire) {
            (1, WIRE_VARINT) => id = Some(reader.read_varint()? as u32),
            (2, WIRE_LEN) => glyph.bitmap = reader.read_bytes()?.to_vec(),
            (3, WIRE_VARINT) => glyph.width = reader.read_varint()? as u32,
            (4, WIRE_VARINT) => glyph.height = reader.read_varint()? as u32,
            (5, WIRE_VARINT) => glyph.left = zigzag_decode(reader.read_varint()?),
            (6, WIRE_VARINT) => glyph.top = zigzag_decode(reader.read_varint()?),
            (7, WIRE_VARINT) => glyph.advance = reader.read_varint()? as u32,
            (_, wire) => reader.skip(wire)?,
        }
    }
    Ok(id.map(|id| (id, glyph)))
}

fn zigzag_decode(v: u64) -> i32 {
    ((v >> 1) as i64 ^ -((v & 1) as i64)) as i32
}

const WIRE_VARINT: u8 = 0;
const WIRE_LEN: u8 = 2;
const WIRE_32BIT: u8 = 5;
const WIRE_64BIT: u8 = 1;

/// Minimal protobuf reader covering just what `glyphs.proto` needs:
/// varints, length-delimited fields, and skipping fields of any wire type
/// (forward-compatibility with fields this schema doesn't know about).
struct PbfReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> PbfReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_varint(&mut self) -> Result<u64, TileError> {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| TileError::Glyph("truncated varint".into()))?;
            self.pos += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(TileError::Glyph("varint too long".into()));
            }
        }
    }

    /// Reads a field tag, returning `None` at end of input.
    fn read_tag(&mut self) -> Result<Option<(u32, u8)>, TileError> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let tag = self.read_varint()?;
        Ok(Some(((tag >> 3) as u32, (tag & 7) as u8)))
    }

    fn read_bytes(&mut self) -> Result<&'a [u8], TileError> {
        let len = self.read_varint()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.buf.len())
            .ok_or_else(|| TileError::Glyph("length-delimited field out of bounds".into()))?;
        let bytes = &self.buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn skip(&mut self, wire: u8) -> Result<(), TileError> {
        match wire {
            WIRE_VARINT => {
                self.read_varint()?;
            }
            WIRE_LEN => {
                self.read_bytes()?;
            }
            WIRE_32BIT => {
                self.pos = self
                    .pos
                    .checked_add(4)
                    .filter(|&p| p <= self.buf.len())
                    .ok_or_else(|| TileError::Glyph("truncated 32-bit field".into()))?;
            }
            WIRE_64BIT => {
                self.pos = self
                    .pos
                    .checked_add(8)
                    .filter(|&p| p <= self.buf.len())
                    .ok_or_else(|| TileError::Glyph("truncated 64-bit field".into()))?;
            }
            other => return Err(TileError::Glyph(format!("unsupported wire type {other}"))),
        }
        Ok(())
    }
}

// ── GlyphFetcher ──────────────────────────────────────────────────────────

const OPENFREEMAP_GLYPHS_URL_TEMPLATE: &str =
    "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf";

/// One 256-codepoint range's worth of decoded glyphs, ready for a caller's
/// own glyph cache.
pub struct GlyphRangeReady {
    pub fontstack: String,
    pub range_start: u32,
    pub glyphs: Arc<HashMap<u32, Glyph>>,
}

type RangeKey = (String, u32);

/// Fetches + caches glyph PBF ranges from OpenFreeMap's glyph server.
/// Mirrors `VectorTileFetcher`'s shape: `request` is fire-and-forget, with
/// results delivered asynchronously via `receiver`.
pub struct GlyphFetcher {
    cache: Mutex<HashMap<RangeKey, Arc<HashMap<u32, Glyph>>>>,
    in_flight: Mutex<HashSet<RangeKey>>,
    sender: Sender<GlyphRangeReady>,
    pub receiver: Receiver<GlyphRangeReady>,
}

impl GlyphFetcher {
    pub fn new() -> Arc<Self> {
        let (sender, receiver) = async_channel::bounded(64);
        Arc::new(Self {
            cache: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(HashSet::new()),
            sender,
            receiver,
        })
    }

    /// Returns the already-cached range containing `codepoint` for
    /// `fontstack`, if any -- doesn't trigger a fetch (see [`Self::request`]
    /// for that).
    pub fn get_cached(&self, fontstack: &str, codepoint: u32) -> Option<Arc<HashMap<u32, Glyph>>> {
        let key = (fontstack.to_string(), glyph_range_start(codepoint));
        self.cache.lock().unwrap().get(&key).cloned()
    }

    /// Requests the 256-codepoint range containing `codepoint` for
    /// `fontstack`, if it isn't already cached or already in flight. The
    /// fetcher updates its own cache once the range arrives (so a later
    /// `get_cached` call sees it); `self.receiver` additionally gets a copy
    /// of every newly-cached range, purely so callers can request a
    /// repaint/wake up without polling `get_cached` every frame. Failures
    /// are ignored (matching the vector/raster tile fetchers' "give up
    /// silently" behaviour) but do clear the in-flight marker, so a later
    /// `request` call for a still-needed glyph will retry.
    pub fn request(self: &Arc<Self>, fontstack: &str, codepoint: u32) {
        let range_start = glyph_range_start(codepoint);
        let key = (fontstack.to_string(), range_start);
        if self.cache.lock().unwrap().contains_key(&key) {
            return;
        }
        if !self.in_flight.lock().unwrap().insert(key.clone()) {
            return;
        }

        let url = OPENFREEMAP_GLYPHS_URL_TEMPLATE
            .replace("{fontstack}", &percent_encode_space(fontstack))
            .replace("{range}", &format!("{range_start}-{}", range_start + 255));

        let this = Arc::clone(self);
        let fontstack_owned = fontstack.to_string();
        let request = ehttp::Request::get(url);
        ehttp::fetch(request, move |result: ehttp::Result<ehttp::Response>| {
            let glyphs = result
                .ok()
                .filter(|r| r.ok)
                .and_then(|r| decode_glyphs(&r.bytes).ok());
            this.in_flight.lock().unwrap().remove(&key);
            if let Some(glyphs) = glyphs {
                let glyphs = Arc::new(glyphs);
                this.cache.lock().unwrap().insert(key, Arc::clone(&glyphs));
                let _ = this.sender.try_send(GlyphRangeReady {
                    fontstack: fontstack_owned,
                    range_start,
                    glyphs,
                });
            }
        });
    }
}

/// Minimal percent-encoding for fontstack names, which only ever contain
/// ASCII letters and spaces (e.g. `"Noto Sans Regular"`) -- not a general
/// URL encoder.
fn percent_encode_space(s: &str) -> String {
    s.replace(' ', "%20")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_start_floors_to_256_boundary() {
        assert_eq!(glyph_range_start(0), 0);
        assert_eq!(glyph_range_start(65), 0);
        assert_eq!(glyph_range_start(255), 0);
        assert_eq!(glyph_range_start(256), 256);
        assert_eq!(glyph_range_start(1024), 1024);
        assert_eq!(glyph_range_start(1279), 1024);
    }

    #[test]
    fn zigzag_round_trips_negative_and_positive() {
        // Standard protobuf sint32 zigzag mapping: 0,-1,1,-2,2 -> 0,1,2,3,4.
        assert_eq!(zigzag_decode(0), 0);
        assert_eq!(zigzag_decode(1), -1);
        assert_eq!(zigzag_decode(2), 1);
        assert_eq!(zigzag_decode(3), -2);
        assert_eq!(zigzag_decode(4), 2);
    }

    /// Decodes a real glyph PBF fetched from OpenFreeMap's glyph server
    /// (`fonts/Noto%20Sans%20Regular/0-255.pbf`) to guard against schema
    /// regressions -- this is the actual wire format servers send, not a
    /// synthetic one.
    #[test]
    fn decodes_real_glyph_range_fixture() {
        let full_path = format!(
            "{}/fixtures/noto_sans_regular_0-255.pbf",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(&full_path).expect("failed to read fixture");
        let glyphs = decode_glyphs(&bytes).expect("decode fixture glyph PBF");

        assert!(glyphs.len() > 100, "expected most of the ASCII range");

        // 'A' (U+0041): known metrics from manual inspection of this
        // fixture, checked so a decoder regression (e.g. buffer/metrics
        // mixed up) is caught even though visual rendering can't be
        // asserted here. `top` is negative -- ink rises *above* the
        // baseline in this glyph-PBF convention, not a positive
        // "how far above" magnitude (a real bug in `rgis-app`'s label
        // baseline centering once assumed the latter; see
        // `glyph_run_baseline_offset` there).
        let a = glyphs.get(&65).expect("'A' glyph present");
        assert_eq!(a.width, 15);
        assert_eq!(a.height, 17);
        assert_eq!(a.top, -9);
        assert_eq!(
            a.bitmap.len() as u32,
            (a.width + 2 * GLYPH_BUFFER) * (a.height + 2 * GLYPH_BUFFER)
        );
    }
}
