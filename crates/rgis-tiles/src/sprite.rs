//! Sprite atlas fetching for a style's `"sprite"` document (e.g.
//! OpenFreeMap liberty's `ofm` sprite), used to render real `icon-image`
//! sprites instead of a placeholder marker dot.
//!
//! A MapLibre/Mapbox sprite is published as a pair of sibling URLs: a JSON
//! index (`{base}.json`, sprite name -> pixel rect within the atlas image)
//! and the packed atlas image itself (`{base}.png`). Only the 1x
//! (`pixelRatio: 1`) variant is fetched here -- real style-spec clients also
//! fetch `{base}@2x.png`/`.json` on high-DPI displays, but a single shared
//! atlas resolution is an acceptable simplification given this renderer's
//! agreed loose visual-parity tolerance.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_channel::{Receiver, Sender};
use serde::Deserialize;

/// One sprite's pixel rect within the atlas image, as published in the
/// sprite JSON index.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct SpriteRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// The fully fetched and decoded sprite atlas: the packed RGBA image and
/// every sprite's rect within it, keyed by sprite name (e.g.
/// `"circle_11_black"`, matching `icon-image` values in the style).
pub struct SpriteAtlas {
    pub image: Arc<image::RgbaImage>,
    pub rects: HashMap<String, SpriteRect>,
}

pub struct SpriteAtlasReady {
    pub atlas: Arc<SpriteAtlas>,
}

/// Fetches a style's sprite atlas once (both the JSON index and the PNG
/// image), delivering the combined result on `receiver` when both parts
/// have arrived. Failures (network error, bad JSON/PNG) are silent --
/// icons then simply never appear, same as any other "ignore failures"
/// tile/glyph fetcher in this crate.
pub struct SpriteFetcher {
    sender: Sender<SpriteAtlasReady>,
    pub receiver: Receiver<SpriteAtlasReady>,
    // Holds whichever of {rects, image} arrives first, waiting for its
    // sibling before combining them into a `SpriteAtlas`.
    partial: Arc<Mutex<Partial>>,
}

#[derive(Default)]
struct Partial {
    rects: Option<HashMap<String, SpriteRect>>,
    image: Option<image::RgbaImage>,
}

impl SpriteFetcher {
    /// `sprite_base_url` is a style document's `"sprite"` field verbatim
    /// (no `.json`/`.png` suffix -- that's appended here).
    pub fn new(sprite_base_url: &str) -> Arc<Self> {
        let (sender, receiver) = async_channel::bounded(1);
        let fetcher = Arc::new(Self {
            sender,
            receiver,
            partial: Arc::new(Mutex::new(Partial::default())),
        });

        let json_fetcher = Arc::clone(&fetcher);
        ehttp::fetch(
            ehttp::Request::get(format!("{sprite_base_url}.json")),
            move |result: ehttp::Result<ehttp::Response>| {
                let Ok(response) = result else { return };
                if !response.ok {
                    return;
                }
                let Ok(rects) =
                    serde_json::from_slice::<HashMap<String, SpriteRect>>(&response.bytes)
                else {
                    return;
                };
                json_fetcher.on_rects(rects);
            },
        );

        let png_fetcher = Arc::clone(&fetcher);
        ehttp::fetch(
            ehttp::Request::get(format!("{sprite_base_url}.png")),
            move |result: ehttp::Result<ehttp::Response>| {
                let Ok(response) = result else { return };
                if !response.ok {
                    return;
                }
                let Ok(decoded) = image::load_from_memory(&response.bytes) else {
                    return;
                };
                png_fetcher.on_image(decoded.to_rgba8());
            },
        );

        fetcher
    }

    fn on_rects(&self, rects: HashMap<String, SpriteRect>) {
        let image = {
            let mut partial = self.partial.lock().unwrap();
            partial.rects = Some(rects);
            partial.image.take()
        };
        if let Some(image) = image {
            self.deliver(self.partial.lock().unwrap().rects.clone().unwrap(), image);
        }
    }

    fn on_image(&self, image: image::RgbaImage) {
        let rects = {
            let mut partial = self.partial.lock().unwrap();
            partial.image = Some(image.clone());
            partial.rects.clone()
        };
        if let Some(rects) = rects {
            self.deliver(rects, image);
        }
    }

    fn deliver(&self, rects: HashMap<String, SpriteRect>, image: image::RgbaImage) {
        let atlas = Arc::new(SpriteAtlas {
            image: Arc::new(image),
            rects,
        });
        let _ = self.sender.try_send(SpriteAtlasReady { atlas });
    }
}
