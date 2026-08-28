//! wasm-only Web Worker pool that offloads MVT decode + tessellation
//! (`rgis_tiles::decode_vector_tile` + `rgis_render::build_tile_mesh`) to a
//! small pool of dedicated Web Workers, so it no longer runs on the main
//! thread (where `Promise::spawn_local` has no real background thread to
//! run on -- see `RgisApp::drain_ready_tiles`).
//!
//! ## Worker asset
//! The worker's own code lives in a separate binary, `rgis-web`'s
//! `src/bin/tile_worker.rs`, built by Trunk as a `data-type="worker"` asset
//! (see `rgis-web/index.html`) and served as plain (non-hashed, per Trunk's
//! own handling of worker assets) `tile_worker.js` / `tile_worker_bg.wasm`
//! files at the site root -- `worker_new` below references those names
//! directly.
//!
//! ## Message protocol
//! Main -> worker: a `js_sys::Array` `[z, x, y, bytes: Uint8Array]` (the
//! tile coord and raw MVT bytes, transferred).
//! Worker -> main:
//! - `[]` (empty array): the worker has finished loading its wasm module
//!   and is ready to receive a job (sent once, right after startup --
//!   matches Trunk's own webworker example's convention, since messages
//!   sent to a worker before this point are silently dropped).
//! - `[z, x, y]` (length 3): MVT decode failed for that tile.
//! - `[z, x, y, fill_vertices, fill_indices, line_vertices, line_indices]`
//!   (length 7, all four as typed arrays, transferred) or
//!   `[z, x, y, fill_vertices, fill_indices, line_vertices, line_indices,
//!   labels_json]` (length 8, `labels_json` a plain JSON string of
//!   `Vec<rgis_render::TileLabel>`): the tessellated
//!   [`TileMeshWire`]. Older/newer workers omitting the 8th element just
//!   get no labels for that tile.
//!
//! Each worker only ever has one job in flight at a time (enforced by this
//! pool), so a worker's own identity is enough to route its reply back to
//! the right caller -- no request IDs are needed.
//!
//! Note: this only covers tiles arriving as raw network bytes (the
//! dominant, network-driven zoom/pan case). Tiles that already have a
//! decoded [`VectorTile`](rgis_tiles::VectorTile) available (an in-memory
//! decoded-tile cache hit, e.g. from `VectorTileFetcher`) still tessellate
//! inline via `Promise::spawn_local`, since there's no raw bytes left to
//! ship to a worker at that point -- this path is rare enough (only hit
//! when the final mesh cache evicted a tile that the fetcher's own,
//! separate decode cache still has) not to be worth complicating the
//! message protocol for.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use js_sys::{Array, Float32Array, Uint8Array, Uint32Array};
use rgis_render::{TileMesh, TileMeshWire};
use rgis_tiles::TileCoord;
use wasm_bindgen::{JsCast, prelude::Closure};
use web_sys::{Blob, BlobPropertyBag, MessageEvent, Url, Worker};

/// Number of dedicated workers kept warm for tile decode+tessellation.
const POOL_SIZE: usize = 4;

/// Max jobs allowed to sit in `PoolState::queue` awaiting a free worker.
/// Without this, a sustained burst of tile fetches outpacing the 4 workers'
/// tessellation throughput (e.g. panning/zooming quickly across many
/// detailed tiles for a while) queues an ever-growing backlog of raw MVT
/// byte buffers, eventually exhausting the wasm heap. When the queue is
/// full, `tessellate` gives up on the newest request immediately (treated
/// the same as a decode failure by callers) rather than growing it further
/// -- the tile just gets re-requested later if it's still visible.
const MAX_QUEUE_LEN: usize = POOL_SIZE * 4;

type Job = (TileCoord, Vec<u8>);
type ReplyTx = async_channel::Sender<Option<TileMesh>>;

struct PoolState {
    workers: Vec<Worker>,
    /// `Some` while a worker has a job in flight; completed by that
    /// worker's own `onmessage` handler.
    inflight: Vec<Option<ReplyTx>>,
    /// Indices of workers that have finished loading and have no job in
    /// flight, ready to be handed the next queued job.
    free: VecDeque<usize>,
    queue: VecDeque<(Job, ReplyTx)>,
}

/// Owns a small pool of dedicated Web Workers used to decode + tessellate
/// vector tiles off the main thread. See the module docs for the message
/// protocol.
pub struct TileWorkerPool {
    state: Rc<RefCell<PoolState>>,
    // Keeps every worker's `onmessage` closure alive for the pool's
    // lifetime (an `RgisApp` field, effectively the whole app's lifetime).
    _onmessages: Vec<Closure<dyn FnMut(MessageEvent)>>,
}

impl TileWorkerPool {
    pub fn new() -> Self {
        let state = Rc::new(RefCell::new(PoolState {
            workers: Vec::with_capacity(POOL_SIZE),
            inflight: Vec::with_capacity(POOL_SIZE),
            free: VecDeque::new(),
            queue: VecDeque::new(),
        }));
        let mut onmessages = Vec::with_capacity(POOL_SIZE);

        for worker_idx in 0..POOL_SIZE {
            let worker = worker_new("tile_worker");
            let state_clone = Rc::clone(&state);
            let onmessage = Closure::wrap(Box::new(move |msg: MessageEvent| {
                handle_message(&state_clone, worker_idx, msg);
            }) as Box<dyn FnMut(MessageEvent)>);
            worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

            let mut s = state.borrow_mut();
            s.workers.push(worker);
            s.inflight.push(None);
            drop(s);

            onmessages.push(onmessage);
        }

        Self {
            state,
            _onmessages: onmessages,
        }
    }

    /// Decodes and tessellates `bytes` (raw MVT for `coord`) on a pooled
    /// worker, returning `None` if decoding failed (or if the pool's queue
    /// is already saturated, see [`MAX_QUEUE_LEN`]).
    pub async fn tessellate(&self, coord: TileCoord, bytes: Vec<u8>) -> Option<TileMesh> {
        let (tx, rx) = async_channel::bounded(1);
        {
            let mut s = self.state.borrow_mut();
            if s.queue.len() >= MAX_QUEUE_LEN {
                return None;
            }
            s.queue.push_back(((coord, bytes), tx));
        }
        pump(&self.state);
        rx.recv().await.ok().flatten()
    }
}

/// Constructs (and starts) a dedicated Worker running the `tile_worker`
/// binary, loaded via a small inline `importScripts`+`wasm_bindgen(...)`
/// bootstrap script -- matches the classic (`no-modules`) Trunk worker
/// example, needing no `data-loader-shim`/`data-bindgen-target` attributes.
///
/// Worker filenames are resolved relative to the current page's own URL
/// (not just `location.origin`), since Trunk serves assets under a
/// configurable base path (`public_url` in `Trunk.toml`, e.g. `/rgis/` for
/// GitHub Pages) rather than always at the site root. A worker spawned from
/// a `Blob:` URL has its own `location` pointing at that blob, not the
/// page, so the resolution must happen using the page's href captured here
/// on the main thread, passed into the bootstrap script as an explicit
/// `new URL(..., base)` call rather than relying on the worker's own
/// (unrelated) `location`.
fn worker_new(name: &str) -> Worker {
    let page_href = web_sys::window()
        .expect("window to be available")
        .location()
        .href()
        .expect("href to be available");

    let script = Array::new();
    script.push(
        &format!(
            r#"importScripts(new URL("{name}.js", "{page_href}").href);wasm_bindgen(new URL("{name}_bg.wasm", "{page_href}").href);"#
        )
        .into(),
    );
    let blob_options = BlobPropertyBag::new();
    blob_options.set_type("text/javascript");
    let blob = Blob::new_with_str_sequence_and_options(&script, &blob_options)
        .expect("blob creation succeeds");
    let url = Url::create_object_url_with_blob(&blob).expect("url creation succeeds");

    Worker::new(&url).expect("failed to spawn tile worker")
}

fn handle_message(state: &Rc<RefCell<PoolState>>, worker_idx: usize, msg: MessageEvent) {
    let data = Array::from(&msg.data());
    let len = data.length();

    if len >= 3 {
        let mesh = if len >= 7 {
            let labels_json = if len >= 8 {
                data.get(7).as_string().unwrap_or_default()
            } else {
                String::new()
            };
            let wire = TileMeshWire {
                fill_vertices: Float32Array::from(data.get(3)).to_vec(),
                fill_indices: Uint32Array::from(data.get(4)).to_vec(),
                line_vertices: Float32Array::from(data.get(5)).to_vec(),
                line_indices: Uint32Array::from(data.get(6)).to_vec(),
                labels_json,
            };
            Some(wire.into_tile_mesh())
        } else {
            None
        };
        let mut s = state.borrow_mut();
        if let Some(tx) = s.inflight[worker_idx].take() {
            let _ = tx.try_send(mesh);
        }
        s.free.push_back(worker_idx);
    } else {
        // Zero-length "ready" ping.
        state.borrow_mut().free.push_back(worker_idx);
    }
    pump(state);
}

/// Dispatches as many queued jobs as there are free workers.
fn pump(state: &Rc<RefCell<PoolState>>) {
    loop {
        let mut s = state.borrow_mut();
        let Some(worker_idx) = s.free.pop_front() else {
            break;
        };
        let Some(((coord, bytes), reply_tx)) = s.queue.pop_front() else {
            s.free.push_front(worker_idx);
            break;
        };
        s.inflight[worker_idx] = Some(reply_tx);

        let msg = Array::new();
        msg.push(&(coord.z as f64).into());
        msg.push(&(coord.x as f64).into());
        msg.push(&(coord.y as f64).into());
        let bytes_array = Uint8Array::from(bytes.as_slice());
        let transfer = Array::new();
        transfer.push(&bytes_array.buffer());
        msg.push(&bytes_array);

        let _ = s.workers[worker_idx].post_message_with_transfer(&msg.into(), &transfer.into());
    }
}
