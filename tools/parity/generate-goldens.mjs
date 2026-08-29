// One-off generator for maplibre-gl-js golden screenshots, used by rgis's
// visual-parity Rust test (`crates/rgis-render/tests/parity.rs`) to check
// that rgis's own MapLibre-style renderer produces pixels that loosely
// match the reference implementation for the OpenFreeMap "liberty" style.
//
// This is intentionally NOT wired into CI: it fetches live tiles from
// https://tiles.openfreemap.org and is meant to be re-run by hand only when
// the goldens need regenerating (e.g. after bumping the pinned style/tile
// data, or after deliberately changing what's being compared). Run with:
//
//   cd tools/parity && npm install && npm run generate
//
// Scope note: symbol layers (text labels + icons) are hidden on both sides
// of the comparison (see rgis-render's parity test for the Rust-side
// equivalent). Label placement/font rendering differs meaningfully between
// a browser's text shaper and rgis's SDF glyph pipeline regardless of
// whether the underlying style evaluation is correct, so comparing them
// pixel-for-pixel would produce noisy failures unrelated to actual style
// parity. This keeps the comparison focused on fill/line/background/
// fill-extrusion/raster rendering, which is where real parity bugs
// (wrong colors, opacities, filters, zoom interpolation, etc.) would show
// up.
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import { dirname, join, extname } from 'node:path';
import { readFileSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { createServer } from 'node:http';

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const STYLE_URL = 'https://tiles.openfreemap.org/styles/liberty';
const GOLDENS_DIR = join(__dirname, '..', '..', 'crates', 'rgis-render', 'tests', 'parity', 'goldens');
const viewports = JSON.parse(readFileSync(join(__dirname, 'viewports.json'), 'utf8'));

const maplibreDir = dirname(require.resolve('maplibre-gl/dist/maplibre-gl.js'));

const html = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="/maplibre-gl.css">
<style>html,body,#map{margin:0;padding:0;width:100%;height:100%;}</style>
</head>
<body>
<div id="map"></div>
<script src="/maplibre-gl.js"></script>
</body>
</html>`;

const MIME = { '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html' };

// A real browser navigation (not `page.setContent`, which uses an opaque
// `about:blank`-ish origin that Chromium refuses to load local `file://`
// scripts from) is needed so maplibre-gl's UMD bundle actually executes --
// so a tiny static file server stands in for a real origin here.
const server = createServer((req, res) => {
  if (req.url === '/' || req.url === '/index.html') {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end(html);
    return;
  }
  const filePath = join(maplibreDir, req.url);
  try {
    const body = readFileSync(filePath);
    res.writeHead(200, { 'content-type': MIME[extname(filePath)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const { port } = server.address();

mkdirSync(GOLDENS_DIR, { recursive: true });

const browser = await chromium.launch();
try {
  for (const vp of viewports) {
    console.log(`Generating golden for viewport "${vp.name}"...`);
    const page = await browser.newPage({ viewport: { width: vp.width, height: vp.height } });
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded' });
    await page.evaluate(
      ({ styleUrl, center, zoom }) => {
        window.__map = new maplibregl.Map({
          container: 'map',
          style: styleUrl,
          center,
          zoom,
          bearing: 0,
          pitch: 0,
          attributionControl: false,
          fadeDuration: 0,
        });
      },
      { styleUrl: STYLE_URL, center: [vp.lon, vp.lat], zoom: vp.zoom },
    );
    // Wait for the style to load, then hide every symbol (text/icon) layer
    // -- see the scope note above -- before waiting for tiles to settle.
    await page.evaluate(() => new Promise((resolve) => {
      const map = window.__map;
      map.once('load', () => {
        for (const layer of map.getStyle().layers) {
          if (layer.type === 'symbol') {
            map.setLayoutProperty(layer.id, 'visibility', 'none');
          }
        }
        resolve();
      });
    }));
    await page.evaluate(() => new Promise((resolve) => {
      const map = window.__map;
      if (map.isStyleLoaded() && !map.isMoving()) {
        // `idle` may already have fired; give layout one more tick then resolve.
        map.once('idle', resolve);
        setTimeout(resolve, 4000);
      } else {
        map.once('idle', resolve);
      }
    }));
    const outPath = join(GOLDENS_DIR, `${vp.name}.png`);
    await page.locator('#map canvas').screenshot({ path: outPath });
    console.log(`  wrote ${outPath}`);
    await page.close();
  }
} finally {
  await browser.close();
  server.close();
}
