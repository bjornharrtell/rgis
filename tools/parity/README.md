# Parity golden generator

Generates the reference screenshots consumed by rgis's visual-parity Rust
test (`crates/rgis-render/tests/parity.rs`), by loading the real
maplibre-gl-js library against the same style (`liberty`, served from
`https://tiles.openfreemap.org/styles/liberty`) and viewports rgis's own
renderer is tested against.

This is a one-off tool, **not run in CI**: it fetches live tiles over the
network and its output (the PNGs under
`../../crates/rgis-render/tests/parity/goldens/`) is committed to the repo
as a fixture. Re-run it only when you deliberately want to refresh the
goldens (e.g. after the upstream style or tile data changes, or after
changing what the parity test compares).

## Usage

```sh
npm install
npx playwright install chromium   # first time only
npm run generate
```

Viewports (center/zoom/size) are defined once in `viewports.json` and must
be kept in sync with the `VIEWPORTS` constant in
`../../crates/rgis-render/tests/parity.rs` (both read from/mirror the same
values so the two renderers are compared at literally the same camera).

## Scope

Symbol layers (text labels and icons) are hidden on both the maplibre-gl-js
side (here) and the rgis side (in the Rust test) before comparing. Label
placement and font rendering differ meaningfully between a browser's text
shaper and rgis's SDF glyph pipeline regardless of whether the underlying
MapLibre style *evaluation* is correct, so including them would produce
noisy diffs unrelated to real style-parity bugs. This keeps the comparison
focused on fill / line / background / fill-extrusion / raster rendering,
which is where actual parity bugs (wrong colors, opacities, filters, zoom
interpolation, etc.) would show up.
