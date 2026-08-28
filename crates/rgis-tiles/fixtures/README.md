# Glyph fixtures

Real, raw `.pbf` glyph-range payloads fetched once from OpenFreeMap's
public glyph server, used by `src/glyphs.rs`'s
`decodes_real_glyph_range_fixture` test to validate the hand-rolled PBF
decoder against actual wire data (schema per
[`glyphs.proto`](https://github.com/mapbox/glyph-pbf-composite)), rather
than only synthetic bytes.

Re-fetch with, e.g.:

```sh
curl -s "https://tiles.openfreemap.org/fonts/Noto%20Sans%20Regular/0-255.pbf" \
  -o noto_sans_regular_0-255.pbf
```

| file | fontstack | codepoint range |
|---|---|---|
| noto_sans_regular_0-255.pbf | Noto Sans Regular | 0-255 (basic Latin + Latin-1 supplement) |
