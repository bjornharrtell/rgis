# Tile fixtures

Real, raw `.pbf` (Mapbox Vector Tile) payloads fetched once from
OpenFreeMap's public tile service, used by
`src/basemap.rs`'s `tile_mesh_byte_budget` test to measure tessellation
output deterministically and offline, without a browser stress test.

Chosen as dense-urban z12/z14 tiles (OpenFreeMap's basemap tops out at
z14 client-side, same as most OpenMapTiles-schema sources; higher zooms
are achieved by over-zooming the z14 tile) since these were the tiles
that triggered multi-GB memory usage during the OOM investigation.

Re-fetch (data version segment changes over time) with, e.g.:

```sh
BASE=$(curl -s https://tiles.openfreemap.org/planet | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["tiles"][0].rsplit("/",3)[0])')
curl -s "$BASE/14/8299/5636.pbf" -o paris_14.pbf
```

| file | z/x/y | city |
|---|---|---|
| paris_12.pbf | 12/2074/1409 | Paris |
| paris_14.pbf | 14/8299/5636 | Paris |
| london_14.pbf | 14/8186/5448 | London |
| nyc_14.pbf | 14/4823/6160 | New York City |
| tokyo_14.pbf | 14/14549/6451 | Tokyo |
