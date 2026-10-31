import json
from pathlib import Path

import rasterio

root = Path("datasets/live")

metadata_path = root / "scene_metadata_diagnostic.json"
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print("=== Diagnostic request metadata ===")
    print("Catalog scene:", metadata.get("catalog_scene_id"))
    print("Catalog datetime:", metadata.get("catalog_datetime"))

    for kind in ("rgb", "spectral"):
        tiles = metadata.get(kind, {}).get("tiles", [])
        print(f"\n{kind.upper()} source tiles: {len(tiles)}")
        for tile in tiles:
            print(
                "  date:", tile.get("date"),
                "| shId:", tile.get("shId"),
                "| product:", tile.get("sentinel2ProductId")
                or tile.get("productId"),
            )
else:
    print("Diagnostic metadata file not found:", metadata_path)

files = [
    root / "rgi2000_v7_0_g_15_00467_debris_rgb.tif",
    root / "rgi2000_v7_0_g_15_00467_debris_spectral.tif",
]

print("\n=== Saved raster metadata ===")
for path in files:
    print(f"\n{path}")
    if not path.exists():
        print("  FILE NOT FOUND")
        continue

    with rasterio.open(path) as ds:
        print("  Shape:", ds.height, "x", ds.width)
        print("  CRS:", ds.crs)
        print("  Transform:", ds.transform)
        print("  Bounds:", ds.bounds)
        print("  Band descriptions:", ds.descriptions)
        print("  TIFF tags:", ds.tags())
