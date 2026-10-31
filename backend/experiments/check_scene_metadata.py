from pathlib import Path
import rasterio

files = [
    Path("datasets/live/rgi2000_v7_0_g_15_00467_debris_spectral.tif"),
    Path("datasets/live/rgi2000_v7_0_g_15_00467_debris_rgb.tif"),
    Path("datasets/historical/rgi2000-v7.0-g-15-00467/historical_results.json"),
]

for path in files:
    print(f"\n--- {path} ---")
    print("Exists:", path.exists())

    if not path.exists() or path.suffix.lower() != ".tif":
        continue

    with rasterio.open(path) as ds:
        print("Shape:", ds.height, "x", ds.width)
        print("CRS:", ds.crs)
        print("Dataset tags:", ds.tags())
        for i in range(1, ds.count + 1):
            print(f"Band {i} tags:", ds.tags(i))
