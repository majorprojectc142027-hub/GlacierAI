from pathlib import Path
import rasterio

root = Path("datasets/historical/rgi2000-v7.0-g-15-00467")

for year in (2025, 2026):
    path = root / "satellite" / (
        f"rgi2000-v7.0-g-15-00467_{year}.tif"
    )

    print(f"\n--- {year} historical RGB ---")
    print("Path:", path)
    print("Exists:", path.exists())

    if not path.exists():
        continue

    with rasterio.open(path) as ds:
        print("Shape:", ds.height, "x", ds.width)
        print("Bands:", ds.count)
        print("CRS:", ds.crs)
        print("Bounds:", ds.bounds)
        print("Dtype:", ds.dtypes)
        print("Descriptions:", ds.descriptions)
        print("Tags:", ds.tags())

        for i in range(1, ds.count + 1):
            band = ds.read(i)
            print(
                f"Band {i}: min={band.min()}, "
                f"p50={__import__('numpy').median(band)}, "
                f"max={band.max()}"
            )
