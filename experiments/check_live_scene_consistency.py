
from pathlib import Path
import rasterio
import numpy as np

FILES = {
    "October RGB": Path(
        "datasets/live/"
        "rgi2000_v7_0_g_15_00467_debris_rgb.tif"
    ),
    "October spectral": Path(
        "datasets/live/"
        "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
    ),
    "September RGB": Path(
        "datasets/historical/satellite/"
        "gangotri_2025-09-22.tif"
    ),
}

for name, path in FILES.items():
    print(f"\n{'=' * 55}")
    print(name)
    print("File:", path)

    if not path.exists():
        print("NOT FOUND")
        continue

    with rasterio.open(path) as ds:
        print("Shape:", (ds.height, ds.width))
        print("Bands:", ds.count)
        print("CRS:", ds.crs)
        print("Transform:", ds.transform)
        print("Bounds:", ds.bounds)
        print("Descriptions:", ds.descriptions)
        print("Scales:", ds.scales)
        print("Offsets:", ds.offsets)
        print("Tags:", ds.tags())

        for i in range(1, ds.count + 1):
            band = ds.read(i)
            valid = np.isfinite(band)
            values = band[valid]

            if values.size == 0:
                print(f"Band {i}: no finite values")
                continue

            print(
                f"Band {i}: dtype={band.dtype}, "
                f"min={values.min()}, "
                f"p50={np.percentile(values, 50):.2f}, "
                f"p95={np.percentile(values, 95):.2f}, "
                f"max={values.max()}, "
                f"zero_fraction={np.mean(values == 0):.4f}, "
                f"max_fraction={np.mean(values == values.max()):.4f}"
            )
