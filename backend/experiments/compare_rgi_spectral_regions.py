
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"

gid = "rgi2000_v7_0_g_15_00467"

spectral_path = LIVE / f"{gid}_debris_spectral.tif"
glacier_path = PRED / f"{gid}_debris_glacier_mask.tif"
reference_path = PRED / f"{gid}_rgi_reference_mask.tif"

with rasterio.open(spectral_path) as src:
    bands = src.read().astype(np.float32)
    descriptions = [
        src.descriptions[i] or f"Band {i + 1}"
        for i in range(src.count)
    ]
    spectral_profile = (src.crs, src.transform, src.shape)

def read_mask(path):
    with rasterio.open(path) as src:
        if (src.crs, src.transform, src.shape) != spectral_profile:
            raise ValueError(f"Grid mismatch: {path}")
        return src.read(1) > 0

pred = read_mask(glacier_path)
ref = read_mask(reference_path)

regions = {
    "Overlap (green)": pred & ref,
    "RGI only (blue)": ref & ~pred,
    "U-Net only (red)": pred & ~ref,
}

print("Spectral bands:", descriptions)
print("Pixel values are stored as in the input raster.")
print("Check scale/reflectance metadata before interpreting thresholds.")

for label, mask in regions.items():
    print(f"\n--- {label}: {int(mask.sum())} pixels ---")
    if not mask.any():
        print("No pixels.")
        continue

    for i, name in enumerate(descriptions):
        values = bands[i][mask]
        values = values[np.isfinite(values)]

        if values.size:
            print(
                f"{name}: median={np.median(values):.4f}, "
                f"p25={np.percentile(values, 25):.4f}, "
                f"p75={np.percentile(values, 75):.4f}"
            )
