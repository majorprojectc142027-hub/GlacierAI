import rasterio
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

files = [
    "datasets/historical/predictions/gangotri_2023_rgi_mask.tif",
    "datasets/historical/predictions/gangotri_2024_rgi_mask.tif",
    "datasets/historical/predictions/gangotri_2025_rgi_mask.tif",
    "datasets/historical/predictions/gangotri_2026_rgi_mask.tif",
    "datasets/live/predictions/gangotri_rgi_glacier_mask.tif",
    "datasets/live/predictions/rgi2000_v7_0_g_15_00467_rgi_reference_mask.tif",
]

for relative_path in files:
    path = ROOT / relative_path

    if not path.exists():
        print(f"\nNOT FOUND: {relative_path}")
        continue

    with rasterio.open(path) as src:
        mask = src.read(1)

        print(f"\nFile: {relative_path}")
        print(f"Shape: {mask.shape}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        print(f"Pixels > 0: {np.count_nonzero(mask)}")
        print(f"Unique values: {np.unique(mask)[:20]}")
