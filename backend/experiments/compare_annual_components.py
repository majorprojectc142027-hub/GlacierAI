
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = (
    ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467"
    / "historical_results.json"
)

with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

print("\nANNUAL HISTORICAL MASK COMPARISON")
print("-" * 95)
print(
    f"{'Year':<8}{'Pixels':>12}{'Area km²':>14}"
    f"{'Components':>14}{'Largest component':>22}{'IoU vs 2023':>15}"
)

masks = {}
reference_grid = None

for obs in sorted(data["observations"], key=lambda x: x["year"]):
    year = int(obs["year"])

    path = Path(obs["mask_path"])
    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        print(f"{year:<8} MISSING MASK: {path}")
        continue

    with rasterio.open(path) as src:
        mask = src.read(1) > 0
        grid = (
            src.crs,
            src.transform,
            src.width,
            src.height,
        )
        pixel_area = abs(src.transform.a * src.transform.e)

    if reference_grid is None:
        reference_grid = grid
    elif grid != reference_grid:
        print(f"{year}: SKIPPED — raster grid differs from other years.")
        continue

    masks[year] = mask

    _, count = ndimage.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(
        ndimage.label(
            mask, structure=np.ones((3, 3), dtype=np.uint8)
        )[0].ravel()
    )[1:]

    largest = int(sizes.max()) if sizes.size else 0
    area = mask.sum() * pixel_area / 1_000_000

    print(
        f"{year:<8}{int(mask.sum()):>12,}{area:>14.3f}"
        f"{count:>14}{largest:>22,}"
        f"{'—':>15}"
    )

if 2023 in masks:
    print("\nIoU against 2023:")
    for year, mask in sorted(masks.items()):
        if year == 2023:
            continue
        intersection = np.logical_and(masks[2023], mask).sum()
        union = np.logical_or(masks[2023], mask).sum()
        iou = intersection / union if union else 1.0
        print(f"2023 vs {year}: IoU = {iou:.4f}")
