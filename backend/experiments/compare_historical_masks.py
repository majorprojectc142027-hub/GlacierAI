
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

years = [2023, 2024, 2025, 2026]
masks = {}

for year in years:
    path = (
        ROOT
        / "datasets"
        / "historical"
        / "predictions"
        / f"gangotri_{year}_rgi_mask.tif"
    )

    with rasterio.open(path) as src:
        masks[year] = src.read(1) > 0

        print(
            f"{year}: {int(masks[year].sum())} positive pixels"
        )

print("\nPairwise historical mask comparison")

for i, year_a in enumerate(years):
    for year_b in years[i + 1:]:
        a = masks[year_a]
        b = masks[year_b]

        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()

        iou = intersection / union if union else 0.0
        dice = (
            2 * intersection / (a.sum() + b.sum())
            if (a.sum() + b.sum()) else 0.0
        )

        print(
            f"{year_a} vs {year_b}: "
            f"IoU={iou:.4f}, Dice={dice:.4f}, "
            f"intersection={intersection}, union={union}"
        )
