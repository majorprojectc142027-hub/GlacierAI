
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = (
    ROOT
    / "datasets/historical/rgi2000-v7.0-g-15-00467"
    / "historical_results.json"
)

with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

observations = sorted(
    data["observations"],
    key=lambda item: item["year"],
)

masks = {}

for obs in observations:
    year = obs["year"]
    path = Path(obs["mask_path"])

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        mask = src.read(1) > 0
        masks[year] = mask

        print(
            f"{year}: pixels={int(mask.sum())}, "
            f"area={obs['area_km2']:.6f} km², "
            f"shape={mask.shape}"
        )

print("\nActual historical mask comparison")

years = sorted(masks)

for i, year_a in enumerate(years):
    for year_b in years[i + 1:]:
        a = masks[year_a]
        b = masks[year_b]

        if a.shape != b.shape:
            print(f"{year_a} vs {year_b}: DIFFERENT SHAPES")
            continue

        intersection = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())

        iou = intersection / union if union else 0.0
        denominator = int(a.sum() + b.sum())
        dice = (
            2 * intersection / denominator
            if denominator else 0.0
        )

        print(
            f"{year_a} vs {year_b}: "
            f"IoU={iou:.4f}, Dice={dice:.4f}"
        )
