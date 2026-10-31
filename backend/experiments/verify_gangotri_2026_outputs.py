
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467"
LIVE = ROOT / "datasets/live/predictions"

with open(BASE / "historical_results.json", encoding="utf-8") as f:
    data = json.load(f)

obs = next(o for o in data["observations"] if int(o["year"]) == 2026)

historical_mask_path = Path(obs["mask_path"])
if not historical_mask_path.is_absolute():
    historical_mask_path = ROOT / historical_mask_path

historical_image_path = Path(obs["satellite_path"])
if not historical_image_path.is_absolute():
    historical_image_path = ROOT / historical_image_path

live_mask_path = LIVE / "rgi2000-v7.0-g-15-00467_2026_glacier_mask.tif"
probability_path = LIVE / "rgi2000-v7.0-g-15-00467_2026_probability.npy"

for path in (
    historical_mask_path,
    historical_image_path,
    live_mask_path,
    probability_path,
):
    if not path.exists():
        raise FileNotFoundError(path)

with rasterio.open(historical_mask_path) as src:
    historical_mask = src.read(1) > 0
    hist_grid = (src.crs, src.transform, src.width, src.height)

with rasterio.open(live_mask_path) as src:
    live_mask = src.read(1) > 0
    live_grid = (src.crs, src.transform, src.width, src.height)

probability = np.load(probability_path)

print("\n--- GANGOTRI 2026 OUTPUT VERIFICATION ---")
print("Historical image:", historical_image_path)
print("Historical mask:", historical_mask_path)
print("Live mask:", live_mask_path)
print("Probability file:", probability_path)

print("\nHistorical grid:", hist_grid)
print("Live grid:", live_grid)
print("Probability shape:", probability.shape)

print("\nHistorical mask pixels:", int(historical_mask.sum()))
print("Live mask pixels:", int(live_mask.sum()))

if hist_grid == live_grid and historical_mask.shape == live_mask.shape:
    intersection = np.logical_and(historical_mask, live_mask).sum()
    union = np.logical_or(historical_mask, live_mask).sum()
    print("Historical/live mask IoU:", round(intersection / union, 5) if union else 1.0)
else:
    print("Historical/live masks use different grids; direct IoU skipped.")

if probability.shape == live_mask.shape:
    print("Probability/live mask shape matches: YES")
    print("Probability range:", float(np.nanmin(probability)), float(np.nanmax(probability)))
    print("Pixels >= 0.20:", int((probability >= 0.20).sum()))
    print("Pixels >= 0.50:", int((probability >= 0.50).sum()))
else:
    print("Probability/live mask shape matches: NO")

print("\nNo files were modified.")
