
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467"
LIVE = ROOT / "datasets/live/predictions"

PROB_PATH = LIVE / "rgi2000-v7.0-g-15-00467_2026_probability.npy"
MASK_PATH = (
    BASE / "predictions/rgi2000-v7.0-g-15-00467_2026_rgi_mask.tif"
)

if not PROB_PATH.exists() or not MASK_PATH.exists():
    raise FileNotFoundError("Required probability or historical mask is missing.")

prob = np.load(PROB_PATH)

with rasterio.open(MASK_PATH) as src:
    saved_mask = src.read(1) > 0
    transform = src.transform
    grid = (src.crs, src.transform, src.width, src.height)

if prob.shape != saved_mask.shape:
    raise ValueError("Probability map and saved mask dimensions differ.")

# The existing final mask is restricted to the RGI inventory footprint.
# For this experiment, hold that spatial constraint constant.
with rasterio.open(MASK_PATH) as src:
    historical_image_shape = (src.height, src.width)

# Read the inventory geometry on the same raster grid.
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import mapping

RGI_PATH = ROOT / "datasets/processed/global_rgi.gpkg"
gdf = gpd.read_file(RGI_PATH)

id_column = next(
    (c for c in gdf.columns if c.lower() in {"rgiid", "rgi_id", "rgi_id_v7"}),
    None,
)
if id_column is None:
    raise ValueError(f"RGI ID column not found: {list(gdf.columns)}")

selected = gdf[
    gdf[id_column].astype(str) == "RGI2000-v7.0-G-15-00467"
]
if selected.empty or gdf.crs is None:
    raise ValueError("RGI feature not found or inventory CRS is missing.")

with rasterio.open(MASK_PATH) as src:
    rgi_geometry = selected.to_crs(src.crs).geometry.union_all()
    rgi_mask = rasterize(
        [(mapping(rgi_geometry), 1)],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1_000_000

print("\n2026 THRESHOLD SENSITIVITY")
print("This is an offline experiment, not accuracy validation.")
print(f"{'Threshold':>12}{'Pixels':>14}{'Area km²':>14}{'IoU vs saved':>16}")

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    candidate = (prob >= threshold) & rgi_mask

    # Simple threshold comparison only: deliberately do not apply
    # morphology or component filtering.
    pixels = int(candidate.sum())
    area = pixels * pixel_area_km2

    intersection = int(np.logical_and(candidate, saved_mask).sum())
    union = int(np.logical_or(candidate, saved_mask).sum())
    iou = intersection / union if union else 1.0

    print(f"{threshold:>12.2f}{pixels:>14,}{area:>14.3f}{iou:>16.4f}")

print("\nSaved historical mask pixels:", int(saved_mask.sum()))
print("No production files were changed.")
