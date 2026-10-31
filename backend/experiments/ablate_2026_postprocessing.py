
from pathlib import Path

import cv2
import numpy as np
import rasterio
from scipy import ndimage
from rasterio.features import rasterize
from shapely.geometry import mapping
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467"
LIVE = ROOT / "datasets/live/predictions"
RGI_PATH = ROOT / "datasets/processed/global_rgi.gpkg"

PROB_PATH = LIVE / "rgi2000-v7.0-g-15-00467_2026_probability.npy"
HIST_MASK_PATH = (
    BASE / "predictions/rgi2000-v7.0-g-15-00467_2026_rgi_mask.tif"
)

THRESHOLD = 0.20
CORE_THRESHOLD = 0.50
MIN_COMPONENT_PIXELS = 4
MIN_COMPONENT_MEAN_PROBABILITY = 0.30
CLOSE_SIZE = 5
OPEN_SIZE = 3

for path in (PROB_PATH, HIST_MASK_PATH, RGI_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

prob = np.load(PROB_PATH).astype(np.float32)

with rasterio.open(HIST_MASK_PATH) as src:
    historical_mask = src.read(1) > 0
    transform = src.transform
    crs = src.crs
    shape = (src.height, src.width)
    pixel_area_km2 = abs(transform.a * transform.e) / 1_000_000

if prob.shape != shape:
    raise ValueError(
        f"Probability shape {prob.shape} differs from mask grid {shape}."
    )

# Rasterize the RGI polygon on the historical raster grid.
gdf = gpd.read_file(RGI_PATH)

id_column = next(
    (c for c in gdf.columns if c.lower() in {"rgiid", "rgi_id", "rgi_id_v7"}),
    None,
)
if id_column is None:
    raise ValueError(f"Cannot identify RGI ID column: {list(gdf.columns)}")

selected = gdf[
    gdf[id_column].astype(str) == "RGI2000-v7.0-G-15-00467"
]
if selected.empty:
    raise ValueError("Gangotri RGI feature not found.")

if gdf.crs is None:
    raise ValueError("RGI inventory CRS is missing.")

geometry = selected.to_crs(crs).geometry.union_all()

rgi_mask = rasterize(
    [(mapping(geometry), 1)],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8",
).astype(bool)

candidate_raw = (prob >= THRESHOLD)
candidate_clipped = candidate_raw & rgi_mask
core = (prob >= CORE_THRESHOLD) & rgi_mask

close_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE)
)
open_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (OPEN_SIZE, OPEN_SIZE)
)

morph = cv2.morphologyEx(
    candidate_clipped.astype(np.uint8),
    cv2.MORPH_CLOSE,
    close_kernel,
    iterations=1,
)
morph = cv2.morphologyEx(
    morph, cv2.MORPH_OPEN, open_kernel, iterations=1
).astype(bool)

# Match the production behavior of restricting the result to RGI.
morph &= rgi_mask

# Apply the component retention rules from predict_geotiff.py.
labels, count = ndimage.label(
    morph, structure=np.ones((3, 3), dtype=np.uint8)
)

filtered = np.zeros(shape, dtype=bool)
kept = 0
removed = 0

for component_id in range(1, count + 1):
    component = labels == component_id
    size = int(component.sum())
    mean_prob = float(prob[component].mean())
    has_core = bool(np.any(core[component]))

    keep = (
        has_core
        or (
            size >= MIN_COMPONENT_PIXELS
            and mean_prob >= MIN_COMPONENT_MEAN_PROBABILITY
        )
    )

    if keep:
        filtered[component] = True
        kept += 1
    else:
        removed += 1

filtered &= rgi_mask

stages = [
    ("Raw probability >= 0.20", candidate_raw),
    ("After RGI clipping", candidate_clipped),
    ("After morphology + RGI", morph),
    ("After component filtering", filtered),
    ("Historical saved mask", historical_mask),
]

print("\n--- 2026 POST-PROCESSING ABLATION ---")
print(f"Grid: {shape}, CRS: {crs}")
print(f"RGI footprint pixels: {int(rgi_mask.sum()):,}")
print(f"Core pixels >= {CORE_THRESHOLD:.2f} inside RGI: {int(core.sum()):,}")
print(f"Components examined: {count}")
print(f"Components kept: {kept}")
print(f"Components removed: {removed}")
print()

for name, mask in stages:
    pixels = int(mask.sum())
    area = pixels * pixel_area_km2
    print(f"{name:<32} pixels={pixels:>9,} area={area:>9.3f} km²")

print("\n--- AGREEMENT WITH SAVED HISTORICAL MASK ---")
for name, mask in stages[:-1]:
    intersection = int(np.logical_and(mask, historical_mask).sum())
    union = int(np.logical_or(mask, historical_mask).sum())
    iou = intersection / union if union else 1.0
    print(f"{name:<32} IoU={iou:.5f}")

print("\nNo source files or historical results were modified.")
