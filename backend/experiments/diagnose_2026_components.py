
import json
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import mapping
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = (
    ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467"
    / "historical_results.json"
)

RGI_PATH = ROOT / "datasets/processed/global_rgi.gpkg"

with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

obs = next(item for item in data["observations"] if item["year"] == 2026)

mask_path = Path(obs["mask_path"])
image_path = Path(obs["satellite_path"])

# Resolve relative paths from the project root.
if not mask_path.is_absolute():
    mask_path = ROOT / mask_path
if not image_path.is_absolute():
    image_path = ROOT / image_path

for path in (mask_path, image_path, RGI_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

with rasterio.open(image_path) as src:
    image = reshape_as_image(src.read([1, 2, 3])).astype(np.float32)
    image = np.clip(image / 10000.0, 0, 1)
    transform = src.transform
    crs = src.crs
    shape = (src.height, src.width)

with rasterio.open(mask_path) as src:
    prediction = src.read(1) > 0

    if (
        src.crs != crs
        or src.transform != transform
        or prediction.shape != shape
    ):
        raise ValueError(
            "Prediction and satellite image grids do not match."
        )

# Connected components: 8-connected pixels count as one component.
structure = np.ones((3, 3), dtype=np.uint8)
labels, component_count = ndimage.label(
    prediction, structure=structure
)
component_sizes = np.bincount(labels.ravel())[1:]
component_sizes = np.sort(component_sizes)[::-1]

print("\n--- 2026 PREDICTION DIAGNOSTICS ---")
print("Mask file:", mask_path)
print("Satellite image:", image_path)
print("CRS:", crs)
print("Image shape:", shape)
print("Predicted pixels:", int(prediction.sum()))
print(
    "Predicted area (km²):",
    round(prediction.sum() * abs(transform.a * transform.e) / 1_000_000, 4),
)
print("Connected components:", int(component_count))
print("Largest 10 component sizes (pixels):", component_sizes[:10].tolist())

# Find the Gangotri RGI feature by the exact inventory ID.
gdf = gpd.read_file(RGI_PATH)

id_column = next(
    (c for c in gdf.columns if c.lower() in {"rgiid", "rgi_id", "rgi_id_v7"}),
    None,
)

if id_column is None:
    raise ValueError(
        f"Could not identify the RGI ID column. Available columns: "
        f"{list(gdf.columns)}"
    )

matches = gdf[gdf[id_column].astype(str) == "RGI2000-v7.0-G-15-00467"]

if matches.empty:
    raise ValueError(
        "Gangotri RGI ID not found. Check the RGI ID column and inventory file."
    )

if gdf.crs is None:
    raise ValueError("RGI inventory has no CRS.")

matches = matches.to_crs(crs)
geometry = matches.geometry.union_all()

rgi_mask = rasterize(
    [(mapping(geometry), 1)],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8",
) > 0

intersection = prediction & rgi_mask
union = prediction | rgi_mask

print("\n--- RGI BOUNDARY COMPARISON ---")
print("RGI pixels on this grid:", int(rgi_mask.sum()))
print("Prediction pixels inside RGI:", int(intersection.sum()))
print("Prediction pixels outside RGI:", int((prediction & ~rgi_mask).sum()))
print(
    "Prediction inside RGI (%):",
    round(100 * intersection.sum() / max(prediction.sum(), 1), 2),
)
print("Prediction/RGI IoU:", round(intersection.sum() / max(union.sum(), 1), 4))

# RGB overlay: green = predicted glacier, red outline = RGI inventory boundary.
overlay = image.copy()
overlay[prediction] = (
    0.35 * image[prediction]
    + 0.65 * np.array([0.0, 1.0, 0.0])
)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

axes[0].imshow(image)
axes[0].contour(rgi_mask, levels=[0.5], colors=["red"], linewidths=1)
axes[0].set_title("2026 RGB + RGI boundary")

axes[1].imshow(prediction, cmap="gray")
axes[1].contour(rgi_mask, levels=[0.5], colors=["red"], linewidths=1)
axes[1].set_title("Prediction + RGI boundary")

axes[2].imshow(overlay)
axes[2].contour(rgi_mask, levels=[0.5], colors=["red"], linewidths=1)
axes[2].set_title("Green: prediction | Red: RGI boundary")

for ax in axes:
    ax.axis("off")

fig.tight_layout()

output = (
    ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467"
    / "diagnose_2026_components.png"
)
fig.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)

print("\nSaved diagnostic image:", output)
