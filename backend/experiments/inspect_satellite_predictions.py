
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

root = Path("datasets/historical/rgi2000-v7.0-g-15-00467")
sat = root / "satellite"
pred = root / "predictions"

stem25 = "rgi2000-v7.0-g-15-00467_2025"
stem26 = "rgi2000-v7.0-g-15-00467_2026"

# Load satellite RGB images and glacier masks
with rasterio.open(sat / f"{stem25}.tif") as ds:
    img25 = ds.read([1, 2, 3]).astype(np.float32)
    valid25 = np.any(img25 != 0, axis=0)
    transform25 = ds.transform
    crs25 = ds.crs

with rasterio.open(sat / f"{stem26}.tif") as ds:
    img26 = ds.read([1, 2, 3]).astype(np.float32)
    valid26 = np.any(img26 != 0, axis=0)
    transform26 = ds.transform
    crs26 = ds.crs

with rasterio.open(pred / f"{stem25}_rgi_mask.tif") as ds:
    mask25 = ds.read(1) > 0

with rasterio.open(pred / f"{stem26}_rgi_mask.tif") as ds:
    mask26 = ds.read(1) > 0

assert img25.shape == img26.shape
assert mask25.shape == mask26.shape
assert transform25 == transform26
assert crs25 == crs26

# Display RGB consistently across both dates
def make_rgb(image, valid):
    rgb = np.moveaxis(image, 0, -1)
    pixels = rgb[valid]

    if len(pixels) == 0:
        raise ValueError("No valid satellite pixels found")

    # Robust stretch for visualization only
    low, high = np.percentile(pixels, [2, 98])
    if high <= low:
        high = low + 1

    rgb = np.clip((rgb - low) / (high - low), 0, 1)
    rgb[~valid] = 0
    return rgb

rgb25 = make_rgb(img25, valid25)
rgb26 = make_rgb(img26, valid26)

common = valid25 & valid26

# Shared = green; 2025-only = blue; 2026-only = red
shared = common & mask25 & mask26
only25 = common & mask25 & ~mask26
only26 = common & ~mask25 & mask26

fig, axes = plt.subplots(2, 2, figsize=(16, 13))

# Original satellite images
axes[0, 0].imshow(rgb25)
axes[0, 0].set_title("2025 Satellite Image (7 June)")
axes[0, 1].imshow(rgb26)
axes[0, 1].set_title("2026 Satellite Image (19 September)")

# Boundaries overlaid on the satellite imagery
axes[1, 0].imshow(rgb25)
axes[1, 0].contour(mask25.astype(float), levels=[0.5],
                   colors=["cyan"], linewidths=0.7)
axes[1, 0].set_title("2025 Image + Predicted Boundary")

axes[1, 1].imshow(rgb26)
axes[1, 1].contour(mask26.astype(float), levels=[0.5],
                   colors=["yellow"], linewidths=0.7)
axes[1, 1].set_title("2026 Image + Predicted Boundary")

for ax in axes.flat:
    ax.axis("off")

fig.tight_layout()
out1 = root / "satellite_prediction_overlay.png"
fig.savefig(out1, dpi=160, bbox_inches="tight")
plt.close(fig)

# Common-coverage disagreement overlay
comparison = np.zeros(common.shape, dtype=np.uint8)
comparison[only25] = 1
comparison[only26] = 2
comparison[shared] = 3

cmap = ListedColormap([
    "black", "dodgerblue", "red", "limegreen"
])
display = np.ma.masked_where(~common, comparison)

fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(display, cmap=cmap, vmin=0, vmax=3)
ax.set_title("Prediction Differences Within Common Valid Coverage")
ax.axis("off")

out2 = root / "satellite_prediction_disagreement.png"
fig.savefig(out2, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved:", out1)
print("Saved:", out2)
print("Common valid pixels:", int(common.sum()))
print("Shared glacier pixels:", int(shared.sum()))
print("2025-only pixels:", int(only25.sum()))
print("2026-only pixels:", int(only26.sum()))
print("Note: RGB stretching is for visualization only.")
