
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image

ROOT = Path(__file__).resolve().parents[2]

RGB_FILE = ROOT / "datasets/live/rgi2000_v7_0_g_15_00467_debris_rgb.tif"
PRED_FILE = ROOT / "datasets/live/predictions/rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
REF_FILE = ROOT / "datasets/live/predictions/rgi2000_v7_0_g_15_00467_rgi_reference_mask.tif"

with rasterio.open(RGB_FILE) as src:
    rgb = reshape_as_image(src.read([1, 2, 3]))
    rgb = rgb.astype(np.float32)
    rgb = np.clip(rgb / 10000.0, 0, 1)

with rasterio.open(PRED_FILE) as src:
    pred = src.read(1) > 0

with rasterio.open(REF_FILE) as src:
    ref = src.read(1) > 0

if rgb.shape[:2] != pred.shape or pred.shape != ref.shape:
    raise ValueError("Raster dimensions do not match.")

# Green = overlap; blue = RGI only; red = U-Net only.
overlay = np.zeros((*pred.shape, 3), dtype=np.float32)
overlay[pred & ref] = [0, 1, 0]
overlay[ref & ~pred] = [0, 0.35, 1]
overlay[pred & ~ref] = [1, 0, 0]

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB")

axes[1].imshow(ref, cmap="gray")
axes[1].set_title("RGI reference mask")

axes[2].imshow(rgb)
axes[2].imshow(overlay, alpha=0.75)
axes[2].set_title("Overlap comparison")

for ax in axes:
    ax.axis("off")

fig.tight_layout()

out = ROOT / "datasets/live/predictions/gangotri_rgi_unet_comparison.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved comparison:", out)
print("Green = both masks")
print("Blue = RGI reference only")
print("Red = U-Net prediction only")
