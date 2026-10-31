
import json
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image

ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = (
    ROOT
    / "datasets/historical/rgi2000-v7.0-g-15-00467"
    / "historical_results.json"
)

with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

obs = next(
    item for item in data["observations"]
    if item["year"] == 2026
)

mask_path = Path(obs["mask_path"])
image_path = Path(obs["satellite_path"])

if not mask_path.exists():
    raise FileNotFoundError(mask_path)

if not image_path.exists():
    raise FileNotFoundError(image_path)

with rasterio.open(image_path) as src:
    rgb = reshape_as_image(src.read([1, 2, 3]))
    rgb = rgb.astype(np.float32)

    # Display normalization for visualization only.
    rgb = np.clip(rgb / 10000.0, 0, 1)

    image_transform = src.transform
    image_crs = src.crs
    image_shape = (src.height, src.width)

with rasterio.open(mask_path) as src:
    mask = src.read(1) > 0

    if (
        src.crs != image_crs
        or src.transform != image_transform
        or mask.shape != image_shape
    ):
        raise ValueError(
            "Mask and satellite image grids do not match."
        )

overlay = rgb.copy()
overlay[mask] = (
    0.35 * rgb[mask]
    + 0.65 * np.array([0.0, 1.0, 0.0])
)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB — 2026")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Actual 2026 U-Net mask")

axes[2].imshow(overlay)
axes[2].set_title("2026 prediction overlay")

for ax in axes:
    ax.axis("off")

fig.tight_layout()

output = (
    ROOT
    / "datasets/historical/rgi2000-v7.0-g-15-00467"
    / "prediction_2026_diagnostic.png"
)

fig.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved:", output)
print("Mask pixels:", int(mask.sum()))
print("Mask area:", round(mask.sum() * 144 / 1_000_000, 6), "km²")
