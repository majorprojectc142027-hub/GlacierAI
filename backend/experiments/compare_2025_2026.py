
import json
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467"
JSON_PATH = BASE / "historical_results.json"

with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

observations = {int(o["year"]): o for o in data["observations"]}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_mask(year):
    path = resolve(observations[year]["mask_path"])

    with rasterio.open(path) as src:
        mask = src.read(1) > 0
        grid = (src.crs, src.transform, src.width, src.height)

    return mask, grid


# Load actual historical masks referenced by historical_results.json.
mask25, grid25 = read_mask(2025)
mask26, grid26 = read_mask(2026)

if grid25 != grid26:
    raise ValueError("2025 and 2026 masks use different grids.")

only25 = mask25 & ~mask26
only26 = mask26 & ~mask25
common = mask25 & mask26

print("\n--- 2025 vs 2026 SPATIAL DIFFERENCE ---")
print("2025 pixels:", int(mask25.sum()))
print("2026 pixels:", int(mask26.sum()))
print("Common pixels:", int(common.sum()))
print("2025-only pixels:", int(only25.sum()))
print("2026-only pixels:", int(only26.sum()))

# Connected-component analysis of 2026-only pixels.
structure = np.ones((3, 3), dtype=np.uint8)
labels, count = ndimage.label(only26, structure=structure)
sizes = np.bincount(labels.ravel())[1:]
largest = np.sort(sizes)[::-1]

print("2026-only connected components:", int(count))
print("Largest 10 2026-only components:", largest[:10].tolist())

# Load the 2026 Sentinel-2 RGB image.
obs26 = observations[2026]
image_path = resolve(obs26["satellite_path"])

with rasterio.open(image_path) as src:
    rgb = reshape_as_image(src.read([1, 2, 3])).astype(np.float32)
    rgb = np.clip(rgb / 10000.0, 0, 1)

if rgb.shape[:2] != mask26.shape:
    raise ValueError("2026 image and mask dimensions differ.")

# Colors:
# Green = predicted in both years
# Blue  = predicted only in 2025
# Red   = predicted only in 2026
green = np.array([0.0, 0.85, 0.0], dtype=np.float32)
blue = np.array([0.0, 0.35, 1.0], dtype=np.float32)
red = np.array([1.0, 0.1, 0.1], dtype=np.float32)

diff = np.zeros((*mask26.shape, 3), dtype=np.float32)
diff[common] = green
diff[only25] = blue
diff[only26] = red

overlay = rgb.copy()
overlay[common] = 0.45 * rgb[common] + 0.55 * green
overlay[only25] = 0.35 * rgb[only25] + 0.65 * blue
overlay[only26] = 0.35 * rgb[only26] + 0.65 * red

fig, axes = plt.subplots(1, 3, figsize=(19, 7))

axes[0].imshow(rgb)
axes[0].set_title("2026 Sentinel-2 RGB")

axes[1].imshow(diff)
axes[1].set_title(
    "Mask difference: green=both, blue=2025, red=2026"
)

axes[2].imshow(overlay)
axes[2].set_title("Spatial difference over RGB")

for ax in axes:
    ax.axis("off")

fig.tight_layout()

output = BASE / "compare_2025_2026.png"
fig.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)

print("\nSaved:", output)
