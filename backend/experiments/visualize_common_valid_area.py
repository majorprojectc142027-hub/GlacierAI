
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

# Load satellite images and valid-data regions
with rasterio.open(sat / f"{stem25}.tif") as ds:
    image25 = ds.read([1, 2, 3])
    valid25 = np.any(image25 != 0, axis=0)
    transform25 = ds.transform
    crs25 = ds.crs

with rasterio.open(sat / f"{stem26}.tif") as ds:
    image26 = ds.read([1, 2, 3])
    valid26 = np.any(image26 != 0, axis=0)
    transform26 = ds.transform
    crs26 = ds.crs

# Load existing annual masks
with rasterio.open(pred / f"{stem25}_rgi_mask.tif") as ds:
    mask25 = ds.read(1) > 0

with rasterio.open(pred / f"{stem26}_rgi_mask.tif") as ds:
    mask26 = ds.read(1) > 0

assert image25.shape == image26.shape
assert mask25.shape == mask26.shape
assert transform25 == transform26
assert crs25 == crs26

# Compare only pixels valid in both satellite images
common = valid25 & valid26

# Categories:
# 0 = neither year predicts glacier
# 1 = 2025 only
# 2 = 2026 only
# 3 = both years predict glacier
comparison = np.zeros(common.shape, dtype=np.uint8)
comparison[common & mask25 & ~mask26] = 1
comparison[common & ~mask25 & mask26] = 2
comparison[common & mask25 & mask26] = 3

display = np.ma.masked_where(~common, comparison)

cmap = ListedColormap([
    "black",
    "dodgerblue",
    "red",
    "limegreen",
])

fig, ax = plt.subplots(figsize=(14, 9))
ax.imshow(display, cmap=cmap, vmin=0, vmax=3)
ax.set_title("Gangotri Glacier: 2025 vs 2026 (Common Valid Coverage)")
ax.axis("off")

output = root / "common_valid_area_comparison.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
plt.close(fig)

# Print exact counts and areas
pixel_area_km2 = abs(
    transform25.a * transform25.e
    - transform25.b * transform25.d
) / 1_000_000

shared = common & mask25 & mask26
only25 = common & mask25 & ~mask26
only26 = common & ~mask25 & mask26

print("Comparison saved to:", output)
print("Common valid pixels:", int(common.sum()))
print("Shared glacier pixels:", int(shared.sum()))
print("2025-only pixels:", int(only25.sum()))
print("2026-only pixels:", int(only26.sum()))
print("Shared area (km^2):", round(shared.sum() * pixel_area_km2, 3))
print("2025-only area (km^2):", round(only25.sum() * pixel_area_km2, 3))
print("2026-only area (km^2):", round(only26.sum() * pixel_area_km2, 3))
