
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"

gid = "rgi2000_v7_0_g_15_00467"

rgb_path = LIVE / f"{gid}_debris_rgb.tif"
spectral_path = LIVE / f"{gid}_debris_spectral.tif"
glacier_path = PRED / f"{gid}_debris_glacier_mask.tif"
debris_path = PRED / f"{gid}_debris_affected_mask.tif"

latitude = 30.8760783581705
longitude = 79.0053687387015
radius = 64

paths = [rgb_path, spectral_path, glacier_path, debris_path]
for path in paths:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

with rasterio.open(spectral_path) as src:
    x, y = transform(
        "EPSG:4326", src.crs, [longitude], [latitude]
    )
    row, col = src.index(x[0], y[0])
    print(f"Coordinate pixel: row={row}, col={col}")
    print(f"CRS: {src.crs}; resolution: {src.res}")

r0, r1 = max(0, row-radius), row+radius
c0, c1 = max(0, col-radius), col+radius

with rasterio.open(rgb_path) as src:
    rgb = src.read([1, 2, 3], window=((r0, r1), (c0, c1)))
    rgb = np.moveaxis(rgb, 0, -1).astype(np.float32)

# Stretch each band for visual inspection.
for band in range(3):
    lo, hi = np.percentile(rgb[:, :, band], [2, 98])
    rgb[:, :, band] = np.clip(
        (rgb[:, :, band] - lo) / max(hi - lo, 1e-6), 0, 1
    )

with rasterio.open(glacier_path) as src:
    glacier = src.read(1, window=((r0, r1), (c0, c1))) > 0

with rasterio.open(debris_path) as src:
    debris = src.read(1, window=((r0, r1), (c0, c1))) > 0

# Confirm the coordinate agrees with the mask grids.
assert glacier.shape == debris.shape == rgb.shape[:2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(rgb)
axes[0].set_title("Original RGB crop")

axes[1].imshow(rgb)
axes[1].imshow(
    np.ma.masked_where(~glacier, glacier),
    cmap="winter", alpha=0.55, vmin=0, vmax=1
)
axes[1].set_title("Predicted glacier overlay")

axes[2].imshow(rgb)
axes[2].imshow(
    np.ma.masked_where(~glacier, glacier),
    cmap="winter", alpha=0.4, vmin=0, vmax=1
)
axes[2].imshow(
    np.ma.masked_where(~debris, debris),
    cmap="Reds", alpha=0.85, vmin=0, vmax=1
)
axes[2].set_title("Glacier + predicted debris")

# Mark the supplied coordinate at the crop centre.
for ax in axes:
    ax.scatter(
        [col-c0], [row-r0],
        marker="+", s=150, linewidths=2, color="yellow"
    )
    ax.axis("off")

fig.suptitle(
    "Sample 01 coordinate check — yellow cross = supplied coordinate"
)
fig.tight_layout()

out = PRED / "debris_samples" / "sample01_coordinate_check.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Saved crop: {out}")
print(f"Glacier pixels in crop: {int(glacier.sum())}")
print(f"Debris pixels in crop: {int(debris.sum())}")
