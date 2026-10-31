from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from backend.app.services.rgi_service import find_glacier_by_name

ROOT = Path(__file__).resolve().parents[1]
SAT = ROOT / "datasets" / "historical" / "satellite"
PRED = ROOT / "datasets" / "historical" / "predictions"
OUT = ROOT / "experiments" / "gangotri_2025_2026_comparison.png"

glacier = find_glacier_by_name("Gangotri")
if glacier is None:
    raise RuntimeError("Gangotri was not found in RGI.")

geometry = glacier["geometry"]

def load_year(year):
    image_path = SAT / f"gangotri_{year}-09-{22 if year == 2025 else 19}.tif"
    mask_path = PRED / (
        f"gangotri_{year}_rgi_mask.tif"
    )

    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3]).astype(np.float32)
        image = np.moveaxis(image, 0, -1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    # Display-only percentile stretch; does not affect model predictions.
    rgb = np.zeros_like(image, dtype=np.float32)
    for band in range(3):
        values = image[..., band]
        lo, hi = np.percentile(values, [2, 98])
        rgb[..., band] = np.clip(
            (values - lo) / max(float(hi - lo), 1e-6), 0, 1
        )

    with rasterio.open(mask_path) as src:
        mask = src.read(1) > 0
        if (
            src.shape != mask.shape
            or src.crs != crs
            or src.transform != transform
        ):
            print(f"WARNING: {year} mask and image grids differ.")

    rgi_geom = transform_geom("EPSG:4326", crs, geometry)
    rgi_mask = rasterize(
        [(rgi_geom, 1)],
        out_shape=mask.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ) > 0

    return rgb, mask, rgi_mask, bounds

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

for row, year in enumerate([2025, 2026]):
    rgb, pred, rgi, bounds = load_year(year)

    axes[row, 0].imshow(rgb)
    axes[row, 0].set_title(f"{year}: Satellite RGB (display stretch)")

    axes[row, 1].imshow(rgb)
    axes[row, 1].imshow(
        np.ma.masked_where(~pred, pred),
        cmap="autumn",
        alpha=0.65,
        vmin=0,
        vmax=1,
    )
    axes[row, 1].set_title(f"{year}: Predicted mask overlay")

    axes[row, 2].imshow(rgb)
    axes[row, 2].contour(
        rgi.astype(np.uint8),
        levels=[0.5],
        colors=["cyan"],
        linewidths=0.8,
    )
    axes[row, 2].contour(
        pred.astype(np.uint8),
        levels=[0.5],
        colors=["red"],
        linewidths=0.8,
    )
    axes[row, 2].set_title(
        f"{year}: RGI boundary (cyan), prediction (red)"
    )

    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle(
    "Gangotri Glacier: Historical Segmentation Diagnostic",
    fontsize=16,
)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved diagnostic image:", OUT)


