
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"

GLACIER_ID = "rgi2000_v7_0_g_15_00467"

RGB_PATH = LIVE / f"{GLACIER_ID}_debris_rgb.tif"
GLACIER_PATH = PRED / f"{GLACIER_ID}_debris_glacier_mask.tif"
DEBRIS_PATH = PRED / f"{GLACIER_ID}_debris_affected_mask.tif"

OUTPUT = PRED / "glacier_geometry_overview.png"


def main():
    with rasterio.open(RGB_PATH) as src:
        rgb_raw = src.read([1, 2, 3]).astype(np.float32)
        profile = src.profile.copy()

    with rasterio.open(GLACIER_PATH) as src:
        if (
            src.transform != profile["transform"]
            or src.crs != profile["crs"]
            or src.shape != (rgb_raw.shape[1], rgb_raw.shape[2])
        ):
            raise ValueError("Glacier mask and RGB raster are misaligned.")
        glacier = src.read(1) > 0

    with rasterio.open(DEBRIS_PATH) as src:
        if (
            src.transform != profile["transform"]
            or src.crs != profile["crs"]
            or src.shape != (rgb_raw.shape[1], rgb_raw.shape[2])
        ):
            raise ValueError("Debris mask and RGB raster are misaligned.")
        debris = src.read(1) > 0

    low, high = np.nanpercentile(rgb_raw, [2, 98])
    rgb = np.clip((rgb_raw - low) / (high - low + 1e-6), 0, 1)
    rgb = np.moveaxis(rgb, 0, -1)

    labels, n = ndimage.label(
        glacier, structure=np.ones((3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    # Show only the 10 largest components in distinct label colours.
    largest_ids = np.argsort(sizes)[-10:]
    component_view = np.zeros_like(labels)
    for new_id, old_id in enumerate(largest_ids, start=1):
        if sizes[old_id] > 0:
            component_view[labels == old_id] = new_id

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")

    axes[1].imshow(glacier, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Glacier mask ({n} components)")

    axes[2].imshow(rgb)
    axes[2].contour(
        glacier.astype(np.uint8),
        levels=[0.5],
        colors=["cyan"],
        linewidths=0.5,
    )
    overlay = np.zeros((*glacier.shape, 4), dtype=np.float32)
    overlay[debris & glacier] = [1.0, 0.1, 0.1, 0.9]
    axes[2].imshow(overlay)
    axes[2].set_title("Cyan = glacier boundary; red = debris")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Connected components: {n}")
    print(f"Saved overview: {OUTPUT}")
    print("No production rasters were changed.")


if __name__ == "__main__":
    main()
