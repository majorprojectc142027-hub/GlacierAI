
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"

RGB_PATH = LIVE / "rgi2000_v7_0_g_15_00467_debris_rgb.tif"
RGI_PATH = ROOT / "datasets" / "masks" / "gangotri_mask.tif"
UNET_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"

OUTPUT = PRED / "rgi_unet_mask_comparison.png"


def main():
    for path in [RGB_PATH, RGI_PATH, UNET_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with rasterio.open(RGB_PATH) as src:
        rgb_data = src.read([1, 2, 3]).astype(np.float32)
        profile = src.profile.copy()
        target_shape = (src.height, src.width)

    low, high = np.nanpercentile(rgb_data, [2, 98])
    rgb = np.clip((rgb_data - low) / (high - low + 1e-6), 0, 1)
    rgb = np.moveaxis(rgb, 0, -1)

    with rasterio.open(RGI_PATH) as src:
        rgi_raw = src.read(1)
        rgi = np.zeros(target_shape, dtype=np.uint8)

        # Align the original RGI mask to the RGB raster grid.
        reproject(
            source=rgi_raw,
            destination=rgi,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.nearest,
        )

    with rasterio.open(UNET_PATH) as src:
        if (
            src.shape != target_shape
            or src.transform != profile["transform"]
            or src.crs != profile["crs"]
        ):
            raise ValueError("U-Net mask and RGB grids differ.")
        unet = (src.read(1) > 0).astype(np.uint8)

    # IMPORTANT: assumes the original RGI mask uses 0 for background
    # and nonzero values for glacier pixels. Confirm its unique values.
    rgi_binary = rgi > 0
    unet_binary = unet > 0

    intersection = rgi_binary & unet_binary
    union = rgi_binary | unet_binary

    rgi_count = int(rgi_binary.sum())
    unet_count = int(unet_binary.sum())
    intersection_count = int(intersection.sum())
    union_count = int(union.sum())

    iou = intersection_count / union_count if union_count else 0
    dice = (
        2 * intersection_count / (rgi_count + unet_count)
        if rgi_count + unet_count else 0
    )

    print("=== RGI vs U-Net MASK COMPARISON ===")
    print("RGB grid:", target_shape)
    print("Aligned RGI glacier pixels:", rgi_count)
    print("U-Net glacier pixels:", unet_count)
    print("Intersection pixels:", intersection_count)
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Sentinel-2 RGB")

    axes[0, 1].imshow(rgi_binary, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("RGI reference mask")

    axes[1, 0].imshow(unet_binary, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("U-Net predicted mask")

    # Green = overlap; red = RGI only; blue = U-Net only.
    comparison = np.zeros((*target_shape, 3), dtype=np.float32)
    comparison[rgi_binary & ~unet_binary] = [1, 0, 0]
    comparison[unet_binary & ~rgi_binary] = [0, 0, 1]
    comparison[intersection] = [0, 1, 0]

    axes[1, 1].imshow(comparison)
    axes[1, 1].set_title(
        "Overlap: green | RGI only: red | U-Net only: blue"
    )

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison image: {OUTPUT}")
    print("No original masks or model files were modified.")


if __name__ == "__main__":
    main()
