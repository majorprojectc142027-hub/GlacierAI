
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"

GLACIER_ID = "rgi2000_v7_0_g_15_00467"

RGB_PATH = LIVE / f"{GLACIER_ID}_debris_rgb.tif"
GLACIER_PATH = PRED / f"{GLACIER_ID}_debris_glacier_mask.tif"
DEBRIS_PATH = PRED / f"{GLACIER_ID}_debris_affected_mask.tif"

CSV_PATH = PRED / "debris_validation_samples.csv"
OUTPUT_DIR = PRED / "debris_mask_comparison"

CROP_SIZE = 256


def main():
    paths = [RGB_PATH, GLACIER_PATH, DEBRIS_PATH, CSV_PATH]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    samples = pd.read_csv(CSV_PATH)
    required = {"sample_id", "group", "row", "col"}
    if not required.issubset(samples.columns):
        raise ValueError(f"CSV must contain: {sorted(required)}")

    # Inspect the five spatial validation samples from the SWIR analysis.
    spatial_csv = PRED / "debris_spatial_validation_samples.csv"
    if spatial_csv.exists():
        spatial = pd.read_csv(spatial_csv)
        if required.issubset(spatial.columns):
            samples = spatial[
                spatial["sample_id"].astype(str).isin(
                    ["V007", "V008", "V009", "V010", "V011"]
                )
            ].copy()

    if samples.empty:
        raise ValueError("No requested samples were found.")

    with rasterio.open(RGB_PATH) as src:
        rgb_raw = src.read([1, 2, 3]).astype(np.float32)
        profile = src.profile.copy()
        height, width = src.height, src.width

    with rasterio.open(GLACIER_PATH) as src:
        if (
            src.crs != profile["crs"]
            or src.transform != profile["transform"]
            or src.height != height
            or src.width != width
        ):
            raise ValueError("Glacier mask is not aligned with RGB raster.")
        glacier = src.read(1) > 0

    with rasterio.open(DEBRIS_PATH) as src:
        if (
            src.crs != profile["crs"]
            or src.transform != profile["transform"]
            or src.height != height
            or src.width != width
        ):
            raise ValueError("Debris mask is not aligned with RGB raster.")
        debris = src.read(1) > 0

    # Use one consistent stretch for all RGB crops.
    low, high = np.nanpercentile(rgb_raw, [2, 98])
    rgb = np.clip((rgb_raw - low) / (high - low + 1e-6), 0, 1)
    rgb = np.moveaxis(rgb, 0, -1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    half = CROP_SIZE // 2

    for _, sample in samples.iterrows():
        sid = str(sample["sample_id"])
        row, col = int(sample["row"]), int(sample["col"])

        if not (0 <= row < height and 0 <= col < width):
            print(f"Skipping {sid}: coordinates outside raster")
            continue

        r0, r1 = max(0, row-half), min(height, row+half)
        c0, c1 = max(0, col-half), min(width, col+half)

        rgb_crop = rgb[r0:r1, c0:c1]
        glacier_crop = glacier[r0:r1, c0:c1]
        debris_crop = debris[r0:r1, c0:c1]

        # Overlay: cyan outline = glacier mask; red = predicted debris.
        overlay = rgb_crop.copy()
        overlay[debris_crop] = (
            0.75 * overlay[debris_crop]
            + 0.25 * np.array([1.0, 0.0, 0.0])
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(rgb_crop)
        axes[0].set_title("Original RGB")

        axes[1].imshow(glacier_crop, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("U-Net glacier mask")

        axes[2].imshow(overlay)
        axes[2].contour(
            glacier_crop.astype(np.uint8),
            levels=[0.5],
            colors=["cyan"],
            linewidths=0.7,
        )
        axes[2].set_title("Red = predicted debris\nCyan = glacier boundary")

        x, y = col - c0, row - r0
        for ax in axes:
            ax.scatter(
                [x], [y], marker="+", s=150,
                c="yellow", linewidths=2
            )
            ax.axis("off")

        debris_pixels = int(debris_crop.sum())
        glacier_pixels = int(glacier_crop.sum())

        fig.suptitle(
            f"{sid} | row={row}, col={col} | "
            f"debris pixels={debris_pixels}, "
            f"glacier pixels={glacier_pixels}"
        )
        fig.tight_layout()

        out = OUTPUT_DIR / f"{sid}_mask_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
