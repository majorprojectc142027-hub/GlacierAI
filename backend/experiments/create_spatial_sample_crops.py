
from pathlib import Path
import math

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

RGB_PATH = (
    ROOT / "datasets" / "live"
    / "rgi2000_v7_0_g_15_00467_debris_rgb.tif"
)

CSV_PATH = (
    ROOT / "datasets" / "live" / "predictions"
    / "debris_spatial_validation_samples.csv"
)

OUTPUT_DIR = (
    ROOT / "datasets" / "live" / "predictions"
    / "debris_spatial_sample_crops"
)

CROP_SIZE = 256
SAMPLES_PER_SHEET = 25
COLS_PER_SHEET = 5


def main():
    if not RGB_PATH.exists():
        raise FileNotFoundError(f"RGB raster not found: {RGB_PATH}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required = {"sample_id", "group", "row", "col"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain: {sorted(required)}"
        )

    print("Samples by group:")
    print(df["group"].value_counts().to_string())

    if df.empty:
        raise ValueError("The spatial validation CSV is empty.")

    with rasterio.open(RGB_PATH) as src:
        rgb = src.read()
        height, width = src.height, src.width

    if rgb.shape[0] < 3:
        raise ValueError("RGB raster must contain at least 3 bands.")

    rgb = np.moveaxis(rgb[:3], 0, -1).astype(np.float32)

    low, high = np.nanpercentile(rgb, [2, 98])
    rgb = np.clip(
        (rgb - low) / (high - low + 1e-6),
        0,
        1,
    )

    half = CROP_SIZE // 2
    saved = []

    for _, sample in df.iterrows():
        sample_id = str(sample["sample_id"])
        group = str(sample["group"])
        row = int(sample["row"])
        col = int(sample["col"])

        if not (0 <= row < height and 0 <= col < width):
            print(f"Skipping {sample_id}: coordinates out of bounds")
            continue

        r0 = max(0, row - half)
        r1 = min(height, row + half)
        c0 = max(0, col - half)
        c1 = min(width, col + half)

        crop = rgb[r0:r1, c0:c1]

        crop_path = OUTPUT_DIR / f"{sample_id}.png"

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(crop)

        x = col - c0
        y = row - r0

        ax.scatter(
            [x], [y],
            s=180,
            c="yellow",
            marker="+",
            linewidths=2.5,
        )

        ax.axvline(x, color="yellow", linewidth=0.7, alpha=0.8)
        ax.axhline(y, color="yellow", linewidth=0.7, alpha=0.8)

        ax.set_title(
            f"{sample_id} | {group}\nRow={row}, Col={col}",
            fontsize=9,
        )
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(crop_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

        saved.append({
            "sample_id": sample_id,
            "group": group,
            "row": row,
            "col": col,
            "crop_file": crop_path.name,
        })

    if not saved:
        raise ValueError("No valid crops were generated.")

    manifest = pd.DataFrame(saved)
    manifest.to_csv(OUTPUT_DIR / "crop_manifest.csv", index=False)

    for start in range(0, len(manifest), SAMPLES_PER_SHEET):
        subset = manifest.iloc[start:start + SAMPLES_PER_SHEET]
        nrows = math.ceil(len(subset) / COLS_PER_SHEET)

        fig, axes = plt.subplots(
            nrows,
            COLS_PER_SHEET,
            figsize=(16, 3.8 * nrows),
            squeeze=False,
        )

        for ax in axes.flat:
            ax.axis("off")

        for ax, (_, item) in zip(axes.flat, subset.iterrows()):
            image_path = OUTPUT_DIR / item["crop_file"]
            ax.imshow(plt.imread(image_path))
            ax.set_title(
                f'{item["sample_id"]} | {item["group"]}',
                fontsize=8,
            )
            ax.axis("off")

        page = start // SAMPLES_PER_SHEET + 1
        sheet_path = OUTPUT_DIR / f"contact_sheet_{page:02d}.png"

        fig.tight_layout()
        fig.savefig(sheet_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {sheet_path}")

    print(f"\nIndividual crops created: {len(manifest)}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
