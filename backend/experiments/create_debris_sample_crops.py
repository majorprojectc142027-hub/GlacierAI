
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
    / "debris_validation_samples.csv"
)

OUTPUT_DIR = (
    ROOT / "datasets" / "live" / "predictions"
    / "debris_sample_crops"
)

CROP_SIZE = 256
SAMPLES_PER_SHEET = 25
COLS_PER_SHEET = 5


def main():
    if not RGB_PATH.exists():
        raise FileNotFoundError(f"RGB raster not found: {RGB_PATH}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Sample CSV not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load only the additional experimental detections.
    df = pd.read_csv(CSV_PATH)

    required = {"sample_id", "group", "row", "col"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {sorted(required)}"
        )

    df = df[
        df["group"] == "additional_experimental"
    ].copy()

    print(f"Experimental samples to review: {len(df)}")

    if df.empty:
        raise ValueError(
            "No additional experimental samples found in the CSV."
        )

    with rasterio.open(RGB_PATH) as src:
        rgb = src.read()
        height, width = src.height, src.width

    if rgb.shape[0] < 3:
        raise ValueError("RGB raster must have at least 3 bands.")

    # Prepare RGB display using consistent image-wide scaling.
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

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(crop)

        # Mark the exact sampled pixel with a yellow crosshair.
        sample_x = col - c0
        sample_y = row - r0

        ax.scatter(
            [sample_x],
            [sample_y],
            s=180,
            c="yellow",
            marker="+",
            linewidths=2.5,
        )

        ax.axvline(
            sample_x,
            color="yellow",
            linewidth=0.7,
            alpha=0.8,
        )
        ax.axhline(
            sample_y,
            color="yellow",
            linewidth=0.7,
            alpha=0.8,
        )

        ax.set_title(
            f"{sample_id} | {group}\n"
            f"Row={row}, Col={col}",
            fontsize=10,
        )
        ax.axis("off")

        fig.tight_layout()
        fig.savefig(
            crop_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        saved.append({
            "sample_id": sample_id,
            "group": group,
            "row": row,
            "col": col,
            "crop_file": crop_path.name,
        })

    if not saved:
        raise ValueError("No valid sample crops were generated.")

    manifest = pd.DataFrame(saved)
    manifest.to_csv(
        OUTPUT_DIR / "crop_manifest.csv",
        index=False,
    )

    # Create contact sheets containing up to 25 samples each.
    for start in range(0, len(manifest), SAMPLES_PER_SHEET):
        subset = manifest.iloc[
            start:start + SAMPLES_PER_SHEET
        ]

        n = len(subset)
        nrows = math.ceil(n / COLS_PER_SHEET)

        fig, axes = plt.subplots(
            nrows,
            COLS_PER_SHEET,
            figsize=(16, 3.8 * nrows),
            squeeze=False,
        )

        for ax in axes.flat:
            ax.axis("off")

        for ax, (_, item) in zip(
            axes.flat,
            subset.iterrows(),
        ):
            image_path = OUTPUT_DIR / item["crop_file"]
            image = plt.imread(image_path)

            ax.imshow(image)
            ax.set_title(
                f'{item["sample_id"]} | additional',
                fontsize=9,
            )
            ax.axis("off")

        page = start // SAMPLES_PER_SHEET + 1
        sheet_path = (
            OUTPUT_DIR / f"contact_sheet_{page:02d}.png"
        )

        fig.tight_layout()
        fig.savefig(
            sheet_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Saved: {sheet_path}")

    print(f"\nIndividual crops created: {len(manifest)}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Yellow crosshairs mark the sampled pixels.")
    print("Review contact_sheet_01.png first.")


if __name__ == "__main__":
    main()
