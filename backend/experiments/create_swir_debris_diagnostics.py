
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "live"
PRED = DATA / "predictions"

SPECTRAL_PATH = (
    DATA / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
)

RGB_PATH = (
    DATA / "rgi2000_v7_0_g_15_00467_debris_rgb.tif"
)

SAMPLES_PATH = PRED / "debris_spatial_validation_samples.csv"

OUTPUT_DIR = PRED / "swir_debris_diagnostics"

SAMPLE_IDS = ["V007", "V008", "V009", "V010", "V011"]
CROP_SIZE = 256


def stretch_band(band, low_pct=2, high_pct=98):
    """Stretch one band for visual display."""
    band = band.astype(np.float32)
    valid = np.isfinite(band)

    if not valid.any():
        return np.zeros_like(band)

    low, high = np.percentile(
        band[valid], [low_pct, high_pct]
    )

    if high <= low:
        return np.zeros_like(band)

    result = np.clip((band - low) / (high - low), 0, 1)
    result[~valid] = 0
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [SPECTRAL_PATH, RGB_PATH, SAMPLES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with rasterio.open(SPECTRAL_PATH) as src:
        spectral = src.read()
        spectral_profile = src.profile.copy()
        band_descriptions = src.descriptions

    with rasterio.open(RGB_PATH) as src:
        rgb_raw = src.read()
        rgb_profile = src.profile.copy()

    if (
        spectral_profile["height"] != rgb_profile["height"]
        or spectral_profile["width"] != rgb_profile["width"]
        or spectral_profile["transform"] != rgb_profile["transform"]
        or spectral_profile["crs"] != rgb_profile["crs"]
    ):
        raise ValueError(
            "RGB and spectral rasters are not on the same grid."
        )

    if spectral.shape[0] < 3 or rgb_raw.shape[0] < 3:
        raise ValueError("Both rasters must have at least 3 bands.")

    print("Spectral band descriptions:", band_descriptions)
    print("Assuming spectral band order: B08, B11, B12")

    # The project's spectral raster uses B08, B11, B12.
    nir, swir1, swir2 = [
        spectral[i].astype(np.float32)
        for i in range(3)
    ]

    # Convert integer-scaled reflectance to reflectance units.
    # Keep floating-point data unchanged.
    if np.issubdtype(spectral.dtype, np.integer):
        nir /= 10000.0
        swir1 /= 10000.0
        swir2 /= 10000.0

    # SWIR false-colour composite:
    # Red = SWIR2 (B12), Green = SWIR1 (B11), Blue = NIR (B08).
    # Use a separate robust stretch for each band.
    false_colour = np.stack(
        [
            stretch_band(swir2),
            stretch_band(swir1),
            stretch_band(nir),
        ],
        axis=-1,
    )

    rgb = np.moveaxis(
        rgb_raw[:3].astype(np.float32), 0, -1
    )
    low, high = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - low) / (high - low + 1e-6), 0, 1)

    samples = pd.read_csv(SAMPLES_PATH)
    samples = samples[
        samples["sample_id"].astype(str).isin(SAMPLE_IDS)
    ].copy()

    found = set(samples["sample_id"].astype(str))
    missing = set(SAMPLE_IDS) - found

    if missing:
        raise ValueError(f"Samples missing from CSV: {sorted(missing)}")

    half = CROP_SIZE // 2
    summaries = []

    for _, sample in samples.iterrows():
        sample_id = str(sample["sample_id"])
        row = int(sample["row"])
        col = int(sample["col"])

        height, width = nir.shape

        if not (0 <= row < height and 0 <= col < width):
            print(f"Skipping {sample_id}: coordinates out of bounds")
            continue

        r0 = max(0, row - half)
        r1 = min(height, row + half)
        c0 = max(0, col - half)
        c1 = min(width, col + half)

        rgb_crop = rgb[r0:r1, c0:c1]
        swir_crop = false_colour[r0:r1, c0:c1]

        x = col - c0
        y = row - r0

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(rgb_crop)
        axes[0].set_title("Sentinel-2 RGB")

        axes[1].imshow(swir_crop)
        axes[1].set_title(
            "SWIR false colour\n"
            "Red=B12, Green=B11, Blue=B08"
        )

        for ax in axes:
            ax.scatter(
                [x], [y],
                marker="+",
                s=180,
                c="yellow",
                linewidths=2.5,
            )
            ax.axis("off")

        n = float(nir[row, col])
        s1 = float(swir1[row, col])
        s2 = float(swir2[row, col])

        ratio1 = s1 / n if n > 0 else np.nan
        ratio2 = s2 / n if n > 0 else np.nan

        fig.suptitle(
            f"{sample_id} | row={row}, col={col}\n"
            f"B08={n:.4f}, B11={s1:.4f}, B12={s2:.4f} | "
            f"B11/B08={ratio1:.4f}, B12/B08={ratio2:.4f}"
        )
        fig.tight_layout()

        output_path = OUTPUT_DIR / f"{sample_id}_swir.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summaries.append({
            "sample_id": sample_id,
            "row": row,
            "col": col,
            "nir_b08": n,
            "swir1_b11": s1,
            "swir2_b12": s2,
            "swir1_nir_ratio": ratio1,
            "swir2_nir_ratio": ratio2,
        })

        print(f"Saved: {output_path}")

    summary_path = OUTPUT_DIR / "swir_sample_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print(f"\nSummary saved: {summary_path}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
