
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from rasterio.features import sieve
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "live"
PRED = DATA / "predictions"

RGB_PATH = DATA / "rgi2000_v7_0_g_15_00467_debris_rgb.tif"
SPECTRAL_PATH = DATA / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
GLACIER_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
CURRENT_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_affected_mask.tif"
CSV_PATH = PRED / "debris_spatial_validation_samples.csv"

OUTPUT_DIR = PRED / "experimental_debris_diagnostics"

SAMPLE_IDS = ["V007", "V008", "V009", "V010", "V011"]
CROP_SIZE = 256

# Experimental threshold settings used in the comparison.
SWIR1_NIR_MIN = 0.30
SWIR2_NIR_MIN = 0.25
NIR_MAX = 0.55
MIN_COMPONENT_SIZE = 20


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(), src.profile.copy()


def check_same_grid(a, b, name_a, name_b):
    if (
        a["height"] != b["height"]
        or a["width"] != b["width"]
        or a["transform"] != b["transform"]
        or a["crs"] != b["crs"]
    ):
        raise ValueError(
            f"Raster grids differ: {name_a} and {name_b}"
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [
        RGB_PATH, SPECTRAL_PATH, GLACIER_PATH,
        CURRENT_PATH, CSV_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with rasterio.open(RGB_PATH) as src:
        rgb_raw = src.read()
        rgb_profile = src.profile.copy()

    with rasterio.open(SPECTRAL_PATH) as src:
        spectral_raw = src.read()
        spectral_profile = src.profile.copy()

    with rasterio.open(GLACIER_PATH) as src:
        glacier_raw = src.read()
        glacier_profile = src.profile.copy()

    with rasterio.open(CURRENT_PATH) as src:
        current_raw = src.read()
        current_profile = src.profile.copy()

    check_same_grid(rgb_profile, spectral_profile, "RGB", "spectral")
    check_same_grid(rgb_profile, glacier_profile, "RGB", "glacier mask")
    check_same_grid(rgb_profile, current_profile, "RGB", "current mask")

    if rgb_raw.shape[0] < 3 or spectral_raw.shape[0] < 3:
        raise ValueError("RGB and spectral rasters need at least 3 bands.")

    height, width = rgb_profile["height"], rgb_profile["width"]

    # Display RGB with a consistent image-wide stretch.
    rgb = np.moveaxis(rgb_raw[:3], 0, -1).astype(np.float32)
    low, high = np.nanpercentile(rgb, [2, 98])
    rgb = np.clip((rgb - low) / (high - low + 1e-6), 0, 1)

    # The spectral raster band order is expected to be B08, B11, B12.
    nir, swir1, swir2 = spectral_raw[:3].astype(np.float32)

    # Convert integer-scaled reflectance to reflectance units.
    # If your source is already floating-point reflectance, don't divide.
    if np.issubdtype(spectral_raw.dtype, np.integer):
        nir /= 10000.0
        swir1 /= 10000.0
        swir2 /= 10000.0

    glacier = glacier_raw[0] > 0
    current = current_raw[0] > 0

    valid = (
        glacier
        & np.isfinite(nir)
        & np.isfinite(swir1)
        & np.isfinite(swir2)
        & (nir > 0)
        & (swir1 >= 0)
        & (swir2 >= 0)
    )

    ratio1 = np.divide(
        swir1, nir,
        out=np.zeros_like(nir),
        where=nir > 0,
    )
    ratio2 = np.divide(
        swir2, nir,
        out=np.zeros_like(nir),
        where=nir > 0,
    )

    experimental_raw = (
        valid
        & (ratio1 >= SWIR1_NIR_MIN)
        & (ratio2 >= SWIR2_NIR_MIN)
        & (nir <= NIR_MAX)
    )

    experimental = (
        sieve(
            experimental_raw.astype(np.uint8),
            size=MIN_COMPONENT_SIZE,
            connectivity=8,
        ) > 0
    ) & valid

    df = pd.read_csv(CSV_PATH)
    df = df[df["sample_id"].astype(str).isin(SAMPLE_IDS)]

    if set(df["sample_id"].astype(str)) != set(SAMPLE_IDS):
        raise ValueError(
            "Some requested samples are missing from the spatial CSV."
        )

    half = CROP_SIZE // 2
    summaries = []

    for _, sample in df.iterrows():
        sample_id = str(sample["sample_id"])
        row = int(sample["row"])
        col = int(sample["col"])

        if not (0 <= row < height and 0 <= col < width):
            print(f"Skipping {sample_id}: coordinates out of bounds")
            continue

        r0 = max(0, row - half)
        r1 = min(height, row + half)
        c0 = max(0, col - half)
        c1 = min(width, col + half)

        rgb_crop = rgb[r0:r1, c0:c1]
        glacier_crop = glacier[r0:r1, c0:c1]
        current_crop = current[r0:r1, c0:c1]
        experimental_crop = experimental[r0:r1, c0:c1]
        extra_crop = experimental_crop & ~current_crop

        # Reflectance values at the exact sampled pixel.
        n = float(nir[row, col])
        s1 = float(swir1[row, col])
        s2 = float(swir2[row, col])
        r1_value = float(ratio1[row, col])
        r2_value = float(ratio2[row, col])

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))

        axes[0].imshow(rgb_crop)
        axes[0].set_title("Sentinel-2 RGB")

        axes[1].imshow(rgb_crop)
        axes[1].imshow(
            np.ma.masked_where(~glacier_crop, glacier_crop),
            cmap=ListedColormap(["cyan"]),
            alpha=0.50,
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Glacier mask (cyan)")

        axes[2].imshow(rgb_crop)
        axes[2].imshow(
            np.ma.masked_where(~current_crop, current_crop),
            cmap=ListedColormap(["red"]),
            alpha=0.70,
            vmin=0,
            vmax=1,
        )
        axes[2].set_title("Current debris (red)")

        axes[3].imshow(rgb_crop)
        axes[3].imshow(
            np.ma.masked_where(~extra_crop, extra_crop),
            cmap=ListedColormap(["orange"]),
            alpha=0.85,
            vmin=0,
            vmax=1,
        )
        axes[3].set_title("Additional experimental (orange)")

        x, y = col - c0, row - r0
        for ax in axes[:4]:
            ax.scatter(
                [x], [y],
                s=100, c="yellow", marker="+", linewidths=2,
            )
            ax.axis("off")

        axes[4].axis("off")
        axes[4].text(
            0.02, 0.98,
            f"{sample_id}\n"
            f"Sample row/col: {row}, {col}\n\n"
            f"Inside glacier mask: {bool(glacier[row, col])}\n"
            f"Current mask: {bool(current[row, col])}\n"
            f"Experimental mask: {bool(experimental[row, col])}\n"
            f"Additional-only: {bool(extra_crop[y, x])}\n\n"
            f"NIR B08: {n:.4f}\n"
            f"SWIR1 B11: {s1:.4f}\n"
            f"SWIR2 B12: {s2:.4f}\n"
            f"B11/B08: {r1_value:.4f}\n"
            f"B12/B08: {r2_value:.4f}\n\n"
            f"Thresholds:\n"
            f"B11/B08 >= {SWIR1_NIR_MIN}\n"
            f"B12/B08 >= {SWIR2_NIR_MIN}\n"
            f"B08 <= {NIR_MAX}",
            va="top",
            fontsize=10,
            family="monospace",
        )

        fig.suptitle(f"{sample_id} — debris candidate inspection")
        fig.tight_layout()

        output_path = OUTPUT_DIR / f"{sample_id}_diagnostic.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summaries.append({
            "sample_id": sample_id,
            "row": row,
            "col": col,
            "inside_glacier_mask": bool(glacier[row, col]),
            "current_mask": bool(current[row, col]),
            "experimental_mask": bool(experimental[row, col]),
            "additional_only": bool(extra_crop[y, x]),
            "nir_b08": n,
            "swir1_b11": s1,
            "swir2_b12": s2,
            "swir1_nir_ratio": r1_value,
            "swir2_nir_ratio": r2_value,
        })

        print(f"Saved: {output_path}")

    summary_path = OUTPUT_DIR / "sample_spectral_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print(f"\nSummary saved: {summary_path}")
    print(f"Diagnostics folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
