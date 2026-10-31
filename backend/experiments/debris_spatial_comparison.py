
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.features import sieve


ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"

RGB_PATH = LIVE / "rgi2000_v7_0_g_15_00467_debris_rgb.tif"
SPECTRAL_PATH = LIVE / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
GLACIER_MASK_PATH = (
    PRED / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
)
EXISTING_DEBRIS_PATH = (
    PRED / "rgi2000_v7_0_g_15_00467_debris_affected_mask.tif"
)

OUTPUT_PATH = PRED / "gangotri_debris_threshold_comparison.png"

SCALE = 10000.0
MIN_COMPONENT_PIXELS = 20

# Current production thresholds
CURRENT = (0.35, 0.30, 0.55)

# Experimental thresholds: deliberately more permissive
EXPERIMENTAL = (0.30, 0.25, 0.55)


def same_grid(a, b):
    return (
        a.width == b.width
        and a.height == b.height
        and a.crs == b.crs
        and a.transform == b.transform
    )


def main():
    required = (
        RGB_PATH,
        SPECTRAL_PATH,
        GLACIER_MASK_PATH,
        EXISTING_DEBRIS_PATH,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with rasterio.open(RGB_PATH) as rgb_src:
        if rgb_src.count < 3:
            raise ValueError("RGB raster must have at least 3 bands.")

        rgb_profile = rgb_src.profile.copy()
        rgb = np.moveaxis(
            rgb_src.read([1, 2, 3]), 0, -1
        ).astype(np.float32)

        # Contrast stretch for visualization only.
        display_rgb = np.zeros_like(rgb, dtype=np.float32)

        for band in range(3):
            values = rgb[:, :, band]
            valid_values = values[np.isfinite(values)]

            if valid_values.size:
                low, high = np.percentile(valid_values, [2, 98])
                if high > low:
                    display_rgb[:, :, band] = np.clip(
                        (values - low) / (high - low), 0, 1
                    )

        with rasterio.open(GLACIER_MASK_PATH) as mask_src:
            if not same_grid(rgb_src, mask_src):
                raise ValueError(
                    "Glacier mask and RGB raster grids differ."
                )
            glacier = mask_src.read(1) > 0

        with rasterio.open(EXISTING_DEBRIS_PATH) as debris_src:
            if not same_grid(rgb_src, debris_src):
                raise ValueError(
                    "Existing debris mask and RGB grids differ."
                )
            existing_debris = debris_src.read(1) > 0

        with rasterio.open(SPECTRAL_PATH) as spectral_src:
            if not same_grid(rgb_src, spectral_src):
                raise ValueError(
                    "Spectral raster and RGB grids differ."
                )

            if spectral_src.count != 3:
                raise ValueError("Expected B08, B11 and B12 bands.")

            b08, b11, b12 = [
                spectral_src.read(i).astype(np.float32) / SCALE
                for i in (1, 2, 3)
            ]

            ratio1 = b11 / (b08 + 1e-6)
            ratio2 = b12 / (b08 + 1e-6)

    valid = (
        glacier
        & np.isfinite(b08)
        & np.isfinite(b11)
        & np.isfinite(b12)
        & (b08 > 0)
        & (b11 > 0)
        & (b12 > 0)
    )

    def classify(thresholds):
        t1, t2, nir_max = thresholds

        raw = (
            valid
            & (ratio1 >= t1)
            & (ratio2 >= t2)
            & (b08 <= nir_max)
        )

        cleaned = sieve(
            raw.astype(np.uint8),
            size=MIN_COMPONENT_PIXELS,
            connectivity=8,
        ) > 0

        return raw, cleaned & glacier

    current_raw, current_mask = classify(CURRENT)
    experimental_raw, experimental_mask = classify(EXPERIMENTAL)

    # Highlight only the additional pixels from the experiment.
    additional = experimental_mask & ~current_mask

    pixel_area_km2 = (
        abs(rgb_profile["transform"].a)
        * abs(rgb_profile["transform"].e)
        / 1_000_000
    )

    glacier_pixels = int(glacier.sum())

    print("\n=== SPATIAL COMPARISON ===")
    print(f"Glacier pixels: {glacier_pixels}")

    for name, raw, cleaned in (
        ("Current", current_raw, current_mask),
        ("Experimental", experimental_raw, experimental_mask),
    ):
        count = int(cleaned.sum())
        area = count * pixel_area_km2
        percentage = 100 * count / glacier_pixels if glacier_pixels else 0

        print(f"\n{name} thresholds: {CURRENT if name == 'Current' else EXPERIMENTAL}")
        print(f"Raw candidates: {int(raw.sum())}")
        print(f"Cleaned candidates: {count}")
        print(f"Estimated area: {area:.6f} km²")
        print(f"Glacier percentage: {percentage:.4f}%")

    print(f"\nAdditional experimental pixels: {int(additional.sum())}")
    print(
        "Existing API mask equals current re-computation: "
        f"{np.array_equal(existing_debris, current_mask)}"
    )

    # Display all detections and the additional experimental pixels.
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].imshow(display_rgb)
    axes[0, 0].set_title("Sentinel-2 RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(display_rgb)
    cyan = np.zeros((*glacier.shape, 4), dtype=np.float32)
    cyan[glacier] = [0.0, 0.9, 1.0, 0.22]
    axes[0, 1].imshow(cyan)

    red = np.zeros((*glacier.shape, 4), dtype=np.float32)
    red[current_mask] = [1.0, 0.1, 0.1, 0.95]
    axes[0, 1].imshow(red)
    axes[0, 1].set_title("Current detections: cyan glacier, red debris")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(display_rgb)
    red_exp = np.zeros((*glacier.shape, 4), dtype=np.float32)
    red_exp[experimental_mask] = [1.0, 0.1, 0.1, 0.95]
    axes[1, 0].imshow(cyan)
    axes[1, 0].imshow(red_exp)
    axes[1, 0].set_title("Experimental detections")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(display_rgb)
    added_overlay = np.zeros((*glacier.shape, 4), dtype=np.float32)
    added_overlay[additional] = [1.0, 0.65, 0.0, 1.0]
    axes[1, 1].imshow(cyan)
    axes[1, 1].imshow(added_overlay)
    axes[1, 1].set_title("Additional experimental pixels: orange")
    axes[1, 1].axis("off")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nComparison image saved to:\n{OUTPUT_PATH}")
    print("No production masks or API files were modified.")


if __name__ == "__main__":
    main()
