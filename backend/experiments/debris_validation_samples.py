
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "datasets" / "live"
PRED = DATA / "predictions"

RGB_PATH = DATA / "rgi2000_v7_0_g_15_00467_debris_rgb.tif"
SPECTRAL_PATH = DATA / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
GLACIER_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
DEBRIS_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_affected_mask.tif"

OUTPUT_CSV = PRED / "debris_validation_samples.csv"
OUTPUT_IMAGE = PRED / "debris_validation_sample_map.png"

SAMPLE_PER_GROUP = 50
RANDOM_SEED = 42


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(), src.profile.copy()


def main():
    rgb, rgb_profile = read_raster(RGB_PATH)
    spectral, spectral_profile = read_raster(SPECTRAL_PATH)
    glacier, glacier_profile = read_raster(GLACIER_PATH)
    current, current_profile = read_raster(DEBRIS_PATH)

    # Check that all inputs are on the same raster grid.
    profiles = [
        spectral_profile,
        glacier_profile,
        current_profile,
    ]

    for profile in profiles:
        if (
            profile["height"] != rgb_profile["height"]
            or profile["width"] != rgb_profile["width"]
            or profile["transform"] != rgb_profile["transform"]
            or profile["crs"] != rgb_profile["crs"]
        ):
            raise ValueError(
                "Raster grids differ. Align the inputs before sampling."
            )

    rgb = np.moveaxis(rgb[:3], 0, -1).astype(np.float32)
    spectral = spectral.astype(np.float32)

    glacier = glacier[0] > 0
    current = current[0] > 0

    # Spectral bands are expected in B08, B11, B12 order.
    
# Sentinel-2 reflectance values are stored as scaled integers.
# Convert to reflectance units before applying the NIR threshold.
    nir, swir1, swir2 = spectral[:3].astype(np.float32)

    REFLECTANCE_SCALE = 10000.0

    nir /= REFLECTANCE_SCALE
    swir1 /= REFLECTANCE_SCALE
    swir2 /= REFLECTANCE_SCALE

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

    # Experimental thresholds from our comparison.
    experimental_raw = (
        valid
        & (ratio1 >= 0.30)
        & (ratio2 >= 0.25)
        & (nir <= 0.55)
    )

    # Match the cleanup method used in the comparison experiment.
    from rasterio.features import sieve

    experimental = (
        sieve(
            experimental_raw.astype(np.uint8),
            size=20,
            connectivity=8,
        ) > 0
    ) & valid

    groups = {
        "current_detection": current & valid,
        "additional_experimental": (
            experimental & ~current & valid
        ),
        "neither_detects": (
            ~current & ~experimental & valid
        ),
    }

    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for group_name, group_mask in groups.items():
        rows, cols = np.where(group_mask)

        count = min(SAMPLE_PER_GROUP, len(rows))
        if count == 0:
            print(f"{group_name}: no eligible pixels")
            continue

        selected = rng.choice(
            len(rows), size=count, replace=False
        )

        for idx in selected:
            row = int(rows[idx])
            col = int(cols[idx])

            records.append({
                "sample_id": f"S{len(records) + 1:03d}",
                "group": group_name,
                "row": row,
                "col": col,
                "nir_b08": float(nir[row, col]),
                "swir1_b11": float(swir1[row, col]),
                "swir2_b12": float(swir2[row, col]),
                "swir1_nir_ratio": float(ratio1[row, col]),
                "swir2_nir_ratio": float(ratio2[row, col]),
                "reference_label": "",
                "review_notes": "",
            })

        print(f"{group_name}: selected {count} samples")

    if not records:
        raise ValueError("No eligible validation samples found.")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    # Display sample locations on the RGB image.
    display = rgb.copy()
    low, high = np.nanpercentile(display, [2, 98])
    display = np.clip((display - low) / (high - low + 1e-6), 0, 1)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(display)

    styles = {
        "current_detection": ("red", "Current"),
        "additional_experimental": ("orange", "Additional"),
        "neither_detects": ("lime", "Neither"),
    }

    for group_name, (color, label) in styles.items():
        subset = df[df["group"] == group_name]
        ax.scatter(
            subset["col"],
            subset["row"],
            s=24,
            c=color,
            label=label,
            edgecolors="black",
            linewidths=0.3,
        )

    ax.set_title("Debris Validation Sample Locations")
    ax.legend()
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"\nCSV saved: {OUTPUT_CSV}")
    print(f"Sample map saved: {OUTPUT_IMAGE}")
    print(f"Total samples: {len(df)}")
    print("\nFill reference_label after reviewing each sample.")


if __name__ == "__main__":
    main()
