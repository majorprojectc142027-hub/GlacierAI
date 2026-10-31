
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import sieve


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "datasets" / "live"
PRED = DATA / "predictions"

SPECTRAL_PATH = DATA / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
GLACIER_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
CURRENT_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_affected_mask.tif"

OUTPUT_PATH = PRED / "debris_spatial_validation_samples.csv"

SAMPLES_PER_GROUP = 50
MIN_DISTANCE = 50  # pixels
SEED = 42


def read_band(path):
    with rasterio.open(path) as src:
        return src.read(), src.profile.copy()


def spatial_sample(mask, count, min_distance, rng):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return []

    order = rng.permutation(len(rows))
    selected = []

    # Grid-based rejection sampling avoids comparing every point
    # with every other point.
    cell_size = min_distance
    grid = {}

    for idx in order:
        r, c = int(rows[idx]), int(cols[idx])
        cell = (r // cell_size, c // cell_size)

        too_close = False

        for gr in range(cell[0] - 1, cell[0] + 2):
            for gc in range(cell[1] - 1, cell[1] + 2):
                for sr, sc in grid.get((gr, gc), []):
                    if (r - sr) ** 2 + (c - sc) ** 2 < min_distance ** 2:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break

        if not too_close:
            selected.append((r, c))
            grid.setdefault(cell, []).append((r, c))

            if len(selected) >= count:
                break

    return selected


def main():
    rng = np.random.default_rng(SEED)

    spectral, sp = read_band(SPECTRAL_PATH)
    glacier, gp = read_band(GLACIER_PATH)
    current, cp = read_band(CURRENT_PATH)

    for profile in (gp, cp):
        if (
            profile["height"] != sp["height"]
            or profile["width"] != sp["width"]
            or profile["transform"] != sp["transform"]
            or profile["crs"] != sp["crs"]
        ):
            raise ValueError("Input raster grids do not match.")

    nir, swir1, swir2 = spectral[:3].astype(np.float32)

    # Convert scaled Sentinel-2 reflectance to reflectance units.
    nir /= 10000.0
    swir1 /= 10000.0
    swir2 /= 10000.0

    glacier = glacier[0] > 0
    current = current[0] > 0

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
        swir1, nir, out=np.zeros_like(nir), where=nir > 0
    )
    ratio2 = np.divide(
        swir2, nir, out=np.zeros_like(nir), where=nir > 0
    )

    experimental_raw = (
        valid
        & (ratio1 >= 0.30)
        & (ratio2 >= 0.25)
        & (nir <= 0.55)
    )

    experimental = (
        sieve(
            experimental_raw.astype(np.uint8),
            size=20,
            connectivity=8,
        ) > 0
    ) & valid

    groups = {
        "current_detection": current & valid,
        "additional_experimental": experimental & ~current & valid,
        "neither_detects": ~experimental & ~current & valid,
    }

    records = []

    for group, mask in groups.items():
        samples = spatial_sample(
            mask, SAMPLES_PER_GROUP, MIN_DISTANCE, rng
        )

        print(f"{group}: selected {len(samples)} samples")

        for row, col in samples:
            records.append({
                "sample_id": f"V{len(records) + 1:03d}",
                "group": group,
                "row": row,
                "col": col,
                "reference_label": "",
                "review_notes": "",
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTotal spatially distributed samples: {len(df)}")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Minimum target separation: {MIN_DISTANCE} pixels")


if __name__ == "__main__":
    main()
