
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import sieve


ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"

SPECTRAL_PATH = LIVE / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
MASK_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"

SCALE = 10000.0
MIN_COMPONENT_PIXELS = 20


def main():
    with rasterio.open(SPECTRAL_PATH) as src:
        b08, b11, b12 = [
            src.read(i).astype(np.float32) / SCALE
            for i in (1, 2, 3)
        ]
        transform = src.transform
        spectral_grid = (src.shape, src.transform, src.crs)

    with rasterio.open(MASK_PATH) as src:
        glacier = src.read(1) > 0
        mask_grid = (src.shape, src.transform, src.crs)

    if spectral_grid != mask_grid:
        raise ValueError("Spectral and glacier mask grids differ.")

    valid = (
        glacier
        & np.isfinite(b08)
        & np.isfinite(b11)
        & np.isfinite(b12)
        & (b08 > 0)
        & (b11 > 0)
        & (b12 > 0)
    )

    ratio1 = b11 / (b08 + 1e-6)
    ratio2 = b12 / (b08 + 1e-6)

    pixel_area_km2 = (
        abs(transform.a) * abs(transform.e) / 1_000_000
    )
    glacier_pixels = int(glacier.sum())
    glacier_area = glacier_pixels * pixel_area_km2

    # Original settings are included as the baseline.
    scenarios = [
        (0.35, 0.30, 0.55),
        (0.30, 0.25, 0.50),
        (0.30, 0.25, 0.55),
        (0.35, 0.30, 0.50),
        (0.40, 0.35, 0.55),
        (0.40, 0.35, 0.60),
    ]

    print("\nTHRESHOLD SENSITIVITY — NOT ACCURACY VALIDATION")
    print(f"Glacier pixels: {glacier_pixels}")
    print(f"Glacier area: {glacier_area:.4f} km²")
    print()
    print(
        f"{'SWIR1':>7} {'SWIR2':>7} {'NIR max':>8} "
        f"{'Raw px':>9} {'Clean px':>10} "
        f"{'Area km²':>10} {'Glacier %':>10}"
    )

    for t1, t2, nir_max in scenarios:
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

        cleaned &= glacier

        raw_count = int(raw.sum())
        clean_count = int(cleaned.sum())
        debris_area = clean_count * pixel_area_km2
        percentage = clean_count / glacier_pixels * 100

        print(
            f"{t1:7.2f} {t2:7.2f} {nir_max:8.2f} "
            f"{raw_count:9d} {clean_count:10d} "
            f"{debris_area:10.4f} {percentage:9.4f}%"
        )

    print("\nNo source files or existing masks were modified.")


if __name__ == "__main__":
    main()
