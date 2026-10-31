
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"

SPECTRAL_PATH = LIVE / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
MASK_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
DEBRIS_PATH = PRED / "rgi2000_v7_0_g_15_00467_debris_affected_mask.tif"

SCALE = 10000.0
SWIR1_MIN = 0.35
SWIR2_MIN = 0.30
NIR_MAX = 0.55
MIN_COMPONENT_PIXELS = 20


def describe(name, values):
    values = values[np.isfinite(values)]

    if values.size == 0:
        print(f"{name}: no valid values")
        return

    print(
        f"{name}: "
        f"min={np.min(values):.4f}, "
        f"p25={np.percentile(values, 25):.4f}, "
        f"median={np.median(values):.4f}, "
        f"p75={np.percentile(values, 75):.4f}, "
        f"p90={np.percentile(values, 90):.4f}, "
        f"max={np.max(values):.4f}"
    )


def main():
    with rasterio.open(SPECTRAL_PATH) as src:
        if src.count != 3:
            raise ValueError(f"Expected 3 bands, got {src.count}")

        b08, b11, b12 = [
            src.read(i).astype(np.float32) / SCALE
            for i in (1, 2, 3)
        ]

        spectral_grid = (
            src.shape,
            src.transform,
            src.crs,
        )

    with rasterio.open(MASK_PATH) as src:
        glacier = src.read(1) > 0
        mask_grid = (
            src.shape,
            src.transform,
            src.crs,
        )

    with rasterio.open(DEBRIS_PATH) as src:
        existing_debris = src.read(1) > 0

    if spectral_grid != mask_grid:
        raise ValueError("Spectral image and glacier mask grids differ.")

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

    pass_ratio1 = ratio1 >= SWIR1_MIN
    pass_ratio2 = ratio2 >= SWIR2_MIN
    pass_nir = b08 <= NIR_MAX

    raw_candidates = (
        valid
        & pass_ratio1
        & pass_ratio2
        & pass_nir
    )

    print("\n=== DEBRIS THRESHOLD DIAGNOSTICS ===")
    print(f"Glacier pixels: {int(glacier.sum())}")
    print(f"Valid glacier pixels: {int(valid.sum())}")
    print(f"Existing cleaned debris pixels: {int(existing_debris.sum())}")
    print(f"Raw threshold candidates: {int(raw_candidates.sum())}")

    print("\n=== PIXELS PASSING EACH RULE ===")
    print(f"Pass SWIR1/NIR >= {SWIR1_MIN}: {int((valid & pass_ratio1).sum())}")
    print(f"Pass SWIR2/NIR >= {SWIR2_MIN}: {int((valid & pass_ratio2).sum())}")
    print(f"Pass NIR <= {NIR_MAX}: {int((valid & pass_nir).sum())}")
    print(f"Pass all three rules: {int(raw_candidates.sum())}")

    print("\n=== REFLECTANCE DISTRIBUTIONS ===")
    describe("B08 NIR", b08[valid])
    describe("B11 SWIR1", b11[valid])
    describe("B12 SWIR2", b12[valid])

    print("\n=== RATIO DISTRIBUTIONS ===")
    describe("B11/B08", ratio1[valid])
    describe("B12/B08", ratio2[valid])

    print("\n=== CANDIDATE PIXEL VALUES ===")
    if raw_candidates.any():
        describe("Candidate B08", b08[raw_candidates])
        describe("Candidate B11/B08", ratio1[raw_candidates])
        describe("Candidate B12/B08", ratio2[raw_candidates])
    else:
        print("No pixels pass all three rules.")

    print("\n=== DIAGNOSTIC ONLY ===")
    print("No thresholds or raster files were changed.")
    print("Passing a threshold is not proof of debris cover.")


if __name__ == "__main__":
    main()
