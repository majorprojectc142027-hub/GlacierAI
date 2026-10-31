
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"

GLACIER_ID = "rgi2000_v7_0_g_15_00467"

GLACIER_PATH = PRED / f"{GLACIER_ID}_debris_glacier_mask.tif"
DEBRIS_PATH = PRED / f"{GLACIER_ID}_debris_affected_mask.tif"
SAMPLES_PATH = PRED / "debris_spatial_validation_samples.csv"

OUTPUT_PATH = PRED / "debris_boundary_diagnostic.csv"

# Raster pixel size is approximately 12 metres.
PIXEL_SIZE_M = 12.0
DISTANCE_BINS_M = [24, 60, 120]


def main():
    for path in [GLACIER_PATH, DEBRIS_PATH, SAMPLES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with rasterio.open(GLACIER_PATH) as src:
        glacier = src.read(1) > 0
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    with rasterio.open(DEBRIS_PATH) as src:
        if (
            src.height != height
            or src.width != width
            or src.transform != transform
            or src.crs != crs
        ):
            raise ValueError("Glacier and debris masks are not aligned.")
        debris = src.read(1) > 0

    # Distance inside glacier to the nearest non-glacier pixel.
    # Sampling distances are approximate because pixel dimensions
    # are assumed to be 12 metres.
    inside_distance = distance_transform_edt(
        glacier, sampling=PIXEL_SIZE_M
    )

    debris_inside = debris & glacier
    total = int(debris_inside.sum())

    print("=== DEBRIS DISTANCE FROM GLACIER EDGE ===")
    print(f"Glacier pixels: {int(glacier.sum())}")
    print(f"Debris-mask pixels inside glacier: {total}")

    rows = []
    for distance_m in DISTANCE_BINS_M:
        count = int(
            (debris_inside & (inside_distance <= distance_m)).sum()
        )
        pct = 100 * count / total if total else 0
        print(
            f"Within {distance_m:>3} m of edge: "
            f"{count} pixels ({pct:.2f}%)"
        )
        rows.append({
            "analysis": "debris_edge_distance",
            "distance_m": distance_m,
            "pixels": count,
            "percentage_of_debris": pct,
        })

    samples = pd.read_csv(SAMPLES_PATH)
    required = {"sample_id", "row", "col"}
    if required.issubset(samples.columns):
        print("\n=== VALIDATION SAMPLE LOCATIONS ===")
        for _, sample in samples.iterrows():
            row, col = int(sample["row"]), int(sample["col"])
            if 0 <= row < height and 0 <= col < width:
                rows.append({
                    "analysis": "sample_location",
                    "sample_id": str(sample["sample_id"]),
                    "row": row,
                    "col": col,
                    "inside_glacier": bool(glacier[row, col]),
                    "predicted_debris": bool(debris[row, col]),
                    "distance_to_edge_m": float(
                        inside_distance[row, col]
                    ),
                })

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print("No masks, model weights, or API files were modified.")


if __name__ == "__main__":
    main()
