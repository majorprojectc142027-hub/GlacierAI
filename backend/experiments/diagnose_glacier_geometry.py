
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "datasets" / "live" / "predictions"

GLACIER_PATH = (
    PRED / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
)
DEBRIS_PATH = (
    PRED / "rgi2000_v7_0_g_15_00467_debris_affected_mask.tif"
)

PIXEL_SIZE_M = 12.0


def main():
    for path in [GLACIER_PATH, DEBRIS_PATH]:
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

    # Label connected glacier regions using 8-neighbour connectivity.
    labels, count = ndimage.label(
        glacier,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labels.ravel())[1:]
    sizes = np.sort(sizes)[::-1]

    print("=== GLACIER MASK GEOMETRY ===")
    print(f"Raster dimensions: {height} x {width}")
    print(f"Glacier pixels: {int(glacier.sum())}")
    print(f"Connected glacier components: {count}")
    print(f"Components with >= 20 pixels: {int((sizes >= 20).sum())}")
    print(f"Components with >= 100 pixels: {int((sizes >= 100).sum())}")
    print(f"Components with >= 1000 pixels: {int((sizes >= 1000).sum())}")
    print(f"Largest component sizes: {sizes[:10].tolist()}")

    # Identify glacier boundary pixels with 8-neighbour connectivity.
    eroded = ndimage.binary_erosion(
        glacier,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary = glacier & ~eroded

    print("\n=== BOUNDARY SUMMARY ===")
    print(f"Boundary pixels: {int(boundary.sum())}")
    print(f"Debris pixels inside glacier: {int((debris & glacier).sum())}")
    print(f"Debris pixels on boundary: {int((debris & boundary).sum())}")

    # Report debris pixels by distance to the edge.
    distance = ndimage.distance_transform_edt(
        glacier,
        sampling=PIXEL_SIZE_M,
    )
    debris_inside = debris & glacier
    total = int(debris_inside.sum())

    print("\n=== DEBRIS DISTANCE CHECK ===")
    for limit in [24, 60, 120]:
        selected = debris_inside & (distance <= limit)
        n = int(selected.sum())
        pct = 100 * n / total if total else 0
        print(f"Within {limit:>3} m: {n} pixels ({pct:.2f}%)")

    print("\nDiagnostic only: no raster files were changed.")


if __name__ == "__main__":
    main()
