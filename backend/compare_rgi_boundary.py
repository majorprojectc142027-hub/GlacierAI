
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


ROOT = Path(r"D:\GlacierAI")
MASK_PATH = ROOT / "datasets/live/large_glacier/rgi2000_v7.0_g_15_00467_full_mask.tif"
RGI_PATH = ROOT / "datasets/processed/global_rgi.gpkg"
RGI_ID = "RGI2000-v7.0-G-15-00467"


def main():
    if not MASK_PATH.exists():
        print("ERROR: Prediction mask not found:", MASK_PATH)
        return

    if not RGI_PATH.exists():
        print("ERROR: RGI dataset not found:", RGI_PATH)
        return

    print("Loading RGI inventory...")
    glaciers = gpd.read_file(RGI_PATH)
    print("Available columns:", list(glaciers.columns))

    id_column = next(
        (
            col
            for col in ["rgi_id", "RGIId", "RGI_ID", "glac_id"]
            if col in glaciers.columns
        ),
        None,
    )

    if id_column is None:
        print("Could not identify the RGI ID column.")
        return

    selected = glaciers[
        glaciers[id_column].astype(str).str.strip() == RGI_ID
    ]

    if selected.empty:
        print("RGI ID not found:", RGI_ID)
        print("Check the available columns and actual ID format.")
        return

    with rasterio.open(MASK_PATH) as src:
        prediction = src.read(1) == 1
        transform = src.transform
        crs = src.crs
        pixel_area = abs(transform.a * transform.e)

        selected = selected.to_crs(crs)

        reference = rasterize(
            [
                (geom, 1)
                for geom in selected.geometry
                if geom is not None and not geom.is_empty
            ],
            out_shape=prediction.shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        ).astype(bool)

    pred_count = int(np.count_nonzero(prediction))
    ref_count = int(np.count_nonzero(reference))
    intersection = int(
        np.count_nonzero(prediction & reference)
    )
    union = int(
        np.count_nonzero(prediction | reference)
    )

    predicted_area = pred_count * pixel_area / 1_000_000
    reference_area = ref_count * pixel_area / 1_000_000
    overlap_area = intersection * pixel_area / 1_000_000
    iou = intersection / union if union else 0.0

    print("\n========== RGI COMPARISON ==========")
    print("Glacier:", RGI_ID)
    print(f"Predicted area: {predicted_area:.3f} km²")
    print(f"Reference area in raster: {reference_area:.3f} km²")
    print(f"Overlap area: {overlap_area:.3f} km²")
    print(f"IoU: {iou:.4f} ({iou * 100:.2f}%)")
    print("====================================")
    print("Comparison is limited to the current raster extent.")


if __name__ == "__main__":
    main()
