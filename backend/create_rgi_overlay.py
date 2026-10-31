
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize


ROOT = Path(r"D:\GlacierAI")
PREDICTION_DIR = ROOT / "datasets" / "live" / "predictions"
OUTPUT_DIR = ROOT / "datasets" / "live" / "large_glacier"
RGI_PATH = ROOT / "datasets" / "processed" / "global_rgi.gpkg"
GLACIER_ID = "RGI2000-v7.0-G-15-00467"

# Actual prefix used in your tile filenames.
FILE_PREFIX = "rgi2000_v7.0_g_15_00467_tile_"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RGI_PATH.exists():
        print("ERROR: RGI inventory not found:", RGI_PATH)
        return

    print("Loading RGI reference boundary...")
    inventory = gpd.read_file(RGI_PATH)

    glacier = inventory[
        inventory["rgi_id"].astype(str).str.strip() == GLACIER_ID
    ]

    if glacier.empty:
        print("ERROR: RGI glacier not found:", GLACIER_ID)
        return

    # Match the real filenames shown in your terminal.
    mask_files = sorted(
        PREDICTION_DIR.glob(f"{FILE_PREFIX}*_glacier_mask.tif")
    )

    print("Matching masks found:", len(mask_files))

    if not mask_files:
        print("ERROR: No tile masks found.")
        return

    saved_count = 0

    for mask_path in mask_files:
        rgb_path = PREDICTION_DIR / mask_path.name.replace(
            "_glacier_mask.tif", "_full_rgb.png"
        )

        print("\nProcessing:", mask_path.name)

        if not rgb_path.exists():
            print("RGB image missing:", rgb_path.name)
            continue

        with rasterio.open(mask_path) as src:
            predicted = src.read(1) == 1
            transform = src.transform
            crs = src.crs
            shape = predicted.shape

        rgb = np.array(Image.open(rgb_path).convert("RGB"))

        if rgb.shape[:2] != shape:
            print("Dimension mismatch:", rgb.shape[:2], shape)
            continue

        # Project the RGI outline into the mask's CRS.
        reference_glacier = glacier.to_crs(crs)

        reference = rasterize(
            [
                (geom, 1)
                for geom in reference_glacier.geometry
                if geom is not None and not geom.is_empty
            ],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

        overlay = rgb.copy()

        # Red = predicted glacier outside the RGI boundary.
        false_positive = predicted & ~reference
        overlay[false_positive] = [255, 40, 40]

        # Magenta = predicted glacier inside the RGI boundary.
        overlap = predicted & reference
        overlay[overlap] = [255, 0, 255]

        # Yellow = RGI boundary, using NumPy only.
        if np.any(reference):
            boundary = np.zeros_like(reference, dtype=bool)

            boundary[1:, :] |= reference[1:, :] != reference[:-1, :]
            boundary[:-1, :] |= reference[:-1, :] != reference[1:, :]
            boundary[:, 1:] |= reference[:, 1:] != reference[:, :-1]
            boundary[:, :-1] |= reference[:, :-1] != reference[:, 1:]

            boundary &= reference
            overlay[boundary] = [255, 255, 0]

        output_path = OUTPUT_DIR / mask_path.name.replace(
            "_glacier_mask.tif", "_rgi_comparison.png"
        )

        Image.fromarray(overlay).save(output_path)
        saved_count += 1

        intersection = int(np.count_nonzero(predicted & reference))
        union = int(np.count_nonzero(predicted | reference))
        iou = intersection / union if union else 0.0

        print("Saved:", output_path)
        print("Predicted pixels:", int(predicted.sum()))
        print("Reference pixels:", int(reference.sum()))
        print("Overlap pixels:", intersection)
        print(f"Tile IoU: {iou:.4f} ({iou * 100:.2f}%)")

    print(f"\nFinished. Overlays saved: {saved_count}")
    print("Yellow = RGI boundary")
    print("Red = prediction outside RGI")
    print("Magenta = prediction overlapping RGI")


if __name__ == "__main__":
    main()
