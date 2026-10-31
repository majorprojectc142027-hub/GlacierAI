
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


# Existing project folders
PREDICTION_DIR = Path(
    r"D:\GlacierAI\datasets\live\predictions"
)

OUTPUT_DIR = Path(
    r"D:\GlacierAI\datasets\live\large_glacier"
)

GLACIER_ID = "rgi2000_v7.0_g_15_00467"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(
        PREDICTION_DIR.glob(
            f"{GLACIER_ID}_tile_*_glacier_mask.tif"
        )
    )

    if not mask_files:
        print("ERROR: No glacier mask files found.")
        return

    print("Masks found:", len(mask_files))

    for mask_path in mask_files:
        rgb_path = PREDICTION_DIR / mask_path.name.replace(
            "_glacier_mask.tif",
            "_full_rgb.png",
        )

        if not rgb_path.exists():
            print("RGB image missing:", rgb_path.name)
            continue

        with rasterio.open(mask_path) as src:
            mask = src.read(1) == 1

        rgb = np.array(
            Image.open(rgb_path).convert("RGB")
        )

        if mask.shape != rgb.shape[:2]:
            print("DIMENSION MISMATCH:", mask_path.name)
            print("Mask:", mask.shape)
            print("RGB:", rgb.shape[:2])
            continue

        # Blend predicted glacier pixels with red.
        overlay = rgb.copy()
        red = np.array([255, 40, 40], dtype=np.float32)

        overlay[mask] = (
            rgb[mask].astype(np.float32) * 0.45
            + red * 0.55
        ).clip(0, 255).astype(np.uint8)

        output_path = OUTPUT_DIR / mask_path.name.replace(
            "_glacier_mask.tif",
            "_overlay.png",
        )

        Image.fromarray(overlay).save(output_path)

        print("Saved:", output_path)
        print("Predicted glacier pixels:", int(mask.sum()))

    print("\nFinished creating overlays.")


if __name__ == "__main__":
    main()
