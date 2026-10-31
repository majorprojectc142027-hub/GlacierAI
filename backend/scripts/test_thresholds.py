
from pathlib import Path
import json

import cv2
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "datasets" / "live" / "predictions"

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def clean_mask(binary_mask, valid_mask):
    """Use the same basic cleaning settings as predict_geotiff.py."""
    mask = binary_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=1
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] >= 4:
            cleaned[labels == component_id] = 1

    cleaned[~valid_mask] = 0
    return cleaned


def main():
    probability_files = sorted(
        PRED_DIR.glob("*_probability.npy")
    )

    if not probability_files:
        print("No saved probability maps found.")
        return

    for probability_path in probability_files:
        stem = probability_path.name.removesuffix(
            "_probability.npy"
        )

        analysis_path = PRED_DIR / f"{stem}_analysis.json"
        mask_path = PRED_DIR / f"{stem}_glacier_mask.tif"
        rgi_path = PRED_DIR / f"{stem}_rgi_mask.png"

        # Only evaluate tiles with all required saved files.
        if not all(p.exists() for p in (
            analysis_path, mask_path, rgi_path
        )):
            continue

        probability = np.load(probability_path)

        with open(analysis_path, encoding="utf-8") as f:
            analysis = json.load(f)

        with rasterio.open(mask_path) as src:
            saved_mask = src.read(1)
            pixel_area = analysis["pixel_area_m2"]

        if probability.shape != saved_mask.shape:
            print(f"Skipping {stem}: image shapes do not match.")
            continue

        # 255 marks invalid pixels in the saved GeoTIFF mask.
        valid = saved_mask != 255

        rgi_image = cv2.imread(
            str(rgi_path), cv2.IMREAD_GRAYSCALE
        )
        if rgi_image is None or rgi_image.shape != probability.shape:
            print(f"Skipping {stem}: RGI mask unavailable or misaligned.")
            continue

        rgi = (rgi_image > 0) & valid

        print(f"\n{'=' * 72}")
        print(stem)
        print("Threshold | Area km² | IoU % | Precision % | Recall %")
        print("-" * 72)

        for threshold in THRESHOLDS:
            raw = (probability >= threshold) & valid
            prediction = clean_mask(raw, valid).astype(bool)

            intersection = np.count_nonzero(prediction & rgi)
            union = np.count_nonzero(prediction | rgi)
            predicted_count = np.count_nonzero(prediction)
            reference_count = np.count_nonzero(rgi)

            iou = 100 * intersection / union if union else 0
            precision = (
                100 * intersection / predicted_count
                if predicted_count else 0
            )
            recall = (
                100 * intersection / reference_count
                if reference_count else 0
            )
            area = predicted_count * pixel_area / 1_000_000

            print(
                f"{threshold:8.2f} | {area:8.2f} | "
                f"{iou:6.2f} | {precision:11.2f} | "
                f"{recall:7.2f}"
            )

    print("\nTest complete. Original masks and checkpoints were not changed.")


if __name__ == "__main__":
    main()
