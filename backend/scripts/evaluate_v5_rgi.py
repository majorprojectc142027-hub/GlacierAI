
from pathlib import Path
import sys
import json
import csv
import cv2
import numpy as np
import rasterio
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.scripts import predict_geotiff as inference

HISTORICAL_DIR = PROJECT_ROOT / "datasets" / "historical" / "rgi2000-v7.0-g-15-00467"
SATELLITE_DIR = HISTORICAL_DIR / "satellite"
REFERENCE_DIR = HISTORICAL_DIR / "predictions"
OUTPUT_DIR = HISTORICAL_DIR / "v5_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025]
THRESHOLD = inference.THRESHOLD

def calculate_metrics(prediction, reference, evaluation_mask):
    pred = prediction.astype(bool) & evaluation_mask
    ref = reference.astype(bool) & evaluation_mask
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref & evaluation_mask))
    fn = int(np.count_nonzero(~pred & ref & evaluation_mask))
    tn = int(np.count_nonzero(~pred & ~ref & evaluation_mask))
    dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "evaluated_pixels": int(evaluation_mask.sum())
    }

def save_geotiff(path, array, profile, valid_mask):
    output = array.astype(np.uint8).copy()
    output[~valid_mask] = 255
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="lzw"
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(output, 1)

def process_year(year, model):
    stem = f"rgi2000-v7.0-g-15-00467_{year}"
    image_path = SATELLITE_DIR / f"{stem}.tif"
    reference_path = REFERENCE_DIR / f"{stem}_rgi_mask.tif"

    if not image_path.exists():
        raise FileNotFoundError(f"Satellite image missing: {image_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"RGI reference mask missing: {reference_path}")

    print("\n" + "=" * 65)
    print(f"EVALUATING YEAR {year}")
    print("=" * 65)

    with rasterio.open(image_path) as src:
        if src.count < 4:
            raise ValueError(f"{image_path.name}: expected 4 bands, found {src.count}")

        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)
        validity_band = src.read(4)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

        valid_mask = (
            np.isfinite(red)
            & np.isfinite(green)
            & np.isfinite(blue)
            & np.isfinite(validity_band)
            & (validity_band > 0)
        )

    with rasterio.open(reference_path) as ref_src:
        if ref_src.count < 1:
            raise ValueError(f"No mask band in {reference_path.name}")
        if ref_src.shape != (height, width):
            raise ValueError(f"{year}: satellite and RGI mask dimensions differ")
        if ref_src.crs != crs:
            raise ValueError(f"{year}: satellite and RGI mask CRS differ")
        if not ref_src.transform.almost_equals(transform):
            raise ValueError(f"{year}: satellite and RGI mask transforms differ")
        reference = ref_src.read(1)

    labelled = np.isin(reference, [0, 1])
    evaluation_mask = valid_mask & labelled

    if not np.any(evaluation_mask):
        raise ValueError(f"{year}: no valid labelled pixels to evaluate")

    rgb = inference.build_rgb(red, green, blue, valid_mask)
    rgb[~valid_mask] = 0

    print(f"Image dimensions: {width} x {height}")
    print(f"Valid satellite pixels: {int(valid_mask.sum()):,}")
    print(f"Evaluation pixels: {int(evaluation_mask.sum()):,}")
    print(f"RGI glacier pixels: {int(np.count_nonzero((reference == 1) & evaluation_mask)):,}")
    print("Running V5 tiled inference...")

    probability = inference.infer_probabilities(model, rgb)
    probability[~valid_mask] = 0.0

    raw_mask = (probability >= THRESHOLD) & valid_mask
    cleaned_mask = inference.clean_mask(raw_mask, valid_mask)

    raw_metrics = calculate_metrics(raw_mask, reference == 1, evaluation_mask)
    cleaned_metrics = calculate_metrics(cleaned_mask, reference == 1, evaluation_mask)

    pixel_area_m2 = inference.get_pixel_area_m2(transform, crs)
    reference_pixels = int(np.count_nonzero((reference == 1) & evaluation_mask))
    raw_pixels = int(np.count_nonzero(raw_mask & valid_mask))
    cleaned_pixels = int(np.count_nonzero(cleaned_mask & valid_mask))

    result = {
        "year": year,
        "image": image_path.name,
        "reference_mask": reference_path.name,
        "threshold": THRESHOLD,
        "pixel_area_m2": pixel_area_m2,
        "valid_satellite_pixels": int(valid_mask.sum()),
        "evaluated_pixels": int(evaluation_mask.sum()),
        "reference_glacier_pixels": reference_pixels,
        "raw_predicted_glacier_pixels": raw_pixels,
        "cleaned_predicted_glacier_pixels": cleaned_pixels,
        "reference_area_km2": reference_pixels * pixel_area_m2 / 1_000_000,
        "raw_predicted_area_km2": raw_pixels * pixel_area_m2 / 1_000_000,
        "cleaned_predicted_area_km2": cleaned_pixels * pixel_area_m2 / 1_000_000,
        "raw_metrics": raw_metrics,
        "cleaned_metrics": cleaned_metrics
    }

    save_geotiff(OUTPUT_DIR / f"{year}_v5_raw_mask.tif", raw_mask, profile, valid_mask)
    save_geotiff(OUTPUT_DIR / f"{year}_v5_cleaned_mask.tif", cleaned_mask, profile, valid_mask)

    np.save(OUTPUT_DIR / f"{year}_v5_probability.npy", probability.astype(np.float32))

    reference_png = np.zeros((height, width), dtype=np.uint8)
    reference_png[reference == 1] = 255
    cv2.imwrite(str(OUTPUT_DIR / f"{year}_rgi_reference.png"), reference_png)

    raw_overlay = rgb.copy()
    raw_overlay[raw_mask] = (
        0.55 * raw_overlay[raw_mask] + 0.45 * np.array([255, 0, 0])
    ).astype(np.uint8)
    raw_overlay[~valid_mask] = 0
    cv2.imwrite(
        str(OUTPUT_DIR / f"{year}_v5_overlay.png"),
        cv2.cvtColor(raw_overlay, cv2.COLOR_RGB2BGR)
    )

    print(f"\n{year} RESULTS")
    print(f"Raw Dice:      {raw_metrics['dice']:.4f}")
    print(f"Raw IoU:       {raw_metrics['iou']:.4f}")
    print(f"Raw Precision: {raw_metrics['precision']:.4f}")
    print(f"Raw Recall:    {raw_metrics['recall']:.4f}")
    print(f"Cleaned Dice:  {cleaned_metrics['dice']:.4f}")
    print(f"Cleaned IoU:   {cleaned_metrics['iou']:.4f}")
    print(f"Reference area: {result['reference_area_km2']:.4f} km²")
    print(f"Raw predicted area: {result['raw_predicted_area_km2']:.4f} km²")
    print(f"Saved outputs in: {OUTPUT_DIR}")

    return result

def pooled_metrics(results, metric_key):
    totals = {
        key: sum(item[metric_key][key] for item in results)
        for key in [
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "evaluated_pixels"
        ]
    }
    tp = totals["true_positive"]
    fp = totals["false_positive"]
    fn = totals["false_negative"]
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        **totals,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall
    }

def main():
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_unet_v5.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"V5 checkpoint not found: {checkpoint_path}")

    print("GLOBAL GLACIER MONITORING - V5 RGI EVALUATION")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Evaluation only: no training and no checkpoint modification.")
    print(f"Threshold: {THRESHOLD}")

    model = inference.load_model()
    results = []

    for year in YEARS:
        results.append(process_year(year, model))

    summary = {
        "model": "U-Net V5",
        "checkpoint": checkpoint_path.name,
        "threshold": THRESHOLD,
        "years": YEARS,
        "reference_note": "Agreement with RGI inventory reference masks; not independent annual boundary validation.",
        "per_year_results": results,
        "pooled_raw_metrics": pooled_metrics(results, "raw_metrics"),
        "pooled_cleaned_metrics": pooled_metrics(results, "cleaned_metrics")
    }

    json_path = OUTPUT_DIR / "v5_rgi_evaluation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / "v5_rgi_evaluation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Year", "Raw Dice", "Raw IoU", "Raw Precision", "Raw Recall",
            "Cleaned Dice", "Cleaned IoU", "Reference Area km2",
            "Raw Predicted Area km2"
        ])
        for item in results:
            writer.writerow([
                item["year"],
                item["raw_metrics"]["dice"],
                item["raw_metrics"]["iou"],
                item["raw_metrics"]["precision"],
                item["raw_metrics"]["recall"],
                item["cleaned_metrics"]["dice"],
                item["cleaned_metrics"]["iou"],
                item["reference_area_km2"],
                item["raw_predicted_area_km2"]
            ])

    print("\n" + "=" * 65)
    print("V5 EVALUATION COMPLETE")
    print("=" * 65)
    for label, key in [
        ("RAW PREDICTIONS", "pooled_raw_metrics"),
        ("CLEANED PREDICTIONS", "pooled_cleaned_metrics")
    ]:
        metrics = summary[key]
        print(f"\n{label}")
        print(f"Dice:      {metrics['dice']:.4f}")
        print(f"IoU:       {metrics['iou']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")

    print(f"\nJSON report: {json_path}")
    print(f"CSV report:  {csv_path}")
    print("\nV5 checkpoint was not modified.")

if __name__ == "__main__":
    main()
