
from pathlib import Path
import csv
import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "datasets" / "historical" / "rgi2000-v7.0-g-15-00467"
IMAGE = BASE / "satellite" / "rgi2000-v7.0-g-15-00467_2025.tif"
PROBABILITY = BASE / "v5_evaluation" / "2025_v5_probability.npy"
REFERENCE = BASE / "v5_evaluation" / "2025_full_regional_rgi_mask.tif"
OUTPUT = BASE / "v5_evaluation"

THRESHOLDS = [0.20, 0.30, 0.40, 0.50]

def calculate_metrics(prediction, reference, valid):
    tp = int(np.count_nonzero(prediction & reference & valid))
    fp = int(np.count_nonzero(prediction & ~reference & valid))
    fn = int(np.count_nonzero(~prediction & reference & valid))
    tn = int(np.count_nonzero(~prediction & ~reference & valid))

    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": int(np.count_nonzero(prediction & valid))
    }

def main():
    print("V5 THRESHOLD SENSITIVITY EXPERIMENT")
    print("Evaluation only - checkpoint will not be modified.")

    probability = np.load(PROBABILITY).astype(np.float32)

    with rasterio.open(IMAGE) as src:
        valid = src.read(4) > 0

    with rasterio.open(REFERENCE) as src:
        reference_data = src.read(1)

    if probability.shape != valid.shape or reference_data.shape != valid.shape:
        raise ValueError("Input dimensions do not match")

    valid &= np.isin(reference_data, [0, 1])
    reference = reference_data == 1

    pixel_area_km2 = None
    with rasterio.open(IMAGE) as src:
        pixel_area_km2 = abs(
            src.transform.a * src.transform.e
            - src.transform.b * src.transform.d
        ) / 1_000_000

    results = []

    for threshold in THRESHOLDS:
        prediction = probability >= threshold
        scores = calculate_metrics(prediction, reference, valid)
        scores["threshold"] = threshold
        scores["predicted_area_km2"] = (
            scores["predicted_pixels"] * pixel_area_km2
        )
        results.append(scores)

        print(f"\nTHRESHOLD: {threshold:.2f}")
        print(f"Dice:      {scores['dice']:.4f}")
        print(f"IoU:       {scores['iou']:.4f}")
        print(f"Precision: {scores['precision']:.4f}")
        print(f"Recall:    {scores['recall']:.4f}")
        print(f"Predicted area: {scores['predicted_area_km2']:.4f} km²")

    csv_path = OUTPUT / "2025_v5_threshold_sensitivity.csv"
    fields = [
        "threshold", "dice", "iou", "precision", "recall",
        "tp", "fp", "fn", "tn", "predicted_pixels",
        "predicted_area_km2"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    metrics_to_plot = ["dice", "iou", "precision", "recall"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for metric in metrics_to_plot:
        axes[0].plot(
            [r["threshold"] for r in results],
            [r[metric] for r in results],
            marker="o",
            label=metric.capitalize()
        )

    axes[0].set_title("V5 Metrics at Different Thresholds")
    axes[0].set_xlabel("Prediction Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        [r["threshold"] for r in results],
        [r["predicted_area_km2"] for r in results],
        marker="o"
    )
    axes[1].set_title("Predicted Glacier Area")
    axes[1].set_xlabel("Prediction Threshold")
    axes[1].set_ylabel("Area (km²)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT / "2025_v5_threshold_sensitivity.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()

    print("\nEXPERIMENT COMPLETE")
    print("CSV:", csv_path)
    print("Chart:", plot_path)
    print("V5 checkpoint was not modified.")

if __name__ == "__main__":
    main()
