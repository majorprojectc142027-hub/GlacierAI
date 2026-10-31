
from pathlib import Path
import csv
import json
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "datasets" / "historical" / "rgi2000-v7.0-g-15-00467"
SATELLITE_DIR = BASE / "satellite"
OUTPUT = BASE / "v5_evaluation"
RGI_FILE = ROOT / "datasets" / "raw" / "RGI" / "RGI2000-v7.0-G-15_south_asia_east" / "RGI2000-v7.0-G-15_south_asia_east.shp"

YEARS = [2023, 2024, 2025]
THRESHOLDS = [0.20, 0.30, 0.40, 0.50]


def calculate_metrics(tp, fp, fn, tn, predicted_pixels, reference_pixels, pixel_area_km2=None):
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    scores = {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "predicted_pixels": int(predicted_pixels),
        "reference_pixels": int(reference_pixels)
    }

    if pixel_area_km2 is not None:
        scores["predicted_area_km2"] = float(predicted_pixels * pixel_area_km2)
        scores["reference_area_km2"] = float(reference_pixels * pixel_area_km2)

    return scores


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("GLOBAL GLACIER MONITORING - V5 MULTI-YEAR RGI EVALUATION")
    print("Evaluation only. No training or checkpoint modification.")
    print("=" * 65)

    if not RGI_FILE.exists():
        raise FileNotFoundError(f"RGI shapefile not found: {RGI_FILE}")

    gdf = gpd.read_file(RGI_FILE)

    if gdf.crs is None:
        raise ValueError("RGI shapefile has no CRS")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("RGI shapefile contains no valid geometries")

    yearly_data = {}

    for year in YEARS:
        print(f"\nPreparing {year}...")

        stem = f"rgi2000-v7.0-g-15-00467_{year}"
        image_path = SATELLITE_DIR / f"{stem}.tif"
        probability_path = OUTPUT / f"{year}_v5_probability.npy"

        if not image_path.exists():
            raise FileNotFoundError(f"Satellite image not found: {image_path}")

        if not probability_path.exists():
            raise FileNotFoundError(f"Probability map not found: {probability_path}")

        probability = np.load(probability_path).astype(np.float32)

        with rasterio.open(image_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()
            bounds = src.bounds

            if src.count < 4:
                raise ValueError(f"{year}: expected 4 satellite bands, found {src.count}")

            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)
            validity_band = src.read(4)

            valid = (
                np.isfinite(red)
                & np.isfinite(green)
                & np.isfinite(blue)
                & np.isfinite(validity_band)
                & (validity_band > 0)
                & np.isfinite(probability)
            )

            pixel_area_km2 = abs(
                transform.a * transform.e - transform.b * transform.d
            ) / 1_000_000

        if probability.shape != (height, width):
            raise ValueError(
                f"{year}: probability shape {probability.shape} "
                f"does not match satellite image {(height, width)}"
            )

        if crs is None:
            raise ValueError(f"{year}: satellite image has no CRS")

        year_gdf = gdf.to_crs(crs)

        image_bounds = box(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top
        )

        intersecting = year_gdf[
            year_gdf.geometry.intersects(image_bounds)
        ]

        if intersecting.empty:
            raise ValueError(f"{year}: no RGI polygons intersect satellite image")

        reference = rasterize(
            [(geom, 1) for geom in intersecting.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False
        ).astype(bool)

        valid_reference_pixels = int(np.count_nonzero(reference & valid))

        reference_output = reference.astype(np.uint8)
        reference_output[~valid] = 255

        mask_path = OUTPUT / f"{year}_full_regional_rgi_mask.tif"

        mask_profile = profile.copy()
        mask_profile.update(
            count=1,
            dtype="uint8",
            nodata=255,
            compress="lzw"
        )

        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(reference_output, 1)

        yearly_data[year] = {
            "probability": probability,
            "reference": reference,
            "valid": valid,
            "pixel_area_km2": pixel_area_km2,
            "polygon_count": len(intersecting),
            "valid_pixels": int(valid.sum()),
            "reference_pixels": valid_reference_pixels
        }

        print(f"Intersecting RGI polygons: {len(intersecting)}")
        print(f"Valid satellite pixels: {int(valid.sum()):,}")
        print(f"Reference glacier pixels in valid area: {valid_reference_pixels:,}")
        print(f"Pixel area: {pixel_area_km2:.8f} km²")
        print(f"Saved reference mask: {mask_path.name}")

    all_results = []
    pooled_results = []

    for threshold in THRESHOLDS:
        print(f"\n{'-' * 55}")
        print(f"THRESHOLD: {threshold:.2f}")
        print(f"{'-' * 55}")

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        total_predicted = 0
        total_reference = 0

        for year in YEARS:
            data = yearly_data[year]

            probability = data["probability"]
            reference = data["reference"]
            valid = data["valid"]

            prediction = probability >= threshold

            tp = int(np.count_nonzero(prediction & reference & valid))
            fp = int(np.count_nonzero(prediction & ~reference & valid))
            fn = int(np.count_nonzero(~prediction & reference & valid))
            tn = int(np.count_nonzero(~prediction & ~reference & valid))

            predicted_pixels = tp + fp
            reference_pixels = tp + fn

            scores = calculate_metrics(
                tp,
                fp,
                fn,
                tn,
                predicted_pixels,
                reference_pixels,
                data["pixel_area_km2"]
            )

            row = {
                "year": year,
                "threshold": threshold,
                **scores
            }

            all_results.append(row)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            total_predicted += predicted_pixels
            total_reference += reference_pixels

            print(
                f"{year}: Dice={scores['dice']:.4f} | "
                f"IoU={scores['iou']:.4f} | "
                f"Precision={scores['precision']:.4f} | "
                f"Recall={scores['recall']:.4f} | "
                f"Predicted area={scores['predicted_area_km2']:.3f} km²"
            )

        # Pooled metrics combine pixel counts across years.
        # Area fields are intentionally excluded because they would
        # sum repeated observations of the same geographic region.
        pooled = calculate_metrics(
            total_tp,
            total_fp,
            total_fn,
            total_tn,
            total_predicted,
            total_reference
        )

        pooled["threshold"] = threshold
        pooled_results.append(pooled)

        print(
            f"POOLED: Dice={pooled['dice']:.4f} | "
            f"IoU={pooled['iou']:.4f} | "
            f"Precision={pooled['precision']:.4f} | "
            f"Recall={pooled['recall']:.4f}"
        )

    # Save yearly results CSV
    csv_path = OUTPUT / "v5_multiyear_threshold_results.csv"

    fields = [
        "year", "threshold", "dice", "iou", "precision", "recall",
        "tp", "fp", "fn", "tn", "predicted_pixels",
        "reference_pixels", "predicted_area_km2", "reference_area_km2"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    # Save pooled results CSV
    pooled_csv = OUTPUT / "v5_multiyear_pooled_results.csv"

    pooled_fields = [
        "threshold", "dice", "iou", "precision", "recall",
        "tp", "fp", "fn", "tn", "predicted_pixels", "reference_pixels"
    ]

    with open(pooled_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=pooled_fields)
        writer.writeheader()
        writer.writerows(pooled_results)

    # Save JSON report
    report = {
        "checkpoint": "best_unet_v5.pth",
        "years": YEARS,
        "thresholds": THRESHOLDS,
        "reference": "All intersecting regional RGI G-15 glacier polygons",
        "yearly_results": all_results,
        "pooled_results": pooled_results,
        "note": (
            "Metrics measure agreement with RGI inventory polygons, "
            "not independently verified annual glacier boundaries. "
            "Pooled scores are pixel-weighted across valid pixels. "
            "Pooled area values are intentionally excluded."
        )
    }

    json_path = OUTPUT / "v5_multiyear_threshold_results.json"

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    # Generate comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for year in YEARS:
        rows = [
            row for row in all_results
            if row["year"] == year
        ]

        axes[0].plot(
            [row["threshold"] for row in rows],
            [row["dice"] for row in rows],
            marker="o",
            label=str(year)
        )

    axes[0].set_title("V5 Dice Score by Year")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Dice Score")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title="Year")

    axes[1].plot(
        [row["threshold"] for row in pooled_results],
        [row["dice"] for row in pooled_results],
        marker="o",
        label="Dice"
    )

    axes[1].plot(
        [row["threshold"] for row in pooled_results],
        [row["iou"] for row in pooled_results],
        marker="s",
        label="IoU"
    )

    axes[1].set_title("Pooled V5 Performance")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    chart_path = OUTPUT / "v5_multiyear_threshold_comparison.png"

    plt.savefig(
        chart_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 65)
    print("MULTI-YEAR EVALUATION COMPLETE")
    print("=" * 65)
    print("Yearly CSV:", csv_path)
    print("Pooled CSV:", pooled_csv)
    print("JSON report:", json_path)
    print("Chart:", chart_path)
    print("V5 checkpoint was not modified.")


if __name__ == "__main__":
    main()
