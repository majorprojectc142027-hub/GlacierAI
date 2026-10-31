
from pathlib import Path
import json
import cv2
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "datasets" / "historical" / "rgi2000-v7.0-g-15-00467"
IMAGE = BASE / "satellite" / "rgi2000-v7.0-g-15-00467_2025.tif"
OLD_MASK = BASE / "predictions" / "rgi2000-v7.0-g-15-00467_2025_rgi_mask.tif"
PROBABILITY = BASE / "v5_evaluation" / "2025_v5_probability.npy"
RGI_FILE = PROJECT_ROOT / "datasets" / "raw" / "RGI" / "RGI2000-v7.0-G-15_south_asia_east" / "RGI2000-v7.0-G-15_south_asia_east.shp"
OUTPUT = BASE / "v5_evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50

def metrics(pred, ref, valid):
    pred = pred.astype(bool) & valid
    ref = ref.astype(bool) & valid
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref & valid))
    fn = int(np.count_nonzero(~pred & ref & valid))
    tn = int(np.count_nonzero(~pred & ~ref & valid))
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evaluated_pixels": int(valid.sum())
    }

def main():
    print("V5 EVALUATION WITH COMPLETE REGIONAL RGI GLACIER POLYGONS")

    probability = np.load(PROBABILITY).astype(np.float32)

    with rasterio.open(IMAGE) as src:
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)
        validity_band = src.read(4)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        height, width = src.height, src.width
        bounds = src.bounds

        valid = (
            np.isfinite(red)
            & np.isfinite(green)
            & np.isfinite(blue)
            & np.isfinite(validity_band)
            & (validity_band > 0)
        )

    if probability.shape != (height, width):
        raise ValueError("Probability map dimensions do not match satellite image")

    gdf = gpd.read_file(RGI_FILE)

    if gdf.crs is None:
        raise ValueError("RGI shapefile has no CRS")

    gdf = gdf.to_crs(crs)
    image_box = gpd.GeoSeries.from_wkt(
        [f"POLYGON(({bounds.left} {bounds.bottom},{bounds.right} {bounds.bottom},{bounds.right} {bounds.top},{bounds.left} {bounds.top},{bounds.left} {bounds.bottom}))"],
        crs=crs
    ).iloc[0]

    intersecting = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geometry.intersects(image_box)
    ].copy()

    if intersecting.empty:
        raise ValueError("No RGI glacier polygons intersect the image")

    print("Regional RGI glacier polygons intersecting image:", len(intersecting))

    geometries = [
        (geom, 1)
        for geom in intersecting.geometry
        if geom is not None and not geom.is_empty
    ]

    reference = rasterize(
        geometries,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False
    )

    evaluation_mask = valid.copy()
    prediction = probability >= THRESHOLD

    full_metrics = metrics(prediction, reference == 1, evaluation_mask)

    with rasterio.open(OLD_MASK) as old_src:
        if old_src.shape != (height, width):
            raise ValueError("Original single-glacier mask dimensions differ")
        old_reference = old_src.read(1)

    old_valid = evaluation_mask & np.isin(old_reference, [0, 1])
    old_metrics = metrics(prediction, old_reference == 1, old_valid)

    print("\nREFERENCE MASK COUNTS")
    print("All-RGI glacier pixels:", int(reference.sum()))
    print("Original single-glacier pixels:", int(np.count_nonzero((old_reference == 1) & old_valid)))
    print("Valid evaluation pixels:", int(evaluation_mask.sum()))

    print("\nORIGINAL SINGLE-GLACIER REFERENCE")
    for key in ["dice", "iou", "precision", "recall"]:
        print(f"{key}: {old_metrics[key]:.4f}")

    print("\nCOMPLETE REGIONAL RGI REFERENCE")
    for key in ["dice", "iou", "precision", "recall"]:
        print(f"{key}: {full_metrics[key]:.4f}")

    print("\nCONFUSION COUNTS")
    for key in ["tp", "fp", "fn", "tn"]:
        print(f"{key}: {full_metrics[key]:,}")

    reference_path = OUTPUT / "2025_full_regional_rgi_mask.tif"
    output_mask = reference.copy()
    output_mask[~valid] = 255
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(reference_path, "w", **out_profile) as dst:
        dst.write(output_mask, 1)

    cv2.imwrite(
        str(OUTPUT / "2025_full_regional_rgi_mask.png"),
        reference * 255
    )

    import matplotlib.pyplot as plt
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb_display = np.zeros_like(rgb, dtype=np.uint8)

    for i in range(3):
        values = rgb[:, :, i]
        band_valid = valid & np.isfinite(values)
        if band_valid.any():
            p2, p98 = np.percentile(values[band_valid], [2, 98])
            if p98 > p2:
                rgb_display[:, :, i] = (
                    np.clip((values - p2) / (p98 - p2), 0, 1) * 255
                ).astype(np.uint8)

    overlay = rgb_display.copy()
    overlay[reference == 1] = (
        0.55 * overlay[reference == 1]
        + 0.45 * np.array([0, 255, 0])
    ).astype(np.uint8)
    overlay[~valid] = 0

    cv2.imwrite(
        str(OUTPUT / "2025_full_rgi_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    summary = {
        "year": 2025,
        "checkpoint": "best_unet_v5.pth",
        "threshold": THRESHOLD,
        "reference": "All intersecting RGI G-15 glacier polygons",
        "intersecting_polygon_count": len(intersecting),
        "single_glacier_metrics": old_metrics,
        "full_regional_rgi_metrics": full_metrics,
        "note": "RGI inventory agreement, not independent annual boundary validation."
    }

    report = OUTPUT / "2025_full_rgi_evaluation.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved full reference mask:", reference_path)
    print("Saved overlay:", OUTPUT / "2025_full_rgi_overlay.png")
    print("Saved report:", report)
    print("\nV5 checkpoint was not modified.")

if __name__ == "__main__":
    main()
