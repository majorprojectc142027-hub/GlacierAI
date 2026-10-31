
import json
from pathlib import Path

import rasterio

from backend.app.services.rgi_service import find_glacier_by_name
from backend.scripts.historical_analysis import (
    download_historical_image,
    apply_rgi_polygon,
)
from backend.scripts.predict_geotiff import predict_geotiff


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RGI_ID = "RGI2000-v7.0-G-15-00467"
GLACIER_SLUG = RGI_ID.lower()

HISTORICAL_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
    / GLACIER_SLUG
)

RESULTS_PATH = HISTORICAL_DIR / "historical_results.json"
TEST_DIR = HISTORICAL_DIR / "september-2025-test"
SATELLITE_DIR = TEST_DIR / "satellite"
PREDICTION_DIR = TEST_DIR / "predictions"

OBSERVATION = {
    "datetime": "2025-09-22T05:30:31Z",
    "id": (
        "S2C_MSIL2A_20250922T051701_N0511_R062_"
        "T44RLV_20250922T103422.SAFE"
    ),
    "cloud_cover": 0.9,
}

TEST_STEM = "gangotri_2025-09-22_test"


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Historical results not found: {RESULTS_PATH}"
        )

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        historical_results = json.load(file)

    glacier = historical_results.get("glacier", {})
    if glacier.get("rgi_id") != RGI_ID:
        raise RuntimeError(
            "Historical results do not match the expected RGI ID."
        )

    rgi_glacier = find_glacier_by_name(RGI_ID)
    if not rgi_glacier:
        raise RuntimeError(f"RGI glacier not found: {RGI_ID}")

    geometry = rgi_glacier.get("geometry")
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        raise RuntimeError("RGI geometry is missing or invalid.")

    latitude = float(rgi_glacier["latitude"])
    longitude = float(rgi_glacier["longitude"])

    SATELLITE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    # Use a unique slug and directory to protect existing yearly imagery.
    test_slug = TEST_STEM

    print("Downloading selected Sentinel-2 observation...")
    download_info = download_historical_image(
        observation=OBSERVATION,
        geometry=geometry,
        latitude=latitude,
        longitude=longitude,
        output_dir=SATELLITE_DIR,
        glacier_slug=test_slug,
    )

    image_path = Path(download_info["path"])
    if not image_path.exists():
        raise FileNotFoundError(
            f"Downloaded image was not created: {image_path}"
        )

    print("\nRunning the existing U-Net model...")
    prediction = predict_geotiff(
        image_path=image_path,
        output_stem=TEST_STEM,
        rgi_geometry=geometry,
    )

    raw_mask_path = Path(prediction["mask_path"])
    if not raw_mask_path.exists():
        raise FileNotFoundError(
            f"Prediction mask was not created: {raw_mask_path}"
        )

    # Store the clipped mask only in the isolated test directory.
    clipped_mask_path = (
        PREDICTION_DIR / f"{TEST_STEM}_rgi_mask.tif"
    )

    analysis = apply_rgi_polygon(
        mask_path=raw_mask_path,
        geometry=geometry,
        output_path=clipped_mask_path,
    )

    with rasterio.open(clipped_mask_path) as src:
        mask = src.read(1)
        pixel_area_m2 = abs(src.transform.a * src.transform.e)
        glacier_pixels = int((mask > 0).sum())
        area_km2 = glacier_pixels * pixel_area_m2 / 1_000_000

    report = {
        "rgi_id": RGI_ID,
        "observation_date": OBSERVATION["datetime"],
        "scene_id": OBSERVATION["id"],
        "catalog_cloud_percent": OBSERVATION["cloud_cover"],
        "model": prediction.get("model", "U-Net"),
        "checkpoint": prediction.get("checkpoint"),
        "threshold": prediction.get("threshold"),
        "glacier_pixels": glacier_pixels,
        "pixel_area_m2": pixel_area_m2,
        "area_km2": round(area_km2, 6),
        "satellite_path": str(image_path),
        "raw_prediction_mask": str(raw_mask_path),
        "rgi_clipped_mask": str(clipped_mask_path),
        "rgi_polygon_applied": True,
    }

    report_path = TEST_DIR / "september_2025_test_result.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n========== TEST RESULT ==========")
    print("Scene:", OBSERVATION["id"])
    print("Date:", OBSERVATION["datetime"])
    print("Cloud cover:", OBSERVATION["cloud_cover"], "%")
    print("Glacier pixels:", glacier_pixels)
    print("Estimated area:", round(area_km2, 6), "km²")
    print("Clipped mask:", clipped_mask_path)
    print("Test report:", report_path)
    print("Original historical results were not modified.")
    print("=================================")


if __name__ == "__main__":
    main()
