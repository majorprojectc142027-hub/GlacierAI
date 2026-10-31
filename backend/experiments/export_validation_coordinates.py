
from pathlib import Path

import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]

PRED_DIR = ROOT / "datasets" / "live" / "predictions"

CSV_PATH = PRED_DIR / "debris_spatial_validation_samples.csv"

RASTER_PATH = (
    ROOT / "datasets" / "live"
    / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
)

OUTPUT_PATH = PRED_DIR / "debris_validation_coordinates.csv"

# Samples to inspect first in Google Earth
SAMPLES_TO_REVIEW = ["V007", "V008", "V009", "V010", "V011"]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Sample CSV not found: {CSV_PATH}")

    if not RASTER_PATH.exists():
        raise FileNotFoundError(f"Spectral raster not found: {RASTER_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = {"sample_id", "group", "row", "col"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found in the input CSV.")

    with rasterio.open(RASTER_PATH) as src:
        if src.crs is None:
            raise ValueError("Spectral raster has no CRS.")

        transformer = Transformer.from_crs(
            src.crs,
            "EPSG:4326",
            always_xy=True,
        )

        latitudes = []
        longitudes = []

        for _, sample in df.iterrows():
            row = int(sample["row"])
            col = int(sample["col"])

            if not (0 <= row < src.height and 0 <= col < src.width):
                raise ValueError(
                    f"Sample {sample['sample_id']} is outside the raster: "
                    f"row={row}, col={col}"
                )

            x, y = src.xy(row, col)
            longitude, latitude = transformer.transform(x, y)

            latitudes.append(latitude)
            longitudes.append(longitude)

    df["latitude"] = latitudes
    df["longitude"] = longitudes

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Exported {len(df)} sample coordinates")
    print(f"Saved: {OUTPUT_PATH}")

    print("\n=== FIRST FIVE SAMPLES ===")
    print(
        df[["sample_id", "group", "latitude", "longitude"]]
        .head()
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\n=== GOOGLE EARTH REVIEW SAMPLES ===")

    review = df[df["sample_id"].isin(SAMPLES_TO_REVIEW)].copy()

    review["review_order"] = review["sample_id"].map(
        {sample_id: i for i, sample_id in enumerate(SAMPLES_TO_REVIEW)}
    )
    review = review.sort_values("review_order")

    found = set(review["sample_id"])
    missing_samples = set(SAMPLES_TO_REVIEW) - found

    print(
        review[
            ["sample_id", "group", "latitude", "longitude"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    if missing_samples:
        print(f"\nWARNING: Samples not found: {sorted(missing_samples)}")

    print("\nCopy latitude, longitude into Google Earth's search bar.")
    print("Review each point against imagery from a suitable date.")


if __name__ == "__main__":
    main()
