
from pathlib import Path

import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box


ROOT = Path(r"D:\GlacierAI")
DATA = ROOT / "datasets"
LIVE = DATA / "live"
PRED = LIVE / "predictions"

INVENTORY = DATA / "processed" / "global_rgi.gpkg"

GLACIER_ID = "RGI2000-v7.0-G-15-00467"
PREFIX = "rgi2000_v7.0_g_15_00467_tile_"


def main():
    print("=" * 65)
    print("GLACIERAI SATELLITE GEOGRAPHIC VERIFICATION")
    print("=" * 65)

    if not INVENTORY.exists():
        raise FileNotFoundError(
            f"RGI inventory not found: {INVENTORY}"
        )

    inventory = gpd.read_file(INVENTORY)

    glacier_rows = inventory[
        inventory["rgi_id"].astype(str) == GLACIER_ID
    ]

    if glacier_rows.empty:
        raise ValueError(
            f"Glacier {GLACIER_ID} not found in inventory."
        )

    glacier = glacier_rows.iloc[0]
    glacier_geometry = gpd.GeoSeries(
        [glacier.geometry],
        crs=inventory.crs,
    )

    print("\nGlacier:", GLACIER_ID)
    print("Inventory CRS:", inventory.crs)
    print("Inventory area km2:", glacier.get("area_km2"))
    print("Original geographic bounds:", glacier.geometry.bounds)

    for tile_id in ("001", "002", "003"):
        name = PREFIX + tile_id
        satellite_path = LIVE / f"{name}.tif"
        prediction_path = PRED / f"{name}_glacier_mask.tif"

        print("\n" + "-" * 65)
        print("TILE", tile_id)
        print("-" * 65)

        if not satellite_path.exists():
            print("MISSING SATELLITE:", satellite_path)
            continue

        with rasterio.open(satellite_path) as src:
            print("File:", satellite_path.name)
            print("Dimensions:", src.width, "x", src.height)
            print("Bands:", src.count)
            print("CRS:", src.crs)
            print("Transform:", src.transform)
            print("Bounds:", src.bounds)
            print("Pixel size:", src.res)
            print("NoData:", src.nodata)
            print("Band descriptions:", src.descriptions)
            print("Band scales:", src.scales)
            print("Band offsets:", src.offsets)

            for band_index in range(1, min(src.count, 3) + 1):
                band = src.read(band_index)
                valid = np.isfinite(band)

                print(
                    f"Band {band_index}: "
                    f"min={float(np.min(band[valid])):.3f}, "
                    f"max={float(np.max(band[valid])):.3f}, "
                    f"mean={float(np.mean(band[valid])):.3f}, "
                    f"nonzero={int(np.count_nonzero(band))}"
                )

            if src.crs is None:
                print("ERROR: satellite CRS is missing.")
                continue

            # Compare the raster footprint against the original glacier.
            glacier_projected = glacier_geometry.to_crs(src.crs).iloc[0]

            raster_footprint = box(
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            )

            intersection = glacier_projected.intersection(
                raster_footprint
            )

            print(
                "Raster footprint intersects glacier:",
                not intersection.is_empty,
            )

            if not intersection.is_empty:
                print(
                    "Intersection area in raster CRS (km2):",
                    round(intersection.area / 1_000_000, 4),
                )

            print(
                "Raster footprint covered by glacier (%):",
                round(
                    100 * intersection.area / raster_footprint.area,
                    4,
                ),
            )

        if prediction_path.exists():
            with rasterio.open(prediction_path) as pred:
                with rasterio.open(satellite_path) as sat:
                    print(
                        "Prediction grid matches satellite:",
                        pred.shape == sat.shape
                        and pred.crs == sat.crs
                        and pred.transform == sat.transform,
                    )
        else:
            print("Prediction TIFF missing:", prediction_path)

    print("\nVerification complete. No files were modified.")


if __name__ == "__main__":
    main()
