from pathlib import Path
import argparse
import sys

import rasterio
from rasterio.merge import merge
import numpy as np

from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

# ==========================================================
# PROJECT IMPORTS
# ==========================================================

from backend.app.services.rgi_service import (
    find_glacier_by_name,
)

from backend.scripts.predict_geotiff import (
    predict_geotiff,
)

from backend.scripts.sentinel2_downloader import (
    download_sentinel2_tile,
    Sentinel2DownloadError,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TILE_SIZE_METERS = 20_000

DAYS_BACK = 60

PREDICTION_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "live"
    / "predictions"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "live"
    / "large_glacier"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# SAFE FILE NAME
# ==========================================================

def safe_name(name: str) -> str:

    return (
        name
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
    )


def glacier_output_key(glacier: dict) -> str:
    """
    Create a stable, collision-resistant file key.
    Prefer the RGI ID so glaciers with similar names do not
    reuse each other's cached tile predictions.
    """
    identity = glacier.get("rgi_id") or glacier.get("name") or "glacier"
    return safe_name(str(identity))


# ==========================================================
# FIND GLACIER
# ==========================================================

def get_glacier(glacier_name: str):

    print()
    print("Searching RGI...")
    print()

    glacier = find_glacier_by_name(
        query=glacier_name,
        limit=10,
    )

    if glacier is None:

        raise RuntimeError(
            f"Glacier not found: {glacier_name}"
        )

    if not glacier.get("geometry"):

        raise RuntimeError(
            "RGI glacier geometry was not returned."
        )

    return glacier


# ==========================================================
# UTM CRS HELPER
# ==========================================================

def get_utm_crs(latitude, longitude):
    """
    Return the WGS84 UTM EPSG code for a geographic coordinate.
    Gangotri is in UTM zone 44N (EPSG:32644).
    """
    zone = int((float(longitude) + 180) // 6) + 1
    zone = max(1, min(zone, 60))
    epsg = (32600 if float(latitude) >= 0 else 32700) + zone
    return f"EPSG:{epsg}"


# ==========================================================
# TRANSFORM RGI GEOMETRY
# ==========================================================

def transform_geometry_to_utm(
    geometry,
    latitude,
    longitude,
):

    target_crs = get_utm_crs(
        latitude,
        longitude,
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True,
    )

    utm_geometry = transform(
        transformer.transform,
        geometry,
    )

    return (
        utm_geometry,
        target_crs,
    )


# ==========================================================
# CREATE 20 KM TILES
# ==========================================================

def create_tiles(
    glacier_geometry,
):

    min_x, min_y, max_x, max_y = (
        glacier_geometry.bounds
    )

    tiles = []

    x = min_x

    while x < max_x:

        y = min_y

        while y < max_y:

            tile = box(
                x,
                y,
                min(
                    x + TILE_SIZE_METERS,
                    max_x,
                ),
                min(
                    y + TILE_SIZE_METERS,
                    max_y,
                ),
            )

            if tile.intersects(
                glacier_geometry
            ):

                intersection = (
                    tile.intersection(
                        glacier_geometry
                    )
                )

                if not intersection.is_empty:

                    tiles.append(
                        {
                            "bbox": [
                                tile.bounds[0],
                                tile.bounds[1],
                                tile.bounds[2],
                                tile.bounds[3],
                            ],
                            "geometry": intersection,
                        }
                    )

            y += TILE_SIZE_METERS

        x += TILE_SIZE_METERS

    return tiles


# ==========================================================
# CONVERT TILE GEOMETRY BACK TO WGS84
# ==========================================================

def tile_geometry_to_wgs84(
    tile_geometry,
    target_crs,
):

    transformer = Transformer.from_crs(
        target_crs,
        "EPSG:4326",
        always_xy=True,
    )

    return transform(
        transformer.transform,
        tile_geometry,
    )


# ==========================================================
# GET EXPECTED MASK PATH
# ==========================================================

def get_mask_path(
    tile_name: str,
):

    return (
        PREDICTION_DIR
        / f"{tile_name}_glacier_mask.tif"
    )


# ==========================================================
# PROCESS ONE TILE
# ==========================================================

def process_tile(
    tile,
    tile_index,
    glacier,
    target_crs,
    force_repredict=False,
    auto_download=True,
):

    glacier_name = glacier["name"]

    glacier_safe = glacier_output_key(glacier)

    tile_name = (
        f"{glacier_safe}_tile_"
        f"{tile_index:03d}"
    )

    image_path = (
        PROJECT_ROOT
        / "datasets"
        / "live"
        / f"{tile_name}.tif"
    )

    mask_path = get_mask_path(
        tile_name
    )

    print()
    print("==========================================")
    print(
        f" PROCESSING TILE {tile_index}"
    )
    print("==========================================")
    print()

    # ======================================================
    # REUSE EXISTING MASK UNLESS RE-PREDICTION WAS REQUESTED
    # ======================================================
    if mask_path.exists() and not force_repredict:
        print("Existing prediction mask found.")
        print("Reusing:", mask_path)

        return {
            "mask": str(mask_path),
            "tile": tile_name,
            "reused": True,
        }

    # ======================================================
    # USE EXISTING IMAGE OR DOWNLOAD FROM COPERNICUS CDSE
    # ======================================================
    satellite_metadata = None
    if not image_path.exists():
        if not auto_download:
            raise FileNotFoundError(
                "Expected satellite TIFF is missing and automatic download is disabled: "
                f"{image_path}"
            )

        print("Satellite TIFF is missing. Searching Copernicus Sentinel-2 L2A...")
        tile_geometry_wgs84 = tile_geometry_to_wgs84(
            tile["geometry"],
            target_crs,
        )
        satellite_metadata = download_sentinel2_tile(
            tile_geometry_wgs84=tile_geometry_wgs84,
            tile_bounds_projected=tuple(tile["bbox"]),
            target_crs=target_crs,
            output_path=image_path,
        )
        print("Sentinel-2 tile prepared:", image_path)
        print("Satellite metadata:", satellite_metadata)
    else:
        print("Using existing satellite image:")
        print(image_path)

    # ======================================================
    # CONVERT TILE RGI GEOMETRY TO WGS84
    # ======================================================

    tile_geometry_wgs84 = (
        tile_geometry_to_wgs84(
            tile["geometry"],
            target_crs,
        )
    )

    tile_geojson = mapping(
        tile_geometry_wgs84
    )

    # ======================================================
    # U-NET INFERENCE
    # ======================================================

    print()
    print(
        "Running U-Net prediction..."
    )

    prediction_result = (
        predict_geotiff(
            image_path=str(
                image_path
            ),
            output_stem=tile_name,
            rgi_geometry=tile_geojson,
        )
    )

    # ======================================================
    # IMPORTANT:
    # predict_geotiff creates:
    #
    # *_glacier_mask.tif
    #
    # We use the actual generated file.
    # ======================================================

    mask_path = get_mask_path(
        tile_name
    )

    if not mask_path.exists():

        raise RuntimeError(
            "U-Net completed but the "
            f"mask was not found:\n{mask_path}"
        )

    print()
    print(
        "Tile prediction complete."
    )

    print(
        "Mask:",
        mask_path,
    )

    return {
        "mask": str(mask_path),
        "tile": tile_name,
        "reused": False,
    }


# ==========================================================
# MERGE MASKS
# ==========================================================

def merge_masks(
    prediction_results,
    glacier_name,
    glacier_id=None,
):

    mask_files = []

    for result in prediction_results:

        mask_path = result.get(
            "mask"
        )

        if not mask_path:
            continue

        path = Path(
            mask_path
        )

        if path.exists():

            mask_files.append(
                path
            )

    if not mask_files:

        raise RuntimeError(
            "No glacier mask files were found."
        )

    print()
    print("==========================================")
    print(" MERGING GLACIER MASKS")
    print("==========================================")
    print()

    print(
        "Masks found:",
        len(mask_files),
    )

    # Normalize each source mask before merging. The model writes
    # glacier pixels as 1, background as 0, and NoData as 255.
    # Since 255 is numerically larger than 1, method="max" would
    # otherwise incorrectly turn NoData into glacier pixels.
    import tempfile

    datasets = []

    with tempfile.TemporaryDirectory(
        prefix="glacier_mask_merge_",
        dir=str(OUTPUT_DIR),
    ) as temporary_dir:

        normalized_files = []

        try:
            for index, path in enumerate(mask_files, start=1):
                print("Loading:", path.name)

                with rasterio.open(path) as source:
                    source_mask = source.read(1)
                    source_nodata = source.nodata

                    # Treat the dataset's declared NoData value as
                    # background for binary mask merging.
                    if source_nodata is not None:
                        source_mask[
                            source_mask == source_nodata
                        ] = 0

                    # Also explicitly clear 255, which is the
                    # NoData value observed in the current tile masks.
                    source_mask[source_mask == 255] = 0

                    # Ensure output is strictly binary: 0 or 1.
                    source_mask = (source_mask == 1).astype(np.uint8)

                    normalized_path = (
                        Path(temporary_dir)
                        / f"normalized_{index:03d}.tif"
                    )

                    profile = source.profile.copy()
                    profile.update(
                        {
                            "driver": "GTiff",
                            "dtype": "uint8",
                            "count": 1,
                            "nodata": 0,
                            "compress": "lzw",
                        }
                    )

                    with rasterio.open(
                        normalized_path,
                        "w",
                        **profile,
                    ) as normalized:
                        normalized.write(source_mask, 1)

                normalized_files.append(normalized_path)

            for normalized_path in normalized_files:
                datasets.append(rasterio.open(normalized_path))

            # Use max so a glacier pixel in an overlapping tile remains
            # glacier. NoData has already been normalized to 0.
            mosaic, mosaic_transform = merge(
                datasets,
                method="max",
                nodata=0,
            )

            reference = datasets[0]
            output_key = safe_name(str(glacier_id or glacier_name))
            output_path = OUTPUT_DIR / f"{output_key}_full_mask.tif"

            # Final binary output should contain only 0 and 1.
            merged_mask = (mosaic[0] == 1).astype(np.uint8)

            profile = reference.profile.copy()
            profile.update(
                {
                    "driver": "GTiff",
                    "height": merged_mask.shape[0],
                    "width": merged_mask.shape[1],
                    "transform": mosaic_transform,
                    "count": 1,
                    "dtype": "uint8",
                    "compress": "lzw",
                    "nodata": 0,
                }
            )

            with rasterio.open(
                output_path,
                "w",
                **profile,
            ) as destination:
                destination.write(merged_mask, 1)

            print()
            print("Full glacier mask created:")
            print(output_path)
            print(
                "Merged values:",
                np.unique(merged_mask, return_counts=True),
            )

            return output_path

        finally:
            for dataset in datasets:
                dataset.close()


# ==========================================================
# CALCULATE AREA
# ==========================================================

def calculate_area(
    mask_path,
):

    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(1)

        # The merged binary mask uses 1 for glacier and 0 for background.
        # Count only 1s, never NoData or other unexpected positive values.
        positive_pixels = int(
            np.count_nonzero(mask == 1)
        )

        pixel_width = abs(
            src.transform.a
        )

        pixel_height = abs(
            src.transform.e
        )

        pixel_area_m2 = (
            pixel_width
            * pixel_height
        )

        area_m2 = (
            positive_pixels
            * pixel_area_m2
        )

        area_km2 = (
            area_m2
            / 1_000_000
        )

    return (
        positive_pixels,
        area_m2,
        area_km2,
    )


# ==========================================================
# CREATE FINAL RGB/GLACIER OVERLAY
# ==========================================================

def create_final_overlay(
    glacier_name,
    mask_path,
    glacier_id=None,
):

    glacier_safe = safe_name(str(glacier_id or glacier_name))

    # ------------------------------------------------------
    # Find all tile RGB images
    # ------------------------------------------------------

    rgb_files = sorted(
        PREDICTION_DIR.glob(
            f"{glacier_safe}_tile_*_full_rgb.png"
        )
    )

    if not rgb_files:

        print(
            "No tile RGB images found."
        )

        return None

    print()
    print(
        "Tile RGB images available:",
        len(rgb_files),
    )

    # ------------------------------------------------------
    # Read final mask
    # ------------------------------------------------------

    with rasterio.open(
        mask_path
    ) as src:

        mask = src.read(1)

        transform_final = (
            src.transform
        )

        crs_final = src.crs

        width = src.width
        height = src.height

    # ------------------------------------------------------
    # For now, save a PNG version of
    # the final binary mask.
    # ------------------------------------------------------

    from PIL import Image

    mask_png_path = (
        OUTPUT_DIR
        / (
            glacier_safe
            + "_full_mask.png"
        )
    )

    mask_image = (
        (mask == 1)
        .astype(np.uint8)
        * 255
    )

    Image.fromarray(
        mask_image
    ).save(
        mask_png_path
    )

    print(
        "Final mask PNG:",
        mask_png_path,
    )

    return mask_png_path


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

def analyze_large_glacier(
    glacier_name: str | None = None,
    glacier_record: dict | None = None,
    force_repredict: bool = False,
    auto_download: bool = True,
):
    """
    Run tiled segmentation for a glacier, downloading missing Sentinel-2 tiles when enabled.

    API usage:
        analyze_large_glacier(glacier_record=resolved_rgi_record)

    Standalone usage:
        analyze_large_glacier("Gangotri Glacier")
    """

    print()
    print("==========================================")
    print(" GLACIERAI - FULL GLACIER ANALYSIS")
    print("==========================================")
    print()

    # ======================================================
    # RGI
    # ======================================================

    if glacier_record is not None:
        # Reuse the exact record already resolved by the API.
        # This avoids searching the RGI inventory a second time.
        glacier = glacier_record
    elif glacier_name and str(glacier_name).strip():
        # Standalone/CLI use can search by glacier name or RGI ID.
        glacier = get_glacier(str(glacier_name).strip())
    else:
        raise ValueError(
            "Provide either glacier_name or a resolved glacier_record."
        )

    required_fields = ("name", "rgi_id", "latitude", "longitude", "geometry")
    missing_fields = [
        field for field in required_fields
        if glacier.get(field) is None
    ]
    if missing_fields:
        raise ValueError(
            "Resolved glacier record is missing required fields: "
            + ", ".join(missing_fields)
        )

    print(
        "Glacier       :",
        glacier["name"],
    )

    print(
        "RGI ID        :",
        glacier["rgi_id"],
    )

    print(
        "Inventory area:",
        glacier.get(
            "area_km2"
        ),
        "km²",
    )

    print(
        "Latitude      :",
        glacier["latitude"],
    )

    print(
        "Longitude     :",
        glacier["longitude"],
    )

    # ======================================================
    # UTM GEOMETRY
    # ======================================================

    utm_geometry, target_crs = (
        transform_geometry_to_utm(
            glacier["geometry"],
            glacier["latitude"],
            glacier["longitude"],
        )
    )

    print(
        "Target CRS    :",
        target_crs,
    )

    print(
        "Glacier bounds:",
        utm_geometry.bounds,
    )

    # ======================================================
    # CREATE TILES
    # ======================================================

    tiles = create_tiles(
        utm_geometry
    )

    print()
    print(
        "Tiles required:",
        len(tiles),
    )

    print(
        "Tile size     :",
        TILE_SIZE_METERS,
        "m",
    )

    print()

    if not tiles:

        raise RuntimeError(
            "No tiles intersect the glacier."
        )

    # ======================================================
    # PROCESS TILES
    # ======================================================

    prediction_results = []
    failed_tiles = []

    for index, tile in enumerate(
        tiles,
        start=1,
    ):

        try:

            result = process_tile(
                tile=tile,
                tile_index=index,
                glacier=glacier,
                target_crs=target_crs,
                force_repredict=force_repredict,
                auto_download=auto_download,
            )

            prediction_results.append(
                result
            )

        except Exception as error:

            print()
            print(
                f"Tile {index} failed:"
            )

            print(
                str(error)
            )

            failed_tiles.append({
                "tile_index": index,
                "error": str(error),
            })

            print(
                "Continuing..."
            )

    # ======================================================
    # CHECK
    # ======================================================

    if not prediction_results:

        raise RuntimeError(
            "No tile predictions were available."
        )

    # ======================================================
    # MERGE
    # ======================================================

    full_mask = merge_masks(
        prediction_results,
        glacier["name"],
        glacier_id=glacier["rgi_id"],
    )

    # ======================================================
    # AREA
    # ======================================================

    pixels, area_m2, area_km2 = (
        calculate_area(
            full_mask
        )
    )

    # ======================================================
    # FINAL VISUALIZATION
    # ======================================================

    final_mask_png = (
        create_final_overlay(
            glacier_name=glacier["name"],
            mask_path=full_mask,
            glacier_id=glacier["rgi_id"],
        )
    )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    print()
    print("==========================================")
    print(" FINAL GLACIER RESULT")
    print("==========================================")
    print()

    print(
        "Glacier          :",
        glacier["name"],
    )

    print(
        "RGI ID           :",
        glacier["rgi_id"],
    )

    print(
        "RGI inventory    :",
        glacier.get(
            "area_km2"
        ),
        "km²",
    )

    print(
        "Predicted pixels :",
        pixels,
    )

    print(
        "Predicted area   :",
        round(
            area_km2,
            4,
        ),
        "km²",
    )

    print(
        "Predicted area   :",
        round(
            area_m2,
            2,
        ),
        "m²",
    )

    print(
        "Tiles processed  :",
        len(prediction_results),
        "/",
        len(tiles),
    )

    print(
        "Tiles failed     :",
        len(failed_tiles),
    )

    print(
        "Full mask        :",
        full_mask,
    )

    if final_mask_png:

        print(
            "Final mask PNG   :",
            final_mask_png,
        )

    print()

    return {
        "glacier":
            glacier["name"],

        "rgi_id":
            glacier["rgi_id"],

        "rgi_area_km2":
            glacier.get(
                "area_km2"
            ),

        "predicted_area_km2":
            area_km2,

        "predicted_area_m2":
            area_m2,

        "predicted_pixels":
            pixels,

        "tiles_total":
            len(tiles),

        "tiles_processed":
            len(prediction_results),

        "tiles_failed":
            len(failed_tiles),

        "failed_tiles":
            failed_tiles,

        "complete":
            len(prediction_results) == len(tiles),

        "mask":
            str(full_mask),

        "mask_png":
            str(final_mask_png)
            if final_mask_png
            else None,
    }


# ==========================================================
# COMMAND-LINE ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run tiled glacier segmentation by glacier name or exact RGI ID, "
            "with optional automatic Sentinel-2 downloads."
        )
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--name",
        help="Glacier name to search in the RGI inventory.",
    )
    selection.add_argument(
        "--rgi-id",
        help="Exact RGI ID, for example RGI2000-v7.0-G-15-00467.",
    )

    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Fail when a tile TIFF is missing instead of downloading Sentinel-2 imagery.",
    )

    parser.add_argument(
        "--force-repredict",
        action="store_true",
        help=(
            "Regenerate prediction masks from existing satellite TIFFs even if masks already exist."
        ),
    )

    args = parser.parse_args()
    query = args.rgi_id or args.name

    try:
        result = analyze_large_glacier(
            glacier_name=query,
            force_repredict=args.force_repredict,
            auto_download=not args.no_auto_download,
        )

        print()
        print("==========================================")
        print(" FULL ANALYSIS COMPLETED")
        print("==========================================")
        print()
        print(result)

        if not result.get("complete", False):
            print()
            print(
                "WARNING: The result is PARTIAL because one or more "
                "tiles failed. Do not treat its area as a complete "
                "glacier-area estimate."
            )

    except Exception as error:
        print()
        print("==========================================")
        print(" FULL ANALYSIS FAILED")
        print("==========================================")
        print()
        print("Error:", error)
        raise
