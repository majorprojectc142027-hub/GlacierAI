from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import requests
import rasterio
from rasterio.merge import merge
import numpy as np
import re
from rasterio.features import rasterize
from shapely.geometry import mapping
from backend.app.services.sentinel_service import (
    get_access_token,
)
from backend.app.services.sentinel_catalog import (
    search_historical_sentinel2,
)
from backend.app.services.sentinel_download import (
    get_utm_crs,
    get_crs_url,
    create_rgi_utm_bbox,
)
from backend.app.services.rgi_service import (
    find_glacier_by_name,
)
from backend.app.services.glims_service import (
    get_glims_for_rgi,
)
from backend.scripts.predict_geotiff import (
    predict_geotiff,
)
# ==========================================================
# PROJECT PATHS
# ==========================================================
PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)
HISTORICAL_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
)
SATELLITE_DIR = (
    HISTORICAL_DIR
    / "satellite"
)
PREDICTION_DIR = (
    HISTORICAL_DIR
    / "predictions"
)
SATELLITE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
# ==========================================================
# COPERNICUS PROCESS API
# ==========================================================
PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu/"
    "process/v1"
)
# ==========================================================
# CONFIGURATION
# ==========================================================
PIXEL_SIZE = 12
DEFAULT_START_YEAR = 2023
DEFAULT_END_YEAR = 2026
DEFAULT_SEASON_START_MONTH = 8
DEFAULT_SEASON_START_DAY = 1
DEFAULT_SEASON_END_MONTH = 9
DEFAULT_SEASON_END_DAY = 30
DEFAULT_MAX_CLOUD_COVER = 30
# ==========================================================
# EVALSCRIPT
# ==========================================================
#
# IMPORTANT:
# We use TILE mosaicking so that Sentinel-2 scenes/tiles
# covering the requested AOI can be combined.
#
# dataMask is used to select valid pixels.
#
# Reflectance is multiplied by 10000 before storing as UINT16.
#
# ==========================================================
def build_evalscript(target_datetime, target_scene_id):
    """
    Build an evalscript that filters TILE acquisitions to the selected
    Catalog observation's ±30-minute timestamp window.

    Filter tiles by both the selected Catalog product ID and acquisition
    timestamp. If the Process API exposes a different product ID format,
    the request will return no valid imagery rather than silently using
    another product.
    """
    target_timestamp_ms = int(target_datetime.timestamp() * 1000)
    target_product_id = json.dumps(str(target_scene_id))

    return f"""
//VERSION=3
function preProcessScenes(collections) {{
    var targetTime = {target_timestamp_ms};
    var targetProductId = {target_product_id};
    var toleranceMs = 30 * 60 * 1000;

    collections.scenes.tiles = collections.scenes.tiles.filter(function(tile) {{
        var tileTime = Date.parse(tile.date);
        return tile.productId === targetProductId &&
               isFinite(tileTime) &&
               Math.abs(tileTime - targetTime) <= toleranceMs;
    }});

    return collections;
}}

function setup() {{
    return {{
        input: [{{
            bands: ["B04", "B03", "B02", "dataMask"],
            units: "REFLECTANCE"
        }}],
        mosaicking: "TILE",
        output: {{
            id: "default",
            bands: 4,
            sampleType: "UINT16"
        }}
    }};
}}

function evaluatePixel(samples) {{
    for (var i = 0; i < samples.length; i++) {{
        if (samples[i].dataMask == 1) {{
            return [
                samples[i].B04 * 10000,
                samples[i].B03 * 10000,
                samples[i].B02 * 10000,
                10000
            ];
        }}
    }}
    return [0, 0, 0, 0];
}}
"""
# ==========================================================
# DOWNLOAD ONE HISTORICAL IMAGE
# ==========================================================
def validate_satellite_tiff(image_path):
    """
    Validate a Sentinel-2 RGB GeoTIFF and report possible no-data gaps.
    Zero-valued pixels are reported as a diagnostic rather than automatically
    rejected: zero can indicate no-data, but must be interpreted with the
    source mask and image footprint.
    """
    image_path = Path(image_path)
    if not image_path.exists() or image_path.stat().st_size == 0:
        raise RuntimeError(
            f"Satellite image is missing or empty: {image_path}"
        )
    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise RuntimeError(
                f"Expected at least 3 bands, found {src.count}: {image_path}"
            )
        image = src.read(indexes=[1, 2, 3], masked=True)
        valid_pixels = image.compressed()
        if valid_pixels.size == 0:
            raise RuntimeError(
                f"Satellite image contains no valid pixels: {image_path}"
            )
        if not np.any(valid_pixels != 0):
            raise RuntimeError(
                f"Satellite image contains only zero values: {image_path}"
            )
        # A pixel is considered zero-valued only when all three RGB bands are
        # zero. Masked pixels are excluded from the denominator.
        pixel_values = image.filled(0)
        pixel_valid = ~np.any(np.ma.getmaskarray(image), axis=0)
        valid_pixel_count = int(np.count_nonzero(pixel_valid))
        zero_pixel_count = int(
            np.count_nonzero(np.all(pixel_values == 0, axis=0) & pixel_valid)
        )
        total_pixel_count = int(src.width * src.height)
        zero_fraction = (
            zero_pixel_count / valid_pixel_count
            if valid_pixel_count else 1.0
        )
        valid_fraction = (
            valid_pixel_count / total_pixel_count
            if total_pixel_count else 0.0
        )
        stats = {
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs) if src.crs else None,
            "valid_pixel_count": valid_pixel_count,
            "total_pixel_count": total_pixel_count,
            "valid_pixel_fraction": round(valid_fraction, 6),
            "zero_pixel_count": zero_pixel_count,
            "zero_pixel_fraction_of_valid": round(zero_fraction, 6),
        }
        print(
            f"Validated satellite image: {image_path.name} | "
            f"Dimensions: {src.width}x{src.height} | "
            f"Valid values: {valid_pixels.size} | "
            f"All-zero pixels: {zero_fraction:.1%}"
        )
        if zero_fraction > 0.05:
            print(
                "WARNING: More than 5% of unmasked pixels are zero in all "
                "RGB bands. Inspect image footprint/no-data handling before "
                "trusting the glacier-area estimate."
            )
        return stats
def download_historical_image(
    observation,
    geometry,
    latitude,
    longitude,
    output_dir,
    glacier_slug,
):
    """
    Download Sentinel-2 imagery around the Catalog-selected observation.

    The Process API uses a time range rather than a direct Catalog scene ID.
    Therefore, the downloaded imagery is not guaranteed to come exclusively
    from the selected Catalog scene.

    Large glacier bounding boxes are split into smaller requests and merged
    into one 12 m GeoTIFF.
    """
    
    datetime_string = observation.get("datetime")
    scene_id = observation.get("id")
    cloud_cover = observation.get("cloud_cover")
    if not datetime_string:
        raise RuntimeError("Selected historical observation has no datetime.")
    if not scene_id:
        raise RuntimeError("Selected historical observation has no scene ID.")
    observation_datetime = datetime.fromisoformat(
        datetime_string.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    year = observation_datetime.year
    print()
    print("==========================================")
    print(f" DOWNLOADING HISTORICAL SENTINEL-2 - {year}")
    print("==========================================")
    print()
    print("Scene ID              :", scene_id)
    print("Observation datetime  :", datetime_string)
    print("Catalog cloud         :", cloud_cover, "%")
    # The Process API accepts a time window rather than this Catalog item ID.
    # Keep the ±30-minute window because narrower windows previously returned
    # blank imagery in this workflow. The least-cloud mosaicking setting means
    # the downloaded pixels are not guaranteed to come exclusively from the
    # Catalog scene identified above; retain the Catalog ID as provenance.
    start_datetime = observation_datetime - timedelta(minutes=30)
    end_datetime = observation_datetime + timedelta(minutes=30)
    evalscript = build_evalscript(observation_datetime, scene_id)
    print("Catalog time used:", observation_datetime.isoformat())
    print(
        "Process API time range:",
        start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to",
        end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    target_crs = get_utm_crs(latitude, longitude)
    print("Target CRS            :", target_crs)
    bbox = create_rgi_utm_bbox(
        geometry=geometry,
        target_crs=target_crs,
    )
    print("AOI source            : RGI 7.0 glacier bbox")
    print("UTM BBOX              :", bbox)
    token = get_access_token()
    selected_cloud = (
        float(cloud_cover)
        if cloud_cover is not None
        else DEFAULT_MAX_CLOUD_COVER
    )
    # Copernicus Process API limit is 2500 pixels per dimension.
    # Use 2400 as a safety margin while preserving 12 m resolution.
    MAX_PIXELS = 2400
    TILE_SIZE_METERS = MAX_PIXELS * PIXEL_SIZE
    min_x, min_y, max_x, max_y = bbox
    width_pixels = (max_x - min_x) / PIXEL_SIZE
    height_pixels = (max_y - min_y) / PIXEL_SIZE
    print()
    print("Requested width (pixels) :", round(width_pixels, 2))
    print("Requested height (pixels):", round(height_pixels, 2))
    # Build tile bboxes.
    tile_bboxes = []
    x = min_x
    while x < max_x:
        tile_max_x = min(x + TILE_SIZE_METERS, max_x)
        y = min_y
        while y < max_y:
            tile_max_y = min(y + TILE_SIZE_METERS, max_y)
            tile_bboxes.append(
                (x, y, tile_max_x, tile_max_y)
            )
            y = tile_max_y
        x = tile_max_x
    print()
    print("Process API tiles required:", len(tile_bboxes))
    tile_paths = []
    for tile_index, tile_bbox in enumerate(tile_bboxes, start=1):
        tile_min_x, tile_min_y, tile_max_x, tile_max_y = tile_bbox
        tile_width_pixels = int(
            np.ceil((tile_max_x - tile_min_x) / PIXEL_SIZE)
        )
        tile_height_pixels = int(
            np.ceil((tile_max_y - tile_min_y) / PIXEL_SIZE)
        )
        print()
        print(f"Downloading tile {tile_index}/{len(tile_bboxes)}")
        print("Tile BBOX:", tile_bbox)
        print("Tile size:", tile_width_pixels, "x", tile_height_pixels)
        payload = {
            "input": {
                "bounds": {
                    "bbox": [
                        tile_min_x,
                        tile_min_y,
                        tile_max_x,
                        tile_max_y,
                    ],
                    "properties": {
                        "crs": get_crs_url(target_crs)
                    },
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": start_datetime.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ"
                                ),
                                "to": end_datetime.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ"
                                ),
                            },
                            "mosaickingOrder": "leastCC",
                            "maxCloudCoverage": selected_cloud,
                        },
                        "processing": {
                            "upsampling": "BILINEAR",
                            "downsampling": "BILINEAR",
                        },
                    }
                ],
            },
            "output": {
                "resx": PIXEL_SIZE,
                "resy": PIXEL_SIZE,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/tiff"},
                    }
                ],
            },
            "evalscript": evalscript,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        }
        response = requests.post(
            PROCESS_URL,
            json=payload,
            headers=headers,
            timeout=180,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Historical Sentinel-2 tile request failed "
                f"(tile {tile_index}/{len(tile_bboxes)}, "
                f"HTTP {response.status_code}):\n"
                f"{response.text}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        tile_path = (
            output_dir
            / f"{glacier_slug}_{year}_tile_{tile_index}.tif"
        )
        temp_tile_path = tile_path.with_name(tile_path.stem + ".download.tmp.tif")
        if len(response.content) == 0:
            raise RuntimeError(
                f"Historical Sentinel-2 tile {tile_index} response was empty."
            )
        # Write to a temporary file so a failed/blank download never destroys
        # a previously valid tile for this year.
        with open(temp_tile_path, "wb") as file:
            file.write(response.content)
        try:
            validate_satellite_tiff(temp_tile_path)
            temp_tile_path.replace(tile_path)
        except Exception:
            try:
                temp_tile_path.unlink()
            except OSError:
                pass
            raise
        tile_paths.append(tile_path)
        print("Tile downloaded:", tile_path)
        print("Tile response size:", len(response.content), "bytes")
    final_path = output_dir / f"{glacier_slug}_{year}.tif"
    if len(tile_paths) == 1:
        tile_paths[0].replace(final_path)
    else:
        print()
        print("Merging", len(tile_paths), "Sentinel-2 tiles...")
        datasets = []
        try:
            for tile_path in tile_paths:
                datasets.append(rasterio.open(tile_path))
            mosaic, mosaic_transform = merge(datasets)
            profile = datasets[0].profile.copy()
            profile.update(
                {
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": mosaic_transform,
                    "driver": "GTiff",
                    "compress": "deflate",
                }
            )
            with rasterio.open(
                final_path,
                "w",
                **profile,
            ) as destination:
                destination.write(mosaic)
        finally:
            for dataset in datasets:
                dataset.close()
        for tile_path in tile_paths:
            try:
                tile_path.unlink()
            except Exception:
                pass
    if not final_path.exists():
        raise RuntimeError(
            "Historical Sentinel-2 final GeoTIFF was not created."
        )
    print()
    # Reject blank output before it can reach the U-Net model.
    validate_satellite_tiff(final_path)
    print("Historical Sentinel-2 download successful!")
    print("Final file:", final_path)
    print("Final size:", final_path.stat().st_size, "bytes")
    
    return {
        "path": final_path,
        "catalog_scene_id": scene_id,
        "catalog_datetime": datetime_string,
        "process_time_from": start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "process_time_to": end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_method": "Process API time-window request; exact Catalog scene not guaranteed",
        "cloud_cover": cloud_cover,
        "year": year,
    }

# ==========================================================
# APPLY RGI POLYGON TO PREDICTION
# ==========================================================
def apply_rgi_polygon(
    mask_path,
    geometry,
    output_path,
):
    print()
    print(
        "Applying RGI glacier polygon..."
    )
    with rasterio.open(
        mask_path
    ) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width
        profile = src.profile.copy()

    # Area in km² is only valid when the raster uses a projected CRS whose
    # horizontal units are metres. Refuse to silently calculate square
    # degrees or other non-metric units as square metres.
    if crs is None:
        raise RuntimeError(
            f"Prediction mask has no CRS; cannot calculate glacier area: {mask_path}"
        )
    # Rasterio CRS objects do not expose axis_info in all versions.
    # Convert to pyproj.CRS before checking projection and linear units.
    from pyproj import CRS as PyprojCRS
    crs_info = PyprojCRS.from_user_input(crs)
    if not crs_info.is_projected:
        raise RuntimeError(
            f"Prediction mask CRS must be projected for area calculation; got {crs}."
        )
    axis_info = crs_info.axis_info
    if len(axis_info) < 2 or any(
        not np.isclose(float(axis.unit_conversion_factor or 0), 1.0)
        for axis in axis_info[:2]
    ):
        raise RuntimeError(
            f"Prediction mask CRS must use metres for area calculation; got {crs}."
        )

    # ------------------------------------------------------
    # Convert RGI geometry from EPSG:4326
    # to image CRS
    # ------------------------------------------------------
    from rasterio.warp import transform_geom
    transformed_geometry = transform_geom(
        "EPSG:4326",
        crs,
        geometry,
    )
    # ------------------------------------------------------
    # Rasterize RGI polygon
    # ------------------------------------------------------
    rgi_mask = rasterize(
        [
            (
                transformed_geometry,
                1
            )
        ],
        out_shape=(
            height,
            width,
        ),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    # ------------------------------------------------------
    # Clip AI prediction to RGI glacier while preserving nodata.
    # Predictor mask values: 1=glacier, 0=valid non-glacier,
    # 255=invalid satellite pixel. Never count 255 as glacier.
    clipped_mask = np.zeros((height, width), dtype=np.uint8)
    valid_pixels = mask != 255
    clipped_mask[valid_pixels & (rgi_mask == 1) & (mask == 1)] = 1
    clipped_mask[~valid_pixels] = 255
    # Save clipped mask; retain 255 as nodata.
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=255,
        compress="lzw",
    )
    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(clipped_mask, 1)
    # Area calculation: count only pixels explicitly labelled 1.
    glacier_pixels = int(np.count_nonzero(clipped_mask == 1))
    # The affine determinant handles rotated/sheared rasters as well as
    # north-up rasters. abs(a) * abs(e) alone can be wrong when b or d
    # are non-zero.
    pixel_area_m2 = abs(
        transform.a * transform.e - transform.b * transform.d
    )
    if not np.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise RuntimeError(
            f"Prediction mask has an invalid pixel area: {pixel_area_m2}"
        )
    area_m2 = (
        glacier_pixels
        * pixel_area_m2
    )
    area_km2 = (
        area_m2
        / 1_000_000
    )
    return {
        "glacier_pixels":
            glacier_pixels,
        "pixel_area_m2":
            pixel_area_m2,
        "area_m2":
            area_m2,
        "area_km2":
            area_km2,
        "mask_path":
            str(output_path),
    }
# ==========================================================
# PROCESS ONE HISTORICAL OBSERVATION
# ==========================================================
def process_observation(
    observation,
    geometry,
    latitude,
    longitude,
    glacier_slug,
    satellite_dir,
    prediction_dir,
    rgi_area_km2=None,
):
    """
    Download one selected historical Sentinel-2 observation,
    run U-Net V3.1, clip the prediction to the RGI polygon,
    and calculate glacier area.
    """
    year = int(
        observation["year"]
    )
    download_info = download_historical_image(
        observation=observation,
        geometry=geometry,
        latitude=latitude,
        longitude=longitude,
        output_dir=satellite_dir,
        glacier_slug=glacier_slug,
    )
    image_path = download_info["path"]
    satellite_quality = validate_satellite_tiff(image_path)
    print()
    print("==========================================")
    print(f" RUNNING U-NET V3.1 - {year}")
    print("==========================================")
    prediction = predict_geotiff(
        image_path=image_path,
        output_stem=f"{glacier_slug}_{year}",
        rgi_geometry=geometry,
    )
    print()
    print(
        "Raw U-Net V3.1 area:",
        prediction["area_km2"],
        "km²",
    )
    clipped_mask_path = (
        prediction_dir
        / f"{glacier_slug}_{year}_rgi_mask.tif"
    )
    analysis = apply_rgi_polygon(
        mask_path=prediction["mask_path"],
        geometry=geometry,
        output_path=clipped_mask_path,
    )
    result = {
        "year": year,
        "date": observation.get("datetime"),
        "scene_id": observation.get("id"),
        "scene_id_source": (
            "Process API TILE acquisitions are filtered by exact Catalog productId "
            "and selected acquisition timestamp within ±30 minutes."
        ),
        "catalog_cloud_percent": observation.get("cloud_cover"),
        "area_km2": round(
            analysis["area_km2"],
            6,
        ),
        "area_m2": round(
            analysis["area_m2"],
            2,
        ),
        "glacier_pixels": analysis["glacier_pixels"],
        "mask_path": analysis["mask_path"],
        "satellite_path": str(image_path),
        "model": "U-Net V3.1",
        "rgi_polygon_applied": True,
        "satellite_quality": satellite_quality,
        "area_quality": {
            "rgi_area_km2": (
                round(float(rgi_area_km2), 6)
                if rgi_area_km2 is not None else None
            ),
            "fraction_of_rgi_area": (
                round(analysis["area_km2"] / float(rgi_area_km2), 6)
                if rgi_area_km2 is not None and float(rgi_area_km2) > 0
                else None
            ),
            "flags": [],
        },
    }
    # These are review flags, not automatic claims that a prediction is wrong.
    if (
        rgi_area_km2 is not None
        and float(rgi_area_km2) > 0
        and analysis["area_km2"] / float(rgi_area_km2) > 0.80
    ):
        result["area_quality"]["flags"].append(
            "Predicted area exceeds 80% of the RGI inventory area; visually "
            "review the mask before interpreting change."
        )
    if (
        rgi_area_km2 is not None
        and float(rgi_area_km2) > 0
        and analysis["area_km2"] > float(rgi_area_km2)
    ):
        result["area_quality"]["flags"].append(
            "Predicted area exceeds the full RGI inventory area; check CRS, "
            "pixel size, mask alignment, and segmentation before using this result."
        )
    if analysis["glacier_pixels"] == 0:
        result["area_quality"]["flags"].append(
            "No glacier pixels remain inside the RGI polygon; inspect the "
            "prediction, image footprint, nodata mask, and CRS alignment."
        )
    if satellite_quality["zero_pixel_fraction_of_valid"] > 0.05:
        result["area_quality"]["flags"].append(
            "Satellite RGB contains more than 5% all-zero unmasked pixels; "
            "review no-data and image-footprint handling."
        )
    if result["area_quality"]["flags"]:
        print("QUALITY REVIEW FLAGS:")
        for flag in result["area_quality"]["flags"]:
            print("-", flag)
    print()
    print("==========================================")
    print(f" {year} ANALYSIS COMPLETE")
    print("==========================================")
    print("Date:", observation.get("datetime"))
    print("Scene:", observation.get("id"))
    print(
        "Cloud:",
        observation.get("cloud_cover"),
        "%",
    )
    print(
        "Glacier pixels:",
        analysis["glacier_pixels"],
    )
    print(
        "Estimated glacier area:",
        round(
            analysis["area_km2"],
            6,
        ),
        "km²",
    )
    return result
# ==========================================================
# GLIMS HISTORICAL REFERENCE
# ==========================================================
def get_glims_historical_reference(
    rgi_id,
):
    """
    Fetch historical GLIMS records for the
    exact RGI glacier.
    GLIMS is kept as an independent historical
    reference source and is not merged directly
    into the Sentinel-2/U-Net observations.
    """
    if not rgi_id:
        return {
            "status": "unavailable",
            "records": [],
        }
    try:
        glims_result = get_glims_for_rgi(
            rgi_id
        )
        if not glims_result.get(
            "success"
        ):
            return {
                "status": "unavailable",
                "records": [],
                "message": glims_result.get(
                    "message",
                    "GLIMS data unavailable.",
                ),
            }
        records = []
        for item in glims_result.get(
            "historical_records",
            [],
        ):
            records.append(
                {
                    "year": item.get("year"),
                    "date": item.get("date"),
                    "area_km2": item.get("area_km2"),
                    "glims_id": item.get("glims_id"),
                    "analysis_id": item.get("analysis_id"),
                    "submission_id": item.get("submission_id"),
                    "status": item.get("status"),
                    "source": item.get("source"),
                    "glacier_name": item.get(
                        "glacier_name"
                    ),
                }
            )
        return {
            "status": "available",
            "glims_id": glims_result.get(
                "glims_id"
            ),
            "records": records,
        }
    except Exception as error:
        print(
            "GLIMS historical reference failed:",
            error,
        )
        return {
            "status": "unavailable",
            "records": [],
            "message": str(error),
        }
# ==========================================================
# RUN HISTORICAL ANALYSIS FOR ANY GLACIER
# ==========================================================
def run_historical_analysis(
    glacier_name,
    rgi_id=None,
    start_year=DEFAULT_START_YEAR,
    end_year=DEFAULT_END_YEAR,
    season_start_month=DEFAULT_SEASON_START_MONTH,
    season_start_day=DEFAULT_SEASON_START_DAY,
    season_end_month=DEFAULT_SEASON_END_MONTH,
    season_end_day=DEFAULT_SEASON_END_DAY,
    max_cloud_cover=DEFAULT_MAX_CLOUD_COVER,
):
    """
    Run the complete historical Sentinel-2 + U-Net V3.1 pipeline
    for any glacier found in the global RGI 7.0 inventory.
    """
    print()
    print("==========================================")
    print(" GLACIERAI - HISTORICAL GLACIER ANALYSIS")
    print("==========================================")
    print()
    print("Requested glacier:", glacier_name)
    print("Years:", start_year, "-", end_year)
    print()
    # ------------------------------------------------------
    # Resolve the exact RGI glacier
    # ------------------------------------------------------
    # Prefer the exact RGI ID when supplied. This is important
    # because multiple RGI polygons can share the same name.
    if rgi_id:
        rgi_glacier = find_glacier_by_name(rgi_id)
    else:
        rgi_glacier = find_glacier_by_name(glacier_name)
    if rgi_glacier is None:
        raise RuntimeError(
            f'Glacier "{glacier_name}" was not found in RGI 7.0.'
        )
    rgi_id = rgi_glacier["rgi_id"]
    rgi_name = rgi_glacier["name"]
    latitude = float(rgi_glacier["latitude"])
    longitude = float(rgi_glacier["longitude"])
    rgi_area_km2 = rgi_glacier.get("area_km2")
    geometry = mapping(
        rgi_glacier["geometry"]
    )
    glacier_slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        rgi_name.lower(),
    ).strip("_")
    # Use the exact RGI ID as the storage key for on-demand
    # global analysis. This prevents same-named glaciers from
    # overwriting each other. The API falls back to legacy
    # name-based folders for existing processed datasets.
    if rgi_id:
        glacier_slug = rgi_id.strip().lower().replace(" ", "_")
    glacier_dir = (
        HISTORICAL_DIR
        / glacier_slug
    )
    satellite_dir = (
        glacier_dir
        / "satellite"
    )
    prediction_dir = (
        glacier_dir
        / "predictions"
    )
    satellite_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    print("RGI ID:", rgi_id)
    print("RGI name:", rgi_name)
    print("Latitude:", latitude)
    print("Longitude:", longitude)
    print("RGI area:", rgi_area_km2, "km²")
    print("Output directory:", glacier_dir)
    # ------------------------------------------------------
    # Search real Sentinel-2 observations
    # ------------------------------------------------------
    print()
    print("Searching Sentinel-2 historical observations...")
    observations = search_historical_sentinel2(
        latitude=latitude,
        longitude=longitude,
        start_year=start_year,
        end_year=end_year,
        season_start_month=season_start_month,
        season_start_day=season_start_day,
        season_end_month=season_end_month,
        season_end_day=season_end_day,
        max_cloud_cover=max_cloud_cover,
    )
    # Keep every candidate so a blank/invalid best-scoring scene does not
    # prevent us from trying another scene from the same year.
    candidate_observations = sorted(
        observations,
        key=lambda item: (
            int(item.get("year") or str(item.get("datetime", "0000")[:4])),
            float(item.get("cloud_cover"))
            if item.get("cloud_cover") is not None
            else float("inf"),
            item.get("datetime") or "",
        ),
    )
    selected_years = sorted({
        int(item.get("year") or str(item.get("datetime", "0000")[:4]))
        for item in candidate_observations
    })
    print()
    print("Historical scene candidates:", len(candidate_observations))
    print("Years with candidates:", len(selected_years))
    if not candidate_observations:
        glims_reference = (
            get_glims_historical_reference(
                rgi_id
            )
        )
        result_payload = {
            "status": "no_observations",
            "glacier": {
                "name": rgi_name,
                "rgi_id": rgi_id,
                "latitude": latitude,
                "longitude": longitude,
                "rgi_area_km2": rgi_area_km2,
            },
            "observations": [],
            "glims_reference": glims_reference,
        }
        results_file = (
            glacier_dir
            / "historical_results.json"
        )
        with open(
            results_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result_payload,
                file,
                indent=4,
            )
        print(
            "No suitable Sentinel-2 observations were found."
        )
        print(
            "Results saved to:",
            results_file,
        )
        return result_payload
    # ------------------------------------------------------
    # Try candidate scenes in ascending cloud-cover order per year
    # ------------------------------------------------------
    results = []
    observations_by_year = {}
    for candidate in candidate_observations:
        candidate_year = int(
            candidate.get("year")
            or str(candidate.get("datetime", "0000")[:4])
        )
        observations_by_year.setdefault(candidate_year, []).append(candidate)
    for year in sorted(observations_by_year):
        year_candidates = observations_by_year[year]
        failures = []
        print()
        print(
            f"Trying {len(year_candidates)} candidate scene(s) for {year}, "
            "lowest cloud cover first."
        )
        year_succeeded = False
        for candidate_index, observation in enumerate(year_candidates, start=1):
            print()
            print(
                f"Candidate {candidate_index}/{len(year_candidates)} "
                f"for {year}: {observation.get('id')} "
                f"(cloud={observation.get('cloud_cover')}%)"
            )
            try:
                result = process_observation(
                    observation=observation,
                    geometry=geometry,
                    latitude=latitude,
                    longitude=longitude,
                    glacier_slug=glacier_slug,
                    satellite_dir=satellite_dir,
                    prediction_dir=prediction_dir,
                    rgi_area_km2=rgi_area_km2,
                )
                result["candidate_attempts"] = candidate_index
                result["candidate_scenes_for_year"] = len(year_candidates)
                results.append(result)
                year_succeeded = True
                print(f"Selected successful scene for {year}: {observation.get('id')}")
                break
            except Exception as error:
                message = str(error)
                failures.append({
                    "scene_id": observation.get("id"),
                    "date": observation.get("datetime"),
                    "cloud_cover": observation.get("cloud_cover"),
                    "error": message,
                })
                print()
                print(f"Candidate {candidate_index} failed for {year}: {message}")
                if candidate_index < len(year_candidates):
                    print("Trying the next available scene for this year...")
        if not year_succeeded:
            first = year_candidates[0]
            results.append({
                "year": year,
                "date": first.get("datetime"),
                "scene_id": first.get("id"),
                "catalog_cloud_percent": first.get("cloud_cover"),
                "status": "failed",
                "candidate_attempts": len(year_candidates),
                "candidate_failures": failures,
                "error": (
                    f"All {len(year_candidates)} candidate scene(s) failed "
                    f"for {year}; see candidate_failures for details."
                ),
            })
    # ------------------------------------------------------
    # Save structured historical results
    # ------------------------------------------------------
    successful_results = [
        item
        for item in results
        if "area_km2" in item
    ]
    # ------------------------------------------------------
    # GLIMS HISTORICAL REFERENCE
    # ------------------------------------------------------
    print()
    print(
        "Fetching GLIMS historical reference..."
    )
    glims_reference = (
        get_glims_historical_reference(
            rgi_id
        )
    )
    print(
        "GLIMS status:",
        glims_reference.get(
            "status"
        ),
    )
    print(
        "GLIMS records:",
        len(
            glims_reference.get(
                "records",
                [],
            )
        ),
    )
    result_payload = {
        "status": (
            "completed"
            if successful_results
            else "failed"
        ),
        "glacier": {
            "name": rgi_name,
            "rgi_id": rgi_id,
            "latitude": latitude,
            "longitude": longitude,
            "rgi_area_km2": rgi_area_km2,
        },
        "historical_config": {
            "start_year": start_year,
            "scene_selection": (
                "Same August-September seasonal window for each year; "
                "candidates are attempted from lowest to highest cloud cover."
                if (season_start_month, season_start_day, season_end_month, season_end_day)
                == (8, 1, 9, 30)
                else "Same configured seasonal window for each year; "
                     "candidates are attempted from lowest to highest cloud cover."
            ),
            "process_api_scene_note": (
                "The Catalog scene ID is passed to the Process API evalscript, which "
                "filters TILE acquisitions by matching productId and the selected "
                "acquisition timestamp within ±30 minutes. A productId mismatch "
                "will result in no matching tiles rather than silently using another product."
            ),
            "end_year": end_year,
            "season": {
                "start": (
                    f"{season_start_month:02d}-"
                    f"{season_start_day:02d}"
                ),
                "end": (
                    f"{season_end_month:02d}-"
                    f"{season_end_day:02d}"
                ),
            },
            "max_cloud_cover": max_cloud_cover,
        },
        "model": "U-Net V3.1",
        "observations": results,
        "glims_reference": glims_reference,
    }
    results_file = (
        glacier_dir
        / "historical_results.json"
    )
    with open(
        results_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result_payload,
            file,
            indent=4,
        )
    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------
    print()
    print("==========================================")
    print(" HISTORICAL GLACIER ANALYSIS SUMMARY")
    print("==========================================")
    print()
    for result in results:
        if "area_km2" in result:
            print(
                f'{result["year"]} | '
                f'{result["date"]} | '
                f'Area: '
                f'{result["area_km2"]} km²'
            )
        else:
            print(
                f'{result["year"]} | '
                f'FAILED | '
                f'{result.get("error")}'
            )
    print()
    print("Successful observations:", len(successful_results))
    print("Results saved to:")
    print(results_file)
    print()
    print("Historical analysis finished.")
    return result_payload
# ==========================================================
# MAIN
# ==========================================================
def main():
    glacier_name = input("Enter a glacier name or RGI ID: ").strip()
    if not glacier_name:
        raise RuntimeError("A glacier name or RGI ID is required.")

    run_historical_analysis(
        glacier_name=glacier_name,
        start_year=DEFAULT_START_YEAR,
        end_year=DEFAULT_END_YEAR,
        season_start_month=DEFAULT_SEASON_START_MONTH,
        season_start_day=DEFAULT_SEASON_START_DAY,
        season_end_month=DEFAULT_SEASON_END_MONTH,
        season_end_day=DEFAULT_SEASON_END_DAY,
        max_cloud_cover=DEFAULT_MAX_CLOUD_COVER,
    )
if __name__ == "__main__":
    main()
