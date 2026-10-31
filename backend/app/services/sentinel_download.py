from pathlib import Path

from datetime import datetime, timedelta, timezone

import math

import requests

from pyproj import Transformer

from shapely.geometry import shape

from typing import Sequence

from backend.app.services.sentinel_service import (

    get_access_token,

)

from backend.app.services.sentinel_catalog import (

    search_sentinel2,

)

# ==========================================================

# PROJECT CONFIGURATION

# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (

    PROJECT_ROOT

    / "datasets"

    / "live"

)

OUTPUT_DIR.mkdir(

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

# ==========================================================

# GENERAL LIVE SENTINEL-2 SEARCH STRATEGY

# ==========================================================

#

# This is deliberately not glacier-specific.

# Every glacier is searched using the same progressive strategy:

#

# 1. Recent + low cloud

# 2. Wider period + moderate cloud

# 3. Up to one year + progressively relaxed cloud

#

# The catalog finds a recent suitable observation. The Process API does not

# accept a catalog product ID directly, so request a wider time window around

# the observation to avoid empty outputs from overly narrow timestamp filters.

SEARCH_STRATEGIES = [

    {"days_back": 2, "max_cloud_cover": 40},

    {"days_back": 7, "max_cloud_cover": 50},

    {"days_back": 15, "max_cloud_cover": 60},

    {"days_back": 30, "max_cloud_cover": 70},

    {"days_back": 90, "max_cloud_cover": 80},

    {"days_back": 180, "max_cloud_cover": 90},

    {"days_back": 365, "max_cloud_cover": 100},

]

DEFAULT_DAYS_BACK = 2

MAX_CLOUD_COVERAGE = 40

# Fixed live-analysis AOI.

#

# Using the complete RGI glacier bounding box can create

# images larger than the Copernicus 2500 x 2500 limit.

#

# This fixed window keeps the live request manageable.

LIVE_AOI_HALF_SIZE_DEGREES = 0.08

# Fallback AOI if this script is used independently.

HALF_SIZE_DEGREES = 0.02

# ==========================================================

# UTM CRS

# ==========================================================

def get_utm_crs(

    latitude: float,

    longitude: float,

) -> str:

    zone = int(

        math.floor(

            (longitude + 180) / 6

        )

    ) + 1

    zone = max(

        1,

        min(

            60,

            zone,

        ),

    )

    if latitude >= 0:

        epsg = 32600 + zone

    else:

        epsg = 32700 + zone

    return f"EPSG:{epsg}"

# ==========================================================

# EPSG → OGC CRS URL

# ==========================================================

def get_crs_url(

    epsg: str,

) -> str:

    code = epsg.split(":")[-1]

    return (

        "http://www.opengis.net/"

        "def/crs/EPSG/0/"

        f"{code}"

    )

# ==========================================================

# CREATE UTM BBOX

# ==========================================================

def create_utm_bbox(

    latitude: float,

    longitude: float,

    target_crs: str,

    half_size_degrees: float = HALF_SIZE_DEGREES,

):

    lon_min = (

        longitude

        - half_size_degrees

    )

    lat_min = (

        latitude

        - half_size_degrees

    )

    lon_max = (

        longitude

        + half_size_degrees

    )

    lat_max = (

        latitude

        + half_size_degrees

    )

    transformer = Transformer.from_crs(

        "EPSG:4326",

        target_crs,

        always_xy=True,

    )

    x_min, y_min = transformer.transform(

        lon_min,

        lat_min,

    )

    x_max, y_max = transformer.transform(

        lon_max,

        lat_max,

    )

    return [

        x_min,

        y_min,

        x_max,

        y_max,

    ]

# ==========================================================

# CREATE UTM BBOX FROM RGI GLACIER POLYGON

# ==========================================================

def create_rgi_utm_bbox(

    geometry: dict,

    target_crs: str,

):

    """Create a UTM bounding box from an RGI glacier polygon."""

    polygon = shape(geometry)

    if polygon.is_empty:

        raise ValueError("RGI glacier geometry is empty.")

    if not polygon.is_valid:

        polygon = polygon.buffer(0)

    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    transformer = Transformer.from_crs(

        "EPSG:4326",

        target_crs,

        always_xy=True,

    )

    x1, y1 = transformer.transform(min_lon, min_lat)

    x2, y2 = transformer.transform(max_lon, max_lat)

    x_min = min(x1, x2)

    x_max = max(x1, x2)

    y_min = min(y1, y2)

    y_max = max(y1, y2)

    padding = 100

    return [

        x_min - padding,

        y_min - padding,

        x_max + padding,

        y_max + padding,

    ]

# ==========================================================

# DOWNLOAD SENTINEL-2

# ==========================================================

def download_sentinel_geotiff(

    latitude: float,

    longitude: float,

    output_name: str = "glacier_latest.tif",

    days_back: int = DEFAULT_DAYS_BACK,

    geometry: dict | None = None,

):

    token = get_access_token()

    now = datetime.now(

        timezone.utc

    )

    # ======================================================

    # TARGET UTM CRS

    # ======================================================

    target_crs = get_utm_crs(

        latitude,

        longitude,

    )

    print()

    print("==========================================")

    print(" GLACIERAI - SENTINEL-2 LIVE DOWNLOAD")

    print("==========================================")

    print()

    print(

        "Latitude       :",

        latitude,

    )

    print(

        "Longitude      :",

        longitude,

    )

    print(

        "Target CRS     :",

        target_crs,

    )

    print(

        "Resolution     :",

        PIXEL_SIZE,

        "m",

    )

    print(

        "Cloud limit    :",

        MAX_CLOUD_COVERAGE,

        "%",

    )

    print(

        "Search period  :",

        days_back,

        "days",

    )

    # ======================================================

    # FIND ACTUAL AVAILABLE SENTINEL-2 OBSERVATION

    # ======================================================

    selected_observation = None

    selected_days_back = None

    selected_cloud_limit = None

    print()

    print("Searching Sentinel-2 catalog...")

    for strategy in SEARCH_STRATEGIES:

        search_days = strategy["days_back"]

        search_cloud = strategy["max_cloud_cover"]

        print(

            f"Trying {search_days} days / "

            f"cloud <= {search_cloud}%..."

        )

        try:

            results = search_sentinel2(

                latitude=latitude,

                longitude=longitude,

                days_back=search_days,

                max_cloud_cover=search_cloud,

                limit=20,

            )

        except Exception as error:

            print(

                "Catalog search failed:",

                error,

            )

            continue

        if results:

            # search_sentinel2 returns newest first.

            selected_observation = results[0]

            selected_days_back = search_days

            selected_cloud_limit = search_cloud

            break

    if selected_observation is None:

        raise RuntimeError(

            "No suitable Sentinel-2 L2A observation was found "

            "for this glacier after all fallback searches."

        )

    selected_datetime = selected_observation.get(

        "datetime"

    )

    selected_scene_id = selected_observation.get(

        "id"

    )

    selected_cloud_cover = selected_observation.get(

        "cloud_cover"

    )

    if not selected_datetime:

        raise RuntimeError(

            "Sentinel-2 catalog returned an observation "

            "without an acquisition datetime."

        )

    print()

    print("Selected Sentinel-2 observation:")

    print("Scene ID      :", selected_scene_id)

    print("Acquisition   :", selected_datetime)

    print("Cloud cover   :", selected_cloud_cover, "%")

    print("Search window :", selected_days_back, "days")

    print("Cloud limit   :", selected_cloud_limit, "%")

    print()

    # ======================================================

    # AOI

    # ======================================================

    # We intentionally use a fixed centroid window instead

    # of the complete RGI glacier bounding box.

    #

    # This prevents large glaciers such as Baltoro and

    # Columbia from exceeding the Copernicus 2500 x 2500

    # pixel limit.

    print(

        "AOI source     : fixed centroid window"

    )

    bbox = create_utm_bbox(

        latitude=latitude,

        longitude=longitude,

        target_crs=target_crs,

        half_size_degrees=LIVE_AOI_HALF_SIZE_DEGREES,

    )

    print(

        "AOI size       : approximately "

        "0.16° × 0.16°"

    )

    print()

    print(

        "UTM BBOX        :",

        bbox,

    )

    print()

    print(

        "Requesting latest suitable Sentinel-2..."

    )

    # ======================================================

    # EVALSCRIPT

    # ======================================================

    # IMPORTANT:

    #

    # Sentinel-2 REFLECTANCE values are returned in the

    # normalized 0-1 range when units="REFLECTANCE".

    #

    # The U-Net inference pipeline expects the Sentinel

    # GeoTIFF to contain values scaled approximately

    # 0-10000 because predict_geotiff.py performs:

    #

    #     band / 10000.0

    #

    # Therefore we explicitly scale reflectance by 10000

    # before writing UINT16 output.

    evalscript = """

//VERSION=3

function setup() {

    return {

        input: [

            {

                bands: [

                    "B04",

                    "B03",

                    "B02"

                ],

                units: "REFLECTANCE"

            }

        ],

        output: {

            id: "default",

            bands: 3,

            sampleType: "UINT16"

        }

    };

}

function evaluatePixel(sample) {

    return [

        sample.B04 * 10000,

        sample.B03 * 10000,

        sample.B02 * 10000

    ];

}

"""

    # ======================================================

    # REQUEST PAYLOAD

    # ======================================================

    payload = {

        "input": {

            "bounds": {

                "bbox": bbox,

                "properties": {

                    "crs":

                        get_crs_url(

                            target_crs

                        )

                },

            },

            "data": [

                {

                    "type":

                        "sentinel-2-l2a",

                    "dataFilter": {

                        "timeRange": {

                            "from":

                                (

                                    datetime.fromisoformat(

                                        selected_datetime.replace(

                                            "Z",

                                            "+00:00"

                                        )

                                    )

                                    - timedelta(hours=12)

                                ).strftime(

                                    "%Y-%m-%dT%H:%M:%SZ"

                                ),

                            "to":

                                (

                                    datetime.fromisoformat(

                                        selected_datetime.replace(

                                            "Z",

                                            "+00:00"

                                        )

                                    )

                                    + timedelta(hours=12)

                                ).strftime(

                                    "%Y-%m-%dT%H:%M:%SZ"

                                ),

                        },

                        "mosaickingOrder":

                            "mostRecent",

                        "maxCloudCoverage":

                            selected_cloud_limit,

                    },

                    "processing": {

                        "upsampling":

                            "BILINEAR",

                        "downsampling":

                            "BILINEAR",

                    },

                }

            ],

        },

        "output": {

            "resx":

                PIXEL_SIZE,

            "resy":

                PIXEL_SIZE,

            "responses": [

                {

                    "identifier":

                        "default",

                    "format": {

                        "type":

                            "image/tiff"

                    },

                }

            ],

        },

        "evalscript":

            evalscript,

    }

    # ======================================================

    # HTTP REQUEST

    # ======================================================

    headers = {

        "Authorization":

            f"Bearer {token}",

        "Content-Type":

            "application/json",

        "Accept":

            "image/tiff",

    }

    response = requests.post(

        PROCESS_URL,

        json=payload,

        headers=headers,

        timeout=120,

    )

    # Diagnostic logging: distinguish an API response issue from a valid
    # GeoTIFF that contains no usable reflectance pixels.
    print("Sentinel Hub HTTP status:", response.status_code)
    print("Sentinel Hub content type:", response.headers.get("Content-Type"))
    print("Sentinel Hub response bytes:", len(response.content))
    print("Selected scene:", selected_scene_id)
    print("Selected acquisition:", selected_datetime)
    print("Requested AOI bbox:", bbox)
    print(
        "Requested time range:",
        payload["input"]["data"][0]["dataFilter"]["timeRange"],
    )

    # ======================================================

    # ERROR

    # ======================================================

    if response.status_code != 200:

        raise RuntimeError(

            "Sentinel-2 GeoTIFF request failed "

            f"(HTTP {response.status_code}):\n"

            f"{response.text}"

        )

    if not response.content:

        raise RuntimeError(

            f"Sentinel-2 Process API returned an empty response body for "

            f"scene {selected_scene_id}."

        )



    # ======================================================

    # SAVE

    # ======================================================

    output_path = (

        OUTPUT_DIR

        / output_name

    )

    with open(

        output_path,

        "wb",

    ) as file:

        file.write(

            response.content

        )

    # ======================================================

    # VALIDATE THE RETURNED RASTER

    # ======================================================

    #

    # HTTP 200 does not guarantee useful satellite pixels.

    # Reject empty/black rasters instead of sending them to U-Net.

    try:

        import rasterio

        import numpy as np

        with rasterio.open(output_path) as dataset:

            valid_data = False

            for band_index in range(

                1,

                min(dataset.count, 3) + 1,

            ):

                band = dataset.read(

                    band_index

                )

                if np.any(

                    np.isfinite(band)

                    & (band > 0)

                ):

                    valid_data = True

                    break

    except Exception as error:

        output_path.unlink(

            missing_ok=True

        )

        raise RuntimeError(

            "Sentinel-2 response could not be validated "

            f"as a GeoTIFF: {error}"

        )

    if not valid_data:

        output_path.unlink(

            missing_ok=True

        )

        raise RuntimeError(

            "Sentinel-2 returned an empty raster for scene "

            f"{selected_scene_id}. "

            "No valid non-zero reflectance pixels were returned."

        )

    # ======================================================

    # SUCCESS

    # ======================================================

    print()

    print("==========================================")

    print(" SENTINEL-2 DOWNLOAD SUCCESSFUL")

    print("==========================================")

    print()

    print(

        "File       :",

        output_path,

    )

    print(

        "File size  :",

        len(response.content),

        "bytes",

    )

    print(

        "Resolution :",

        PIXEL_SIZE,

        "m",

    )

    print(

        "CRS        :",

        target_crs,

    )

    print(

        "AOI        :",

        "fixed centroid window",

    )

    print(

        "Reflectance :",

        "scaled ×10000 for UINT16",

    )

    print()

    return {

        "path":

            str(output_path),

        "crs":

            target_crs,

        "resolution_m":

            PIXEL_SIZE,

        "cloud_limit_percent":

            selected_cloud_limit,

        "days_back":

            selected_days_back,

        "selected_scene_id":

            selected_scene_id,

        "selected_datetime":

            selected_datetime,

        "selected_cloud_cover":

            selected_cloud_cover,

        "aoi_source":

            "fixed_centroid_window",

        "reflectance_scale":

            10000,

    }

# ==========================================================

# DOWNLOAD SENTINEL-2 SPECTRAL DATA FOR DEBRIS ANALYSIS

# ==========================================================

def download_sentinel_spectral_geotiff(

    latitude: float,

    longitude: float,

    selected_datetime: str,

    selected_scene_id: str,

    selected_cloud_limit=None,

    output_name: str = "glacier_debris_spectral.tif",

):

    """Download Sentinel-2 B08, B11 and B12 for debris analysis.

    Saves the current Process API response to a temporary GeoTIFF,

    validates it, applies band/reflectance metadata, and only then

    replaces the destination file.

    The time filter narrows the request around the selected acquisition

    time; it does not independently guarantee the exact catalog scene ID.

    """

    import os

    import tempfile

    import numpy as np

    import rasterio

    token = get_access_token()

    target_crs = get_utm_crs(latitude, longitude)

    print()

    print("==========================================")

    print(" GLACIERAI - DEBRIS SPECTRAL DOWNLOAD")

    print("==========================================")

    print()

    print("Scene ID     :", selected_scene_id)

    print("Acquisition  :", selected_datetime)

    print("Target CRS   :", target_crs)

    print("Resolution   :", PIXEL_SIZE, "m")

    bbox = create_utm_bbox(

        latitude=latitude,

        longitude=longitude,

        target_crs=target_crs,

        half_size_degrees=LIVE_AOI_HALF_SIZE_DEGREES,

    )

    evalscript = """

//VERSION=3

function setup() {

    return {

        input: [{

            bands: ["B08", "B11", "B12"],

            units: "REFLECTANCE"

        }],

        output: {

            id: "default",

            bands: 3,

            sampleType: "UINT16"

        }

    };

}

function evaluatePixel(sample) {

    return [

        sample.B08 * 10000,

        sample.B11 * 10000,

        sample.B12 * 10000

    ];

}

"""

    observation_time = datetime.fromisoformat(

        selected_datetime.replace("Z", "+00:00")

    )

    data_filter = {

        "timeRange": {

            "from": (

                observation_time - timedelta(seconds=1)

            ).strftime("%Y-%m-%dT%H:%M:%SZ"),

            "to": (

                observation_time + timedelta(seconds=1)

            ).strftime("%Y-%m-%dT%H:%M:%SZ"),

        },

        "mosaickingOrder": "mostRecent",

    }

    if selected_cloud_limit is not None:

        data_filter["maxCloudCoverage"] = selected_cloud_limit

    payload = {

        "input": {

            "bounds": {

                "bbox": bbox,

                "properties": {"crs": get_crs_url(target_crs)},

            },

            "data": [{

                "type": "sentinel-2-l2a",

                "dataFilter": data_filter,

                "processing": {

                    "upsampling": "BILINEAR",

                    "downsampling": "BILINEAR",

                },

            }],

        },

        "output": {

            "resx": PIXEL_SIZE,

            "resy": PIXEL_SIZE,

            "responses": [{

                "identifier": "default",

                "format": {"type": "image/tiff"},

            }],

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

        timeout=120,

    )

    if response.status_code != 200:

        raise RuntimeError(

            "Sentinel-2 spectral download failed "

            f"(HTTP {response.status_code}):\n{response.text}"

        )

    if not response.content:

        raise RuntimeError("Sentinel Hub returned an empty response body.")
    print("Spectral response bytes:", len(response.content))
    print("Spectral response type:", response.headers.get("Content-Type"))
    print("Selected scene:", selected_scene_id)
    print("Selected acquisition:", selected_datetime)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / output_name

    temp_path = None

    try:

        # Save the current response before opening it with Rasterio.

        with tempfile.NamedTemporaryFile(

            prefix=f"{output_path.stem}_",

            suffix=".tif",

            dir=OUTPUT_DIR,

            delete=False,

        ) as temp_file:

            temp_path = Path(temp_file.name)

            temp_file.write(response.content)

        with rasterio.open(temp_path, "r+") as dataset:

            if dataset.count != 3:

                raise RuntimeError(

                    f"Expected 3 bands, found {dataset.count}."

                )

            if dataset.crs is None:

                raise RuntimeError(

                    "Downloaded spectral GeoTIFF has no CRS."

                )

            if dataset.width <= 0 or dataset.height <= 0:

                raise RuntimeError(

                    "Downloaded spectral GeoTIFF has invalid dimensions."

                )

            dataset.set_band_description(1, "B08_NIR")

            dataset.set_band_description(2, "B11_SWIR1")

            dataset.set_band_description(3, "B12_SWIR2")

            dataset.scales = (0.0001, 0.0001, 0.0001)

            dataset.offsets = (0.0, 0.0, 0.0)

            dataset.update_tags(

                REFLECTANCE_SCALE="10000",

                REFLECTANCE_UNITS="unitless",

                BAND_ORDER="B08,B11,B12",

                REQUESTED_SCENE_ID=str(selected_scene_id),

                REQUESTED_ACQUISITION_DATETIME=str(selected_datetime),

            )

        # Reopen and validate pixel data before replacing the old output.

        with rasterio.open(temp_path) as dataset:

            valid_data = False

            for band_index in range(1, 4):

                band = dataset.read(band_index)

                if np.any(np.isfinite(band) & (band > 0)):

                    valid_data = True

                    break

            if not valid_data:
                print("Raster dimensions:", dataset.width, dataset.height)
                print("Raster bands:", dataset.count)
                for band_index in range(1, dataset.count + 1):
                    band = dataset.read(band_index)
                    print(
                    f"Band {band_index}: min={np.nanmin(band)}, "
                    f"max={np.nanmax(band)}, "
                    f"nonzero={np.count_nonzero(band)}"
                    )
            
                raise RuntimeError(

                    "Sentinel Hub returned an empty spectral raster."

                )

            width = dataset.width

            height = dataset.height

            actual_crs = dataset.crs

            actual_transform = dataset.transform

        # Atomic replacement keeps a previously valid file if earlier

        # downloading, metadata writing, or validation fails.

        os.replace(temp_path, output_path)

        temp_path = None

    except Exception as error:

        raise RuntimeError(

            "Failed to save or validate the spectral GeoTIFF: "

            f"{error}"

        ) from error

    finally:

        if temp_path is not None:

            temp_path.unlink(missing_ok=True)

    print()

    print("==========================================")

    print(" SPECTRAL DOWNLOAD SUCCESSFUL")

    print("==========================================")

    print("File       :", output_path)

    print("Bands      :", "B08, B11, B12")

    print("Dimensions :", f"{width} x {height}")

    print("Resolution :", PIXEL_SIZE, "m")

    print("CRS        :", actual_crs)

    print("Reflectance:", "raw values scaled x10000")

    print()

    return {

        "path": str(output_path),

        "crs": str(actual_crs),

        "resolution_m": PIXEL_SIZE,

        "selected_scene_id": selected_scene_id,

        "selected_datetime": selected_datetime,

        "cloud_limit_percent": selected_cloud_limit,

        "bands": ["B08", "B11", "B12"],

        "reflectance_scale": 10000,

        "aoi_source": "fixed_centroid_window",

        "width": width,

        "height": height,

        "transform": list(actual_transform)[:6],

    }

def download_sentinel_tile(

    bbox: Sequence[float],

    target_crs: str,

    output_name: str,

    days_back: int = DEFAULT_DAYS_BACK,

):

    """

    Download one manageable Sentinel-2 tile.

    bbox:

        [min_x, min_y, max_x, max_y] in target_crs.

    This function is used by the large-glacier tiled pipeline.

    The existing download_sentinel_geotiff() function is kept

    unchanged for normal live analysis.

    """

    token = get_access_token()

    now = datetime.now(timezone.utc)

    start = now - timedelta(days=days_back)

    print()

    print("==========================================")

    print(" SENTINEL-2 TILE DOWNLOAD")

    print("==========================================")

    print()

    print("Target CRS     :", target_crs)

    print("Resolution     :", PIXEL_SIZE, "m")

    print("Cloud limit    :", MAX_CLOUD_COVERAGE, "%")

    print("Search period  :", days_back, "days")

    print("Tile BBOX      :", bbox)

    # ======================================================

    # EVALSCRIPT

    # ======================================================

    # Keep this identical to the working live pipeline.

    # Sentinel reflectance is scaled to 0-10000 because

    # predict_geotiff.py expects that scale.

    evalscript = """

//VERSION=3

function setup() {

    return {

        input: [

            {

                bands: [

                    "B04",

                    "B03",

                    "B02"

                ],

                units: "REFLECTANCE"

            }

        ],

        output: {

            id: "default",

            bands: 3,

            sampleType: "UINT16"

        }

    };

}

function evaluatePixel(sample) {

    return [

        sample.B04 * 10000,

        sample.B03 * 10000,

        sample.B02 * 10000

    ];

}

"""

    # ======================================================

    # REQUEST PAYLOAD

    # ======================================================

    payload = {

        "input": {

            "bounds": {

                "bbox": list(bbox),

                "properties": {

                    "crs":

                        get_crs_url(target_crs)

                },

            },

            "data": [

                {

                    "type":

                        "sentinel-2-l2a",

                    "dataFilter": {

                        "timeRange": {

                            "from":

                                start.strftime(

                                    "%Y-%m-%dT%H:%M:%SZ"

                                ),

                            "to":

                                now.strftime(

                                    "%Y-%m-%dT%H:%M:%SZ"

                                ),

                        },

                        "mosaickingOrder":

                            "mostRecent",

                        "maxCloudCoverage":

                            MAX_CLOUD_COVERAGE,

                    },

                    "processing": {

                        "upsampling":

                            "BILINEAR",

                        "downsampling":

                            "BILINEAR",

                    },

                }

            ],

        },

        "output": {

            "resx":

                PIXEL_SIZE,

            "resy":

                PIXEL_SIZE,

            "responses": [

                {

                    "identifier":

                        "default",

                    "format": {

                        "type":

                            "image/tiff"

                    },

                }

            ],

        },

        "evalscript":

            evalscript,

    }

    # ======================================================

    # HTTP REQUEST

    # ======================================================

    headers = {

        "Authorization":

            f"Bearer {token}",

        "Content-Type":

            "application/json",

        "Accept":

            "image/tiff",

    }

    response = requests.post(

        PROCESS_URL,

        json=payload,

        headers=headers,

        timeout=120,

    )

    # ======================================================

    # ERROR HANDLING

    # ======================================================

    if response.status_code != 200:

        raise RuntimeError(

            "Sentinel-2 tile request failed "

            f"(HTTP {response.status_code}):\n"

            f"{response.text}"

        )

    # ======================================================

    # SAVE

    # ======================================================

    output_path = OUTPUT_DIR / output_name

    with open(

        output_path,

        "wb",

    ) as file:

        file.write(

            response.content

        )

    # ======================================================

    # SUCCESS

    # ======================================================

    print()

    print("Tile download successful")

    print("File       :", output_path)

    print("File size  :", len(response.content), "bytes")

    print("Resolution :", PIXEL_SIZE, "m")

    print("CRS        :", target_crs)

    print()

    return {

        "path":

            str(output_path),

        "crs":

            target_crs,

        "resolution_m":

            PIXEL_SIZE,

        "cloud_limit_percent":

            MAX_CLOUD_COVERAGE,

        "days_back":

            days_back,

        "bbox":

            list(bbox),

        "reflectance_scale":

            10000,

    }

# ==========================================================

# TEST

# ==========================================================

if __name__ == "__main__":

    try:

        result = download_sentinel_geotiff(

            latitude=30.843,

            longitude=79.089,

            output_name=

                "gangotri_latest.tif",

        )

        print(

            "Test completed successfully."

        )

        print(

            "GeoTIFF:",

            result["path"],

        )

    except Exception as error:

        print()

        print(

            "Download status : FAILED"

        )
        print()
        print(
            "Error:",
            error,
        )
