
from datetime import datetime, timedelta
from io import BytesIO
import json
import tarfile
from pathlib import Path

import requests
import rasterio

from backend.app.services.sentinel_download import (
    PROCESS_URL,
    PIXEL_SIZE,
    create_utm_bbox,
    get_utm_crs,
    get_crs_url,
)
from backend.app.services.sentinel_service import get_access_token
from backend.app.services.sentinel_catalog import search_sentinel2


LATITUDE = 30.825184
LONGITUDE = 79.073847

OUTPUT_DIR = Path("datasets/diagnostics/sentinel_combined")
BANDS = ["B04", "B08"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Searching Sentinel-2 catalog...")
    results = search_sentinel2(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        days_back=30,
        max_cloud_cover=30,
        limit=20,
    )
    if not results:
        raise RuntimeError("No Sentinel-2 observations found.")

    selected = results[0]
    selected_datetime = selected["datetime"]
    selected_scene_id = selected.get("id", "unknown")

    observation_time = datetime.fromisoformat(
        selected_datetime.replace("Z", "+00:00")
    )

    target_crs = get_utm_crs(LATITUDE, LONGITUDE)
    bbox = create_utm_bbox(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        target_crs=target_crs,
        half_size_degrees=0.01,
    )

    time_from = (
        observation_time - timedelta(minutes=2)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    time_to = (
        observation_time + timedelta(minutes=2)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("Catalog scene:", selected_scene_id)
    print("Catalog time :", selected_datetime)
    print("Target CRS   :", target_crs)
    print("AOI half-size: 0.01 degrees")
    print("Requesting B04 and B08 in one Process API request...")

    evalscript = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B04", "B08"],
            units: "REFLECTANCE"
        }],
        mosaicking: Mosaicking.TILE,
        output: {
            id: "default",
            bands: 2,
            sampleType: "UINT16"
        }
    };
}

function evaluatePixel(samples) {
    for (let i = 0; i < samples.length; i++) {
        const s = samples[i];

        if (s.B04 > 0 && s.B08 > 0) {
            return [
                s.B04 * 10000,
                s.B08 * 10000
            ];
        }
    }

    return [0, 0];
}

function updateOutputMetadata(
    scenes, inputMetadata, outputMetadata
) {
    outputMetadata.userData = {
        tiles: scenes.tiles,
        serviceVersion: inputMetadata.serviceVersion
    };
}
"""

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": get_crs_url(target_crs)
                },
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": time_from,
                        "to": time_to,
                    },
                    "mosaickingOrder": "mostRecent",
                    "maxCloudCoverage": 30,
                },
                "processing": {
                    "upsampling": "BILINEAR",
                    "downsampling": "BILINEAR",
                },
            }],
        },
        "output": {
            "resx": PIXEL_SIZE,
            "resy": PIXEL_SIZE,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                },
                {
                    "identifier": "userdata",
                    "format": {"type": "application/json"},
                },
            ],
        },
        "evalscript": evalscript,
    }

    response = requests.post(
        PROCESS_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/x-tar",
        },
        timeout=(30, 300),
    )
    response.raise_for_status()

    raster_bytes = None
    user_data = None

    with tarfile.open(
        fileobj=BytesIO(response.content),
        mode="r:*",
    ) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                continue

            content = extracted.read()
            name = member.name.lower()

            if name.endswith("userdata.json"):
                user_data = json.loads(content.decode("utf-8"))
            elif name.endswith((".tif", ".tiff")):
                raster_bytes = content

    if raster_bytes is None:
        raise RuntimeError(
            "No GeoTIFF found in Process API TAR response."
        )

    if user_data is None:
        raise RuntimeError(
            "No userdata.json found in Process API TAR response."
        )

    raster_path = OUTPUT_DIR / "combined_b04_b08.tif"
    metadata_path = OUTPUT_DIR / "combined_request_metadata.json"

    # Save only inside the separate diagnostics directory.
    raster_path.write_bytes(raster_bytes)

    try:
        with rasterio.open(raster_path, "r+") as ds:
            if ds.count != 2:
                raise RuntimeError(
                    f"Expected two bands, got {ds.count}."
                )

            if ds.crs is None:
                raise RuntimeError("Downloaded raster has no CRS.")

            for index, band_name in enumerate(BANDS, start=1):
                ds.set_band_description(index, band_name)

            ds.scales = (0.0001, 0.0001)
            ds.offsets = (0.0, 0.0)
            ds.update_tags(
                REQUESTED_SCENE_ID=str(selected_scene_id),
                REQUESTED_ACQUISITION_DATETIME=str(selected_datetime),
                BAND_ORDER=",".join(BANDS),
                REFLECTANCE_SCALE="10000",
                PROVENANCE_NOTE=(
                    "Requested scene metadata does not prove "
                    "exact per-pixel source provenance."
                ),
            )

            grid = {
                "width": ds.width,
                "height": ds.height,
                "crs": str(ds.crs),
                "transform": list(ds.transform)[:6],
                "bounds": list(ds.bounds),
                "band_descriptions": list(ds.descriptions),
            }

            valid_pixels = 0
            for index in (1, 2):
                valid_pixels += int((ds.read(index) > 0).sum())

            if valid_pixels == 0:
                raise RuntimeError(
                    "The returned raster contains no positive pixels."
                )

    except Exception:
        raster_path.unlink(missing_ok=True)
        raise

    tiles = user_data.get("tiles", [])

    report = {
        "catalog_selected_scene_id": selected_scene_id,
        "catalog_selected_datetime": selected_datetime,
        "request_bands": BANDS,
        "request_time_range": {
            "from": time_from,
            "to": time_to,
        },
        "source_tiles_returned_by_same_request": tiles,
        "source_tile_count": len(tiles),
        "service_version": user_data.get("serviceVersion"),
        "raster": str(raster_path),
        "grid": grid,
        "interpretation": (
            "B04 and B08 were requested together. Returned tile metadata "
            "describes tiles participating in this request; it does not "
            "prove exact per-pixel contribution or the provenance of older TIFFs."
        ),
    }

    metadata_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nCOMBINED REQUEST COMPLETED")
    print("Raster  :", raster_path)
    print("Metadata:", metadata_path)
    print(
        "Grid    :",
        grid["width"],
        "x",
        grid["height"],
        grid["crs"],
    )
    print("Bands   :", grid["band_descriptions"])
    print("Tiles returned:", len(tiles))

    for tile in tiles:
        print(
            " -", tile.get("date"),
            "| shId:", tile.get("shId"),
            "| product:",
            tile.get("sentinel2ProductId") or tile.get("productId"),
        )


if __name__ == "__main__":
    main()
