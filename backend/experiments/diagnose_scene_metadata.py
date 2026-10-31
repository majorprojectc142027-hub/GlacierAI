from datetime import datetime, timedelta
from io import BytesIO
import json
import tarfile

import requests

from backend.app.services.sentinel_download import (
    PROCESS_URL,
    PIXEL_SIZE,
    LIVE_AOI_HALF_SIZE_DEGREES,
    create_utm_bbox,
    get_utm_crs,
    get_crs_url,
)
from backend.app.services.sentinel_service import get_access_token
from backend.app.services.sentinel_catalog import search_sentinel2

LATITUDE = 30.825184
LONGITUDE = 79.073847


def get_scene_metadata(bands, selected_datetime, bbox, target_crs, token):
    observation_time = datetime.fromisoformat(
        selected_datetime.replace("Z", "+00:00")
    )

    evalscript = f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{ bands: {json.dumps(bands)} }}],
        mosaicking: Mosaicking.TILE,
        output: {{ id: "default", bands: 1, sampleType: "UINT8" }}
    }};
}}
function evaluatePixel(sample) {{
    return [0];
}}
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {{
    outputMetadata.userData = {{
        tiles: scenes.tiles,
        serviceVersion: inputMetadata.serviceVersion
    }};
}}
"""

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": get_crs_url(target_crs)},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": (
                            observation_time - timedelta(minutes=2)
                        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "to": (
                            observation_time + timedelta(minutes=2)
                        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                    "mosaickingOrder": "mostRecent",
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
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/x-tar",
        },
        timeout=120,
    )
    response.raise_for_status()

    with tarfile.open(fileobj=BytesIO(response.content), mode="r:*") as archive:
        member = next(
            (item for item in archive.getmembers()
             if item.name.endswith("userdata.json")),
            None,
        )
        if member is None:
            raise RuntimeError(
                "No userdata.json found. Response members: "
                + ", ".join(archive.getnames())
            )
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("Could not read userdata.json")
        return json.load(extracted)


def main():
    token = get_access_token()
    target_crs = get_utm_crs(LATITUDE, LONGITUDE)
    bbox = create_utm_bbox(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        target_crs=target_crs,
        half_size_degrees=LIVE_AOI_HALF_SIZE_DEGREES,
    )

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

    print("Catalog scene:", selected.get("id"))
    print("Catalog datetime:", selected_datetime)
    print("\nRequesting RGB source metadata...")
    rgb = get_scene_metadata(
        ["B04", "B03", "B02"],
        selected_datetime, bbox, target_crs, token,
    )

    print("Requesting spectral source metadata...")
    spectral = get_scene_metadata(
        ["B08", "B11", "B12"],
        selected_datetime, bbox, target_crs, token,
    )

    with open(
        "datasets/live/scene_metadata_diagnostic.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "catalog_scene_id": selected.get("id"),
                "catalog_datetime": selected_datetime,
                "rgb": rgb,
                "spectral": spectral,
            },
            file,
            indent=2,
        )

    rgb_tiles = rgb.get("tiles", [])
    spectral_tiles = spectral.get("tiles", [])

    def tile_keys(tiles):
        return {
            (
                tile.get("date"),
                tile.get("shId"),
                tile.get("dataPath"),
            )
            for tile in tiles
        }

    rgb_keys = tile_keys(rgb_tiles)
    spectral_keys = tile_keys(spectral_tiles)

    print("\nRGB tiles:", len(rgb_tiles))
    print("Spectral tiles:", len(spectral_tiles))
    print("Matching tile metadata:", rgb_keys == spectral_keys)
    print("RGB only:", sorted(map(str, rgb_keys - spectral_keys)))
    print("Spectral only:", sorted(map(str, spectral_keys - rgb_keys)))
    print("\nFull metadata saved to datasets/live/scene_metadata_diagnostic.json")


if __name__ == "__main__":
    main()
