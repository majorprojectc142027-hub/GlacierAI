
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import io
import json
import math
import tarfile
import requests

from backend.app.services.sentinel_service import get_access_token
from backend.app.services.sentinel_catalog import search_historical_sentinel2
from backend.app.services.rgi_service import find_glacier_by_name
from backend.app.services.sentinel_download import (
    get_utm_crs,
    get_crs_url,
    create_rgi_utm_bbox,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
PIXEL_SIZE = 12
MAX_PIXELS = 2400
TIME_TOLERANCE_MINUTES = 30

def build_diagnostic_evalscript(target_datetime, target_scene_id):
    target_timestamp_ms = int(target_datetime.timestamp() * 1000)
    target_product_id = json.dumps(str(target_scene_id))
    return f"""
//VERSION=3
var tilesBeforeFilter = [];

function preProcessScenes(collections) {{
    var targetTime = {target_timestamp_ms};
    var targetProductId = {target_product_id};
    var toleranceMs = {TIME_TOLERANCE_MINUTES} * 60 * 1000;

    tilesBeforeFilter = JSON.parse(JSON.stringify(collections.scenes.tiles || []));

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
        if (samples[i].dataMask === 1) {{
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

function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {{
    outputMetadata.userData = {{
        "tilesBeforeFilter": tilesBeforeFilter,
        "tilesAfterFilter": scenes.tiles || [],
        "inputMetadata": inputMetadata,
        "targetProductId": {target_product_id},
        "targetDatetime": "{target_datetime.isoformat()}"
    }};
}}
"""

def choose_observation(observations, year):
    candidates = [
        item for item in observations
        if int(item.get("year") or str(item.get("datetime", ""))[:4]) == year
    ]
    if not candidates:
        raise RuntimeError(f"No Catalog observations found for {year}.")
    return min(
        candidates,
        key=lambda item: (
            float("inf")
            if item.get("cloud_cover") is None
            else float(item["cloud_cover"])
        ),
    )

def make_tile_bboxes(bbox):
    min_x, min_y, max_x, max_y = bbox
    tile_size = MAX_PIXELS * PIXEL_SIZE
    result = []
    x = min_x
    while x < max_x:
        tile_max_x = min(x + tile_size, max_x)
        y = min_y
        while y < max_y:
            tile_max_y = min(y + tile_size, max_y)
            result.append((x, y, tile_max_x, tile_max_y))
            y = tile_max_y
        x = tile_max_x
    return result

def extract_userdata(response_content):
    with tarfile.open(fileobj=io.BytesIO(response_content), mode="r:*") as archive:
        members = archive.getmembers()
        userdata_member = next(
            (member for member in members if Path(member.name).name == "userdata.json"),
            None,
        )
        if userdata_member is None:
            raise RuntimeError(
                "TAR response did not contain userdata.json. "
                f"Archive members: {[member.name for member in members]}"
            )
        file_obj = archive.extractfile(userdata_member)
        if file_obj is None:
            raise RuntimeError("Could not read userdata.json from TAR response.")
        return json.load(file_obj)

def main():
    parser = argparse.ArgumentParser(
        description="Verify Sentinel-2 Catalog scene selection against Process API metadata."
    )
    parser.add_argument(
        "--glacier",
        required=True,
        help="Glacier name or exact RGI ID.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Year to test (default: 2026).",
    )
    parser.add_argument("--max-cloud", type=float, default=30)
    args = parser.parse_args()

    print("\nResolving glacier:", args.glacier)
    glacier = find_glacier_by_name(args.glacier)

    if not glacier:
        raise RuntimeError(f"Glacier not found: {args.glacier}")

    latitude = float(glacier["latitude"])
    longitude = float(glacier["longitude"])
    geometry = glacier["geometry"]
    rgi_id = glacier.get("rgi_id")
    glacier_name = glacier.get("name", args.glacier)

    print("Glacier:", glacier_name)
    print("RGI ID:", rgi_id)
    print("Coordinates:", latitude, longitude)

    print(f"\nSearching Catalog for {args.year}...")
    observations = search_historical_sentinel2(
        latitude=latitude,
        longitude=longitude,
        start_year=args.year,
        end_year=args.year,
        max_cloud_cover=args.max_cloud,
    )

    observation = choose_observation(observations, args.year)
    scene_id = observation.get("id")
    datetime_string = observation.get("datetime")
    cloud_cover = observation.get("cloud_cover")

    if not scene_id or not datetime_string:
        raise RuntimeError("Selected Catalog observation has no ID or datetime.")

    target_datetime = datetime.fromisoformat(
        datetime_string.replace("Z", "+00:00")
    ).astimezone(timezone.utc)

    start_datetime = target_datetime - timedelta(minutes=TIME_TOLERANCE_MINUTES)
    end_datetime = target_datetime + timedelta(minutes=TIME_TOLERANCE_MINUTES)

    print("\n========== CATALOG SELECTION ==========")
    print("Scene ID:", scene_id)
    print("Acquisition:", datetime_string)
    print("Cloud cover:", cloud_cover)
    print("Process API window:", start_datetime.isoformat(), "to", end_datetime.isoformat())

    target_crs = get_utm_crs(latitude, longitude)
    bbox = create_rgi_utm_bbox(
        geometry=geometry,
        target_crs=target_crs,
    )
    tile_bboxes = make_tile_bboxes(bbox)

    print("\nTarget CRS:", target_crs)
    print("Glacier bbox:", bbox)
    print("Process API requests:", len(tile_bboxes))

    token = get_access_token()
    evalscript = build_diagnostic_evalscript(target_datetime, scene_id)

    diagnostic = {
        "glacier": glacier_name,
        "rgi_id": rgi_id,
        "year": args.year,
        "catalog_selection": {
            "id": scene_id,
            "datetime": datetime_string,
            "cloud_cover": cloud_cover,
        },
        "process_api": {
            "time_from": start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_to": end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "max_cloud_cover": (
                float(cloud_cover)
                if cloud_cover is not None
                else args.max_cloud
            ),
            "mosaicking_order": "leastCC",
            "target_crs": target_crs,
            "pixel_size": PIXEL_SIZE,
        },
        "tile_results": [],
    }

    selected_cloud = (
        float(cloud_cover)
        if cloud_cover is not None
        else args.max_cloud
    )

    for index, tile_bbox in enumerate(tile_bboxes, start=1):
        print(f"\nRequesting Process API tile {index}/{len(tile_bboxes)}")
        print("BBOX:", tile_bbox)

        payload = {
            "input": {
                "bounds": {
                    "bbox": list(tile_bbox),
                    "properties": {"crs": get_crs_url(target_crs)},
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "to": end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        },
                        "mosaickingOrder": "leastCC",
                        "maxCloudCoverage": selected_cloud,
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

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/x-tar",
        }

        response = requests.post(
            PROCESS_URL,
            json=payload,
            headers=headers,
            timeout=180,
        )

        if response.status_code != 200:
            result = {
                "tile_index": index,
                "bbox": list(tile_bbox),
                "http_status": response.status_code,
                "error": response.text[:3000],
            }
            diagnostic["tile_results"].append(result)
            print("Request failed:", response.status_code)
            print(response.text[:1000])
            continue

        try:
            userdata = extract_userdata(response.content)
        except Exception as error:
            diagnostic["tile_results"].append({
                "tile_index": index,
                "bbox": list(tile_bbox),
                "http_status": response.status_code,
                "metadata_error": str(error),
            })
            print("Metadata extraction failed:", error)
            continue

        before = userdata.get("tilesBeforeFilter", [])
        after = userdata.get("tilesAfterFilter", [])

        result = {
            "tile_index": index,
            "bbox": list(tile_bbox),
            "http_status": response.status_code,
            "response_bytes": len(response.content),
            "tiles_before_filter": before,
            "tiles_after_filter": after,
            "input_metadata": userdata.get("inputMetadata"),
            "target_product_id": userdata.get("targetProductId"),
            "target_datetime": userdata.get("targetDatetime"),
        }
        diagnostic["tile_results"].append(result)

        print("Tiles before filter:", len(before))
        print("Tiles after filter:", len(after))

        for tile in before:
            print(
                "  BEFORE:",
                "productId =", tile.get("productId"),
                "| date =", tile.get("date"),
                "| dataPath =", tile.get("dataPath"),
                "| shId =", tile.get("shId"),
            )

        for tile in after:
            print(
                "  AFTER :",
                "productId =", tile.get("productId"),
                "| date =", tile.get("date"),
                "| dataPath =", tile.get("dataPath"),
                "| shId =", tile.get("shId"),
            )

    successful = [
        item for item in diagnostic["tile_results"]
        if item.get("http_status") == 200 and "tiles_after_filter" in item
    ]
    before_all = [
        tile
        for item in successful
        for tile in item["tiles_before_filter"]
    ]
    after_all = [
        tile
        for item in successful
        for tile in item["tiles_after_filter"]
    ]

    diagnostic["summary"] = {
        "successful_process_requests": len(successful),
        "failed_process_requests": len(diagnostic["tile_results"]) - len(successful),
        "total_tiles_before_filter": len(before_all),
        "total_tiles_after_filter": len(after_all),
        "exact_product_id_matches_after_filter": sum(
            tile.get("productId") == scene_id for tile in after_all
        ),
        "catalog_scene_id": scene_id,
        "catalog_scene_id_found_before_filter": any(
            tile.get("productId") == scene_id for tile in before_all
        ),
    }

    output_dir = (
        PROJECT_ROOT
        / "datasets"
        / "diagnostics"
        / "process_api_scene_verification"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"scene_verification_{args.year}_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(diagnostic, file, indent=2, ensure_ascii=False)

    print("\n========== VERIFICATION SUMMARY ==========")
    print(json.dumps(diagnostic["summary"], indent=2))
    print("\nFull metadata saved to:")
    print(output_path)

if __name__ == "__main__":
    main()
