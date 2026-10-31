
from pathlib import Path
import re

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.debris_service import analyze_debris
from backend.app.services.rgi_service import find_glacier_by_name
from backend.app.services.sentinel_download import (
    download_sentinel_geotiff,
    download_sentinel_spectral_geotiff,
)
from backend.scripts.predict_geotiff import predict_geotiff
import traceback

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LIVE_DIR = PROJECT_ROOT / "datasets" / "live"
PREDICTION_DIR = LIVE_DIR / "predictions"


def safe_rgi_name(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.strip().lower(),
    ).strip("_")



def resolve_glacier(glacier: str | None, rgi_id: str | None):
    for value in (rgi_id, glacier):
        if not value:
            continue

        result = find_glacier_by_name(value.strip(), limit=20)

        if not result:
            continue

        # The RGI service returns a single dictionary.
        if isinstance(result, dict):
            return result

        # Keep compatibility if the service ever returns a list.
        if isinstance(result, list):
            if rgi_id:
                exact = next(
                    (
                        item
                        for item in result
                        if item.get("rgi_id", "").lower()
                        == rgi_id.strip().lower()
                    ),
                    None,
                )
                if exact:
                    return exact

            if result:
                return result[0]

    return None



@router.get("/analyze")
def analyze_debris_api(
    glacier: str | None = Query(None),
    rgi_id: str | None = Query(None),
    latitude: float | None = Query(None),
    longitude: float | None = Query(None),
):
    try:
        if not glacier and not rgi_id:
            raise HTTPException(
                status_code=400,
                detail="Provide either glacier or rgi_id.",
            )

        resolved = resolve_glacier(glacier, rgi_id)

        if not resolved:
            raise HTTPException(
                status_code=404,
                detail=f"Could not resolve glacier '{glacier or rgi_id}'.",
            )

        resolved_name = resolved.get("name") or glacier or rgi_id
        resolved_rgi_id = resolved.get("rgi_id")

        if not resolved_rgi_id:
            raise HTTPException(
                status_code=404,
                detail="Resolved glacier has no RGI ID.",
            )

        lat = latitude if latitude is not None else resolved.get("latitude")
        lon = longitude if longitude is not None else resolved.get("longitude")
        geometry = resolved.get("geometry")

        if lat is None or lon is None:
            raise HTTPException(
                status_code=400,
                detail="Glacier latitude and longitude are required.",
            )

        if geometry is None:
            raise HTTPException(
                status_code=400,
                detail="RGI glacier geometry is unavailable.",
            )

        safe_name = safe_rgi_name(resolved_rgi_id)

        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Download fresh RGB imagery and get its selected observation metadata.
        rgb_result = download_sentinel_geotiff(
            latitude=float(lat),
            longitude=float(lon),
            output_name=f"{safe_name}_debris_rgb.tif",
        )

        rgb_path = Path(rgb_result["path"])

        # 2. Generate a fresh U-Net glacier mask without requiring Live Analysis.
        prediction = predict_geotiff(
            image_path=rgb_path,
            output_stem=f"{safe_name}_debris",
            rgi_geometry=geometry,
        )

        mask_path = Path(prediction["mask_path"])

        if not mask_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"U-Net mask was not created: {mask_path}",
            )

        # 3. Download spectral bands using the RGB observation's time.
        # The current downloader uses a time window; it does not guarantee
        # an exact scene-ID match.
        selected_datetime = rgb_result.get("selected_datetime")
        selected_scene_id = rgb_result.get("selected_scene_id")
        selected_cloud_limit = rgb_result.get("cloud_limit_percent")

        if not selected_datetime or not selected_scene_id:
            raise HTTPException(
                status_code=502,
                detail="RGB downloader did not return observation metadata.",
            )

        spectral_result = download_sentinel_spectral_geotiff(
            latitude=float(lat),
            longitude=float(lon),
            selected_datetime=selected_datetime,
            selected_scene_id=selected_scene_id,
            selected_cloud_limit=selected_cloud_limit,
            output_name=f"{safe_name}_debris_spectral.tif",
        )

        spectral_path = Path(spectral_result["path"])

        # 4. Estimate debris-covered pixels using the existing service.
        output_path = PREDICTION_DIR / (
            f"{safe_name}_debris_affected_mask.tif"
        )

        result = analyze_debris(
            spectral_path=spectral_path,
            glacier_mask_path=mask_path,
            output_path=output_path,
        )

        result.update(
            {
                "glacier": resolved_name,
                "rgi_id": resolved_rgi_id,
                "rgb_image": str(rgb_path),
                "glacier_mask": str(mask_path),
                "spectral_image": str(spectral_path),
                "sentinel": {
                    "rgb_scene_id": selected_scene_id,
                    "rgb_acquisition_datetime": selected_datetime,
                    "spectral_scene_id_reported": (
                        spectral_result.get("selected_scene_id")
                    ),
                    "spectral_acquisition_datetime_reported": (
                        spectral_result.get("selected_datetime")
                    ),
                    "exact_scene_match_verified": False,
                    "note": (
                        "The current spectral downloader uses a time-window "
                        "filter and does not independently verify the exact "
                        "scene ID. Interpret the spectral scene metadata "
                        "accordingly."
                    ),
                },
                "rgb_download": rgb_result,
                "spectral_download": spectral_result,
            }
        )

        return result

    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")