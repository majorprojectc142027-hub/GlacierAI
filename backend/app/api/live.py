from pathlib import Path
import re

from fastapi import APIRouter, HTTPException, Query

from backend.app.services.rgi_service import (
    find_glacier_by_name,
    find_nearest_glacier,
)
from backend.scripts.predict_large_glacier import (
    analyze_large_glacier,
    glacier_output_key,
    PREDICTION_DIR,
    OUTPUT_DIR,
)
from backend.services.glacier_classifier_service import (
    classify_glacier,
)

# ==========================================================
# ROUTER
# ==========================================================
router = APIRouter()
# ==========================================================
# PROJECT PATH
# ==========================================================
PROJECT_ROOT = (
    Path(__file__).resolve().parents[3]
)
# ==========================================================
# NAME CLEANING
# ==========================================================
def create_safe_name(
    name: str,
) -> str:
    safe_name = (
        name.lower().strip()
    )
    safe_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        safe_name,
    )
    safe_name = (
        safe_name.strip("_")
    )
    if not safe_name:
        safe_name = "glacier"
    return safe_name
# ==========================================================
# LIVE GLACIER ANALYSIS
# ==========================================================
@router.get("/analyze-live")
def analyze_live(
    name: str = Query(
        ...,
        description="Glacier name",
    ),
    rgi_id: str | None = Query(
        None,
        description="Exact RGI 7.0 glacier ID",
    ),
    latitude: float | None = Query(
        None,
        description="Optional latitude",
    ),
    longitude: float | None = Query(
        None,
        description="Optional longitude",
    ),
):
    try:
      # ==================================================
        # NORMALIZE RGI ID
        # ==================================================
        # Accept both:
        #
        # RGI2000-v7.0-G-15-01064
        #
        # and:
        #
        # RGI ID: RGI2000-v7.0-G-15-01064
        #
        # Internally we always use the clean RGI ID.
        if rgi_id:
            rgi_id = rgi_id.strip()
            if rgi_id.upper().startswith("RGI ID:"):
                rgi_id = (
                    rgi_id.split(":", 1)[1]
                    .strip()
                )
            elif rgi_id.upper().startswith("RGI:"):
                rgi_id = (
                    rgi_id.split(":", 1)[1]
                    .strip()
                )
        # ==================================================
        # CREATE SAFE FILE NAME
        # ==================================================
        # Use RGI ID whenever available.
        # This prevents filename collisions for glaciers
        # having the same name.
        glacier_id = create_safe_name(
            rgi_id or name
        )
        image_filename = (
            f"{glacier_id}_latest.tif"
        )
        # ==================================================
        # START
        # ==================================================
        print()
        print("==========================================")
        print(" GLACIERAI - LIVE GLACIER ANALYSIS")
        print("==========================================")
        print()
        print(
            "Requested glacier :",
            name,
        )
        print(
            "Requested RGI ID  :",
            rgi_id,
        )
        print(
            "Latitude          :",
            latitude,
        )
        print(
            "Longitude         :",
            longitude,
        )
        # ==================================================
        # STEP 1
        # FIND RGI GLACIER
        # ==================================================
        print()
        print(
            "Step 1/4 : Finding glacier in RGI 7.0..."
        )
        # --------------------------------------------------
        # FIRST: EXACT RGI ID
        # --------------------------------------------------
        rgi_glacier = None
        if rgi_id:
            print(
                "Searching global RGI by exact ID:",
                rgi_id,
            )
            rgi_glacier = find_glacier_by_name(
                query=rgi_id,
                limit=1,
            )
            if rgi_glacier is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"RGI glacier '{rgi_id}' "
                        "was not found in RGI 7.0."
                    ),
                )
        # --------------------------------------------------
        # SECOND: SEARCH BY NAME
        # --------------------------------------------------
        if rgi_glacier is None and name:
            print(
                "Searching global RGI by name:",
                name,
            )
            rgi_glacier = find_glacier_by_name(
                query=name,
                limit=20,
            )
        # --------------------------------------------------
        # THIRD: FALLBACK TO COORDINATES
        # --------------------------------------------------
        if rgi_glacier is None:
            if (
                latitude is not None
                and
                longitude is not None
            ):
                print(
                    "Name search failed."
                )
                print(
                    "Falling back to coordinates..."
                )
                rgi_glacier = find_nearest_glacier(
                    latitude=latitude,
                    longitude=longitude,
                    max_distance_km=20.0,
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No RGI glacier found "
                        f"with the name '{name}'. "
                        f"Please enter an actual "
                        f"glacier name or provide "
                        f"coordinates."
                    ),
                )
        # --------------------------------------------------
        # STILL NOT FOUND
        # --------------------------------------------------
        if rgi_glacier is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No glacier matching "
                    f"'{name}' was found in "
                    f"RGI 7.0."
                ),
            )
        # ==================================================
        # EXTRACT RGI INFORMATION
        # ==================================================
        rgi_id = rgi_glacier[
            "rgi_id"
        ]
        rgi_name = rgi_glacier[
            "name"
        ]
        rgi_area_km2 = rgi_glacier[
            "area_km2"
        ]
        rgi_distance_km = rgi_glacier.get(
            "distance_km",
            0,
        )
        matched_by = rgi_glacier.get(
            "matched_by",
            "name",
        )
        # IMPORTANT:
        # Keep the original Shapely geometry
        # for predict_geotiff().
        rgi_geometry = rgi_glacier[
            "geometry"
        ]
        if rgi_geometry is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "RGI glacier geometry "
                    "was not returned."
                ),
            )
        print()
        print(
            "RGI glacier found!"
        )
        print(
            "RGI ID            :",
            rgi_id,
        )
        print(
            "RGI name          :",
            rgi_name,
        )
        print(
            "RGI area          :",
            rgi_area_km2,
            "km²",
        )
        print(
            "Distance          :",
            rgi_distance_km,
            "km",
        )
        print(
            "Matched by        :",
            matched_by,
        )
        # ==================================================
        # STEP 2
        # RUN TILED SENTINEL-2 SEGMENTATION
        # ==================================================
        # Reuse the exact RGI record already resolved above.
        # The tiled pipeline downloads/processes only tiles
        # intersecting this glacier's RGI geometry.
        rgi_latitude = rgi_glacier["latitude"]
        rgi_longitude = rgi_glacier["longitude"]

        print()
        print("Step 2/3 : Running tiled Sentinel-2 segmentation...")
        print("AOI source        : RGI 7.0 glacier polygon")
        print("Analysis mode     : tiled full-glacier coverage")
        print("RGI latitude      :", rgi_latitude)
        print("RGI longitude     :", rgi_longitude)

        tiled_analysis = analyze_large_glacier(
            glacier_record=rgi_glacier
        )

        mask_path = Path(tiled_analysis["mask"])
        mask_png_path = (
            Path(tiled_analysis["mask_png"])
            if tiled_analysis.get("mask_png")
            else None
        )

        # ==================================================
        # STEP 3
        # CNN GLACIER TYPE CLASSIFICATION
        # ==================================================
        # The tiled pipeline creates RGB preview images per tile.
        # Classify the first available RGB tile. This is a tile-level
        # classification, not a classification of the merged mask.
        print()
        print("Step 3/3 : Classifying glacier type with CNN...")

        glacier_key = glacier_output_key(rgi_glacier)
        rgb_candidates = sorted(
            PREDICTION_DIR.glob(
                f"{glacier_key}_tile_*_full_rgb.png"
            )
        )

        classification = None
        if rgb_candidates:
            try:
                classification = classify_glacier(
                    str(rgb_candidates[0])
                )
                classification["classification_scope"] = (
                    "first available satellite tile"
                )
                classification["verification"] = {
                    "rgi_available": True,
                    "rgi_glacier_id": rgi_id,
                    "rgi_glacier_name": rgi_name,
                    "metadata_source": "RGI 7.0",
                    "status": (
                        "CNN tile classification with RGI inventory metadata"
                    ),
                    "note": (
                        "This classifies one available satellite tile; "
                        "it is not a glacier-wide classification."
                    ),
                }
                print(
                    "CNN Glacier type:",
                    classification.get("display_name", "Unavailable"),
                )
                print(
                    "CNN Classification confidence:",
                    classification.get("confidence_percent"),
                    "%",
                )
            except Exception as classification_error:
                # Segmentation should remain usable even if the
                # optional classification model fails.
                print(
                    "CNN classification failed:",
                    str(classification_error),
                )
                classification = {
                    "status": "unavailable",
                    "message": str(classification_error),
                    "classification_scope": None,
                }
        else:
            classification = {
                "status": "unavailable",
                "message": "No RGB tile image was available for classification.",
                "classification_scope": None,
            }

        # ==================================================
        # OUTPUT URLS
        # ==================================================
        # The merged outputs are saved under datasets/live/large_glacier.
        # These URLs assume the existing app serves datasets/live at
        # the /datasets/live URL prefix.
        mask_filename = mask_path.name
        mask_png_filename = mask_png_path.name if mask_png_path else None

        complete = bool(tiled_analysis.get("complete", False))
        status_message = (
            "Analysis completed successfully"
            if complete
            else "Analysis partially completed; one or more tiles failed"
        )

        print()
        print("==========================================")
        print(" TILED GLACIER ANALYSIS FINISHED")
        print("==========================================")
        print("Glacier          :", rgi_name)
        print("Predicted area   :", tiled_analysis["predicted_area_km2"], "km²")
        print(
            "Tiles processed  :",
            tiled_analysis["tiles_processed"],
            "/",
            tiled_analysis["tiles_total"],
        )
        print("Complete         :", complete)

        # ==================================================
        # RETURN JSON
        # ==================================================
        return {
            "status": status_message,
            "classification": classification,
            "glacier": {
                "requested_name": name,
                "requested_rgi_id": rgi_id,
                "rgi_id": rgi_id,
                "rgi_name": rgi_name,
                "latitude": rgi_latitude,
                "longitude": rgi_longitude,
                "rgi_area_km2": rgi_area_km2,
                "distance_from_input_km": rgi_distance_km,
                "matched_by": matched_by,
            },
            "satellite": {
                "source": "Sentinel-2 L2A",
                "processing": "Latest suitable observation downloaded per tile",
                "aoi_source": "RGI 7.0 glacier polygon, tiled",
                "resolution_m": None,
                "crs": None,
                "cloud_limit_percent": None,
                "observation_window_days": 60,
                "scene_id": None,
                "observation_date": None,
                "note": (
                    "The tiled pipeline does not currently return combined "
                    "scene metadata for the full glacier."
                ),
            },
            "model": {
                "name": "U-Net V3",
                "type": "Semantic Segmentation",
                "checkpoint": "best_unet_v3.pth",
                "mode": "tiled",
            },
            "analysis": {
                "glacier_pixels": tiled_analysis["predicted_pixels"],
                "total_pixels": None,
                "coverage_percent": None,
                "area_m2": tiled_analysis["predicted_area_m2"],
                "area_km2": tiled_analysis["predicted_area_km2"],
                "rgi_polygon_applied": True,
                "xai": None,
                "complete": complete,
                "tiles_total": tiled_analysis["tiles_total"],
                "tiles_processed": tiled_analysis["tiles_processed"],
                "tiles_failed": tiled_analysis.get("tiles_failed", 0),
                "failed_tiles": tiled_analysis.get("failed_tiles", []),
                "area_note": (
                    None
                    if complete
                    else (
                        "Area is based on successful tiles only and may be "
                        "an underestimate because processing was incomplete."
                    )
                ),
            },
            "outputs": {
                "mask": f"/datasets/live/large_glacier/{mask_filename}",
                "mask_png": (
                    f"/datasets/live/large_glacier/{mask_png_filename}"
                    if mask_png_filename
                    else None
                ),
                # The tiled pipeline currently creates a binary mask PNG,
                # not a combined RGB overlay or probability products.
                "overlay": (
                    f"/datasets/live/large_glacier/{mask_png_filename}"
                    if mask_png_filename
                    else None
                ),
                "probability_heatmap": None,
                "probability_overlay": None,
                "rgi_mask": None,
            },
        }
    # ======================================================
    # HTTP ERRORS
    # ======================================================
    except HTTPException:
        raise
    # ======================================================
    # GENERAL ERRORS
    # ======================================================
    except Exception as error:
        print()
        print("==========================================")
        print(" LIVE ANALYSIS FAILED")
        print("==========================================")
        print(
            error
        )
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
