from pathlib import Path

import re



from fastapi import APIRouter, HTTPException, Query

from shapely.geometry import mapping



from backend.app.services.rgi_service import (

    find_glacier_by_name,

    find_nearest_glacier,

)



from backend.app.services.sentinel_download import (

    download_sentinel_geotiff,

)



from backend.scripts.predict_geotiff import (

    predict_geotiff,

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

        # RGI GEOMETRY → GEOJSON

        # ==================================================



        # Sentinel downloader receives GeoJSON-compatible

        # geometry.



        rgi_geojson = mapping(

            rgi_geometry

        )



        # ==================================================

        # STEP 2

        # DOWNLOAD SENTINEL-2

        # ==================================================



        print()

        print(

            "Step 2/4 : Downloading Sentinel-2..."

        )



        print(

            "AOI source        : RGI glacier polygon"

        )



        # IMPORTANT:

        # Use RGI centroid instead of optional

        # input coordinates.



        rgi_latitude = rgi_glacier[

            "latitude"

        ]



        rgi_longitude = rgi_glacier[

            "longitude"

        ]



        print(

            "RGI latitude      :",

            rgi_latitude,

        )



        print(

            "RGI longitude     :",

            rgi_longitude,

        )



        download_info = (

            download_sentinel_geotiff(



                latitude=rgi_latitude,



                longitude=rgi_longitude,



                output_name=image_filename,



                geometry=rgi_geojson,



            )

        )



        image_path = Path(

            download_info["path"]

        )



        print()

        print(

            "Satellite image downloaded:"

        )



        print(

            image_path

        )



        # ==================================================

        # STEP 3

        # RUN U-NET V3

        # ==================================================



        print()

        print(

            "Step 3/4 : Running U-Net V3..."

        )



        # IMPORTANT:

        # Pass ORIGINAL Shapely geometry.

        #

        # Do NOT pass rgi_geojson.

        #

        # predict_geotiff() internally transforms

        # the geometry to the GeoTIFF CRS.



        analysis = predict_geotiff(



            image_path=image_path,



            output_stem=glacier_id,



            rgi_geometry=rgi_geometry,



        )



                # ==================================================

        # STEP 4

        # CNN GLACIER TYPE CLASSIFICATION

        # ==================================================



        print()

        print(

            "Step 4/4 : Classifying glacier type with CNN..."

        )



        classification = classify_glacier(

            analysis["rgb_path"]

        )



        # ==================================================

        # RGI VERIFICATION INFORMATION

        # ==================================================

        # RGI confirms the inventory glacier and geometry.

        # It is NOT treated as ground truth for the CNN's

        # three glacier-type classes.



        classification["verification"] = {

            "rgi_available": True,

            "rgi_glacier_id": rgi_id,

            "rgi_glacier_name": rgi_name,

            "metadata_source": "RGI 7.0",

            "status": "CNN prediction with RGI inventory metadata",

            "note": (

                "RGI 7.0 confirms the inventory glacier and "

                "its geometry, but does not automatically "

                "establish the CNN glacier-type class."

            ),

        }



        print(

            "CNN Glacier type:",

            classification["display_name"],

        )



        print(

            "CNN Classification confidence:",

            classification["confidence_percent"],

            "%",

        )



        print(

            "Classification source:",

            classification.get(

                "source",

                "CNN prediction",

            ),

        )



        print(

            "RGI verification:",

            classification["verification"]["status"],

        )



        # ==================================================

        # OUTPUT FILENAMES

        # ==================================================



        mask_filename = Path(

            analysis["mask_path"]

        ).name



        mask_png_filename = Path(

            analysis["mask_png_path"]

        ).name



        overlay_filename = Path(

            analysis["overlay_path"]

        ).name



        probability_heatmap_filename = Path(

            analysis["probability_heatmap_path"]

        ).name



        probability_overlay_filename = Path(

            analysis["probability_overlay_path"]

        ).name



        rgi_mask_filename = Path(

            analysis["rgi_mask_png_path"]

        ).name



        # ==================================================

        # FINAL RESPONSE

        # ==================================================



        print()

        print("==========================================")

        print(" LIVE GLACIER ANALYSIS COMPLETE")

        print("==========================================")

        print()



        print(

            "Glacier:",

            rgi_name,

        )



        print(

            "Predicted area:",

            analysis["area_km2"],

            "km²",

        )



        print(

            "Coverage:",
            analysis["coverage_percent_of_valid_image"],
            "%",

        )



        # ==================================================

        # RETURN JSON

        # ==================================================



        return {



            "status":

                "Analysis completed successfully",

            "classification": classification,



            "glacier": {



                "requested_name":

                    name,



                "requested_rgi_id":

                    rgi_id,



                "rgi_id":

                    rgi_id,



                "rgi_name":

                    rgi_name,



                "latitude":

                    rgi_latitude,



                "longitude":

                    rgi_longitude,



                "rgi_area_km2":

                    rgi_area_km2,



                "distance_from_input_km":

                    rgi_distance_km,



                "matched_by":

                    matched_by,



            },



            "satellite": {



                "source":

                    "Sentinel-2 L2A",



                "processing":

                    "Latest suitable observation",



                "aoi_source":

                    "RGI 7.0 glacier polygon",



                "resolution_m":

                    download_info[

                        "resolution_m"

                    ],



                "crs":

                    download_info[

                        "crs"

                    ],



                "cloud_limit_percent":

                    download_info[

                        "cloud_limit_percent"

                    ],



                "observation_window_days":

                    download_info[

                        "days_back"

                    ],



                "scene_id":

                    download_info.get(

                        "selected_scene_id"

                    ),



                "observation_date":

                    download_info.get(

                        "selected_datetime"

                    ),



            },



            "model": {



                "name":

                    "U-Net V3",



                "type":

                    "Semantic Segmentation",



                "checkpoint":

                    "best_unet_v3.pth",



            },



            "analysis": {



                "glacier_pixels":

                    analysis[

                        "glacier_pixels"

                    ],



                # Number of valid pixels used for coverage calculation.
                "total_pixels": analysis["valid_image_pixels"],



                "coverage_percent": analysis["coverage_percent_of_valid_image"],



                "area_m2":

                    analysis[

                        "area_m2"

                    ],



                "area_km2":

                    analysis[

                        "area_km2"

                    ],



                "rgi_polygon_applied":

                    analysis[

                        "rgi_polygon_applied"

                    ],

                "xai":

                    analysis["xai"],



            },



            "outputs": {



                "mask":

                    f"/datasets/live/"

                    f"predictions/"

                    f"{mask_filename}",



                "mask_png":

                    f"/datasets/live/"

                    f"predictions/"

                    f"{mask_png_filename}",



                "overlay":

                    f"/datasets/live/"

                    f"predictions/"

                    f"{overlay_filename}",



                "probability_heatmap":

                    f"/datasets/live/"

                    f"predictions/"

                    f"{probability_heatmap_filename}",



                "probability_overlay":

                    f"/datasets/live/"

                    f"predictions/"

                    f"{probability_overlay_filename}",



                "rgi_mask":

                    f"/datasets/live/"

                    f"predictions/"

                    f"{rgi_mask_filename}",



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