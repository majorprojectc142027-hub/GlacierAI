from fastapi import APIRouter, Query, HTTPException

from shapely.geometry import mapping



from backend.app.services.rgi_service import (

    search_glaciers,

    find_glacier_by_name,

    find_nearest_glacier,

)

from backend.app.services.glims_service import get_glims_for_rgi



router = APIRouter()





# ==========================================================

# SEARCH GLOBAL RGI 7.0 GLACIERS

# ==========================================================



@router.get("/search")

def search_glaciers_api(

    query: str = Query(

        ...,

        min_length=2,

        description="Glacier name, RGI ID, country, or region",

    ),

    limit: int = Query(10, ge=1, le=50),

):

    query = query.strip()



    if not query:

        raise HTTPException(

            status_code=400,

            detail="Search query cannot be empty.",

        )



    try:

        glaciers = search_glaciers(

            query=query,

            limit=limit,

        )



        results = []



        for glacier in glaciers:

            results.append(

                {

                    "id": str(

                        glacier.get(

                            "rgi_id",

                            glacier.get("id", ""),

                        )

                    ),

                    "name": str(

                        glacier.get(

                            "name",

                            "Unknown Glacier",

                        )

                    ),

                    "latitude": (

                        float(glacier["latitude"])

                        if glacier.get("latitude") is not None

                        else None

                    ),

                    "longitude": (

                        float(glacier["longitude"])

                        if glacier.get("longitude") is not None

                        else None

                    ),

                    "area_km2": (

                        float(glacier["area_km2"])

                        if glacier.get("area_km2") is not None

                        else None

                    ),

                    "region": str(

                        glacier.get(

                            "region",

                            "RGI 7.0",

                        )

                    ),

                    "country": str(

                        glacier.get(

                            "country",

                            "Global",

                        )

                    ),

                    "matched_by": glacier.get(

                        "matched_by",

                        "name",

                    ),

                }

            )



        return {

            "results": results,

            "count": len(results),

            "query": query,

        }



    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=f"RGI glacier search failed: {error}",

        )





# ==========================================================

# FIND NEAREST GLACIER FROM MAP LOCATION

# ==========================================================



@router.get("/nearest")

def get_nearest_glacier(

    latitude: float = Query(

        ...,

        description="Latitude of clicked map location",

    ),

    longitude: float = Query(

        ...,

        description="Longitude of clicked map location",

    ),

    max_distance_km: float = Query(

        50.0,

        gt=0,

        description="Maximum search distance in kilometres",

    ),

):

    try:

        glacier = find_nearest_glacier(

            latitude=latitude,

            longitude=longitude,

            max_distance_km=max_distance_km,

        )



        if not glacier:

            raise HTTPException(

                status_code=404,

                detail=(

                    "No glacier found within "

                    f"{max_distance_km} km of the selected location."

                ),

            )



        geometry_geojson = None

        geometry = glacier.get("geometry")



        if geometry is not None:

            try:

                simplified_geometry = geometry.simplify(

                    0.001,

                    preserve_topology=True,

                )

                geometry_geojson = mapping(

                    simplified_geometry

                )

            except Exception as geometry_error:

                raise HTTPException(

                    status_code=500,

                    detail=(

                        "Failed to convert glacier geometry "

                        f"to GeoJSON: {geometry_error}"

                    ),

                )



        return {

            "status": "success",

            "glacier": {

                "id": str(

                    glacier.get(

                        "rgi_id",

                        glacier.get("id", ""),

                    )

                ),

                "name": str(

                    glacier.get(

                        "name",

                        "Unknown Glacier",

                    )

                ),

                "latitude": (

                    float(glacier["latitude"])

                    if glacier.get("latitude") is not None

                    else None

                ),

                "longitude": (

                    float(glacier["longitude"])

                    if glacier.get("longitude") is not None

                    else None

                ),

                "area_km2": (

                    float(glacier["area_km2"])

                    if glacier.get("area_km2") is not None

                    else None

                ),

                "region": str(

                    glacier.get(

                        "region",

                        "RGI 7.0",

                    )

                ),

                "country": str(

                    glacier.get(

                        "country",

                        "Global",

                    )

                ),

                "distance_km": float(

                    glacier.get(

                        "distance_km",

                        0.0,

                    )

                ),

                "matched_by": glacier.get(

                    "matched_by",

                    "nearest_glacier",

                ),

                "geometry": geometry_geojson,

            },

        }



    except HTTPException:

        raise



    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(

                "Nearest glacier lookup failed: "

                f"{error}"

            ),

        )





# ==========================================================

# GET GLIMS HISTORICAL OUTLINES

# ==========================================================



@router.get("/{rgi_id}/glims")

def get_glacier_glims(rgi_id: str):

    rgi_id = rgi_id.strip()



    if not rgi_id:

        raise HTTPException(

            status_code=400,

            detail="RGI ID cannot be empty.",

        )



    try:

        return get_glims_for_rgi(rgi_id)



    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=f"Failed to load GLIMS data: {error}",

        )





# ==========================================================

# GET COMPLETE GLACIER DETAILS + RGI GEOMETRY

# ==========================================================



@router.get("/{rgi_id}")

def get_glacier_details(rgi_id: str):

    rgi_id = rgi_id.strip()



    if not rgi_id:

        raise HTTPException(

            status_code=400,

            detail="RGI ID cannot be empty.",

        )



    try:

        glacier = find_glacier_by_name(rgi_id)



        if not glacier:

            raise HTTPException(

                status_code=404,

                detail=(

                    f"Glacier with RGI ID "

                    f"'{rgi_id}' was not found."

                ),

            )



        geometry_geojson = None

        geometry = glacier.get("geometry")



        if geometry is not None:

            try:

                simplified_geometry = geometry.simplify(

                    0.001,

                    preserve_topology=True,

                )

                geometry_geojson = mapping(

                    simplified_geometry

                )

            except Exception as geometry_error:

                raise HTTPException(

                    status_code=500,

                    detail=(

                        "Failed to convert glacier geometry "

                        f"to GeoJSON: {geometry_error}"

                    ),

                )



        return {

            "status": "success",

            "glacier": {

                "id": str(

                    glacier.get(

                        "rgi_id",

                        rgi_id,

                    )

                ),

                "name": str(

                    glacier.get(

                        "name",

                        "Unknown Glacier",

                    )

                ),

                "latitude": (

                    float(glacier["latitude"])

                    if glacier.get("latitude") is not None

                    else None

                ),

                "longitude": (

                    float(glacier["longitude"])

                    if glacier.get("longitude") is not None

                    else None

                ),

                "area_km2": (

                    float(glacier["area_km2"])

                    if glacier.get("area_km2") is not None

                    else None

                ),

                "region": str(

                    glacier.get(

                        "region",

                        "RGI 7.0",

                    )

                ),

                "country": str(

                    glacier.get(

                        "country",

                        "Global",

                    )

                ),

                "geometry": geometry_geojson,

            },

        }



    except HTTPException:

        raise



    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(

                "Failed to load glacier details: "

                f"{error}"

            ),

        )
