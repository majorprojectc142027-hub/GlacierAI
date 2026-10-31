from datetime import datetime, timedelta, timezone
import requests
from backend.app.services.sentinel_service import get_access_token
# ==========================================================
# COPERNICUS CATALOG API
# ==========================================================
CATALOG_URL = (
    "https://sh.dataspace.copernicus.eu/catalog/v1/search"
)
# ==========================================================
# DEFAULT SETTINGS
# ==========================================================
COLLECTION = "sentinel-2-l2a"
DEFAULT_CLOUD_COVER = 30
DEFAULT_LIMIT = 100
# ==========================================================
# BUILD BBOX
# ==========================================================
def create_bbox(
    latitude: float,
    longitude: float,
    half_size: float = 0.15,
):
    """
    Create a small bounding box around the glacier.
    The coordinates are:
        min longitude
        min latitude
        max longitude
        max latitude
    """
    return [
        longitude - half_size,
        latitude - half_size,
        longitude + half_size,
        latitude + half_size,
    ]
# ==========================================================
# SEARCH SENTINEL-2
# ==========================================================
def search_sentinel2(
    latitude: float,
    longitude: float,
    days_back: int = 60,
    max_cloud_cover: float = DEFAULT_CLOUD_COVER,
    limit: int = DEFAULT_LIMIT,
):
    """
    Search for recent Sentinel-2 L2A observations.
    This preserves the original live-search behavior,
    while allowing a larger result limit and cloud filter.
    """
    token = get_access_token()
    now = datetime.now(
        timezone.utc
    )
    start = (
        now
        - timedelta(
            days=days_back
        )
    )
    bbox = create_bbox(
        latitude=latitude,
        longitude=longitude,
    )
    payload = {
        "bbox": bbox,
        "datetime": (
            f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
            f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        ),
        "collections": [
            COLLECTION
        ],
        "limit": limit,
        "filter": (
            f"eo:cloud_cover <= "
            f"{max_cloud_cover}"
        ),
    }
    headers = {
        "Authorization":
            f"Bearer {token}",
        "Content-Type":
            "application/json",
    }
    response = requests.post(
        CATALOG_URL,
        json=payload,
        headers=headers,
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Sentinel-2 catalog search failed "
            f"(HTTP {response.status_code}): "
            f"{response.text}"
        )
    data = response.json()
    features = data.get(
        "features",
        []
    )
    results = []
    for feature in features:
        properties = (
            feature.get(
                "properties",
                {}
            )
        )
        cloud_cover = (
            properties.get(
                "eo:cloud_cover"
            )
        )
        results.append(
            {
                "id":
                    feature.get(
                        "id"
                    ),
                "datetime":
                    properties.get(
                        "datetime"
                    ),
                "cloud_cover":
                    cloud_cover,
                "collection":
                    COLLECTION,
            }
        )
    # ------------------------------------------------------
    # Sort newest first
    # ------------------------------------------------------
    results.sort(
        key=lambda item:
            item.get(
                "datetime"
            )
            or "",
        reverse=True,
    )
    print(
        f"Catalog returned {len(results)} observation(s) "
        f"within {days_back} day(s), cloud <= {max_cloud_cover}%."
    )
    if results:
        print(
            "Newest observation:",
            results[0].get("datetime"),
            "| cloud:",
            results[0].get("cloud_cover"),
        )
    return results
# ==========================================================
# SEARCH SENTINEL-2 FOR A SPECIFIC DATE RANGE
# ==========================================================
def search_sentinel2_date_range(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = DEFAULT_CLOUD_COVER,
    limit: int = DEFAULT_LIMIT,
):
    """
    Search Sentinel-2 L2A observations within
    an explicit date range.
    Parameters
    ----------
    latitude:
        Glacier latitude.
    longitude:
        Glacier longitude.
    start_date:
        Start date in YYYY-MM-DD format.
    end_date:
        End date in YYYY-MM-DD format.
    max_cloud_cover:
        Maximum accepted cloud percentage.
    limit:
        Maximum number of catalog results.
    """
    token = get_access_token()
    # ------------------------------------------------------
    # Convert dates into UTC datetime strings
    # ------------------------------------------------------
    start_datetime = (
        f"{start_date}T00:00:00Z"
    )
    end_datetime = (
        f"{end_date}T23:59:59Z"
    )
    # ------------------------------------------------------
    # Create glacier search area
    # ------------------------------------------------------
    bbox = create_bbox(
        latitude=latitude,
        longitude=longitude,
    )
    # ------------------------------------------------------
    # Catalog request
    # ------------------------------------------------------
    payload = {
        "bbox": bbox,
        "datetime": (
            f"{start_datetime}/"
            f"{end_datetime}"
        ),
        "collections": [
            COLLECTION
        ],
        "limit": limit,
        "filter": (
            f"eo:cloud_cover <= "
            f"{max_cloud_cover}"
        ),
    }
    headers = {
        "Authorization":
            f"Bearer {token}",
        "Content-Type":
            "application/json",
    }
    response = requests.post(
        CATALOG_URL,
        json=payload,
        headers=headers,
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Sentinel-2 historical catalog "
            "search failed "
            f"(HTTP {response.status_code}): "
            f"{response.text}"
        )
    data = response.json()
    features = data.get(
        "features",
        []
    )
    results = []
    for feature in features:
        properties = (
            feature.get(
                "properties",
                {}
            )
        )
        results.append(
            {
                "id":
                    feature.get(
                        "id"
                    ),
                "datetime":
                    properties.get(
                        "datetime"
                    ),
                "cloud_cover":
                    properties.get(
                        "eo:cloud_cover"
                    ),
                "collection":
                    COLLECTION,
            }
        )
    # ------------------------------------------------------
    # Sort oldest → newest
    # ------------------------------------------------------
    results.sort(
        key=lambda item:
            item.get(
                "datetime"
            )
            or "",
    )
    return results
# ==========================================================
# HISTORICAL SENTINEL-2 SEARCH
# ==========================================================
def search_historical_sentinel2(
    latitude: float,
    longitude: float,
    start_year: int = 2023,
    end_year: int = 2026,
    season_start_month: int = 8,
    season_start_day: int = 1,
    season_end_month: int = 9,
    season_end_day: int = 30,
    max_cloud_cover: float = DEFAULT_CLOUD_COVER,
):
    """
    Search for Sentinel-2 observations for every year
    in the requested historical period.
    The same seasonal window is used for every year.
    Default:
        August 1 - September 30 for each requested year.
    Using the same seasonal period helps reduce differences
    caused by seasonal glacier conditions.
    """
    if start_year > end_year:
        raise ValueError(
            "start_year cannot be greater "
            "than end_year."
        )
    all_results = []
    print()
    print(
        "=========================================="
    )
    print(
        " HISTORICAL SENTINEL-2 SEARCH"
    )
    print(
        "=========================================="
    )
    print()
    print(
        "Location:"
    )
    print(
        "Latitude  :",
        latitude,
    )
    print(
        "Longitude :",
        longitude,
    )
    print()
    print(
        "Years:",
        start_year,
        "-",
        end_year,
    )
    print(
        "Season:",
        f"{season_start_month:02d}-"
        f"{season_start_day:02d}",
        "to",
        f"{season_end_month:02d}-"
        f"{season_end_day:02d}",
    )
    print()
    # ======================================================
    # SEARCH EACH YEAR
    # ======================================================
    for year in range(
        start_year,
        end_year + 1,
    ):
        start_date = (
            f"{year}-"
            f"{season_start_month:02d}-"
            f"{season_start_day:02d}"
        )
        end_date = (
            f"{year}-"
            f"{season_end_month:02d}-"
            f"{season_end_day:02d}"
        )
        print(
            f"Searching {year}..."
        )
        try:
            year_results = (
                search_sentinel2_date_range(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_cover=max_cloud_cover,
                    limit=DEFAULT_LIMIT,
                )
            )
            print(
                f"  Found {len(year_results)} "
                f"observations"
            )
            for item in year_results:
                item[
                    "year"
                ] = year
                all_results.append(
                    item
                )
        except Exception as error:
            print(
                f"  Search failed for {year}: "
                f"{error}"
            )
    # ======================================================
    # SORT ALL RESULTS
    # ======================================================
    all_results.sort(
        key=lambda item:
            item.get(
                "datetime"
            )
            or "",
    )
    print()
    print(
        "Total historical observations:",
        len(all_results),
    )
    print()
    return all_results
# ==========================================================
# SELECT BEST OBSERVATION FOR EACH YEAR
# ==========================================================
def select_best_observation_per_year(
    observations,
):
    """
    Select one Sentinel-2 observation per year.
    Selection rule:
        lowest cloud cover.
    If cloud cover is unavailable,
    that observation is placed after
    observations with known cloud cover.
    """
    best_by_year = {}
    for observation in observations:
        year = observation.get(
            "year"
        )
        if year is None:
            datetime_value = (
                observation.get(
                    "datetime"
                )
            )
            if datetime_value:
                year = int(
                    datetime_value[
                        :4
                    ]
                )
        if year is None:
            continue
        cloud_cover = (
            observation.get(
                "cloud_cover"
            )
        )
        if cloud_cover is None:
            cloud_sort_value = (
                float("inf")
            )
        else:
            cloud_sort_value = float(
                cloud_cover
            )
        if year not in best_by_year:
            best_by_year[
                year
            ] = observation
            continue
        existing = (
            best_by_year[
                year
            ]
        )
        existing_cloud = (
            existing.get(
                "cloud_cover"
            )
        )
        if existing_cloud is None:
            existing_sort_value = (
                float("inf")
            )
        else:
            existing_sort_value = (
                float(
                    existing_cloud
                )
            )
        if (
            cloud_sort_value
            < existing_sort_value
        ):
            best_by_year[
                year
            ] = observation
    selected = list(
        best_by_year.values()
    )
    selected.sort(
        key=lambda item:
            item.get(
                "year",
                0
            )
    )
    return selected
# ==========================================================
# PRINT OBSERVATIONS
# ==========================================================
def print_observations(
    observations,
    title="Sentinel-2 Observations",
):
    """
    Print observations in a readable format.
    """
    print()
    print(
        "=========================================="
    )
    print(
        f" {title}"
    )
    print(
        "=========================================="
    )
    print()
    if not observations:
        print(
            "No observations found."
        )
        print()
        return
    for index, item in enumerate(
        observations,
        start=1,
    ):
        print(
            f"Observation {index}"
        )
        print(
            "----------------------------"
        )
        print(
            "Year        :",
            item.get(
                "year"
            ),
        )
        print(
            "ID          :",
            item.get(
                "id"
            ),
        )
        print(
            "Date        :",
            item.get(
                "datetime"
            ),
        )
        print(
            "Cloud cover :",
            item.get(
                "cloud_cover"
            ),
        )
        print(
            "Collection  :",
            item.get(
                "collection"
            ),
        )
        print()
# ==========================================================
# DYNAMIC TEST
# ==========================================================
if __name__ == "__main__":
    from backend.app.services.rgi_service import find_glacier_by_name

    print()
    print("==========================================")
    print(" GLACIERAI - HISTORICAL SENTINEL-2 TEST")
    print("==========================================")
    glacier_query = input(
        "Enter a glacier name or RGI ID: "
    ).strip()

    if not glacier_query:
        raise ValueError("A glacier name or RGI ID is required.")

    try:
        glacier = find_glacier_by_name(glacier_query)
        if not glacier:
            raise ValueError(
                f'Glacier "{glacier_query}" was not found in RGI 7.0.'
            )

        latitude = float(glacier["latitude"])
        longitude = float(glacier["longitude"])

        print()
        print("Selected glacier:", glacier["name"])
        print("RGI ID          :", glacier["rgi_id"])
        print("Latitude        :", latitude)
        print("Longitude       :", longitude)

        observations = search_historical_sentinel2(
            latitude=latitude,
            longitude=longitude,
            start_year=2023,
            end_year=2026,
            max_cloud_cover=DEFAULT_CLOUD_COVER,
        )

        print_observations(
            observations,
            title="ALL HISTORICAL OBSERVATIONS",
        )

        best_observations = select_best_observation_per_year(
            observations
        )
        print_observations(
            best_observations,
            title="BEST OBSERVATION PER YEAR",
        )

        print("==========================================")
        print(" HISTORICAL SEARCH SUMMARY")
        print("==========================================")
        print("All observations:", len(observations))
        print("Selected yearly observations:", len(best_observations))
        print()
        for item in best_observations:
            cloud_cover = item.get("cloud_cover")
            warning = (
                " [CLOUD-COVER CAUTION]"
                if cloud_cover is not None
                and float(cloud_cover) > 20.0
                else ""
            )
            print(
                f'- {item.get("year")} | {item.get("datetime")} '
                f'| cloud: {cloud_cover}%{warning}'
            )
        print()
        print(
            "Note: Cloud cover is a scene-level catalog value. "
            "The caution is a screening flag, not a guarantee of "
            "image quality over the glacier."
        )

    except Exception as error:
        print()
        print("Search status: FAILED")
        print("Error:", error)
        raise
