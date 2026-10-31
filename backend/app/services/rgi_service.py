from pathlib import Path
import math
import re
from functools import lru_cache

import geopandas as gpd
import requests
from shapely.geometry import Point, shape

try:
    from backend.app.data.glacier_aliases import GLACIER_ALIASES
except ImportError:
    GLACIER_ALIASES = {}


# ======================================================if alias_name:
# RGI DATASET
# ==========================================================

RGI_FILE = (
    Path(__file__).resolve().parents[3]
    / "datasets"
    / "processed"
    / "global_rgi.gpkg"
)


# ==========================================================
# RGI REGION INFORMATION
#
# RGI v7 glacier IDs contain a two-digit region code:
#
# -01 Alaska
# -02 Western Canada / USA
# -03 Arctic Canada North
# -04 Arctic Canada South
# -05 Greenland
# -06 Iceland
# -07 Svalbard
# -08 Scandinavia
# -09 Russian Arctic
# -10 North Asia
# -11 Central Europe
# -12 Caucasus / Middle East
# -13 Central Asia
# -14 South Asia East
# -15 South Asia West
# -16 Low Latitudes
# -17 Southern Andes
# -18 New Zealand
# -19 Antarctic / Subantarctic
# ==========================================================

RGI_REGIONS = {
    "01": "Alaska",
    "02": "Western Canada / USA",
    "03": "Arctic Canada North",
    "04": "Arctic Canada South",
    "05": "Greenland",
    "06": "Iceland",
    "07": "Svalbard",
    "08": "Scandinavia",
    "09": "Russian Arctic",
    "10": "North Asia",
    "11": "Central Europe",
    "12": "Caucasus / Middle East",
    "13": "Central Asia",
    "14": "South Asia East",
    "15": "South Asia West",
    "16": "Low Latitudes",
    "17": "Southern Andes",
    "18": "New Zealand",
    "19": "Antarctic / Subantarctic",
}


# ==========================================================
# SEARCH ALIASES
#
# These allow natural geographic searches without
# hardcoding individual glacier names.
# ==========================================================

REGION_ALIASES = {
    # RGI region 01
    "alaska": {"01"},
    "alaskan": {"01"},

    # RGI region 02
    "western canada": {"02"},
    "canada": {"02", "03", "04"},

    # RGI region 03 / 04
    "arctic canada": {"03", "04"},

    # RGI region 05
    "greenland": {"05"},

    # RGI region 06
    "iceland": {"06"},
    "icelandic": {"06"},

    # RGI region 07
    "svalbard": {"07"},

    # RGI region 08
    "scandinavia": {"08"},
    "norway": {"08"},
    "sweden": {"08"},

    # RGI region 09
    "russian arctic": {"09"},

    # RGI region 10
    "north asia": {"10"},
    "siberia": {"10"},

    # RGI region 11
    "central europe": {"11"},
    "alps": {"11"},
    "europe": {"11"},

    # RGI region 12
    "caucasus": {"12"},
    "middle east": {"12"},

    # RGI region 13
    "central asia": {"13"},
    "kazakhstan": {"13"},
    "kyrgyzstan": {"13"},
    "tajikistan": {"13"},

    # South Asia
    #
    # RGI regions 14 and 15 cover the South Asian
    # glacier regions. This is intentionally broader
    # than a country-level boundary.
    "south asia": {"14", "15"},
    "himalaya": {"14", "15"},
    "himalayas": {"14", "15"},
    "himalayan": {"14", "15"},

    # India / Nepal / Bhutan are represented within
    # South Asian RGI regions. These aliases therefore
    # search those RGI regions rather than claiming
    # country-level polygon boundaries.
    "india": {"14", "15"},
    "indian": {"14", "15"},
    "nepal": {"15"},
    "bhutan": {"15"},
    "pakistan": {"15"},

    # RGI region 16
    "low latitudes": {"16"},
    "tropics": {"16"},
    "tropical glaciers": {"16"},

    # RGI region 17
    "southern andes": {"17"},
    "andes": {"17"},
    "patagonia": {"17"},
    "chile": {"17"},
    "argentina": {"17"},

    # RGI region 18
    "new zealand": {"18"},
    "nz": {"18"},

    # RGI region 19
    "antarctica": {"19"},
    "antarctic": {"19"},
    "subantarctic": {"19"},
}


# ==========================================================
# LOAD RGI ATTRIBUTES
# ==========================================================

@lru_cache(maxsize=1)
def load_rgi():
    """
    Load the global RGI v7 dataset without geometry.

    Used for:
        - Glacier name search
        - RGI ID search
        - Geographic region search
    """

    if not RGI_FILE.exists():
        raise FileNotFoundError(
            f"RGI file not found: {RGI_FILE}"
        )

    return gpd.read_file(
        RGI_FILE,
        ignore_geometry=True,
    )


# ==========================================================
# EXTRACT RGI REGION CODE
# ==========================================================

def get_rgi_region_code(rgi_id):
    """
    Extract the two-digit RGI region code.

    Example:

        RGI2000-v7.0-G-15-08331
                         ^^
                         15

    Returns:
        "15"

    or None if the ID does not match.
    """

    if rgi_id is None:
        return None

    value = str(rgi_id)

    match = re.search(
        r"-G-(\d{2})-",
        value,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


# ==========================================================
# NORMALIZE SEARCH QUERY
# ==========================================================

def normalize_query(query: str) -> str:
    """
    Normalize a user search query.

    Examples:

        "Iceland glaciers"
            -> "iceland"

        "Alaska glacier"
            -> "alaska"

        "  Khumbu  "
            -> "khumbu"
    """

    query = str(query).strip().lower()

    if not query:
        return ""

    # Remove common search words.
    query = re.sub(
        r"\b(glacier|glaciers|ice|icefield|"
        r"region|regions|area|areas)\b",
        " ",
        query,
    )

    # Normalize whitespace.
    query = re.sub(
        r"\s+",
        " ",
        query,
    ).strip()

    return query


# ==========================================================
# FIND REGION CODES FROM QUERY
# ==========================================================

def get_region_codes_for_query(query: str):
    """
    Determine whether the query refers to an RGI region.

    Supports:
        Iceland
        Iceland glaciers
        Alaska
        Himalayas
        India
        Antarctica
        Patagonia
        etc.
    """

    normalized = normalize_query(query)

    if not normalized:
        return set()

    # Exact alias.
    if normalized in REGION_ALIASES:
        return REGION_ALIASES[
            normalized
        ]

    region_codes = set()

    # Also support phrases contained in the query.
    for alias, codes in REGION_ALIASES.items():
        if alias in normalized:
            region_codes.update(
                codes
            )

    return region_codes


# ==========================================================
# FORMAT SEARCH RESULT
# ==========================================================

def _format_search_result(row):
    """
    Convert an RGI attribute row into a frontend-friendly
    glacier dictionary.
    """

    glacier_name = row.get(
        "glac_name"
    )

    if (
        glacier_name is None
        or str(glacier_name).strip() == ""
    ):
        glacier_name = (
            f"RGI Glacier {row['rgi_id']}"
        )

    # Remove GLIMS internal identifier prefixes from
    # human-readable glacier names.
    # Example:
    #   IN5Q151B0013 Siachen Glacier
    # becomes:
    #   Siachen Glacier
    glacier_name = re.sub(
        r"^[A-Z]{2}\d+[A-Z]\d+\s+",
        "",
        str(glacier_name).strip(),
    )

    rgi_id = str(
        row["rgi_id"]
    )

    region_code = (
        get_rgi_region_code(
            rgi_id
        )
    )

    region_name = (
        RGI_REGIONS.get(
            region_code,
            "Global",
        )
        if region_code
        else "Global"
    )

    return {
        "rgi_id": rgi_id,

        "name": str(
            glacier_name
        ),

        "latitude": round(
            float(row["cenlat"]),
            6,
        ),

        "longitude": round(
            float(row["cenlon"]),
            6,
        ),

        "area_km2": (
            round(
                float(row["area_km2"]),
                4,
            )
            if row.get(
                "area_km2"
            ) is not None
            else None
        ),

        "region": region_name,

        "country": region_name,
    }


# ==========================================================
# LOCAL GLACIER NAME ALIAS FALLBACK
# ==========================================================

def _search_local_glacier_aliases(query: str, limit: int = 20):
    """
    Resolve human-friendly glacier aliases to RGI v7 IDs.

    This is intentionally checked before the live GLIMS name search.
    It keeps important named glaciers searchable even when the GLIMS
    public search service is temporarily unavailable or an RGI record
    has no value in its ``glac_name`` field.

    ``GLACIER_ALIASES`` maps one or more user-facing names to an RGI ID.
    Example::

        {
            "gangotri": "RGI2000-v7.0-G-15-00467",
            "gangotri glacier": "RGI2000-v7.0-G-15-00467",
        }
    """
    if not GLACIER_ALIASES:
        return []

    normalized = normalize_query(query)
    if not normalized:
        return []

    # Match both the raw normalized query and the query with common
    # glacier words removed (e.g. "Gangotri Glacier" -> "gangotri").
    candidates = {str(query).strip().lower(), normalized}

    matched_rgi_id = None
    matched_alias = None

    for alias, value in GLACIER_ALIASES.items():
        alias_normalized = normalize_query(alias)
        alias_raw = str(alias).strip().lower()

        if alias_normalized in candidates or alias_raw in candidates:
            matched_rgi_id = str(value).strip()
            matched_alias = str(alias).strip()
            break

    if not matched_rgi_id:
        return []

    try:
        gdf = load_rgi()
        matches = gdf[
            gdf["rgi_id"].astype(str).str.strip() == matched_rgi_id
        ]
    except Exception as error:
        print(f"Warning: Local glacier alias lookup failed: {error}")
        return []

    if matches.empty:
        print(
            f"Warning: Alias '{matched_alias}' points to RGI ID "
            f"'{matched_rgi_id}', but that ID is not present in the local RGI dataset."
        )
        return []

    glaciers = []
    for _, row in matches.head(limit).iterrows():
        result = _format_search_result(row)

        # RGI may not contain a human-readable name. Use the alias as the
        # user-facing name and append "Glacier" when appropriate.
        display_name = matched_alias or normalized.title()
        if not display_name.lower().endswith("glacier"):
            display_name = f"{display_name.title()} Glacier"

        result["name"] = display_name
        result["matched_by"] = "local_alias"
        glaciers.append(result)

    return glaciers


# ==========================================================
# GLIMS NAME SEARCH FALLBACK
# ==========================================================

GLIMS_WFS_URL = "https://glims.org/geoserver/wfs"
GLIMS_LAYER = "GLIMS:GLIMS_Glacier_Outlines"


def _escape_cql_text(value: str) -> str:
    """Escape text before putting it into a CQL string literal."""

    return str(value).replace("'", "''")


def _search_glims_by_name(
    query: str,
    limit: int = 20,
):
    """
    Search the GLIMS Glacier Outlines WFS by glacier name.

    The GLIMS WFS has the human-readable glacier name in ``glac_name``.
    Different GeoServer deployments can handle CQL/ECQL filters slightly
    differently, so this function tries several equivalent filter forms.

    This function returns raw GLIMS GeoJSON features. The caller resolves
    each feature to the local RGI v7 inventory.
    """

    normalized = normalize_query(query)

    if not normalized:
        return []

    result_count = min(max(int(limit) * 5, 10), 100)

    # Keep the search text safe for a CQL string and prevent a user's
    # "%" or "_" from becoming unintended SQL LIKE wildcards.
    safe_text = re.sub(r"[%_]", " ", normalized)
    safe_text = re.sub(r"\s+", " ", safe_text).strip()
    escaped = _escape_cql_text(safe_text)

    base_params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": GLIMS_LAYER,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "count": result_count,
    }

    # GeoServer's WFS implementation supports CQL/ECQL filters, but
    # some deployments are more permissive with LIKE than ILIKE.
    filter_attempts = [
        {
            **base_params,
            "CQL_FILTER": f"glac_name LIKE '%{escaped}%'",
        },
        {
            **base_params,
            "CQL_FILTER": f"glac_name ILIKE '%{escaped}%'",
        },
    ]

    # Final fallback: use the OGC Filter Encoding 2.0 form. This avoids
    # depending on the server's CQL parser while still filtering by name.
    xml_text = (
        safe_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

    filter_xml = (
        '<fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0">'
        '<fes:PropertyIsLike wildCard="%" singleChar="_" escapeChar="\\">'
        "<fes:ValueReference>glac_name</fes:ValueReference>"
        f"<fes:Literal>%{xml_text}%</fes:Literal>"
        "</fes:PropertyIsLike>"
        "</fes:Filter>"
    )

    filter_attempts.append(
        {
            **base_params,
            "filter": filter_xml,
        }
    )

    features = []

    for attempt_number, params in enumerate(filter_attempts, start=1):
        try:
            response = requests.get(
                GLIMS_WFS_URL,
                params=params,
                timeout=30,
            )

            # A filter syntax rejected by GeoServer should not prevent
            # the next supported filter form from being attempted.
            if response.status_code >= 400:
                print(
                    "GLIMS name search attempt "
                    f"{attempt_number} returned HTTP {response.status_code}."
                )
                continue

            data = response.json()
            candidate_features = data.get("features", [])

            if candidate_features:
                features = candidate_features
                break

        except requests.RequestException as error:
            print(
                "Warning: GLIMS name search attempt "
                f"{attempt_number} failed: {error}"
            )
        except ValueError as error:
            print(
                "Warning: GLIMS returned invalid JSON on name search "
                f"attempt {attempt_number}: {error}"
            )

    if not features:
        return []

    # Prefer usable/current records and then the most recent observation.
    def sort_key(feature):
        props = feature.get("properties", {}) or {}
        status = str(props.get("rec_status") or "").lower()
        date = str(props.get("src_date") or "")

        return (
            0 if status in {"ok", "current"} else 1,
            -(int(date[:4]) if date[:4].isdigit() else 0),
        )

    return sorted(features, key=sort_key)


def _resolve_rgi_from_glims_feature(feature):
    """
    Convert one GLIMS feature into the local RGI v7 record.

    GLIMS may contain an RGI identifier from a different RGI
    version, so an exact ID match is attempted first. If that
    fails, the GLIMS geometry centroid is used to find the nearest
    glacier in the local RGI v7 inventory.
    """

    properties = feature.get("properties", {}) or {}
    geometry_json = feature.get("geometry")

    if not geometry_json:
        return None

    try:
        glims_geometry = shape(geometry_json)
    except Exception:
        return None

    if glims_geometry.is_empty:
        return None

    glims_rgi_id = properties.get("rgiid")
    if glims_rgi_id is not None:
        glims_rgi_id = str(glims_rgi_id).strip()

    # ------------------------------------------------------
    # 1. Exact RGI ID match in the local v7 inventory
    # ------------------------------------------------------

    if glims_rgi_id:
        try:
            gdf = load_rgi()
            matches = gdf[
                gdf["rgi_id"].astype(str).str.strip()
                == glims_rgi_id
            ]

            if not matches.empty:
                row = matches.iloc[0]
                return {
                    "rgi_id": str(row["rgi_id"]),
                    "latitude": float(row["cenlat"]),
                    "longitude": float(row["cenlon"]),
                    "area_km2": (
                        float(row["area_km2"])
                        if row.get("area_km2") is not None
                        else None
                    ),
                    "rgi_name": (
                        str(row.get("glac_name"))
                        if row.get("glac_name") not in (None, "")
                        else None
                    ),
                }
        except Exception as error:
            print(f"Warning: Exact GLIMS/RGI match failed: {error}")

    # ------------------------------------------------------
    # 2. Coordinate fallback to the local RGI v7 inventory
    # ------------------------------------------------------

    centroid = glims_geometry.centroid

    try:
        nearest = find_nearest_glacier(
            latitude=float(centroid.y),
            longitude=float(centroid.x),
            max_distance_km=50.0,
        )
    except Exception as error:
        print(f"Warning: Could not resolve GLIMS glacier to RGI v7: {error}")
        nearest = None

    if nearest:
        return {
            "rgi_id": str(nearest["rgi_id"]),
            "latitude": float(nearest["latitude"]),
            "longitude": float(nearest["longitude"]),
            "area_km2": nearest.get("area_km2"),
            "rgi_name": nearest.get("name"),
        }

    return None


def _search_glims_name_fallback(
    query: str,
    limit: int = 20,
):
    """
    Search GLIMS names and return frontend-compatible results.

    Multiple GLIMS historical outlines can belong to the same
    glacier, so results are deduplicated by the resolved RGI ID.
    """

    try:
        features = _search_glims_by_name(
            query=query,
            limit=limit,
        )
    except requests.RequestException as error:
        print(f"Warning: GLIMS name search failed: {error}")
        return []
    except ValueError as error:
        print(f"Warning: GLIMS returned invalid JSON: {error}")
        return []

    glaciers = []
    seen_rgi_ids = set()

    for feature in features:
        properties = feature.get("properties", {}) or {}
        resolved = _resolve_rgi_from_glims_feature(feature)

        if not resolved:
            continue

        rgi_id = resolved["rgi_id"]

        if rgi_id in seen_rgi_ids:
            continue

        seen_rgi_ids.add(rgi_id)

        glacier_name = (
            properties.get("glac_name")
            or resolved.get("rgi_name")
            or f"RGI Glacier {rgi_id}"
        )

        # GLIMS can prepend an internal identifier to the
        # display name, e.g.:
        #   IN5Q151B0013 Siachen Glacier
        # Keep the identifier internally through rgi_id/glims_id,
        # but show only the human-readable glacier name.
        glacier_name = re.sub(
            r"^[A-Z]{2}\d+[A-Z]\d+\s+",
            "",
            str(glacier_name).strip(),
        )

        region_code = get_rgi_region_code(rgi_id)
        region_name = (
            RGI_REGIONS.get(region_code, "Global")
            if region_code
            else "Global"
        )

        glaciers.append({
            "rgi_id": rgi_id,
            "name": str(glacier_name),
            "latitude": round(resolved["latitude"], 6),
            "longitude": round(resolved["longitude"], 6),
            "area_km2": (
                round(float(resolved["area_km2"]), 4)
                if resolved.get("area_km2") is not None
                else None
            ),
            "region": region_name,
            "country": region_name,
            "matched_by": "glims_name",
        })

        if len(glaciers) >= limit:
            break

    return glaciers


# ==========================================================
# WIKIDATA NAME SEARCH FALLBACK
# ==========================================================

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_HEADERS = {
    "User-Agent": (
        "GlacierAI/1.0 "
        "(AI-based glacier monitoring research project; "
        "contact via project repository)"
    ),
    "Accept": "application/json",
}


def _search_wikidata_by_name(
    query: str,
    limit: int = 20,
):
    """
    Resolve a human glacier name through Wikidata when the local
    RGI attributes and current GLIMS name field do not contain it.

    Wikidata is used only as a name/coordinate discovery layer.
    The actual glacier geometry and RGI identity still come from
    the local RGI v7 dataset by resolving the Wikidata coordinates
    to the nearest local RGI glacier.
    """

    normalized = normalize_query(query)

    if not normalized:
        return []

    try:
        search_response = requests.get(
            WIKIDATA_API_URL,
            params={
                "action": "wbsearchentities",
                "search": normalized,
                "language": "en",
                "uselang": "en",
                "format": "json",
                "limit": min(max(int(limit) * 2, 5), 50),
            },
            headers=WIKIDATA_HEADERS,
            timeout=20,
        )
        search_response.raise_for_status()
        search_data = search_response.json()

        search_results = search_data.get("search", [])
        if not search_results:
            return []

        entity_ids = [
            str(item.get("id"))
            for item in search_results
            if item.get("id")
        ]

        if not entity_ids:
            return []

        entity_response = requests.get(
            WIKIDATA_API_URL,
            params={
                "action": "wbgetentities",
                "ids": "|".join(entity_ids),
                "props": "labels|descriptions|claims",
                "languages": "en",
                "format": "json",
            },
            headers=WIKIDATA_HEADERS,
            timeout=20,
        )
        entity_response.raise_for_status()
        entity_data = entity_response.json()

    except requests.RequestException as error:
        print(f"Warning: Wikidata glacier search failed: {error}")
        return []
    except ValueError as error:
        print(f"Warning: Wikidata returned invalid JSON: {error}")
        return []

    glaciers = []
    seen_rgi_ids = set()

    for entity_id in entity_ids:
        entity = entity_data.get("entities", {}).get(entity_id, {})
        claims = entity.get("claims", {}) or {}

        # Wikidata P625 = coordinate location.
        coordinate_claims = claims.get("P625", [])
        coordinate = None

        for claim in coordinate_claims:
            value = (
                claim.get("mainsnak", {})
                .get("datavalue", {})
                .get("value", {})
            )

            if "latitude" in value and "longitude" in value:
                coordinate = value
                break

        if coordinate is None:
            continue

        latitude = coordinate.get("latitude")
        longitude = coordinate.get("longitude")

        if latitude is None or longitude is None:
            continue

        try:
            nearest = find_nearest_glacier(
                latitude=float(latitude),
                longitude=float(longitude),
                max_distance_km=50.0,
            )
        except Exception as error:
            print(
                f"Warning: Could not resolve Wikidata entity "
                f"{entity_id} to local RGI: {error}"
            )
            nearest = None

        if not nearest:
            continue

        rgi_id = str(nearest["rgi_id"])

        if rgi_id in seen_rgi_ids:
            continue

        seen_rgi_ids.add(rgi_id)

        label = (
            entity.get("labels", {})
            .get("en", {})
            .get("value")
        )

        if not label:
            label = normalized.title()

        # Only accept a result when the Wikidata label is reasonably
        # related to the user's query. This avoids returning an
        # unrelated nearby glacier from a broad Wikidata search.
        label_normalized = normalize_query(label)
        if (
            normalized not in label_normalized
            and label_normalized not in normalized
        ):
            continue

        region_code = get_rgi_region_code(rgi_id)
        region_name = (
            RGI_REGIONS.get(region_code, "Global")
            if region_code
            else "Global"
        )

        glaciers.append({
            "rgi_id": rgi_id,
            "name": (
                label
                if str(label).lower().endswith("glacier")
                else f"{label} Glacier"
            ),
            "latitude": round(float(nearest["latitude"]), 6),
            "longitude": round(float(nearest["longitude"]), 6),
            "area_km2": nearest.get("area_km2"),
            "region": region_name,
            "country": region_name,
            "matched_by": "wikidata_name",
        })

        if len(glaciers) >= limit:
            break

    return glaciers


# ==========================================================
# SEARCH GLOBAL RGI
# ==========================================================

def search_glaciers(
    query: str,
    limit: int = 20,
):
    """
    Search the global RGI v7 glacier inventory.

    Search modes:

        1. Glacier name
        2. RGI ID
        3. RGI geographic region

    Examples:

        Khumbu
        Gangotri
        RGI2000-v7.0-G-15-08331
        Iceland
        Iceland glaciers
        Alaska
        Himalayas
        Antarctica
        Patagonia
    """

    gdf = load_rgi()

    raw_query = str(
        query
    ).strip()

    if not raw_query:
        return []

    query_lower = (
        raw_query.lower()
    )

    normalized_query = normalize_query(
        raw_query
    )

    # ======================================================
    # 1. NAME SEARCH
    # ======================================================

    name_mask = (
        gdf["glac_name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            query_lower,
            regex=False,
        )
    )

    # Also search the normalized query if
    # "glacier" was removed.
    if (
        normalized_query
        and normalized_query != query_lower
    ):
        normalized_name_mask = (
            gdf["glac_name"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                normalized_query,
                regex=False,
            )
        )

        name_mask = (
            name_mask
            | normalized_name_mask
        )

    # ======================================================
    # 2. RGI ID SEARCH
    # ======================================================

    id_mask = (
        gdf["rgi_id"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            query_lower,
            regex=False,
        )
    )

    # ======================================================
    # 3. GEOGRAPHIC REGION SEARCH
    # ======================================================

    region_codes = (
        get_region_codes_for_query(
            raw_query
        )
    )

    region_mask = None

    if region_codes:

        region_series = (
            gdf["rgi_id"]
            .fillna("")
            .astype(str)
            .str.extract(
                r"-G-(\d{2})-",
                expand=False,
            )
        )

        region_mask = (
            region_series.isin(
                region_codes
            )
        )

    # ======================================================
    # COMBINE SEARCHES
    # ======================================================

    if region_mask is not None:

        combined_mask = (
            name_mask
            | id_mask
            | region_mask
        )

    else:

        combined_mask = (
            name_mask
            | id_mask
        )

    results = gdf[
        combined_mask
    ].copy()

    if results.empty:
        # Region and ID searches should remain local. For a normal
        # glacier-name search, fall back to GLIMS because the local
        # RGI v7 attributes may not contain human-readable names.
        if not region_codes and not re.search(
            r"^RGI\d{4}-v\d+\.\d+-G-\d{2}-",
            raw_query,
            flags=re.IGNORECASE,
        ):
            # 1. Use the local alias index first. This is reliable even
            # when the live GLIMS name-search service is unavailable.
            alias_results = _search_local_glacier_aliases(
                query=raw_query,
                limit=limit,
            )
            if alias_results:
                return alias_results

            # 2. Try the live GLIMS name search.
            glims_results = _search_glims_name_fallback(
                query=raw_query,
                limit=limit,
            )

            if glims_results:
                return glims_results

            # 3. Use Wikidata as a global name/coordinate discovery
            # layer when the RGI and GLIMS name fields do not contain
            # the glacier name. The final geometry still comes from
            # the local RGI v7 inventory.
            return _search_wikidata_by_name(
                query=raw_query,
                limit=limit,
            )

        return []

    # ======================================================
    # RESULT PRIORITY
    #
    # Exact name / ID first.
    # Then geographic/name matches.
    # ======================================================

    def priority(row):
        name_value = str(
            row.get(
                "glac_name",
                "",
            )
        ).strip().lower()

        id_value = str(
            row.get(
                "rgi_id",
                "",
            )
        ).strip().lower()

        if name_value == query_lower:
            return 0

        if id_value == query_lower:
            return 0

        if (
            normalized_query
            and name_value
            == normalized_query
        ):
            return 1

        return 2

    results["_search_priority"] = (
        results.apply(
            priority,
            axis=1,
        )
    )

    results = (
        results
        .sort_values(
            "_search_priority"
        )
        .head(limit)
    )

    # ======================================================
    # FORMAT RESULTS
    # ======================================================

    glaciers = []

    for _, row in results.iterrows():

        result = (
            _format_search_result(
                row
            )
        )

        # Tell the frontend what kind of
        # match produced the result.
        if region_codes:

            result[
                "matched_by"
            ] = "region"

        elif (
            str(
                row.get(
                    "glac_name",
                    "",
                )
            ).strip().lower()
            == query_lower
        ):

            result[
                "matched_by"
            ] = "exact_name"

        elif (
            str(
                row.get(
                    "rgi_id",
                    "",
                )
            ).strip().lower()
            == query_lower
        ):

            result[
                "matched_by"
            ] = "exact_rgi_id"

        else:

            result[
                "matched_by"
            ] = "name"

        glaciers.append(
            result
        )

    return glaciers


# ==========================================================
# FIND GLACIER BY NAME / RGI ID / REGION
# ==========================================================

def find_glacier_by_name(
    query: str,
    limit: int = 20,
):
    """
    Find one glacier dynamically.

    Matching priority:

        1. Exact glacier name
        2. Exact RGI ID
        3. Partial glacier name
        4. Geographic region

    For geographic searches such as "Iceland",
    the first RGI glacier in that region is returned.
    """

    query = str(
        query
    ).strip()

    if not query:
        return None

    results = search_glaciers(
        query=query,
        limit=limit,
    )

    if not results:
        return None

    query_clean = (
        query.lower()
    )

    # ======================================================
    # EXACT NAME
    # ======================================================

    exact_name_matches = [
        glacier
        for glacier in results
        if glacier["name"]
        .strip()
        .lower()
        == query_clean
    ]

    if exact_name_matches:

        selected = (
            exact_name_matches[0]
        )

    else:

        # ==================================================
        # EXACT RGI ID
        # ==================================================

        exact_id_matches = [
            glacier
            for glacier in results
            if glacier["rgi_id"]
            .strip()
            .lower()
            == query_clean
        ]

        if exact_id_matches:

            selected = (
                exact_id_matches[0]
            )

        else:

            # ==============================================
            # FIRST SEARCH RESULT
            # ==============================================

            selected = results[0]

    # ======================================================
    # LOAD GEOMETRY
    # ======================================================

    bbox = _calculate_bbox(
        latitude=selected[
            "latitude"
        ],
        longitude=selected[
            "longitude"
        ],
        distance_km=50.0,
    )

    gdf = gpd.read_file(
        RGI_FILE,
        bbox=bbox,
    )

    if gdf.empty:
        return None

    # ======================================================
    # FIND SELECTED RGI RECORD
    # ======================================================

    matches = gdf[
        gdf["rgi_id"]
        .astype(str)
        == str(
            selected["rgi_id"]
        )
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]

    # ======================================================
    # FORMAT COMPLETE RESULT
    # ======================================================

    result = _format_search_result(
        row
    )

    result[
        "distance_km"
    ] = 0.0

    result[
        "matched_by"
    ] = selected.get(
        "matched_by",
        "name",
    )

    # Preserve the human-readable name resolved during search.
    # This is important for glaciers whose RGI record has no
    # value in the "glac_name" field, but whose name was resolved
    # through the local alias table (for example, Kolahoi).
    if selected.get("name"):
        result["name"] = selected["name"]

    result[
        "geometry"
    ] = row.geometry

    return result


# ==========================================================
# CALCULATE SEARCH BOUNDING BOX
# ==========================================================

def _calculate_bbox(
    latitude: float,
    longitude: float,
    distance_km: float,
):
    """
    Convert a radius in kilometres into a conservative
    latitude/longitude bounding box.
    """

    lat_delta = (
        distance_km / 111.0
    )

    cos_lat = max(
        math.cos(
            math.radians(
                latitude
            )
        ),
        0.1,
    )

    lon_delta = (
        distance_km
        / (
            111.0
            * cos_lat
        )
    )

    return (
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta,
    )


# ==========================================================
# FIND NEAREST GLACIER BY COORDINATES
# ==========================================================

def find_nearest_glacier(
    latitude: float,
    longitude: float,
    max_distance_km: float = 20.0,
):
    """
    Find the RGI glacier closest to the supplied
    latitude and longitude.
    """

    # ------------------------------------------------------
    # Validate latitude
    # ------------------------------------------------------

    if not (
        -90 <= latitude <= 90
    ):
        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    # ------------------------------------------------------
    # Validate longitude
    # ------------------------------------------------------

    if not (
        -180 <= longitude <= 180
    ):
        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    # ------------------------------------------------------
    # Validate search radius
    # ------------------------------------------------------

    if max_distance_km <= 0:
        raise ValueError(
            "max_distance_km must be greater than 0."
        )

    # ------------------------------------------------------
    # Geographic point
    # ------------------------------------------------------

    point = Point(
        longitude,
        latitude,
    )

    # ------------------------------------------------------
    # Local bounding box
    # ------------------------------------------------------

    bbox = _calculate_bbox(
        latitude,
        longitude,
        max_distance_km,
    )

    # ------------------------------------------------------
    # Read local geometries only
    # ------------------------------------------------------

    gdf = gpd.read_file(
        RGI_FILE,
        bbox=bbox,
    )

    if gdf.empty:
        return None

    # ------------------------------------------------------
    # Ensure CRS
    # ------------------------------------------------------

    if gdf.crs is None:
        gdf = gdf.set_crs(
            "EPSG:4326"
        )

    # ------------------------------------------------------
    # Point inside glacier polygon
    # ------------------------------------------------------

    inside = gdf[
        gdf.geometry.contains(
            point
        )
    ]

    if not inside.empty:

        row = inside.iloc[0]

        return _format_glacier_result(
            row=row,
            distance_km=0.0,
            matched_by="inside_polygon",
        )

    # ------------------------------------------------------
    # Project local data for distance
    # ------------------------------------------------------

    gdf_projected = (
        gdf.to_crs(
            "EPSG:3857"
        )
    )

    point_projected = (
        gpd.GeoSeries(
            [point],
            crs="EPSG:4326",
        )
        .to_crs(
            "EPSG:3857"
        )
        .iloc[0]
    )

    distances = (
        gdf_projected
        .geometry
        .distance(
            point_projected
        )
    )

    nearest_index = (
        distances.idxmin()
    )

    distance_m = float(
        distances.loc[
            nearest_index
        ]
    )

    distance_km = (
        distance_m / 1000.0
    )

    if (
        distance_km
        > max_distance_km
    ):
        return None

    row = gdf.loc[
        nearest_index
    ]

    return _format_glacier_result(
        row=row,
        distance_km=distance_km,
        matched_by="nearest_glacier",
    )


# ==========================================================
# FORMAT COMPLETE GLACIER RESULT
# ==========================================================

def _format_glacier_result(
    row,
    distance_km: float,
    matched_by: str,
):
    """
    Convert an RGI geometry row into a JSON-friendly
    structure.
    """

    glacier_name = row.get(
        "glac_name"
    )

    # If RGI does not contain a human-readable name, use the
    # local glacier alias table. This keeps named glaciers such
    # as Gangotri Glacier consistent across search and map-click
    # results.
    rgi_id_value = str(row["rgi_id"]).strip()

    if (
        glacier_name is None
        or str(
            glacier_name
        ).strip()
        == ""
    ):
        alias_name = None

        for alias, alias_rgi_id in GLACIER_ALIASES.items():
            if str(alias_rgi_id).strip() == rgi_id_value:
                alias_name = str(alias).strip()
                break

        if alias_name:
            glacier_name = alias_name

            if not glacier_name.lower().endswith("glacier"):
                glacier_name = f"{glacier_name.title()} Glacier"
        else:
            glacier_name = (
                f"RGI Glacier {row['rgi_id']}"
            )

    # Keep internal IDs out of the user-facing glacier name.
    glacier_name = re.sub(
        r"^[A-Z]{2}\d+[A-Z]\d+\s+",
        "",
        str(glacier_name).strip(),
    )

    rgi_id = str(
        row["rgi_id"]
    )

    region_code = (
        get_rgi_region_code(
            rgi_id
        )
    )

    region_name = (
        RGI_REGIONS.get(
            region_code,
            "Global",
        )
        if region_code
        else "Global"
    )

    return {
        "rgi_id": rgi_id,

        "name": str(
            glacier_name
        ),

        "latitude": round(
            float(
                row["cenlat"]
            ),
            6,
        ),

        "longitude": round(
            float(
                row["cenlon"]
            ),
            6,
        ),

        "area_km2": (
            round(
                float(
                    row["area_km2"]
                ),
                4,
            )
            if row.get(
                "area_km2"
            ) is not None
            else None
        ),

        "region": region_name,

        "country": region_name,

        "distance_km": round(
            distance_km,
            3,
        ),

        "matched_by": matched_by,

        "geometry": row.geometry,
    }
