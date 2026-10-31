"""Download and prepare Sentinel-2 L2A RGB imagery for a glacier tile.

Credentials are read from the project .env file (CDSE_USERNAME/CDSE_PASSWORD).
The module searches the CDSE OData catalogue, downloads a selected SAFE archive,
extracts the 10 m B02/B03/B04 bands, and writes an RGB GeoTIFF in the requested CRS.
"""
from __future__ import annotations

import os
import re
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
CACHE_DIR = PROJECT_ROOT / "datasets" / "live" / "sentinel2_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DAYS_BACK = int(os.getenv("CDSE_DAYS_BACK", "60"))
DEFAULT_CLOUD_LIMIT = float(os.getenv("CDSE_CLOUD_LIMIT", "40"))
REQUEST_TIMEOUT = (30, 180)


class Sentinel2DownloadError(RuntimeError):
    """Raised when suitable Sentinel-2 data cannot be found or prepared."""


def _credentials() -> tuple[str, str]:
    # Re-read .env so changes made after process startup can be picked up.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    username = os.getenv("CDSE_USERNAME", "").strip()
    password = os.getenv("CDSE_PASSWORD", "").strip()
    if not username or not password:
        raise Sentinel2DownloadError(
            "Copernicus credentials are missing. Add CDSE_USERNAME and "
            "CDSE_PASSWORD to the project .env file."
        )
    return username, password


def _get_access_token(session: requests.Session) -> str:
    username, password = _credentials()
    try:
        response = session.post(
            TOKEN_URL,
            data={
                "grant_type": "password",
                "client_id": "cdse-public",
                "username": username,
                "password": password,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise Sentinel2DownloadError("Copernicus did not return an access token.")
        return token
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" HTTP {exc.response.status_code}: {exc.response.text[:500]}"
        raise Sentinel2DownloadError(
            "Copernicus authentication failed. Check the credentials in .env "
            "and confirm the account can sign in." + detail
        ) from exc


def _odata_literal(value: str) -> str:
    return value.replace("'", "''")


def _search_products(
    session: requests.Session,
    token: str,
    tile_bbox_wgs84: tuple[float, float, float, float],
    days_back: int,
    cloud_limit: float,
) -> list[dict[str, Any]]:
    minx, miny, maxx, maxy = tile_bbox_wgs84
    # OData Intersects expects longitude/latitude coordinates in SRID 4326.
    polygon = (
        f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},"
        f"{minx} {maxy},{minx} {miny}))"
    )
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(1, days_back))
    start_text = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_text = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    filter_text = (
        "Collection/Name eq 'SENTINEL-2'"
        " and contains(Name,'MSIL2A')"
        f" and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {float(cloud_limit):.2f})"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
        f" and ContentDate/Start ge {start_text}"
        f" and ContentDate/Start le {end_text}"
    )
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "$filter": filter_text,
        "$orderby": "ContentDate/Start desc",
        "$top": "30",
    }
    try:
        response = session.get(CATALOGUE_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        products = response.json().get("value", [])
    except requests.RequestException as exc:
        raise Sentinel2DownloadError(f"Sentinel-2 catalogue search failed: {exc}") from exc

    # Sort defensively by acquisition time, then prefer the lowest cloud cover.
    def cloud_value(product: dict[str, Any]) -> float:
        for attribute in product.get("Attributes", []) or []:
            if str(attribute.get("Name", "")).lower() == "cloudcover":
                try:
                    return float(attribute.get("Value", 100))
                except (TypeError, ValueError):
                    pass
        return 100.0

    products.sort(
        key=lambda item: (
            str((item.get("ContentDate") or {}).get("Start", "")),
            -cloud_value(item),
        ),
        reverse=True,
    )
    return products


def _download_product(session: requests.Session, token: str, product: dict[str, Any]) -> Path:
    product_id = str(product["Id"])
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", product_id)
    archive_path = CACHE_DIR / f"{safe_id}.zip"
    if archive_path.exists() and archive_path.stat().st_size > 1024 * 1024:
        return archive_path

    url = f"{DOWNLOAD_URL}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    temporary_path = archive_path.with_suffix(".zip.part")
    try:
        with session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if not zipfile.is_zipfile(temporary_path):
            raise Sentinel2DownloadError(
                "Copernicus download did not return a valid SAFE ZIP archive."
            )
        temporary_path.replace(archive_path)
        return archive_path
    except requests.RequestException as exc:
        temporary_path.unlink(missing_ok=True)
        raise Sentinel2DownloadError(f"Sentinel-2 product download failed: {exc}") from exc


def _extract_rgb_bands(archive_path: Path, product_id: str) -> list[tuple[Path, Path, Path]]:
    product_dir = CACHE_DIR / f"{archive_path.stem}_bands"
    product_dir.mkdir(parents=True, exist_ok=True)
    bands_by_granule: dict[str, dict[str, Path]] = {}
    band_pattern = re.compile(r"_(B02|B03|B04)_10m\.jp2$", re.IGNORECASE)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                match = band_pattern.search(member)
                if not match:
                    continue
                band = match.group(1).upper()
                # Group by the granule IMG_DATA directory so bands stay aligned.
                marker = "/IMG_DATA/R10m/"
                if marker not in member:
                    continue
                granule_key = member.split(marker, 1)[0]
                destination = product_dir / Path(member).name
                if not destination.exists():
                    with archive.open(member) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                bands_by_granule.setdefault(granule_key, {})[band] = destination
    except (zipfile.BadZipFile, OSError) as exc:
        raise Sentinel2DownloadError(f"Could not extract Sentinel-2 RGB bands: {exc}") from exc

    groups = []
    for bands in bands_by_granule.values():
        if all(key in bands for key in ("B02", "B03", "B04")):
            # Return paths in model input order: red, green, blue.
            groups.append((bands["B04"], bands["B03"], bands["B02"]))
    if not groups:
        raise Sentinel2DownloadError(
            "The downloaded Sentinel-2 product did not contain matching 10 m B02/B03/B04 bands."
        )
    return groups


def _overlap_area(source: rasterio.io.DatasetReader, target_bounds: tuple[float, float, float, float], target_crs: str) -> float:
    from rasterio.warp import transform_bounds
    try:
        bounds = transform_bounds(source.crs, target_crs, *source.bounds, densify_pts=21)
    except Exception:
        return 0.0
    left = max(bounds[0], target_bounds[0])
    bottom = max(bounds[1], target_bounds[1])
    right = min(bounds[2], target_bounds[2])
    top = min(bounds[3], target_bounds[3])
    return max(0.0, right - left) * max(0.0, top - bottom)


def _write_tile_geotiff(
    band_group: tuple[Path, Path, Path],
    output_path: Path,
    target_bounds: tuple[float, float, float, float],
    target_crs: str,
) -> None:
    left, bottom, right, top = target_bounds
    resolution = 10.0
    width = max(1, int(np.ceil((right - left) / resolution)))
    height = max(1, int(np.ceil((top - bottom) / resolution)))
    transform = from_origin(left, top, resolution, resolution)
    output = np.zeros((3, height, width), dtype=np.uint16)

    for band_index, band_path in enumerate(band_group, start=1):
        with rasterio.open(band_path) as source:
            reproject(
                source=rasterio.band(source, 1),
                destination=output[band_index - 1],
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint16",
        "crs": target_crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tif.part")
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(output)
    temporary.replace(output_path)


def download_sentinel2_tile(
    *,
    tile_geometry_wgs84,
    tile_bounds_projected: tuple[float, float, float, float],
    target_crs: str,
    output_path: str | Path,
    days_back: int = DEFAULT_DAYS_BACK,
    cloud_limit: float = DEFAULT_CLOUD_LIMIT,
) -> dict[str, Any]:
    """Find, download and prepare one Sentinel-2 L2A tile as a 10 m RGB GeoTIFF."""
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 1024:
        return {"path": str(output_path), "reused": True}

    bounds = tile_geometry_wgs84.bounds
    minx, miny, maxx, maxy = bounds
    if maxx <= minx or maxy <= miny:
        raise Sentinel2DownloadError("The requested glacier tile has invalid geographic bounds.")

    with requests.Session() as session:
        token = _get_access_token(session)
        products = _search_products(
            session,
            token,
            (minx, miny, maxx, maxy),
            days_back,
            cloud_limit,
        )
        if not products:
            raise Sentinel2DownloadError(
                f"No Sentinel-2 L2A product found in the last {days_back} days "
                f"with cloud cover <= {cloud_limit:.0f}%. Try a longer date window "
                "or a higher cloud limit."
            )

        last_error = None
        for product in products:
            try:
                archive = _download_product(session, token, product)
                groups = _extract_rgb_bands(archive, str(product["Id"]))
                target_area = (
                    (tile_bounds_projected[2] - tile_bounds_projected[0])
                    * (tile_bounds_projected[3] - tile_bounds_projected[1])
                )
                best_group = None
                best_overlap = 0.0
                for group in groups:
                    with rasterio.open(group[0]) as source:
                        overlap = _overlap_area(source, tile_bounds_projected, target_crs)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_group = group
                if best_group is None or best_overlap <= 0:
                    raise Sentinel2DownloadError(
                        "The selected Sentinel-2 product has no 10 m granule overlapping this tile."
                    )
                _write_tile_geotiff(
                    best_group,
                    output_path,
                    tile_bounds_projected,
                    target_crs,
                )
                content_date = (product.get("ContentDate") or {}).get("Start")
                return {
                    "path": str(output_path),
                    "reused": False,
                    "product_id": product.get("Id"),
                    "product_name": product.get("Name"),
                    "observation_date": content_date,
                    "cloud_cover_percent": next(
                        (a.get("Value") for a in product.get("Attributes", []) or []
                         if str(a.get("Name", "")).lower() == "cloudcover"),
                        None,
                    ),
                    "overlap_fraction": min(1.0, best_overlap / target_area) if target_area else None,
                }
            except Exception as exc:
                last_error = exc
                continue

    raise Sentinel2DownloadError(
        "Could not prepare imagery from the available Sentinel-2 products. "
        f"Last error: {last_error}"
    )
