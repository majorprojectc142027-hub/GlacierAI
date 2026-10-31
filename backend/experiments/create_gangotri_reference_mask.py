
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parents[2]

RGI_FILE = ROOT / "datasets" / "processed" / "global_rgi.gpkg"
GRID_FILE = (
    ROOT / "datasets" / "live" / "rgi2000_v7_0_g_15_00467_debris_rgb.tif"
)
OUTPUT_FILE = (
    ROOT / "datasets" / "live" / "predictions"
    / "rgi2000_v7_0_g_15_00467_rgi_reference_mask.tif"
)

GLACIER_ID = "RGI2000-v7.0-G-15-00467"

# Load the exact Gangotri inventory polygon.
gdf = gpd.read_file(RGI_FILE)
gdf = gdf[gdf["rgi_id"].astype(str) == GLACIER_ID].copy()

if len(gdf) != 1:
    raise ValueError(
        f"Expected one polygon for {GLACIER_ID}, found {len(gdf)}"
    )

with rasterio.open(GRID_FILE) as src:
    grid_crs = src.crs
    transform = src.transform
    shape = (src.height, src.width)
    profile = src.profile.copy()

# Reproject the polygon to the prediction raster's CRS.
gdf = gdf.to_crs(grid_crs)

# Rasterize on the exact same grid as the RGB/U-Net raster.
reference = rasterize(
    [(geometry, 1) for geometry in gdf.geometry if geometry is not None],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=False,
)

profile.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=0,
    compress="deflate",
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(OUTPUT_FILE, "w", **profile) as dst:
    dst.write(reference, 1)

print("Reference mask created:", OUTPUT_FILE)
print("Raster shape:", reference.shape)
print("CRS:", grid_crs)
print("Reference glacier pixels:", int(np.count_nonzero(reference)))
print("Reference area on raster grid (km²):",
      round(np.count_nonzero(reference) * abs(transform.a * transform.e) / 1_000_000, 4))
print("Original RGB and U-Net files were not modified.")
