
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

root = Path("datasets/historical/rgi2000-v7.0-g-15-00467")
sat = root / "satellite"
pred = root / "predictions"

spectral_path = Path(
    "datasets/live/rgi2000_v7_0_g_15_00467_debris_spectral.tif"
)

stem25 = "rgi2000-v7.0-g-15-00467_2025"
stem26 = "rgi2000-v7.0-g-15-00467_2026"

# Load reference grid from the 2025 image
with rasterio.open(sat / f"{stem25}.tif") as ref:
    height, width = ref.height, ref.width
    dst_transform = ref.transform
    dst_crs = ref.crs

# Load masks
with rasterio.open(pred / f"{stem25}_rgi_mask.tif") as ds:
    mask25 = ds.read(1) > 0

with rasterio.open(pred / f"{stem26}_rgi_mask.tif") as ds:
    mask26 = ds.read(1) > 0

assert mask25.shape == (height, width)
assert mask26.shape == (height, width)

# Load spectral bands and reproject them to the reference grid
with rasterio.open(spectral_path) as src:
    print("Source band descriptions:", src.descriptions)
    print("Source CRS:", src.crs)
    print("Source bounds:", src.bounds)

    source = src.read([1, 2, 3]).astype(np.float32)
    source_transform = src.transform
    source_crs = src.crs

    # The source has no declared nodata value.
    # Treat pixels with all three bands equal to zero as invalid.
    source_valid = np.any(source != 0, axis=0)

    spectral = np.zeros((3, height, width), dtype=np.float32)

    for band in range(3):
        reproject(
            source=source[band],
            destination=spectral[band],
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=0,
            dst_nodata=0,
        )

    # Reproject coverage separately using nearest-neighbour.
    coverage = np.zeros((height, width), dtype=np.uint8)

    reproject(
        source=source_valid.astype(np.uint8),
        destination=coverage,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )

spectral_valid = (coverage > 0) & np.all(spectral > 0, axis=0)

# Convert scaled digital values to reflectance-like values.
# This scaling does not change normalized ratios.
nir = spectral[0] / 10000.0
swir1 = spectral[1] / 10000.0
swir2 = spectral[2] / 10000.0

denominator = nir + swir1
nir_swir_index = np.full((height, width), np.nan, dtype=np.float32)

ok = spectral_valid & (denominator > 1e-8)
nir_swir_index[ok] = (
    (nir[ok] - swir1[ok]) / denominator[ok]
)

# Restrict the comparison to locations with valid spectral data.
common = spectral_valid

regions = {
    "Shared prediction": common & mask25 & mask26,
    "2025-only prediction": common & mask25 & ~mask26,
    "2026-only prediction": common & ~mask25 & mask26,
    "Neither prediction": common & ~mask25 & ~mask26,
}

print("\n=== COVERAGE ===")
print("Historical grid pixels:", height * width)
print("Pixels with valid spectral coverage:", int(common.sum()))
print(
    "Spectral coverage (% of historical grid):",
    round(100 * common.mean(), 2),
)

def summarize(name, region):
    count = int(region.sum())
    print(f"\n--- {name} ---")
    print("Valid pixels:", count)

    if count == 0:
        print("No valid pixels in this region.")
        return

    for band_name, values in [
        ("B08 NIR reflectance", nir),
        ("B11 SWIR1 reflectance", swir1),
        ("B12 SWIR2 reflectance", swir2),
    ]:
        selected = values[region]
        print(
            band_name,
            "| median:", round(float(np.median(selected)), 4),
            "| p25:", round(float(np.percentile(selected, 25)), 4),
            "| p75:", round(float(np.percentile(selected, 75)), 4),
        )

    selected_index = nir_swir_index[region]
    selected_index = selected_index[np.isfinite(selected_index)]

    if selected_index.size:
        print(
            "Normalized NIR-SWIR index",
            "| median:", round(float(np.median(selected_index)), 4),
            "| p25:", round(float(np.percentile(selected_index, 25)), 4),
            "| p75:", round(float(np.percentile(selected_index, 75)), 4),
        )

for name, region in regions.items():
    summarize(name, region)

print("\n=== INTERPRETATION CAUTION ===")
print(
    "This raster's acquisition date is not established by its metadata. "
    "Spectral differences are exploratory and do not prove glacier identity."
)
print(
    "Standard NDSI cannot be calculated because B03 green is unavailable."
)
