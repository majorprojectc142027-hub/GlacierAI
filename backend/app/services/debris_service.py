from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from rasterio.features import sieve


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LIVE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "live"
)

PREDICTION_DIR = (
    LIVE_DIR
    / "predictions"
)

DEFAULT_SPECTRAL_PATH = (
    LIVE_DIR
    / "gangotri_debris_spectral.tif"
)

DEFAULT_GLACIER_MASK = (
    PREDICTION_DIR
    / "gangotri_rgi_glacier_mask.tif"
)

DEFAULT_OUTPUT = (
    PREDICTION_DIR
    / "gangotri_debris_affected_mask.tif"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

REFLECTANCE_SCALE = 10000.0

# Minimum number of connected pixels retained.
# At 12 m resolution:
#
# 1 pixel = 144 m²
#
# This removes extremely small isolated detections.
MIN_COMPONENT_PIXELS = 20

# Initial spectral thresholds.
#
# These are deliberately treated as an estimation rule,
# NOT as ground-truth debris segmentation.
#
# Features:
#
# SWIR / NIR ratio
# B11 / B08
#
# SWIR2 / NIR ratio
# B12 / B08
#
# Lower NIR relative to SWIR is used as a debris
# candidate characteristic.
SWIR1_NIR_MIN = 0.35
SWIR2_NIR_MIN = 0.30

# NIR reflectance upper limit.
#
# Very bright clean snow/ice generally has strong NIR
# response. This helps avoid classifying bright snow as
# debris.
NIR_MAX = 0.55


# ==========================================================
# LOAD SPECTRAL DATA
# ==========================================================

def load_spectral_data(
    spectral_path: str | Path,
):
    """
    Load B08, B11 and B12 reflectance data.

    Expected order:

        Band 1 -> B08
        Band 2 -> B11
        Band 3 -> B12
    """

    spectral_path = Path(
        spectral_path
    )

    if not spectral_path.exists():

        raise FileNotFoundError(
            f"Spectral raster not found: {spectral_path}"
        )

    with rasterio.open(
        spectral_path
    ) as dataset:

        if dataset.count != 3:

            raise ValueError(
                "Expected exactly 3 spectral bands "
                "(B08, B11, B12), but found "
                f"{dataset.count}."
            )

        # Read as masked arrays so GeoTIFF NoData values and
        # per-band validity masks never enter spectral calculations.
        b08 = (
            dataset.read(1, masked=True).astype(np.float32).filled(np.nan)
            / REFLECTANCE_SCALE
        )
        b11 = (
            dataset.read(2, masked=True).astype(np.float32).filled(np.nan)
            / REFLECTANCE_SCALE
        )
        b12 = (
            dataset.read(3, masked=True).astype(np.float32).filled(np.nan)
            / REFLECTANCE_SCALE
        )

        profile = dataset.profile.copy()

        transform = dataset.transform
        crs = dataset.crs

        shape = dataset.shape

    return (
        b08,
        b11,
        b12,
        profile,
        transform,
        crs,
        shape,
    )


# ==========================================================
# LOAD GLACIER MASK
# ==========================================================

def load_glacier_mask(
    glacier_mask_path: str | Path,
):
    """
    Load a U-Net glacier mask.

    The prediction GeoTIFF uses 1 for glacier, 0 for
    background and 255 for NoData. Only valid pixels
    whose value is exactly 1 are glacier pixels.
    """

    glacier_mask_path = Path(
        glacier_mask_path
    )

    if not glacier_mask_path.exists():

        raise FileNotFoundError(
            f"Glacier mask not found: {glacier_mask_path}"
        )

    with rasterio.open(
        glacier_mask_path
    ) as dataset:

        mask = dataset.read(1)
        data_valid = dataset.read_masks(1) > 0
        nodata = dataset.nodata

        profile = dataset.profile.copy()

        transform = dataset.transform
        crs = dataset.crs

        shape = dataset.shape

    # Do not use mask > 0 here: 255 is the prediction
    # raster's NoData sentinel, not a glacier class.
    glacier_mask = (mask == 1) & data_valid

    if nodata is not None:
        glacier_mask &= mask != nodata

    return (
        glacier_mask,
        profile,
        transform,
        crs,
        shape,
    )


# ==========================================================
# GRID VALIDATION
# ==========================================================

def validate_same_grid(
    spectral_shape,
    spectral_transform,
    spectral_crs,
    mask_shape,
    mask_transform,
    mask_crs,
):
    """
    Ensure spectral data and glacier mask share exactly
    the same raster grid.
    """

    if spectral_shape != mask_shape:

        raise ValueError(
            "Spectral raster and glacier mask have "
            f"different shapes: "
            f"{spectral_shape} vs {mask_shape}"
        )

    if spectral_crs != mask_crs:

        raise ValueError(
            "Spectral raster and glacier mask have "
            f"different CRS: "
            f"{spectral_crs} vs {mask_crs}"
        )

    if spectral_transform != mask_transform:

        raise ValueError(
            "Spectral raster and glacier mask do not "
            "have the same geospatial transform."
        )


# ==========================================================
# SPECTRAL FEATURE CALCULATION
# ==========================================================

def calculate_spectral_features(
    b08,
    b11,
    b12,
):
    """
    Calculate simple spectral features used for
    debris-affected region estimation.

    Features:

        SWIR1/NIR = B11 / B08
        SWIR2/NIR = B12 / B08

    Small epsilon prevents division by zero.
    """

    epsilon = 1e-6

    swir1_nir = (
        b11
        / (
            b08
            + epsilon
        )
    )

    swir2_nir = (
        b12
        / (
            b08
            + epsilon
        )
    )

    return (
        swir1_nir,
        swir2_nir,
    )


# ==========================================================
# DEBRIS CANDIDATE ESTIMATION
# ==========================================================

def estimate_debris_mask(
    glacier_mask,
    b08,
    b11,
    b12,
):
    """
    Estimate debris-affected glacier pixels.

    This is a spectral candidate estimation, not a
    validated debris segmentation model.

    A pixel is considered a debris candidate when:

        1. It belongs to the glacier mask.
        2. SWIR1/NIR exceeds the configured threshold.
        3. SWIR2/NIR exceeds the configured threshold.
        4. NIR reflectance is below the configured upper limit.
    """

    (
        swir1_nir,
        swir2_nir,
    ) = calculate_spectral_features(
        b08,
        b11,
        b12,
    )

    valid_pixel = (
        np.isfinite(b08)
        & np.isfinite(b11)
        & np.isfinite(b12)
    )

    debris_candidate = (
        glacier_mask
        & valid_pixel
        & (b08 > 0)
        & (b11 > 0)
        & (b12 > 0)
        & (
            swir1_nir
            >= SWIR1_NIR_MIN
        )
        & (
            swir2_nir
            >= SWIR2_NIR_MIN
        )
        & (
            b08
            <= NIR_MAX
        )
    )

    return (
        debris_candidate,
        swir1_nir,
        swir2_nir,
    )


# ==========================================================
# REMOVE SMALL COMPONENTS
# ==========================================================

def clean_debris_mask(
    debris_mask,
    min_pixels=MIN_COMPONENT_PIXELS,
):
    """
    Remove very small isolated debris candidates.
    """

    mask_uint8 = (
        debris_mask
        .astype(np.uint8)
    )

    cleaned = sieve(
        mask_uint8,
        size=min_pixels,
        connectivity=8,
    )

    return (
        cleaned > 0
    )


# ==========================================================
# AREA CALCULATION
# ==========================================================

def calculate_area_km2(
    pixel_count,
    transform,
    crs,
):
    """Calculate pixel area using the affine determinant and CRS units."""
    if crs is None:
        raise ValueError("Cannot calculate area: raster CRS is missing.")
    projected_crs = PyprojCRS.from_user_input(crs)
    if not projected_crs.is_projected:
        raise ValueError(
            f"Cannot calculate area from angular/geographic CRS {crs}. "
            "Reproject the raster to an appropriate projected CRS first."
        )
    if len(projected_crs.axis_info) < 2:
        raise ValueError(f"Cannot determine linear units for CRS {crs}.")
    x_factor = projected_crs.axis_info[0].unit_conversion_factor
    y_factor = projected_crs.axis_info[1].unit_conversion_factor
    if not x_factor or not y_factor:
        raise ValueError(f"Unsupported or unknown linear units for CRS {crs}.")
    # Determinant handles rotated/sheared rasters as well as north-up grids.
    pixel_area_native = abs(
        transform.a * transform.e - transform.b * transform.d
    )
    pixel_area_m2 = pixel_area_native * x_factor * y_factor
    if not np.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError(f"Invalid pixel area derived from transform: {pixel_area_m2}")
    return float(pixel_count * pixel_area_m2 / 1_000_000.0)


# ==========================================================
# SAVE DEBRIS MASK
# ==========================================================

def save_debris_mask(
    debris_mask,
    profile,
    output_path,
):
    """
    Save debris-affected mask as a georeferenced
    uint8 GeoTIFF.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_profile = (
        profile.copy()
    )

    output_profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="lzw",
    )

    with rasterio.open(
        output_path,
        "w",
        **output_profile,
    ) as dataset:

        dataset.write(
            debris_mask.astype(
                np.uint8
            ),
            1,
        )


# ==========================================================
# MAIN DEBRIS ANALYSIS
# ==========================================================

def analyze_debris(
    spectral_path: str | Path = DEFAULT_SPECTRAL_PATH,
    glacier_mask_path: str | Path = DEFAULT_GLACIER_MASK,
    output_path: str | Path = DEFAULT_OUTPUT,
):
    """
    Run debris-affected region estimation.

    Returns a dictionary containing:

        debris pixels
        glacier pixels
        debris area
        glacier area
        debris percentage
        output mask
        spectral thresholds

    Scientific note:

        This is an estimation based on spectral
        characteristics inside the valid U-Net glacier mask.
        It is NOT an independently validated debris
        ground-truth segmentation.
    """

    print()
    print("==========================================")
    print(" GLACIERAI - DEBRIS ANALYSIS")
    print("==========================================")
    print()

    print(
        "Spectral raster :",
        spectral_path,
    )

    print(
        "Glacier mask    :",
        glacier_mask_path,
    )

    # ======================================================
    # LOAD SPECTRAL DATA
    # ======================================================

    (
        b08,
        b11,
        b12,
        spectral_profile,
        spectral_transform,
        spectral_crs,
        spectral_shape,
    ) = load_spectral_data(
        spectral_path
    )

    # ======================================================
    # LOAD GLACIER MASK
    # ======================================================

    (
        glacier_mask,
        mask_profile,
        mask_transform,
        mask_crs,
        mask_shape,
    ) = load_glacier_mask(
        glacier_mask_path
    )

    # ======================================================
    # GRID CHECK
    # ======================================================

    validate_same_grid(
        spectral_shape,
        spectral_transform,
        spectral_crs,
        mask_shape,
        mask_transform,
        mask_crs,
    )

    # ======================================================
    # VALID SPECTRAL COVERAGE
    # ======================================================

    spectral_valid = (
        np.isfinite(b08)
        & np.isfinite(b11)
        & np.isfinite(b12)
        & (b08 > 0)
        & (b11 > 0)
        & (b12 > 0)
    )
    glacier_spectral_valid = glacier_mask & spectral_valid
    spectral_valid_glacier_pixels = int(
        np.count_nonzero(glacier_spectral_valid)
    )

    # ======================================================
    # STATISTICS BEFORE CLASSIFICATION
    # ======================================================

    glacier_pixels = int(
        np.count_nonzero(
            glacier_mask
        )
    )

    glacier_area_km2 = (
        calculate_area_km2(
            glacier_pixels,
            spectral_transform,
            spectral_crs,
        )
    )

    print(
        "Glacier pixels :",
        glacier_pixels,
    )

    print(
        "Glacier area   :",
        f"{glacier_area_km2:.6f}",
        "km²",
    )
    print(
        "Spectral-valid glacier pixels:",
        spectral_valid_glacier_pixels,
    )

    if glacier_pixels == 0:

        raise RuntimeError(
            "The supplied glacier mask contains "
            "zero glacier pixels."
        )

    # ======================================================
    # DEBRIS ESTIMATION
    # ======================================================

    (
        debris_candidate,
        swir1_nir,
        swir2_nir,
    ) = estimate_debris_mask(
        glacier_mask,
        b08,
        b11,
        b12,
    )

    # ======================================================
    # CLEAN SMALL ISOLATED AREAS
    # ======================================================

    debris_mask = (
        clean_debris_mask(
            debris_candidate
        )
    )

    # Never allow debris outside glacier mask.
    debris_mask &= glacier_mask

    # ======================================================
    # DEBRIS STATISTICS
    # ======================================================

    debris_pixels = int(
        np.count_nonzero(
            debris_mask
        )
    )

    debris_area_km2 = (
        calculate_area_km2(
            debris_pixels,
            spectral_transform,
            spectral_crs,
        )
    )

    debris_percentage = (
        (debris_pixels / spectral_valid_glacier_pixels) * 100.0
        if spectral_valid_glacier_pixels > 0
        else 0.0
    )

    # ======================================================
    # SAVE
    # ======================================================

    save_debris_mask(
        debris_mask,
        mask_profile,
        output_path,
    )

    # ======================================================
    # SPECTRAL STATISTICS INSIDE GLACIER
    # ======================================================

    glacier_valid = glacier_spectral_valid

    if np.any(glacier_valid):

        mean_b08 = float(
            np.mean(
                b08[glacier_valid]
            )
        )

        mean_b11 = float(
            np.mean(
                b11[glacier_valid]
            )
        )

        mean_b12 = float(
            np.mean(
                b12[glacier_valid]
            )
        )

        mean_swir1_nir = float(
            np.mean(
                swir1_nir[glacier_valid]
            )
        )

        mean_swir2_nir = float(
            np.mean(
                swir2_nir[glacier_valid]
            )
        )

    else:

        mean_b08 = 0.0
        mean_b11 = 0.0
        mean_b12 = 0.0
        mean_swir1_nir = 0.0
        mean_swir2_nir = 0.0

    # ======================================================
    # OUTPUT
    # ======================================================

    result = {

        "status":
            "SUCCESS",

        "analysis_type":
            "debris_affected_region_estimation",

        "scientific_note":
            (
                "Spectral estimation inside the U-Net "
                "glacier mask. This is not independently "
                "validated debris ground-truth segmentation."
            ),

        "spectral_path":
            str(
                Path(
                    spectral_path
                )
            ),

        "glacier_mask_path":
            str(
                Path(
                    glacier_mask_path
                )
            ),

        "output_mask_path":
            str(
                Path(
                    output_path
                )
            ),

        "bands":
            [
                "B08",
                "B11",
                "B12",
            ],

        "glacier_pixels":
            glacier_pixels,

        "spectral_valid_glacier_pixels":
            spectral_valid_glacier_pixels,

        "debris_pixels":
            debris_pixels,

        "glacier_area_km2":
            glacier_area_km2,

        "debris_area_km2":
            debris_area_km2,

        "debris_percentage":
            float(debris_percentage),

        "debris_percentage_denominator":
            "glacier pixels with valid B08, B11 and B12 observations",

        "thresholds":
            {
                "swir1_nir_min":
                    SWIR1_NIR_MIN,

                "swir2_nir_min":
                    SWIR2_NIR_MIN,

                "nir_max":
                    NIR_MAX,

                "minimum_component_pixels":
                    MIN_COMPONENT_PIXELS,
            },

        "spectral_statistics":
            {
                "mean_b08":
                    mean_b08,

                "mean_b11":
                    mean_b11,

                "mean_b12":
                    mean_b12,

                "mean_swir1_nir":
                    mean_swir1_nir,

                "mean_swir2_nir":
                    mean_swir2_nir,
            },

        "pixel_area_m2":
            float(
                abs(spectral_transform.a * spectral_transform.e - spectral_transform.b * spectral_transform.d)
                * PyprojCRS.from_user_input(spectral_crs).axis_info[0].unit_conversion_factor
                * PyprojCRS.from_user_input(spectral_crs).axis_info[1].unit_conversion_factor
            ),

        "crs":
            str(
                spectral_crs
            ),
    }

    print()
    print("==========================================")
    print(" DEBRIS ANALYSIS COMPLETE")
    print("==========================================")
    print()

    print(
        "Glacier pixels :",
        glacier_pixels,
    )

    print(
        "Glacier area   :",
        f"{glacier_area_km2:.6f}",
        "km²",
    )

    print(
        "Debris pixels  :",
        debris_pixels,
    )

    print(
        "Debris area    :",
        f"{debris_area_km2:.6f}",
        "km²",
    )

    print(
        "Debris %       :",
        f"{debris_percentage:.2f}%",
    )

    print(
        "Output mask    :",
        output_path,
    )

    print()

    return result


# ==========================================================
# COMMAND-LINE TEST
# ==========================================================

if __name__ == "__main__":

    try:

        result = analyze_debris()

        print()
        print("Result:")
        print(result)

    except Exception as error:

        print()
        print("==========================================")
        print(" DEBRIS ANALYSIS FAILED")
        print("==========================================")
        print()

        print("Error:",error,)