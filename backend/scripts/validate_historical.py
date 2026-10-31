from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from backend.app.services.rgi_service import find_glacier_by_name


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_DIR = PROJECT_ROOT / "datasets" / "historical"


# ============================================================
# VALIDATION METRICS
# ============================================================

def calculate_validation_metrics(
    mask_path,
    geometry,
    rgi_area_km2,
):
    """
    Compare an existing U-Net segmentation mask with
    the corresponding RGI 7.0 glacier polygon.

    IMPORTANT:
    RGI is used here as an inventory reference, not as
    independent ground-truth segmentation.
    """

    with rasterio.open(mask_path) as src:
        predicted = src.read(1) > 0

        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)

    # --------------------------------------------------------
    # Convert RGI geometry from WGS84 to raster CRS
    # --------------------------------------------------------

    rgi_geometry = transform_geom(
        "EPSG:4326",
        crs,
        geometry,
    )

    # --------------------------------------------------------
    # Rasterize RGI polygon
    # --------------------------------------------------------

    rgi_mask = rasterize(
        [(rgi_geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ) > 0

    # --------------------------------------------------------
    # Pixel counts
    # --------------------------------------------------------

    predicted_pixels = int(
        np.count_nonzero(predicted)
    )

    rgi_pixels = int(
        np.count_nonzero(rgi_mask)
    )

    intersection = int(
        np.count_nonzero(
            predicted & rgi_mask
        )
    )

    union = int(
        np.count_nonzero(
            predicted | rgi_mask
        )
    )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice = (
        2.0 * intersection
        / (predicted_pixels + rgi_pixels + 1e-8)
    )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = (
        intersection
        / (union + 1e-8)
    )

    # --------------------------------------------------------
    # RGI coverage
    # --------------------------------------------------------

    coverage = (
        100.0
        * predicted_pixels
        / (rgi_pixels + 1e-8)
    )

    # --------------------------------------------------------
    # Pixel area
    # --------------------------------------------------------

    pixel_area_m2 = (
        abs(transform.a)
        * abs(transform.e)
    )

    # --------------------------------------------------------
    # Predicted glacier area
    # --------------------------------------------------------

    predicted_area_km2 = (
        predicted_pixels
        * pixel_area_m2
        / 1_000_000
    )

    # --------------------------------------------------------
    # Rasterized RGI area
    # --------------------------------------------------------

    rgi_raster_area_km2 = (
        rgi_pixels
        * pixel_area_m2
        / 1_000_000
    )

    # --------------------------------------------------------
    # Difference from RGI inventory area
    # --------------------------------------------------------

    if rgi_area_km2 is not None:
        area_difference_km2 = (
            predicted_area_km2
            - float(rgi_area_km2)
        )
    else:
        area_difference_km2 = None

    if (
        rgi_area_km2 is not None
        and float(rgi_area_km2) != 0
    ):
        area_difference_percent = (
            100.0
            * area_difference_km2
            / float(rgi_area_km2)
        )
    else:
        area_difference_percent = None

    # --------------------------------------------------------
    # Return metrics
    # --------------------------------------------------------

    return {
        "predicted_pixels": predicted_pixels,
        "rgi_pixels": rgi_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,

        "rgi_overlap_dice": round(
            float(dice),
            6,
        ),

        "rgi_overlap_iou": round(
            float(iou),
            6,
        ),

        "rgi_coverage_percent": round(
            float(coverage),
            4,
        ),

        "predicted_area_from_mask_km2": round(
            float(predicted_area_km2),
            6,
        ),

        "rasterized_rgi_area_km2": round(
            float(rgi_raster_area_km2),
            6,
        ),

        "rgi_inventory_area_km2": (
            round(
                float(rgi_area_km2),
                6,
            )
            if rgi_area_km2 is not None
            else None
        ),

        "area_difference_km2": (
            round(
                float(area_difference_km2),
                6,
            )
            if area_difference_km2 is not None
            else None
        ),

        "area_difference_percent": (
            round(
                float(area_difference_percent),
                4,
            )
            if area_difference_percent is not None
            else None
        ),
    }


# ============================================================
# VALIDATE HISTORICAL GLACIER
# ============================================================

def validate_historical_glacier(glacier_name):
    """
    Validate all already-processed historical observations
    for the requested glacier.

    No Sentinel-2 download or U-Net inference happens here.
    This script only validates existing masks.
    """

    glacier = find_glacier_by_name(
        glacier_name
    )

    if glacier is None:
        raise RuntimeError(
            f'Glacier "{glacier_name}" was not found in RGI 7.0.'
        )

    rgi_name = glacier["name"]
    rgi_id = glacier["rgi_id"]
    geometry = glacier["geometry"]
    rgi_area_km2 = glacier.get("area_km2")

    if geometry is None:
        raise RuntimeError(
            f'RGI geometry is not available for "{rgi_name}".'
        )

    # --------------------------------------------------------
    # Historical directory
    # --------------------------------------------------------

    # Historical results are stored using the resolved RGI ID.
    # This keeps validation compatible with the RGI-based folder
    # structure used by the historical analysis pipeline.
    #
    # Example:
    #   Kolahoi -> RGI2000-v7.0-G-14-34651
    #   datasets/historical/rgi2000-v7.0-g-14-34651/
    #       historical_results.json
    results_file = (
        HISTORICAL_DIR
        / str(rgi_id).lower()
        / "historical_results.json"
    )

    if not results_file.exists():
        raise FileNotFoundError(
            f"Historical results not found:\n{results_file}"
        )

    # --------------------------------------------------------
    # Load historical results
    # --------------------------------------------------------

    with open(
        results_file,
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    print()
    print("==========================================")
    print(" GLACIERAI - HISTORICAL VALIDATION")
    print("==========================================")
    print("Glacier:", rgi_name)
    print("RGI ID:", rgi_id)
    print(
        "RGI inventory area:",
        rgi_area_km2,
        "km²",
    )
    print()

    # --------------------------------------------------------
    # Validate observations
    # --------------------------------------------------------

    validated = 0
    skipped = 0

    observations = payload.get(
        "observations",
        [],
    )

    for result in observations:

        year = result.get(
            "year",
            "Unknown",
        )

        # ----------------------------------------------------
        # Only validate observations that actually have
        # a processed mask
        # ----------------------------------------------------

        mask_value = result.get(
            "mask_path"
        )

        if not mask_value:
            print(
                f"{year} | SKIPPED | "
                "No mask_path in historical result."
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Resolve mask path
        # ----------------------------------------------------

        mask_path = Path(
            mask_value
        )

        if not mask_path.is_absolute():
            mask_path = (
                PROJECT_ROOT
                / mask_path
            )

        if not mask_path.exists():
            print(
                f"{year} | SKIPPED | "
                f"Mask not found: {mask_path}"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Calculate validation metrics
        # ----------------------------------------------------

        try:

            metrics = calculate_validation_metrics(
                mask_path=mask_path,
                geometry=geometry,
                rgi_area_km2=rgi_area_km2,
            )

        except Exception as error:

            print(
                f"{year} | FAILED | "
                f"{error}"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Store validation inside each observation
        # ----------------------------------------------------

        result["validation"] = metrics

        validated += 1

        print(
            f"{year} | "
            f"Area: "
            f"{result.get('area_km2', 0):.6f} km² | "
            f"Coverage: "
            f"{metrics['rgi_coverage_percent']:.2f}% | "
            f"Dice: "
            f"{metrics['rgi_overlap_dice']:.4f} | "
            f"IoU: "
            f"{metrics['rgi_overlap_iou']:.4f} | "
            f"Area diff: "
            f"{metrics['area_difference_percent']:.2f}%"
        )

    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    payload["validation"] = {
        "method": (
            "Existing U-Net V3 mask compared "
            "with RGI 7.0 polygon"
        ),

        "reference": (
            "RGI 7.0 glacier inventory"
        ),

        "note": (
            "Dice and IoU measure spatial overlap "
            "with the RGI inventory polygon. "
            "They are not independent ground-truth "
            "segmentation metrics."
        ),

        "observations_total": len(
            observations
        ),

        "observations_validated": validated,

        "observations_skipped": skipped,

        "validated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    # --------------------------------------------------------
    # Save updated historical results
    # --------------------------------------------------------

    with open(
        results_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=4,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("==========================================")
    print(" VALIDATION COMPLETE")
    print("==========================================")
    print(
        "Validated observations:",
        validated,
    )
    print(
        "Skipped observations:",
        skipped,
    )
    print(
        "Updated:",
        results_file,
    )
    print()


# ============================================================
# COMMAND-LINE ENTRY
# ============================================================

def main():
    """
    Validate historical observations for any glacier.

    Usage:
        python backend/scripts/validate_historical.py "Kolahoi"
        python backend/scripts/validate_historical.py "Gangotri"
        python backend/scripts/validate_historical.py "Zemu Glacier"

    The glacier is resolved through the RGI service, so no
    glacier-specific folder or RGI ID is hardcoded here.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Validate existing historical U-Net masks "
            "against the RGI 7.0 inventory."
        )
    )

    parser.add_argument(
        "glacier_name",
        help="Glacier name, alias, or RGI ID",
    )

    args = parser.parse_args()

    validate_historical_glacier(
        args.glacier_name
    )


if __name__ == "__main__":
    main()
