from pathlib import Path
import json
from typing import Optional
import numpy as np
import rasterio
# ==========================================================
# PROJECT PATH
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
)
RGI_DATABASE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "global_rgi.gpkg"
)
def get_glacier_name_from_rgi(rgi_id: str) -> str:
    """Return the official glacier name for an exact RGI ID."""
    if not rgi_id or not rgi_id.strip():
        raise ValueError("A non-empty RGI ID is required for name lookup.")
    if not RGI_DATABASE.is_file():
        raise FileNotFoundError(
            f"Global RGI database not found: {RGI_DATABASE}"
        )
    import geopandas as gpd
    try:
        rgi_data = gpd.read_file(
            RGI_DATABASE,
            columns=["rgi_id", "glac_name"],
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not read the global RGI database: {error}"
        ) from error
    required_columns = {"rgi_id", "glac_name"}
    missing_columns = required_columns.difference(rgi_data.columns)
    if missing_columns:
        raise ValueError(
            "The global RGI database is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    matches = rgi_data.loc[
        rgi_data["rgi_id"].astype(str).str.strip() == rgi_id.strip(),
        "glac_name",
    ]
    if matches.empty:
        raise ValueError(f"RGI ID not found in the global inventory: {rgi_id}")
    name = matches.iloc[0]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"No official glacier name is recorded for RGI ID: {rgi_id}")
    return name.strip()
# ==========================================================
# HELPERS
# ==========================================================
def safe_float(value) -> Optional[float]:
    """
    Convert a value to float without raising errors.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
def safe_int(value) -> Optional[int]:
    """
    Convert a value to int without raising errors.
    """
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
def glacier_slug(name: str) -> str:
    """
    Convert glacier name into the historical directory format.
    """
    return (
        name
        .strip()
        .lower()
        .replace(" ", "_")
    )
def rgi_slug(rgi_id: str) -> str:
    """Create the stable directory name used for an RGI ID."""
    return (
        rgi_id
        .strip()
        .lower()
        .replace(" ", "_")
    )
def resolve_historical_directory(
    glacier_name: str,
    rgi_id: Optional[str] = None,
) -> tuple[Path, Optional[str]]:
    """
    Resolve historical results by exact RGI ID first.
    Legacy name folders and existing result files are supported
    as fallbacks so older GlacierAI data continues to work.
    """
    if rgi_id:
        directory = HISTORICAL_DIR / rgi_slug(rgi_id)
        if (directory / "historical_results.json").exists():
            return directory, rgi_id
    name_directory = HISTORICAL_DIR / glacier_slug(glacier_name)
    if (name_directory / "historical_results.json").exists():
        try:
            with open(
                name_directory / "historical_results.json",
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)
            glacier = payload.get("glacier", {})
            stored_rgi_id = (
                glacier.get("rgi_id")
                if isinstance(glacier, dict)
                else None
            )
            return name_directory, stored_rgi_id
        except Exception:
            return name_directory, rgi_id
    # Compatibility fallback: find the saved result whose glacier
    # name matches the requested glacier.
    requested_name = glacier_name.strip().casefold()
    if HISTORICAL_DIR.exists():
        for directory in HISTORICAL_DIR.iterdir():
            if not directory.is_dir():
                continue
            results_file = directory / "historical_results.json"
            if not results_file.exists():
                continue
            try:
                with open(
                    results_file,
                    "r",
                    encoding="utf-8",
                ) as file:
                    payload = json.load(file)
                glacier = payload.get("glacier", {})
                stored_name = (
                    glacier.get("name")
                    if isinstance(glacier, dict)
                    else None
                )
                if (
                    isinstance(stored_name, str)
                    and stored_name.strip().casefold()
                    == requested_name
                ):
                    return directory, glacier.get("rgi_id")
            except Exception:
                continue
    if rgi_id:
        return HISTORICAL_DIR / rgi_slug(rgi_id), rgi_id
    return name_directory, None
# ==========================================================
# OBSERVATION QUALITY
# ==========================================================
def assess_observation_quality(observation: dict) -> dict:
    """Assess observation quality without inventing validation metrics."""
    validation = observation.get("validation", {})
    if not isinstance(validation, dict):
        validation = {}
    def safe_metric(value):
        try:
            result = float(value)
            return result if np.isfinite(result) else None
        except (TypeError, ValueError):
            return None
    dice = safe_metric(validation.get("rgi_overlap_dice"))
    iou = safe_metric(validation.get("rgi_overlap_iou"))
    coverage = safe_metric(validation.get("rgi_coverage_percent"))
    area_difference = safe_metric(validation.get("area_difference_percent"))
    area = safe_metric(observation.get("area_km2"))
    satellite = observation.get("satellite_quality", {})
    area_quality = observation.get("area_quality", {})
    if not isinstance(satellite, dict):
        satellite = {}
    if not isinstance(area_quality, dict):
        area_quality = {}
    reasons = []
    for value, label in [
        (dice, "Dice score unavailable"),
        (iou, "IoU score unavailable"),
        (coverage, "RGI coverage unavailable"),
        (area_difference, "Area difference unavailable"),
        (area, "Predicted area unavailable")
    ]:
        if value is None:
            reasons.append(label)
    zero_fraction = safe_metric(satellite.get("zero_pixel_fraction_of_valid"))
    valid_fraction = safe_metric(satellite.get("valid_pixel_fraction"))
    if zero_fraction is not None and zero_fraction > 0.05:
        reasons.append(f"High all-zero pixel fraction: {zero_fraction:.2%}")
    if valid_fraction is not None and valid_fraction < 0.95:
        reasons.append(f"Low valid-pixel fraction: {valid_fraction:.2%}")
    flags = area_quality.get("flags", [])
    if isinstance(flags, list):
        reasons.extend(str(flag) for flag in flags if flag)
    quality = "UNKNOWN"
    if all(value is not None for value in [dice, iou, coverage, area_difference, area]):
        if area <= 0:
            quality = "LOW"
            reasons.append("Predicted glacier area is zero")
        elif dice >= 0.70 and iou >= 0.50 and coverage >= 50 and abs(area_difference) <= 50:
            quality = "HIGH"
        elif dice >= 0.50 and iou >= 0.30 and coverage >= 25 and abs(area_difference) <= 75:
            quality = "MEDIUM"
        else:
            quality = "LOW"
    elif area is not None and area <= 0:
        quality = "LOW"
        reasons.append("Predicted glacier area is zero")
    explanations = {
        "HIGH": "Observation passed the conservative RGI comparison thresholds.",
        "MEDIUM": "Observation passed the minimum RGI comparison thresholds; interpret cautiously.",
        "LOW": "Observation failed quality checks and is unsuitable for a defensible retreat estimate.",
        "UNKNOWN": "RGI comparison metrics are missing, so observation quality cannot be established."
    }
    return {
        "quality": quality,
        "dice": dice,
        "iou": iou,
        "rgi_coverage_percent": coverage,
        "area_difference_percent": area_difference,
        "zero_pixel_fraction": zero_fraction,
        "valid_pixel_fraction": valid_fraction,
        "area_quality_flags": flags if isinstance(flags, list) else [],
        "reasons": list(dict.fromkeys(reasons)),
        "explanation": explanations[quality]
    }
# MASK PATH
# ==========================================================
def resolve_mask_path(
    observation: dict,
    glacier_directory: Path,
) -> Optional[Path]:
    mask_path = observation.get(
        "mask_path"
    )
    if not mask_path:
        return None
    path = Path(
        str(mask_path)
    )
    # Absolute path
    if path.is_absolute() and path.exists():
        return path
    # Relative to project
    project_path = (
        PROJECT_ROOT
        / str(mask_path)
    )
    if project_path.exists():
        return project_path
    # Relative to glacier directory
    glacier_path = (
        glacier_directory
        / str(mask_path)
    )
    if glacier_path.exists():
        return glacier_path
    return None
# ==========================================================
# SPATIAL MASK COMPARISON
# ==========================================================
def compare_masks(
    previous_mask_path: Path,
    current_mask_path: Path,
) -> Optional[dict]:
    """
    Compare two historical glacier masks.
    Returns spatial overlap and changed-area information.
    This is a spatial change indicator, NOT a direct
    terminus-retreat measurement.
    """
    try:
        with rasterio.open(
            previous_mask_path
        ) as previous_src:
            previous_mask = (
                previous_src.read(1) > 0
            )
            previous_transform = (
                previous_src.transform
            )
            previous_crs = (
                previous_src.crs
            )
            previous_resolution = (
                abs(previous_transform.a)
                *
                abs(previous_transform.e)
            )
        with rasterio.open(
            current_mask_path
        ) as current_src:
            current_mask = (
                current_src.read(1) > 0
            )
            current_transform = (
                current_src.transform
            )
            current_crs = (
                current_src.crs
            )
            current_resolution = (
                abs(current_transform.a)
                *
                abs(current_transform.e)
            )
        # --------------------------------------------------
        # CRS check
        # --------------------------------------------------
        if previous_crs != current_crs:
            return {
                "available": False,
                "reason": (
                    "Historical masks use different CRS."
                ),
            }
        # --------------------------------------------------
        # Raster-grid alignment check
        # --------------------------------------------------
        # Equal CRS and array shape do not guarantee that two masks
        # cover the same geographic pixels. Require matching affine
        # transforms before comparing pixel locations.
        if not np.allclose(
            tuple(previous_transform),
            tuple(current_transform),
            rtol=0,
            atol=1e-8,
        ):
            return {
                "available": False,
                "reason": (
                    "Historical masks use different raster transforms "
                    "and must be aligned before spatial comparison."
                ),
            }
        # Shape check
        # --------------------------------------------------
        if previous_mask.shape != current_mask.shape:
            return {
                "available": False,
                "reason": (
                    "Historical masks have different "
                    "dimensions and require reprojection."
                ),
            }
        # --------------------------------------------------
        # Pixel comparison
        # --------------------------------------------------
        intersection = (
            previous_mask
            &
            current_mask
        )
        union = (
            previous_mask
            |
            current_mask
        )
        gained = (
            current_mask
            &
            ~previous_mask
        )
        lost = (
            previous_mask
            &
            ~current_mask
        )
        intersection_pixels = int(
            intersection.sum()
        )
        union_pixels = int(
            union.sum()
        )
        gained_pixels = int(
            gained.sum()
        )
        lost_pixels = int(
            lost.sum()
        )
        pixel_area_m2 = (
            current_resolution
        )
        changed_pixels = (
            gained_pixels
            +
            lost_pixels
        )
        changed_area_km2 = (
            changed_pixels
            *
            pixel_area_m2
            /
            1_000_000
        )
        gained_area_km2 = (
            gained_pixels
            *
            pixel_area_m2
            /
            1_000_000
        )
        lost_area_km2 = (
            lost_pixels
            *
            pixel_area_m2
            /
            1_000_000
        )
        if union_pixels > 0:
            spatial_iou = (
                intersection_pixels
                /
                union_pixels
            )
        else:
            spatial_iou = None
        return {
            "available": True,
            "intersection_pixels":
                intersection_pixels,
            "union_pixels":
                union_pixels,
            "gained_pixels":
                gained_pixels,
            "lost_pixels":
                lost_pixels,
            "gained_area_km2":
                round(
                    gained_area_km2,
                    6
                ),
            "lost_area_km2":
                round(
                    lost_area_km2,
                    6
                ),
            "changed_area_km2":
                round(
                    changed_area_km2,
                    6
                ),
            "spatial_iou":
                (
                    round(
                        spatial_iou,
                        4
                    )
                    if spatial_iou is not None
                    else None
                ),
            "interpretation": (
                "Spatial boundary change detected between "
                "the two model-generated glacier masks."
            ),
            "note": (
                "This measures model-mask spatial change. "
                "It is not a direct terminus-retreat distance."
            ),
        }
    except Exception as error:
        return {
            "available": False,
            "reason":
                str(error),
        }
# ==========================================================
# YEAR-TO-YEAR AREA CHANGE
# ==========================================================
def calculate_area_changes(
    observations: list
) -> list:
    changes = []
    for previous, current in zip(
        observations,
        observations[1:]
    ):
        previous_year = safe_int(
            previous.get("year")
        )
        current_year = safe_int(
            current.get("year")
        )
        previous_area = safe_float(
            previous.get("area_km2")
        )
        current_area = safe_float(
            current.get("area_km2")
        )
        if (
            previous_year is None
            or
            current_year is None
            or
            previous_area is None
            or
            current_area is None
        ):
            continue
        years_elapsed = (
            current_year
            -
            previous_year
        )
        # Duplicate or out-of-order years cannot form a valid annual
        # comparison. Do not emit a misleading change record for them.
        if years_elapsed <= 0:
            continue
        area_change = (
            current_area
            -
            previous_area
        )
        if previous_area != 0:
            percentage_change = (
                area_change
                /
                previous_area
                *
                100
            )
        else:
            percentage_change = None
        if years_elapsed > 0:
            annualized_change = (
                area_change
                /
                years_elapsed
            )
        else:
            annualized_change = None
        current_quality = (
            current.get(
                "_quality",
                {}
            )
        )
        previous_quality = (
            previous.get(
                "_quality",
                {}
            )
        )
        both_suitable = (
            current_quality.get("quality")
            in {"HIGH", "MEDIUM"}
            and
            previous_quality.get("quality")
            in {"HIGH", "MEDIUM"}
        )
        if both_suitable:
            if area_change < 0:
                interpretation = (
                    "Area decrease observed between "
                    "quality-screened observations."
                )
            elif area_change > 0:
                interpretation = (
                    "Area increase observed between "
                    "quality-screened observations."
                )
            else:
                interpretation = (
                    "No area change observed."
                )
        else:
            interpretation = (
                "Change calculated numerically, but "
                "the observation pair is not sufficiently "
                "validated for a physical retreat conclusion."
            )
        changes.append({
            "from_year":
                previous_year,
            "to_year":
                current_year,
            "from_area_km2":
                round(
                    previous_area,
                    6
                ),
            "to_area_km2":
                round(
                    current_area,
                    6
                ),
            "area_change_km2":
                round(
                    area_change,
                    6
                ),
            "percentage_change":
                (
                    round(
                        percentage_change,
                        4
                    )
                    if percentage_change is not None
                    else None
                ),
            "annualized_area_change_km2_per_year":
                (
                    round(
                        annualized_change,
                        6
                    )
                    if annualized_change is not None
                    else None
                ),
            "from_quality":
                previous_quality.get(
                    "quality",
                    "UNKNOWN"
                ),
            "to_quality":
                current_quality.get(
                    "quality",
                    "UNKNOWN"
                ),
            "suitable_for_retreat":
                both_suitable,
            "interpretation":
                interpretation,
        })
    return changes
# ==========================================================
# OVERALL TREND
# ==========================================================
def calculate_overall_trend(
    observations: list
) -> dict:
    suitable = [
        observation
        for observation in observations
        if observation.get(
            "_quality",
            {}
        ).get(
            "quality"
        )
        in {"HIGH", "MEDIUM"}
    ]
    if len(suitable) < 2:
        return {
            "status":
                "INSUFFICIENT_VALIDATION",
            "usable_observations":
                len(suitable),
            "total_observations":
                len(observations),
            "start_year":
                None,
            "end_year":
                None,
            "start_area_km2":
                None,
            "end_area_km2":
                None,
            "area_change_km2":
                None,
            "percentage_change":
                None,
            "annualized_area_change_km2_per_year":
                None,
            "retreat_rate_m_per_year":
                None,
            "note": (
                "A defensible physical retreat rate cannot "
                "be estimated because fewer than two historical "
                "observations passed the validation quality screen."
            ),
        }
    first = suitable[0]
    last = suitable[-1]
    start_year = safe_int(
        first.get("year")
    )
    end_year = safe_int(
        last.get("year")
    )
    start_area = safe_float(
        first.get("area_km2")
    )
    end_area = safe_float(
        last.get("area_km2")
    )
    if (
        start_year is None
        or end_year is None
        or start_area is None
        or end_area is None
    ):
        return {
            "status":
                "INSUFFICIENT_DATA",
            "note":
                "Required historical area information is missing.",
        }
    years_elapsed = (
        end_year
        -
        start_year
    )
    area_change = (
        end_area
        -
        start_area
    )
    if start_area != 0:
        percentage_change = (
            area_change
            /
            start_area
            *
            100
        )
    else:
        percentage_change = None
    if years_elapsed > 0:
        annualized_change = (
            area_change
            /
            years_elapsed
        )
    else:
        annualized_change = None
    if area_change < 0:
        trend = "AREA_DECREASE"
    elif area_change > 0:
        trend = "AREA_INCREASE"
    else:
        trend = "NO_CHANGE"
    return {
        "status":
            "VALIDATED_AREA_TREND",
        "usable_observations":
            len(suitable),
        "total_observations":
            len(observations),
        "start_year":
            start_year,
        "end_year":
            end_year,
        "start_area_km2":
            round(
                start_area,
                6
            ),
        "end_area_km2":
            round(
                end_area,
                6
            ),
        "area_change_km2":
            round(
                area_change,
                6
            ),
        "percentage_change":
            (
                round(
                    percentage_change,
                    4
                )
                if percentage_change is not None
                else None
            ),
        "annualized_area_change_km2_per_year":
            (
                round(
                    annualized_change,
                    6
                )
                if annualized_change is not None
                else None
            ),
        # --------------------------------------------------
        # DO NOT invent m/year.
        # --------------------------------------------------
        "retreat_rate_m_per_year":
            None,
        "trend":
            trend,
        "note": (
            "The current pipeline estimates glacier area "
            "change. A physical terminus retreat rate in "
            "m/year requires validated multi-temporal "
            "boundary/terminus geometry."
        ),
    }
# ==========================================================
# COMPLETE CHANGE ANALYSIS
# ==========================================================
def run_change_analysis(
    glacier_name: str,
    rgi_id: Optional[str] = None,
) -> dict:
    if not rgi_id and not (glacier_name and glacier_name.strip()):
        raise ValueError("Provide either an RGI ID or a glacier name.")
    if rgi_id:
        glacier_name = get_glacier_name_from_rgi(rgi_id)
    glacier_directory, resolved_rgi_id = (
        resolve_historical_directory(
            glacier_name=glacier_name,
            rgi_id=rgi_id,
        )
    )
    if resolved_rgi_id and not rgi_id:
        glacier_name = get_glacier_name_from_rgi(resolved_rgi_id)
    results_file = (
        glacier_directory
        / "historical_results.json"
    )
    if not results_file.exists():
        raise FileNotFoundError(
            f"Historical results not found: "
            f"{results_file}"
        )
    with open(
        results_file,
        "r",
        encoding="utf-8"
    ) as file:
        raw_data = json.load(
            file
        )
    observations = raw_data.get(
        "observations",
        []
    )
    if not isinstance(
        observations,
        list
    ):
        observations = []
    valid_observations = []
    for observation in observations:
        if not isinstance(
            observation,
            dict
        ):
            continue
        area = safe_float(
            observation.get(
                "area_km2"
            )
        )
        if area is None:
            continue
        quality = (
            assess_observation_quality(
                observation
            )
        )
        observation_copy = (
            dict(observation)
        )
        observation_copy[
            "_quality"
        ] = quality
        valid_observations.append(
            observation_copy
        )
    valid_observations.sort(
        key=lambda item:
            safe_int(
                item.get("year")
            )
            or 0
    )
    # ======================================================
    # AREA CHANGES
    # ======================================================
    area_changes = (
        calculate_area_changes(
            valid_observations
        )
    )
    # ======================================================
    # SPATIAL MASK COMPARISON
    # ======================================================
    spatial_changes = []
    for previous, current in zip(
        valid_observations,
        valid_observations[1:]
    ):
        previous_mask = (
            resolve_mask_path(
                previous,
                glacier_directory
            )
        )
        current_mask = (
            resolve_mask_path(
                current,
                glacier_directory
            )
        )
        result = {
            "from_year":
                previous.get(
                    "year"
                ),
            "to_year":
                current.get(
                    "year"
                ),
        }
        if (
            previous_mask is None
            or
            current_mask is None
        ):
            result.update({
                "available":
                    False,
                "reason":
                    "Historical mask files were not available."
            })
        else:
            comparison = (
                compare_masks(
                    previous_mask,
                    current_mask
                )
            )
            result.update(
                comparison
            )
        spatial_changes.append(
            result
        )
    # ======================================================
    # OVERALL TREND
    # ======================================================
    overall = (
        calculate_overall_trend(
            valid_observations
        )
    )
    # ======================================================
    # QUALITY SUMMARY
    # ======================================================
    high = 0
    medium = 0
    low = 0
    unknown = 0
    for observation in valid_observations:
        quality = (
            observation
            .get("_quality", {})
            .get(
                "quality",
                "UNKNOWN"
            )
        )
        if quality == "HIGH":
            high += 1
        elif quality == "MEDIUM":
            medium += 1
        elif quality == "LOW":
            low += 1
        else:
            unknown += 1
    # ======================================================
    # FINAL RESULT
    # ======================================================
    result = {
        "status":
            "success",
        "glacier":
            glacier_name,
        "rgi_id":
            resolved_rgi_id,
        "analysis_type":
            "Area Change and Spatial Retreat Analysis",
        "method":
            "Historical U-Net V3 observations with "
            "validation-aware quality screening",
        "observation_quality": [
            {
                "year":
                    observation.get(
                        "year"
                    ),
                "quality":
                    observation[
                        "_quality"
                    ],
            }
            for observation
            in valid_observations
        ],
        "quality_summary": {
            "high":
                high,
            "medium":
                medium,
            "low":
                low,
            "unknown":
                unknown,
            "total":
                len(valid_observations),
        },
        "year_to_year_area_change":
            area_changes,
        "spatial_boundary_change":
            spatial_changes,
        "overall_trend":
            overall,
        "scientific_note": (
            "Area change is derived from U-Net V3 "
            "segmentation. Physical terminus retreat "
            "in metres/year is not reported unless "
            "validated multi-temporal boundary geometry "
            "supports that measurement."
        ),
    }
    # ======================================================
    # SAVE RESULT
    # ======================================================
    output_file = (
        glacier_directory
        / "change_analysis.json"
    )
    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=4
        )
    print()
    print(
        "=========================================="
    )
    print(
        " GLACIERAI - CHANGE ANALYSIS"
    )
    print(
        "=========================================="
    )
    print(
        "Glacier:",
        glacier_name
    )
    print(
        "Observations:",
        len(valid_observations)
    )
    print(
        "High quality:",
        high
    )
    print(
        "Medium quality:",
        medium
    )
    print(
        "Low quality:",
        low
    )
    print(
        "Overall status:",
        overall.get(
            "status"
        )
    )
    print(
        "Result saved:",
        output_file
    )
    print()
    return result
# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "GlacierAI historical "
            "area-change analysis"
        )
    )
    parser.add_argument(
        "--glacier",
        default=None,
        help="Glacier name (used when --rgi-id is not supplied)"
    )
    parser.add_argument(
        "--rgi-id",
        default=None,
        help="Exact RGI 7.0 glacier ID; official name is read from the RGI database"
    )
    args = parser.parse_args()
    if not args.rgi_id and not args.glacier:
        parser.error("provide either --rgi-id or --glacier")
    run_change_analysis(
        glacier_name=args.glacier or "",
        rgi_id=args.rgi_id,
    )
