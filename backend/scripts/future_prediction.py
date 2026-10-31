"""
GlacierAI - Validation-Aware Future Glacier Area Prediction

Purpose
-------
Generate future glacier-area estimates using a linear trend.

The prediction uses:
1. HIGH and MEDIUM quality observations when validation metadata exists.
2. A lower-confidence fallback using available historical U-Net observations
   when validation metadata is unavailable.

Important
---------
This is a trend-based estimate, NOT a physical climate model.

The script is glacier-independent and works with historical JSON files stored
under:

datasets/historical/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from backend.scripts.change_analysis import assess_observation_quality

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

HISTORICAL_DIR = BASE_DIR / "datasets" / "historical"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def slugify(value: str) -> str:
    """
    Convert a glacier name into a safe filename slug.
    """

    value = str(value).strip().lower()

    cleaned = []

    for char in value:
        if char.isalnum():
            cleaned.append(char)
        elif char in (" ", "-", "_"):
            cleaned.append("_")

    slug = "".join(cleaned)

    while "__" in slug:
        slug = slug.replace("__", "_")

    return slug.strip("_")

def rgi_slug(rgi_id: str | None) -> str:
    """
    Convert an RGI ID into the exact folder format used by GlacierAI.

    RGI folders preserve the standard hyphenated ID, for example:
        RGI2000-v7.0-G-15-00467
        -> rgi2000-v7.0-g-15-00467

    Glacier names still use slugify(); RGI IDs do not.
    """

    if not rgi_id:
        return ""

    return str(rgi_id).strip().lower()

# ============================================================
# FIND HISTORICAL FILE
# ============================================================

def find_historical_file(
    glacier_name: str | None = None,
    rgi_id: str | None = None,
):
    """
    Find the historical JSON file for a glacier.

    Priority:
    1. RGI ID
    2. Glacier name
    3. Matching JSON file
    """

    HISTORICAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. RGI ID
    # --------------------------------------------------------

    if rgi_id:

        rgi_folder = HISTORICAL_DIR / rgi_slug(rgi_id)

        candidate = (
            rgi_folder
            / "historical_results.json"
        )

        if candidate.exists():
            return candidate

        # Support older flat-file layout.
        flat_candidate = (
            HISTORICAL_DIR
            / f"{rgi_slug(rgi_id)}.json"
        )

        if flat_candidate.exists():
            return flat_candidate

    # --------------------------------------------------------
    # 2. Glacier name
    # --------------------------------------------------------

    if glacier_name:

        name_slug = slugify(glacier_name)

        candidate = (
            HISTORICAL_DIR
            / name_slug
            / "historical_results.json"
        )

        if candidate.exists():
            return candidate

        # Support older flat-file layout.
        flat_candidate = (
            HISTORICAL_DIR
            / f"{name_slug}.json"
        )

        if flat_candidate.exists():
            return flat_candidate

    # --------------------------------------------------------
    # 3. Search files
    # --------------------------------------------------------

    json_files = list(
        HISTORICAL_DIR.glob("*.json")
    )

    json_files += list(
        HISTORICAL_DIR.glob(
            "*/historical_results.json"
        )
    )

    if not json_files:
        return None

    glacier_name_slug = (
        slugify(glacier_name)
        if glacier_name
        else ""
    )

    rgi_id_slug = (
        rgi_slug(rgi_id)
        if rgi_id
        else ""
    )

    for file in json_files:

        filename = file.stem.lower()

        relative_text = str(
            file.relative_to(HISTORICAL_DIR)
        ).lower()

        if (
            rgi_id_slug
            and (
                rgi_id_slug.lower() in filename
                or rgi_id_slug.lower() in relative_text
            )
        ):
            return file

        if (
            glacier_name_slug
            and (
                glacier_name_slug.lower() in filename
                or glacier_name_slug.lower() in relative_text
            )
        ):
            return file

    return None

# ============================================================
# LOAD HISTORICAL OBSERVATIONS
# ============================================================

def extract_raw_observations(
    data: dict,
) -> list[dict]:
    """
    Extract historical observations from supported JSON structures.

    Supported keys:
        observations
        records
        results
        historical

    Each observation should contain:
        year
        area_km2
    """

    possible_keys = [
        "observations",
        "records",
        "results",
        "historical",
    ]

    raw_records = None

    for key in possible_keys:

        value = data.get(key)

        if isinstance(value, list):

            raw_records = value

            break

    if not raw_records:
        return []

    # --------------------------------------------------------
    # Normalize and deduplicate by year
    # --------------------------------------------------------

    by_year = {}

    for item in raw_records:

        if not isinstance(item, dict):
            continue

        try:

            year = int(
                item.get("year")
            )

            area = float(
                item.get("area_km2")
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if area < 0:
            continue

        observation = dict(item)

        observation["year"] = year

        observation["area_km2"] = area

        # Keep latest occurrence for duplicate years.
        by_year[year] = observation

    observations = list(
        by_year.values()
    )

    observations.sort(
        key=lambda item: item["year"]
    )

    return observations

# ============================================================
# VALIDATION SCREEN
# ============================================================

def prepare_validated_observations(
    data: dict,
):
    """
    Apply GlacierAI's existing observation-quality assessment.

    HIGH and MEDIUM observations are considered usable.

    LOW and UNKNOWN observations are excluded from the
    validation-based regression.
    """

    raw_observations = (
        extract_raw_observations(data)
    )

    screened = []

    usable = []

    high_quality = 0

    medium_quality = 0

    low_quality = 0

    unknown_quality = 0

    for observation in raw_observations:

        quality = assess_observation_quality(
            observation
        )

        enriched = dict(
            observation
        )

        enriched["_quality"] = quality

        quality_label = quality.get(
            "quality",
            "UNKNOWN",
        )

        screened.append(
            enriched
        )

        if quality_label == "HIGH":

            high_quality += 1

            usable.append(
                enriched
            )

        elif quality_label == "MEDIUM":

            medium_quality += 1

            usable.append(
                enriched
            )

        elif quality_label == "LOW":

            low_quality += 1

        else:

            unknown_quality += 1

    summary = {

        "total_observations":
            len(raw_observations),

        "usable_observations":
            len(usable),

        "high_quality_observations":
            high_quality,

        "medium_quality_observations":
            medium_quality,

        "excluded_low_quality_observations":
            low_quality,

        "excluded_unknown_quality_observations":
            unknown_quality,

        "usable_years": [
            item["year"]
            for item in usable
        ],

        "excluded_years": [
            item["year"]
            for item in screened
            if item not in usable
        ],
    }

    return (
        screened,
        usable,
        summary,
    )

# ============================================================
# FUTURE PREDICTION
# ============================================================

def generate_future_prediction(
    glacier_name: str | None = None,
    rgi_id: str | None = None,
    future_start_year: int | None = None,
    future_end_year: int | None = None,
):
    """
    Generate future glacier-area estimates.

    Returns
    -------
    dict
        Structured prediction result.
    """

    # --------------------------------------------------------
    # Locate historical file
    # --------------------------------------------------------

    historical_file = find_historical_file(
        glacier_name=glacier_name,
        rgi_id=rgi_id,
    )

    if historical_file is None:

        raise FileNotFoundError(
            "Historical data file not found. "
            f"Expected under: {HISTORICAL_DIR}"
        )

    print()
    print(
        "Historical file found:"
    )
    print(
        historical_file
    )

    # --------------------------------------------------------
    # Load historical data
    # --------------------------------------------------------

    with historical_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        historical_data = json.load(
            file
        )

    stored_glacier = historical_data.get(
        "glacier",
        {},
    )

    if not isinstance(
        stored_glacier,
        dict,
    ):
        stored_glacier = {}

    resolved_name = (
        glacier_name
        or stored_glacier.get("name")
        or historical_data.get(
            "glacier_name"
        )
        or historical_file.stem
    )

    resolved_rgi_id = (
        rgi_id
        or stored_glacier.get("rgi_id")
        or historical_data.get("rgi_id")
    )

    # --------------------------------------------------------
    # Validation-aware screening
    # --------------------------------------------------------

    (
        all_observations,
        usable_observations,
        validation_summary,
    ) = prepare_validated_observations(
        historical_data
    )

    # --------------------------------------------------------
    # Basic historical arrays
    # --------------------------------------------------------

    all_years = [
        item["year"]
        for item in all_observations
    ]

    all_areas = [
        item["area_km2"]
        for item in all_observations
    ]

    usable_years = [
        item["year"]
        for item in usable_observations
    ]

    usable_areas = [
        item["area_km2"]
        for item in usable_observations
    ]

    # --------------------------------------------------------
    # Select observations for regression
    # --------------------------------------------------------

    if len(usable_observations) >= 2:

        regression_observations = (
            usable_observations
        )

        prediction_confidence = (
            "validated"
        )

        prediction_method = (
            "Validation-aware Linear Trend Regression"
        )

        prediction_note = (
            "Trend-based estimate fitted to "
            "HIGH/MEDIUM quality historical "
            "observations. This is not a "
            "physical climate prediction."
        )

    elif len(all_observations) >= 2:
        # Use all historical observations only as an explicitly low-confidence fallback.
        # This makes a trend available when quality metadata is missing, without claiming validation.
        regression_observations = all_observations
        prediction_confidence = "low"
        prediction_method = "Low-Confidence Historical U-Net Area Linear Trend"
        prediction_note = (
            "Low-confidence trend-based estimate fitted to available historical observations "
            "because fewer than two observations passed HIGH/MEDIUM quality checks. "
            "Review the input dates, segmentation masks, and area calculations before use. "
            "This is not a physical climate prediction and must not be treated as confirmed retreat."
        )
    else:

        output_data = {

            "status":
                "INSUFFICIENT_HISTORY",

            "glacier": {
                "name":
                    resolved_name,

                "rgi_id":
                    resolved_rgi_id,
            },

            "prediction_method":
                "Historical U-Net Area Linear Trend",

            "historical_years":
                all_years,

            "historical_area_km2":
                all_areas,

            "used_historical_years":
                all_years,

            "used_historical_area_km2":
                all_areas,

            "trend_slope_km2_per_year":
                None,

            "predictions":
                [],

            "predicted_change_km2":
                None,

            "predicted_change_percent":
                None,

            "validation_summary":
                validation_summary,

            "prediction_confidence":
                "unavailable",

            "note":
                "At least two historical observations "
                "are required to estimate a future trend.",
        }

        # IMPORTANT:
        # Save inside the glacier-specific folder.
        output_file = (
            historical_file.parent
            / "future_prediction.json"
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output_data,
                file,
                indent=2,
            )

        return {
            "data":
                output_data,

            "json_path":
                str(output_file),

            "graph_path":
                None,
        }

    # ========================================================
    # REGRESSION DATA
    # ========================================================

    years = np.array(
        [
            item["year"]
            for item in regression_observations
        ],
        dtype=float,
    )

    areas = np.array(
        [
            item["area_km2"]
            for item in regression_observations
        ],
        dtype=float,
    )

    used_years = years.tolist()

    used_areas = areas.tolist()

    # --------------------------------------------------------
    # Determine future range
    # --------------------------------------------------------

    last_historical_year = int(
        max(years)
    )

    if future_start_year is None:

        future_start_year = (
            last_historical_year + 1
        )

    if future_end_year is None:

        future_end_year = (
            future_start_year + 3
        )

    if future_end_year < future_start_year:

        raise ValueError(
            "future_end_year must be greater "
            "than or equal to future_start_year."
        )

    # ========================================================
    # LINEAR TREND
    # ========================================================

    slope, intercept = np.polyfit(
        years,
        areas,
        1,
    )

    future_years = np.arange(
        future_start_year,
        future_end_year + 1,
        dtype=float,
    )

    future_areas = (
        slope * future_years
        + intercept
    )

    # Glacier area cannot be negative.

    future_areas = np.maximum(
        future_areas,
        0.0,
    )

    # ========================================================
    # BUILD PREDICTIONS
    # ========================================================

    predictions = []

    for year, area in zip(
        future_years,
        future_areas,
    ):

        predictions.append(
            {
                "year":
                    int(year),

                "predicted_area_km2":
                    round(
                        float(area),
                        6,
                    ),
            }
        )

    # --------------------------------------------------------
    # Predicted change
    # --------------------------------------------------------

    if len(future_areas) >= 2:

        predicted_change = (
            float(future_areas[-1])
            - float(future_areas[0])
        )

    else:

        predicted_change = 0.0

    first_predicted_area = float(
        future_areas[0]
    )

    if abs(
        first_predicted_area
    ) > 1e-12:

        predicted_change_percent = (
            predicted_change
            / first_predicted_area
            * 100.0
        )

    else:

        predicted_change_percent = None

    # ========================================================
    # OUTPUT DATA
    # ========================================================

    output_data = {

        "status":
            "success",

        "glacier": {
            "name":
                resolved_name,

            "rgi_id":
                resolved_rgi_id,
        },

        "prediction_method":
            prediction_method,

        # All observations retained.
        "historical_years":
            all_years,

        "historical_area_km2":
            all_areas,

        # Observations selected for regression.
        "used_historical_years":
            [
                int(x)
                for x in used_years
            ],

        "used_historical_area_km2":
            used_areas,

        "trend_slope_km2_per_year":
            round(
                float(slope),
                8,
            ),

        "predictions":
            predictions,

        "predicted_change_km2":
            round(
                predicted_change,
                6,
            ),

        "predicted_change_percent":
            (
                round(
                    predicted_change_percent,
                    4,
                )
                if predicted_change_percent
                is not None
                else None
            ),

        "validation_summary":
            validation_summary,

        "prediction_confidence":
            prediction_confidence,

        "note":
            prediction_note,
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    # IMPORTANT FIX:
    #
    # Previously the prediction was saved as:
    #
    # datasets/historical/
    #     historical_results_future_prediction.json
    #
    # That makes it difficult for the API to locate the
    # prediction for a specific glacier.
    #
    # Now it is saved next to that glacier's historical file:
    #
    # datasets/historical/
    #     rgi2000-v7.0-g-01-09291/
    #         historical_results.json
    #         future_prediction.json

    output_file = (
        historical_file.parent
        / "future_prediction.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2,
        )

    # ========================================================
    # GRAPH
    # ========================================================

    graph_file = (
        historical_file.parent
        / "future_prediction.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    # --------------------------------------------------------
    # Plot all historical observations
    # --------------------------------------------------------

    if all_years:

        plt.plot(
            all_years,
            all_areas,
            marker="o",
            linestyle="--",
            alpha=0.5,
            label="All historical observations",
        )

    # --------------------------------------------------------
    # Plot validation-screened observations
    # --------------------------------------------------------

    if usable_observations:

        plt.plot(
            usable_years,
            usable_areas,
            marker="o",
            linewidth=2,
            label="Validation-screened observations",
        )

    # --------------------------------------------------------
    # Plot regression observations if fallback is being used
    # --------------------------------------------------------

    if (
        len(usable_observations) < 2
        and len(all_observations) >= 2
    ):

        plt.plot(
            years,
            areas,
            marker="s",
            linewidth=2,
            label="Observations used for lower-confidence trend",
        )

    # --------------------------------------------------------
    # Trend line
    # --------------------------------------------------------

    trend_years = np.linspace(
        float(min(years)),
        float(max(years)),
        100,
    )

    trend_areas = (
        slope * trend_years
        + intercept
    )

    trend_areas = np.maximum(
        trend_areas,
        0.0,
    )

    plt.plot(
        trend_years,
        trend_areas,
        linestyle=":",
        linewidth=2,
        label="Fitted linear trend",
    )

    # --------------------------------------------------------
    # Future prediction
    # --------------------------------------------------------

    plt.plot(
        future_years,
        future_areas,
        marker="x",
        linestyle="--",
        linewidth=2,
        label="Future estimate",
    )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "Glacier Area (km┬▓)"
    )

    plt.title(
        f"Future Glacier Area Prediction - {resolved_name}"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        graph_file,
        dpi=150,
    )

    plt.close()

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "data":
            output_data,

        "json_path":
            str(output_file),

        "graph_path":
            str(graph_file),
    }

# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate validation-aware future "
            "glacier area predictions."
        )
    )

    parser.add_argument(
        "--glacier",
        type=str,
        default=None,
        help="Glacier name",
    )

    parser.add_argument(
        "--rgi-id",
        type=str,
        default=None,
        help="RGI glacier ID",
    )

    parser.add_argument(
        "--future-start",
        type=int,
        default=None,
        help="First future prediction year",
    )

    parser.add_argument(
        "--future-end",
        type=int,
        default=None,
        help="Last future prediction year",
    )

    args = parser.parse_args()

    result = generate_future_prediction(
        glacier_name=args.glacier,
        rgi_id=args.rgi_id,
        future_start_year=args.future_start,
        future_end_year=args.future_end,
    )

    data = result["data"]

    print()

    print("=" * 70)

    print(
        "GLACIERAI FUTURE PREDICTION"
    )

    print("=" * 70)

    print(
        f"Glacier: "
        f"{data['glacier']['name']}"
    )

    print(
        f"RGI ID: "
        f"{data['glacier'].get('rgi_id')}"
    )

    print(
        f"Status: "
        f"{data['status']}"
    )

    print()

    print(
        "Historical file:"
    )

    print(
        result.get(
            "json_path"
        )
    )

    print()

    print(
        "Validation Summary"
    )

    print("-" * 70)

    validation = data[
        "validation_summary"
    ]

    print(
        "Total observations:",
        validation[
            "total_observations"
        ],
    )

    print(
        "Usable observations:",
        validation[
            "usable_observations"
        ],
    )

    print(
        "HIGH quality:",
        validation[
            "high_quality_observations"
        ],
    )

    print(
        "MEDIUM quality:",
        validation[
            "medium_quality_observations"
        ],
    )

    print(
        "Excluded LOW:",
        validation[
            "excluded_low_quality_observations"
        ],
    )

    print(
        "Excluded UNKNOWN:",
        validation[
            "excluded_unknown_quality_observations"
        ],
    )

    print()

    if data["status"] != "success":

        print(
            "Prediction not generated:"
        )

        print(
            data["note"]
        )

        print()

        print(
            "Output:",
            result["json_path"],
        )

        return

    # --------------------------------------------------------
    # Regression information
    # --------------------------------------------------------

    print(
        "Prediction confidence:",
        data[
            "prediction_confidence"
        ],
    )

    print(
        "Historical years used:",
        data[
            "used_historical_years"
        ],
    )

    print(
        "Historical areas used:",
        data[
            "used_historical_area_km2"
        ],
    )

    print(
        "Trend slope:",
        data[
            "trend_slope_km2_per_year"
        ],
        "km┬▓/year",
    )

    print()

    print(
        "Future Estimates"
    )

    print("-" * 70)

    for prediction in data[
        "predictions"
    ]:

        print(
            prediction["year"],
            "->",
            prediction[
                "predicted_area_km2"
            ],
            "km┬▓",
        )

    print()

    print(
        "Predicted change:",
        data[
            "predicted_change_km2"
        ],
        "km┬▓",
    )

    print(
        "Predicted change percentage:",
        data[
            "predicted_change_percent"
        ],
        "%",
    )

    print()

    print(
        "JSON:",
        result["json_path"],
    )

    print(
        "Graph:",
        result["graph_path"],
    )

    print()

    print(
        data["note"]
    )

    print("=" * 70)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
