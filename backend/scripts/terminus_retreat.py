from pathlib import Path
import json
import math

import numpy as np
import rasterio
from scipy.ndimage import binary_erosion, label


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

FLOW_AZIMUTH_DEG = 315.0

# Ignore tiny disconnected segmentation fragments.
MIN_COMPONENT_PIXELS = 500

# Use the downstream-most 2.5% of the selected
# component's boundary.
TERMINUS_PERCENTILE = 97.5


# ==========================================================
# FLOW VECTOR
# ==========================================================

def flow_vector(azimuth_deg: float):

    theta = math.radians(
        azimuth_deg
    )

    east = math.sin(theta)
    north = math.cos(theta)

    return np.array(
        [east, north],
        dtype=float,
    )


# ==========================================================
# FIND HISTORICAL MASKS
# ==========================================================

def find_historical_masks(rgi_id: str):

    glacier_dir = (
        HISTORICAL_ROOT
        / rgi_id.lower()
    )

    predictions_dir = (
        glacier_dir
        / "predictions"
    )

    if not predictions_dir.exists():

        raise FileNotFoundError(
            f"Prediction directory not found: "
            f"{predictions_dir}"
        )

    masks = {}

    for path in predictions_dir.glob(
        "*_rgi_mask.tif"
    ):

        parts = path.stem.split("_")

        try:
            year = int(parts[-3])
        except (
            ValueError,
            IndexError,
        ):
            continue

        masks[year] = path

    return dict(
        sorted(masks.items())
    )


# ==========================================================
# EXTRACT TERMINUS
# ==========================================================

def extract_terminus_position(
    mask_path: Path,
    flow_azimuth_deg: float = FLOW_AZIMUTH_DEG,
):

    with rasterio.open(mask_path) as src:

        mask = src.read(1) > 0

        transform = src.transform
        crs = src.crs

        if not np.any(mask):

            raise ValueError(
                f"Mask contains no glacier pixels: "
                f"{mask_path}"
            )

        # --------------------------------------------------
        # Connected components
        # --------------------------------------------------

        labels, component_count = label(
            mask,
            structure=np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
        )

        component_sizes = np.bincount(
            labels.ravel()
        )

        # --------------------------------------------------
        # Flow direction
        # --------------------------------------------------

        direction = flow_vector(
            flow_azimuth_deg
        )

        candidates = []

        for component_id in range(
            1,
            component_count + 1,
        ):

            size = component_sizes[
                component_id
            ]

            if size < MIN_COMPONENT_PIXELS:
                continue

            rows, cols = np.where(
                labels == component_id
            )

            xs, ys = rasterio.transform.xy(
                transform,
                rows,
                cols,
                offset="center",
            )

            coordinates = np.column_stack(
                [
                    np.asarray(xs),
                    np.asarray(ys),
                ]
            )

            flow_positions = (
                coordinates @ direction
            )

            candidates.append(
                {
                    "component_id":
                        int(component_id),
                    "pixels":
                        int(size),
                    "flow_min":
                        float(
                            flow_positions.min()
                        ),
                    "flow_max":
                        float(
                            flow_positions.max()
                        ),
                    "coordinates":
                        coordinates,
                }
            )

        if not candidates:

            raise ValueError(
                "No glacier component passed "
                f"MIN_COMPONENT_PIXELS="
                f"{MIN_COMPONENT_PIXELS}"
            )

        # --------------------------------------------------
        # Select the main glacier trunk.
        #
        # The downstream-most component can be a detached
        # segmentation fragment, snow patch, or tributary.
        # For this analysis we track the largest eligible
        # connected component, which represents the main
        # glacier body in the historical segmentation.
        # --------------------------------------------------

        selected = max(
            candidates,
            key=lambda item:
                item["pixels"],
        )

        component_id = (
            selected["component_id"]
        )

        component_mask = (
            labels == component_id
        )

        # --------------------------------------------------
        # Boundary of selected component
        # --------------------------------------------------

        eroded = binary_erosion(
            component_mask,
            structure=np.ones(
                (3, 3),
                dtype=bool,
        ))

        boundary = (
            component_mask
            & ~eroded
        )

        boundary_rows, boundary_cols = (
            np.where(boundary)
        )

        if len(boundary_rows) == 0:

            raise ValueError(
                "No boundary found for selected "
                f"component {component_id}"
            )

        xs, ys = rasterio.transform.xy(
            transform,
            boundary_rows,
            boundary_cols,
            offset="center",
        )

        boundary_coordinates = np.column_stack(
            [
                np.asarray(xs),
                np.asarray(ys),
            ]
        )

        boundary_flow = (
            boundary_coordinates
            @ direction
        )

        # --------------------------------------------------
        # Downstream boundary band
        # --------------------------------------------------

        threshold = np.percentile(
            boundary_flow,
            TERMINUS_PERCENTILE,
        )

        terminus_coordinates = (
            boundary_coordinates[
                boundary_flow >= threshold
            ]
        )

        terminus_flow = (
            boundary_flow[
                boundary_flow >= threshold
            ]
        )

        # Median is more stable than selecting
        # one extreme pixel.
        center = np.median(
            terminus_coordinates,
            axis=0,
        )

        flow_position = float(
            np.median(
                terminus_flow
            )
        )

        return {
            "mask_path": str(
                mask_path
            ),
            "crs": str(crs),

            "flow_azimuth_deg":
                flow_azimuth_deg,

            "selected_component":
                int(component_id),

            "selected_component_pixels":
                int(selected["pixels"]),

            "total_components":
                int(component_count),

            "eligible_components":
                int(len(candidates)),

            "boundary_pixels":
                int(len(boundary_rows)),

            "terminus_band_pixels":
                int(len(
                    terminus_coordinates
                )),

            "terminus_easting_m":
                float(center[0]),

            "terminus_northing_m":
                float(center[1]),

            "flow_coordinate_m":
                flow_position,
        }


# ==========================================================
# MOVEMENT CLASSIFICATION
# ==========================================================

def classify_movement(flow_change_m: float):
    """
    Positive flow-coordinate change means movement in the
    assumed down-glacier direction.

    Negative flow-coordinate change means movement opposite
    the assumed down-glacier direction.

    These are AI-derived apparent movements and are not
    independently validated physical measurements.
    """

    if flow_change_m > 0:
        return "APPARENT ADVANCE"

    if flow_change_m < 0:
        return "APPARENT RETREAT"

    return "NO APPARENT MOVEMENT"


# ==========================================================
# DISTANCE
# ==========================================================

def calculate_distance(
    previous,
    current,
):

    dx = (
        current["terminus_easting_m"]
        - previous["terminus_easting_m"]
    )

    dy = (
        current["terminus_northing_m"]
        - previous["terminus_northing_m"]
    )

    distance = math.sqrt(
        dx * dx
        + dy * dy
    )

    return {
        "distance_m":
            float(distance),

        "east_displacement_m":
            float(dx),

        "north_displacement_m":
            float(dy),
    }


# ==========================================================
# ANALYSIS
# ==========================================================

def run_terminus_analysis(
    glacier_name: str,
    rgi_id: str,
    flow_azimuth_deg:
        float = FLOW_AZIMUTH_DEG,
):

    masks = find_historical_masks(
        rgi_id
    )

    if len(masks) < 2:

        raise RuntimeError(
            "At least two historical masks "
            "are required."
        )

    observations = []

    for year, path in masks.items():

        print()
        print(
            f"Processing {year}: "
            f"{path}"
        )

        result = (
            extract_terminus_position(
                path,
                flow_azimuth_deg,
            )
        )

        result["year"] = year

        observations.append(
            result
        )

        print(
            f"  Components       : "
            f"{result['total_components']}"
        )

        print(
            f"  Eligible          : "
            f"{result['eligible_components']}"
        )

        print(
            f"  Selected component: "
            f"{result['selected_component']}"
        )

        print(
            f"  Component pixels  : "
            f"{result['selected_component_pixels']}"
        )

        print(
            f"  Terminus E        : "
            f"{result['terminus_easting_m']:.2f}"
        )

        print(
            f"  Terminus N        : "
            f"{result['terminus_northing_m']:.2f}"
        )

        print(
            f"  Flow coordinate    : "
            f"{result['flow_coordinate_m']:.2f}"
        )

    # ------------------------------------------------------
    # Year-to-year changes
    # ------------------------------------------------------

    changes = []

    for previous, current in zip(
        observations[:-1],
        observations[1:],
    ):

        displacement = (
            calculate_distance(
                previous,
                current,
            )
        )

        year_delta = (
            current["year"]
            - previous["year"]
        )

        flow_change = (
            current["flow_coordinate_m"]
            - previous["flow_coordinate_m"]
        )

        movement_type = classify_movement(
            flow_change
        )

        flow_rate = (
            flow_change
            / year_delta
        )

        changes.append(
            {
                "from_year":
                    previous["year"],

                "to_year":
                    current["year"],

                "flow_coordinate_change_m":
                    float(flow_change),

                "flow_displacement_rate_m_per_year":
                    float(flow_rate),

                "movement_type":
                    movement_type,

                "terminus_displacement_m":
                    displacement[
                        "distance_m"
                    ],

                "east_displacement_m":
                    displacement[
                        "east_displacement_m"
                    ],

                "north_displacement_m":
                    displacement[
                        "north_displacement_m"
                    ],

                # Kept for backward compatibility with
                # any existing frontend/report consumers.
                "retreat_distance_m":
                    float(-flow_change),

                "retreat_rate_m_per_year":
                    float(-flow_rate),
            }
        )

    # ------------------------------------------------------
    # Overall
    # ------------------------------------------------------

    first = observations[0]
    last = observations[-1]

    total_flow_change = (
        last["flow_coordinate_m"]
        - first["flow_coordinate_m"]
    )

    total_years = (
        last["year"]
        - first["year"]
    )

    net_flow_displacement = (
        total_flow_change
    )

    annual_flow_rate = (
        net_flow_displacement
        / total_years
    )

    overall_movement_type = classify_movement(
        net_flow_displacement
    )

    result = {
        "status":
            "success",

        "validation_status":
            "UNVALIDATED",

        "validation_message":
            "Terminus positions are derived from "
            "AI-generated historical glacier masks "
            "and have not been independently validated "
            "against observed terminus positions.",

        "glacier":
            glacier_name,

        "rgi_id":
            rgi_id,

        "analysis_type":
            "Physical Terminus Position "
            "and Retreat Analysis",

        "method":
            "Largest eligible connected-component "
            "selection to represent the main glacier "
            "trunk, followed by robust downstream "
            "boundary extraction along an approximate "
            "down-glacier flow direction",

        "flow_direction": {
            "azimuth_deg":
                flow_azimuth_deg,

            "description":
                "Approximate down-glacier "
                "direction",
        },

        "configuration": {
            "min_component_pixels":
                MIN_COMPONENT_PIXELS,

            "terminus_percentile":
                TERMINUS_PERCENTILE,
        },

        "terminus_observations":
            observations,

        "year_to_year_changes":
            changes,

        "overall": {
            "start_year":
                first["year"],

            "end_year":
                last["year"],

            "net_flow_displacement_m":
                float(
                    net_flow_displacement
                ),

            "annual_flow_displacement_rate_m_per_year":
                float(
                    annual_flow_rate
                ),

            "movement_type":
                overall_movement_type,

            # Backward-compatible fields.
            "total_retreat_distance_m":
                float(
                    -net_flow_displacement
                ),

            "annualized_retreat_rate_m_per_year":
                float(
                    -annual_flow_rate
                ),
        },

        "scientific_note":
            "Terminus position is derived "
            "from AI-generated historical "
            "glacier masks. The largest eligible "
            "connected component is used to "
            "represent the main glacier trunk, "
            "followed by downstream boundary "
            "extraction. Positive flow-coordinate "
            "change is reported as apparent advance; "
            "negative flow-coordinate change is "
            "reported as apparent retreat. These "
            "movements are UNVALIDATED and must not "
            "be presented as independently validated "
            "physical retreat or advance measurements.",
    }

    output_dir = (
        HISTORICAL_ROOT
        / rgi_id.lower()
    )

    output_file = (
        output_dir
        / "terminus_retreat.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )

    print()
    print("=" * 60)
    print(
        " GLACIERAI - TERMINUS RETREAT ANALYSIS"
    )
    print("=" * 60)

    print(
        "Validation status: UNVALIDATED"
    )

    for change in changes:

        print(
            f"{change['from_year']} -> "
            f"{change['to_year']} | "
            f"Flow displacement: "
            f"{change['flow_coordinate_change_m']:.2f} m | "
            f"Movement: "
            f"{change['movement_type']} | "
            f"Rate: "
            f"{change['flow_displacement_rate_m_per_year']:.2f} "
            f"m/year"
        )

    print()
    print(
        f"Net flow displacement: "
        f"{net_flow_displacement:.2f} m"
    )

    print(
        f"Overall movement: "
        f"{overall_movement_type}"
    )

    print(
        f"Annual flow displacement rate: "
        f"{annual_flow_rate:.2f} m/year"
    )

    print()
    print(
        f"Saved: {output_file}"
    )

    return result


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--glacier",
        default="Gangotri Glacier",
    )

    parser.add_argument(
        "--rgi-id",
        default="RGI2000-v7.0-G-15-00467",
    )

    parser.add_argument(
        "--flow-azimuth",
        type=float,
        default=315.0,
    )

    args = parser.parse_args()

    run_terminus_analysis(
        glacier_name=args.glacier,
        rgi_id=args.rgi_id,
        flow_azimuth_deg=args.flow_azimuth,
    )