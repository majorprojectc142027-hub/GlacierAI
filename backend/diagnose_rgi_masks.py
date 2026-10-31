
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from PIL import Image


# ============================================================
# GLACIERAI - RGI REFERENCE MASK DIAGNOSTIC
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GLACIER_ID = "RGI2000-v7.0-G-15-00467"
GLACIER_KEY = "rgi2000_v7.0_g_15_00467"

SATELLITE_DIR = PROJECT_ROOT / "datasets" / "live"
PREDICTION_DIR = SATELLITE_DIR / "predictions"

# All diagnostic outputs are kept separate from existing files.
DIAGNOSTIC_DIR = (
    SATELLITE_DIR / "large_glacier" / "rgi_diagnostics"
)
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

TILE_IDS = ("001", "002", "003")


# ============================================================
# FIND ORIGINAL RGI INVENTORY
# ============================================================

def find_inventory():
    print("\nSearching for global_rgi.gpkg...")

    candidates = list(PROJECT_ROOT.rglob("global_rgi.gpkg"))

    if not candidates:
        raise FileNotFoundError(
            "Could not find global_rgi.gpkg inside:\n"
            f"{PROJECT_ROOT}\n\n"
            "Place the inventory inside the project or update "
            "the inventory search path in this script."
        )

    # Prefer the inventory under datasets, if multiple copies exist.
    candidates.sort(
        key=lambda p: (
            "datasets" not in str(p).lower(),
            len(str(p))
        )
    )

    inventory_path = candidates[0]
    print("Inventory:", inventory_path)

    if len(candidates) > 1:
        print("Other inventory copies found:")
        for path in candidates[1:]:
            print("  ", path)

    return inventory_path


def find_rgi_geometry(inventory_path):
    print("\nLoading original RGI inventory...")

    inventory = gpd.read_file(inventory_path)

    if inventory.empty:
        raise ValueError("The RGI inventory contains no records.")

    if inventory.crs is None:
        raise ValueError(
            "The inventory has no CRS. Cannot safely align geometry."
        )

    print("Inventory CRS:", inventory.crs)
    print("Inventory columns:", list(inventory.columns))

    # Match the ID column without assuming its exact capitalization.
    id_candidates = [
        column for column in inventory.columns
        if column.lower() in {
            "rgi_id",
            "rgiid",
            "rgi_id_v7",
            "rgi_id_v70",
        }
    ]

    if not id_candidates:
        id_candidates = [
            column for column in inventory.columns
            if "rgi" in column.lower()
            and "id" in column.lower()
        ]

    if not id_candidates:
        raise ValueError(
            "Could not identify the RGI ID column.\n"
            "Check the printed inventory columns and update "
            "the ID column selection in find_rgi_geometry()."
        )

    id_column = id_candidates[0]

    matches = inventory[
        inventory[id_column].astype(str).str.strip() == GLACIER_ID
    ]

    if matches.empty:
        raise ValueError(
            f"Glacier ID {GLACIER_ID} was not found in "
            f"column {id_column!r}."
        )

    if len(matches) > 1:
        print(
            "Warning: multiple matching RGI records found; "
            "using the first."
        )

    glacier = matches.iloc[0]
    geometry = glacier.geometry

    if geometry is None or geometry.is_empty:
        raise ValueError("The selected glacier geometry is empty.")

    if not geometry.is_valid:
        print(
            "Warning: original geometry is invalid. "
            "Attempting a geometry repair."
        )
        geometry = geometry.buffer(0)

    if geometry.is_empty:
        raise ValueError("Geometry is empty after repair.")

    print("Selected glacier:", GLACIER_ID)
    print("Geometry type:", geometry.geom_type)
    print("Inventory area field values:")

    for column in inventory.columns:
        if "area" in column.lower():
            print(f"  {column}: {glacier[column]}")

    # Keep the source CRS information with the geometry.
    geometry_series = gpd.GeoSeries(
        [geometry],
        crs=inventory.crs
    )

    return geometry_series


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(reference, prediction, valid):
    reference = reference.astype(bool) & valid
    prediction = prediction.astype(bool) & valid

    intersection = int(np.count_nonzero(reference & prediction))
    ref_pixels = int(np.count_nonzero(reference))
    pred_pixels = int(np.count_nonzero(prediction))
    union = int(np.count_nonzero(reference | prediction))

    iou = intersection / union if union else 1.0

    denominator = ref_pixels + pred_pixels
    dice = (
        2.0 * intersection / denominator
        if denominator
        else 1.0
    )

    return {
        "reference_pixels": ref_pixels,
        "prediction_pixels": pred_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou_percent": iou * 100.0,
        "dice_percent": dice * 100.0,
    }


def load_existing_reference(path, expected_shape):
    if not path.exists():
        return None

    image = np.asarray(Image.open(path).convert("L"))

    if image.shape != expected_shape:
        print(
            "WARNING: existing RGI PNG has a different shape:",
            image.shape,
            "expected:",
            expected_shape
        )
        return None

    return image > 0


# ============================================================
# PROCESS ONE TILE
# ============================================================

def process_tile(tile_id, glacier_geometry):
    tile_name = f"{GLACIER_KEY}_tile_{tile_id}"

    satellite_path = (
        SATELLITE_DIR / f"{tile_name}.tif"
    )

    prediction_path = (
        PREDICTION_DIR / f"{tile_name}_glacier_mask.tif"
    )

    old_reference_path = (
        PREDICTION_DIR / f"{tile_name}_rgi_mask.png"
    )

    if not satellite_path.exists():
        raise FileNotFoundError(
            f"Satellite TIFF not found:\n{satellite_path}"
        )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction TIFF not found:\n{prediction_path}"
        )

    print("\n" + "=" * 65)
    print("TILE", tile_id)
    print("=" * 65)

    # --------------------------------------------------------
    # Read the actual satellite grid.
    # --------------------------------------------------------

    with rasterio.open(satellite_path) as src:
        shape = (src.height, src.width)
        transform = src.transform
        raster_crs = src.crs
        raster_bounds = src.bounds
        profile = src.profile.copy()

    if raster_crs is None:
        raise ValueError(
            f"Satellite TIFF has no CRS: {satellite_path}"
        )

    print("Satellite shape:", shape)
    print("Satellite CRS:", raster_crs)
    print("Satellite bounds:", raster_bounds)

    # --------------------------------------------------------
    # Read prediction and check its grid.
    # --------------------------------------------------------

    with rasterio.open(prediction_path) as src:
        prediction_values = src.read(1)
        prediction_transform = src.transform
        prediction_crs = src.crs
        prediction_nodata = src.nodata

    if prediction_values.shape != shape:
        raise ValueError(
            "Prediction shape does not match satellite image."
        )

    if prediction_crs != raster_crs:
        raise ValueError(
            "Prediction CRS does not match satellite CRS."
        )

    if prediction_transform != transform:
        raise ValueError(
            "Prediction transform does not match satellite transform."
        )

    # The saved prediction uses 1 for glacier, 0 for background,
    # and may use 255 as NoData.
    valid = np.isfinite(prediction_values)

    if prediction_nodata is not None:
        valid &= prediction_values != prediction_nodata

    valid &= prediction_values != 255

    prediction = prediction_values == 1

    print("Prediction grid matches satellite grid: YES")
    print("Valid prediction pixels:", int(valid.sum()))

    # --------------------------------------------------------
    # Reproject the original inventory geometry to the
    # satellite TIFF CRS and rasterize it on the exact grid.
    # --------------------------------------------------------

    glacier_in_raster_crs = glacier_geometry.to_crs(
        raster_crs
    ).iloc[0]

    # Clip the original polygon to the raster footprint for
    # clarity. Rasterization itself is also limited to this grid.
    tile_footprint = rasterio.coords.BoundingBox(
        left=raster_bounds.left,
        bottom=raster_bounds.bottom,
        right=raster_bounds.right,
        top=raster_bounds.top,
    )

    from shapely.geometry import box

    footprint_geometry = box(
        tile_footprint.left,
        tile_footprint.bottom,
        tile_footprint.right,
        tile_footprint.top,
    )

    clipped_geometry = glacier_in_raster_crs.intersection(
        footprint_geometry
    )

    if clipped_geometry.is_empty:
        print(
            "WARNING: original glacier polygon does not intersect "
            "this tile."
        )
        fresh_reference = np.zeros(shape, dtype=np.uint8)
    else:
        fresh_reference = rasterize(
            [(clipped_geometry, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )

    # --------------------------------------------------------
    # Save the new reference mask to a separate directory.
    # Existing reference and prediction files remain untouched.
    # --------------------------------------------------------

    fresh_png_path = (
        DIAGNOSTIC_DIR / f"{tile_name}_rgi_reference_fresh.png"
    )

    Image.fromarray(
        fresh_reference * 255
    ).save(fresh_png_path)

    fresh_tif_path = (
        DIAGNOSTIC_DIR / f"{tile_name}_rgi_reference_fresh.tif"
    )

    reference_profile = profile.copy()
    reference_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw",
    )

    with rasterio.open(
        fresh_tif_path,
        "w",
        **reference_profile,
    ) as dst:
        dst.write(fresh_reference, 1)

    # --------------------------------------------------------
    # Compare the fresh reference with the prediction.
    # --------------------------------------------------------

    fresh_metrics = calculate_metrics(
        fresh_reference == 1,
        prediction,
        valid,
    )

    print("\nFRESH ORIGINAL-INVENTORY COMPARISON")
    print("Reference pixels:", fresh_metrics["reference_pixels"])
    print("Prediction pixels:", fresh_metrics["prediction_pixels"])
    print("Overlap pixels:", fresh_metrics["intersection_pixels"])
    print("Union pixels:", fresh_metrics["union_pixels"])
    print(f"IoU: {fresh_metrics['iou_percent']:.2f}%")
    print(f"Dice: {fresh_metrics['dice_percent']:.2f}%")

    # --------------------------------------------------------
    # Compare old saved RGI PNG with the newly generated mask.
    # --------------------------------------------------------

    old_reference = load_existing_reference(
        old_reference_path,
        shape,
    )

    old_metrics = None

    if old_reference is not None:
        old_metrics = calculate_metrics(
            old_reference,
            prediction,
            valid,
        )

        ref_difference = int(
            np.count_nonzero(
                (old_reference != (fresh_reference == 1)) & valid
            )
        )

        print("\nEXISTING RGI PNG COMPARISON")
        print(
            "Existing reference pixels:",
            int(np.count_nonzero(old_reference & valid))
        )
        print(
            f"Existing-reference IoU: "
            f"{old_metrics['iou_percent']:.2f}%"
        )
        print(
            "Pixels where old and fresh reference masks differ:",
            ref_difference,
        )
    else:
        print(
            "\nNo compatible existing RGI PNG was found; "
            "skipping old-reference comparison."
        )

    print("\nFresh reference saved:")
    print(fresh_png_path)
    print(fresh_tif_path)

    return {
        "tile": tile_id,
        "shape": shape,
        "fresh_reference_pixels": fresh_metrics["reference_pixels"],
        "prediction_pixels": fresh_metrics["prediction_pixels"],
        "intersection_pixels": fresh_metrics["intersection_pixels"],
        "iou_percent": fresh_metrics["iou_percent"],
        "dice_percent": fresh_metrics["dice_percent"],
        "old_reference_iou_percent": (
            old_metrics["iou_percent"]
            if old_metrics is not None
            else None
        ),
        "fresh_reference_png": str(fresh_png_path),
        "fresh_reference_tif": str(fresh_tif_path),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("GLACIERAI - RGI REFERENCE MASK DIAGNOSTIC")
    print("=" * 65)
    print("Project root:", PROJECT_ROOT)
    print("Diagnostic output:", DIAGNOSTIC_DIR)

    inventory_path = find_inventory()
    glacier_geometry = find_rgi_geometry(inventory_path)

    results = []
    failures = []

    for tile_id in TILE_IDS:
        try:
            result = process_tile(
                tile_id,
                glacier_geometry,
            )
            results.append(result)
        except Exception as exc:
            print(f"\nERROR processing tile {tile_id}: {exc}")
            failures.append({
                "tile": tile_id,
                "error": str(exc),
            })

    # --------------------------------------------------------
    # Print concise summary.
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 65)

    for result in results:
        print(
            f"Tile {result['tile']}: "
            f"IoU={result['iou_percent']:.2f}%, "
            f"Dice={result['dice_percent']:.2f}%, "
            f"reference pixels={result['fresh_reference_pixels']}"
        )

        if result["old_reference_iou_percent"] is not None:
            print(
                "  Existing reference IoU:",
                f"{result['old_reference_iou_percent']:.2f}%"
            )

    if failures:
        print("\nFailed tiles:")
        for failure in failures:
            print(
                f"Tile {failure['tile']}: {failure['error']}"
            )

    summary_path = DIAGNOSTIC_DIR / "rgi_diagnostic_summary.txt"

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("GlacierAI RGI diagnostic summary\n")
        file.write(f"Inventory: {inventory_path}\n")
        file.write(f"Glacier ID: {GLACIER_ID}\n\n")

        for result in results:
            file.write(
                f"Tile {result['tile']}\n"
                f"  Shape: {result['shape']}\n"
                f"  Fresh reference pixels: "
                f"{result['fresh_reference_pixels']}\n"
                f"  Prediction pixels: {result['prediction_pixels']}\n"
                f"  Overlap pixels: {result['intersection_pixels']}\n"
                f"  Fresh-reference IoU: "
                f"{result['iou_percent']:.2f}%\n"
                f"  Fresh-reference Dice: "
                f"{result['dice_percent']:.2f}%\n"
                f"  Existing-reference IoU: "
                f"{result['old_reference_iou_percent']}\n\n"
            )

        if failures:
            file.write("Failures:\n")
            for failure in failures:
                file.write(
                    f"Tile {failure['tile']}: {failure['error']}\n"
                )

    print("\nSummary saved:")
    print(summary_path)
    print("\nDiagnostic finished. Existing prediction masks were not modified.")


if __name__ == "__main__":
    main()
