
from pathlib import Path
import csv
import random

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

LIVE_DIR = ROOT / "datasets" / "live"
PRED_DIR = LIVE_DIR / "predictions"
OUTPUT_DIR = PRED_DIR / "debris_samples"

GLACIER_ID = "rgi2000_v7_0_g_15_00467"

RGB_PATH = LIVE_DIR / f"{GLACIER_ID}_debris_rgb.tif"
GLACIER_PATH = PRED_DIR / f"{GLACIER_ID}_debris_glacier_mask.tif"
DEBRIS_PATH = PRED_DIR / f"{GLACIER_ID}_debris_affected_mask.tif"

PATCH_SIZE = 128
DEBRIS_SAMPLES = 12
RANDOM_GLACIER_SAMPLES = 12
RANDOM_SEED = 42


# --------------------------------------------------
# VALIDATE INPUT FILES
# --------------------------------------------------

def check_files():
    for path in [RGB_PATH, GLACIER_PATH, DEBRIS_PATH]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Gangotri file not found:\n{path}"
            )

    print("Using Gangotri files:")
    print("RGB:", RGB_PATH)
    print("Glacier mask:", GLACIER_PATH)
    print("Debris mask:", DEBRIS_PATH)


# --------------------------------------------------
# LOAD RASTERS AND VERIFY ALIGNMENT
# --------------------------------------------------

def load_rasters():
    with rasterio.open(RGB_PATH) as src:
        if src.count < 3:
            raise ValueError("RGB raster must have at least 3 bands.")

        rgb_raw = src.read([1, 2, 3]).astype(np.float32)
        rgb_profile = src.profile.copy()

    with rasterio.open(GLACIER_PATH) as src:
        glacier_mask = src.read(1) > 0
        glacier_profile = src.profile.copy()

    with rasterio.open(DEBRIS_PATH) as src:
        debris_mask = src.read(1) > 0
        debris_profile = src.profile.copy()

    profiles = [
        ("RGB", rgb_profile),
        ("glacier mask", glacier_profile),
        ("debris mask", debris_profile),
    ]

    for name, profile in profiles[1:]:
        if (
            profile["height"] != rgb_profile["height"]
            or profile["width"] != rgb_profile["width"]
            or profile["transform"] != rgb_profile["transform"]
            or profile["crs"] != rgb_profile["crs"]
        ):
            raise ValueError(
                f"{name} does not align with the RGB raster. "
                "No overlays were generated."
            )

    return rgb_raw, rgb_profile, glacier_mask, debris_mask


# --------------------------------------------------
# PREPARE RGB DISPLAY
# --------------------------------------------------

def prepare_rgb(rgb_raw):
    rgb = np.zeros_like(rgb_raw, dtype=np.float32)

    for band in range(3):
        values = rgb_raw[band]
        valid = np.isfinite(values)

        if not valid.any():
            continue

        low, high = np.percentile(values[valid], [2, 98])

        if high <= low:
            high = low + 1.0

        rgb[band] = np.clip(
            (values - low) / (high - low),
            0,
            1,
        )

    return np.moveaxis(rgb, 0, -1)


# --------------------------------------------------
# SAMPLE CENTRES
# --------------------------------------------------

def select_samples(glacier_mask, debris_mask):
    rng = random.Random(RANDOM_SEED)

    height, width = glacier_mask.shape
    half = PATCH_SIZE // 2

    interior = np.zeros((height, width), dtype=bool)

    if height <= PATCH_SIZE or width <= PATCH_SIZE:
        raise ValueError("Raster is too small for the selected patch size.")

    interior[half:height-half, half:width-half] = True

    glacier_candidates = np.argwhere(glacier_mask & interior)
    debris_candidates = np.argwhere(
        debris_mask & glacier_mask & interior
    )

    if len(glacier_candidates) == 0:
        raise RuntimeError("No glacier pixels available for sampling.")

    rng.shuffle(debris_candidates)
    rng.shuffle(glacier_candidates)

    selected_debris = []
    selected_random = []

    # Keep sample centres separated so patches are less redundant.
    min_spacing_sq = PATCH_SIZE ** 2

    for row, col in debris_candidates:
        point = (int(row), int(col))

        if all(
            (point[0] - r) ** 2 + (point[1] - c) ** 2
            >= min_spacing_sq
            for r, c in selected_debris
        ):
            selected_debris.append(point)

        if len(selected_debris) >= DEBRIS_SAMPLES:
            break

    # Select additional random glacier centres away from debris centres.
    for row, col in glacier_candidates:
        point = (int(row), int(col))

        if any(
            (point[0] - r) ** 2 + (point[1] - c) ** 2
            < min_spacing_sq
            for r, c in selected_debris + selected_random
        ):
            continue

        selected_random.append(point)

        if len(selected_random) >= RANDOM_GLACIER_SAMPLES:
            break

    samples = []

    for row, col in selected_debris:
        samples.append((row, col, "debris_prediction_centre"))

    for row, col in selected_random:
        samples.append((row, col, "random_glacier_centre"))

    if not samples:
        raise RuntimeError("No sample patches could be selected.")

    print("\nDebris-centred samples:", len(selected_debris))
    print("Random glacier samples:", len(selected_random))
    print("Total patches:", len(samples))

    return samples


# --------------------------------------------------
# GENERATE PATCHES AND CONTACT SHEET
# --------------------------------------------------

def create_outputs(
    rgb,
    profile,
    glacier_mask,
    debris_mask,
    samples,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    half = PATCH_SIZE // 2
    columns = 4
    rows = (len(samples) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 4.8 * rows),
        squeeze=False,
    )

    metadata = []

    cyan = np.array([0.15, 0.85, 0.95])
    red = np.array([1.0, 0.10, 0.10])

    for index, (row, col, sample_type) in enumerate(samples, start=1):
        r0, r1 = row - half, row + half
        c0, c1 = col - half, col + half

        rgb_patch = rgb[r0:r1, c0:c1]
        glacier_patch = glacier_mask[r0:r1, c0:c1]
        debris_patch = debris_mask[r0:r1, c0:c1]

        overlay = rgb_patch.copy()

        overlay[glacier_patch] = (
            0.55 * overlay[glacier_patch]
            + 0.45 * cyan
        )

        overlay[debris_patch] = (
            0.20 * overlay[debris_patch]
            + 0.80 * red
        )

        ax = axes[(index - 1) // columns][
            (index - 1) % columns
        ]

        ax.imshow(overlay)
        ax.set_title(
            f"Sample {index:02d} | {sample_type}\n"
            f"Debris pixels: {int(debris_patch.sum())}",
            fontsize=9,
        )
        ax.axis("off")

        patch_path = OUTPUT_DIR / f"sample_{index:02d}.png"
        plt.imsave(patch_path, overlay)

        x, y = rasterio.transform.xy(
            profile["transform"],
            row,
            col,
            offset="center",
        )

        longitude = ""
        latitude = ""

        if profile["crs"] is not None:
            from rasterio.warp import transform

            lon, lat = transform(
                profile["crs"],
                "EPSG:4326",
                [x],
                [y],
            )
            longitude = lon[0]
            latitude = lat[0]

        metadata.append({
            "sample": index,
            "type": sample_type,
            "row": row,
            "col": col,
            "x": x,
            "y": y,
            "longitude": longitude,
            "latitude": latitude,
            "glacier_pixels": int(glacier_patch.sum()),
            "debris_prediction_pixels": int(debris_patch.sum()),
            "image_file": patch_path.name,
            "manual_label": "",
            "notes": "",
        })

    for index in range(len(samples), rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.suptitle(
        "GlacierAI — Gangotri Debris Sample Review\n"
        "Cyan = predicted glacier | Red = predicted debris",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    contact_path = OUTPUT_DIR / "debris_samples_contact_sheet.png"
    fig.savefig(contact_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    csv_path = OUTPUT_DIR / "debris_samples_metadata.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metadata[0].keys(),
        )
        writer.writeheader()
        writer.writerows(metadata)

    print("\nCompleted successfully.")
    print("Contact sheet:", contact_path)
    print("Individual patches:", OUTPUT_DIR)
    print("Sample metadata:", csv_path)
    print("\nThese are model-selected samples, not verified labels.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    check_files()

    rgb_raw, profile, glacier_mask, debris_mask = load_rasters()

    # Keep the original masks untouched.
    debris_mask = debris_mask & glacier_mask

    print("\nRaster shape:", glacier_mask.shape)
    print("Glacier pixels:", int(glacier_mask.sum()))
    print("Debris prediction pixels:", int(debris_mask.sum()))

    rgb = prepare_rgb(rgb_raw)

    samples = select_samples(glacier_mask, debris_mask)

    create_outputs(
        rgb,
        profile,
        glacier_mask,
        debris_mask,
        samples,
    )


if __name__ == "__main__":
    main()
