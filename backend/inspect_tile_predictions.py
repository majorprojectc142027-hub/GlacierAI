
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(r"D:\GlacierAI")
PRED = ROOT / "datasets" / "live" / "predictions"
DIAG = (
    ROOT / "datasets" / "live" / "large_glacier"
    / "rgi_diagnostics"
)
DIAG.mkdir(parents=True, exist_ok=True)

PREFIX = "rgi2000_v7.0_g_15_00467_tile_"


def save_binary_mask(mask, path):
    Image.fromarray(
        mask.astype(np.uint8) * 255
    ).save(path)


for tile in ("001", "002", "003"):
    print("\n" + "=" * 55)
    print("TILE", tile)

    stem = PREFIX + tile

    probability_path = PRED / f"{stem}_probability.npy"
    prediction_path = PRED / f"{stem}_glacier_mask.tif"
    reference_path = DIAG / f"{stem}_rgi_reference_fresh.png"
    rgb_path = PRED / f"{stem}_full_rgb.png"

    if not probability_path.exists():
        print("Missing probability file:", probability_path)
        continue

    if not prediction_path.exists() or not reference_path.exists():
        print("Missing prediction or fresh reference mask.")
        continue

    probability = np.load(probability_path)

    with rasterio.open(prediction_path) as src:
        prediction_values = src.read(1)
        nodata = src.nodata

    reference = (
        np.asarray(Image.open(reference_path).convert("L")) > 0
    )

    if probability.shape != reference.shape:
        print("ERROR: probability/reference shapes differ.")
        print(probability.shape, reference.shape)
        continue

    valid = np.isfinite(probability)

    if nodata is not None:
        valid &= prediction_values != nodata

    valid &= prediction_values != 255

    prediction = prediction_values == 1

    print("Shape:", probability.shape)
    print("Valid pixels:", int(valid.sum()))
    print("Probability min:", float(probability[valid].min()))
    print("Probability max:", float(probability[valid].max()))
    print("Probability mean:", float(probability[valid].mean()))

    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        count = int(np.count_nonzero(
            (probability >= threshold) & valid
        ))
        print(f"Pixels with probability >= {threshold:.1f}: {count}")

    intersection = reference & prediction & valid
    outside = (~reference) & prediction & valid

    print("Reference pixels:", int((reference & valid).sum()))
    print("Prediction pixels:", int((prediction & valid).sum()))
    print("Overlap pixels:", int(intersection.sum()))
    print("Predicted pixels outside reference:", int(outside.sum()))

    # Export visualizations with matching dimensions.
    save_binary_mask(
        reference,
        DIAG / f"{stem}_reference_view.png"
    )
    save_binary_mask(
        prediction,
        DIAG / f"{stem}_prediction_view.png"
    )

    # Green = reference only; red = prediction only;
    # yellow = overlapping pixels.
    comparison = np.zeros(
        (*reference.shape, 3), dtype=np.uint8
    )

    comparison[reference & ~prediction] = [0, 255, 0]
    comparison[prediction & ~reference] = [255, 0, 0]
    comparison[reference & prediction] = [255, 255, 0]

    Image.fromarray(comparison).save(
        DIAG / f"{stem}_comparison_view.png"
    )

    if rgb_path.exists():
        rgb = np.asarray(
            Image.open(rgb_path).convert("RGB")
        )

        if rgb.shape[:2] == reference.shape:
            overlay = rgb.copy()
            overlay[reference & ~prediction] = [0, 255, 0]
            overlay[prediction & ~reference] = [255, 0, 0]
            overlay[reference & prediction] = [255, 255, 0]

            Image.fromarray(overlay).save(
                DIAG / f"{stem}_comparison_on_rgb.png"
            )

    print("Saved diagnostic images to:", DIAG)

print("\nInspection complete.")
