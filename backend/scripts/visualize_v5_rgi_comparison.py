
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "datasets" / "historical" / "rgi2000-v7.0-g-15-00467"
IMAGE = BASE / "satellite" / "rgi2000-v7.0-g-15-00467_2025.tif"
PROBABILITY = BASE / "v5_evaluation" / "2025_v5_probability.npy"
RGI_MASK = BASE / "v5_evaluation" / "2025_full_regional_rgi_mask.tif"
OUTPUT = BASE / "v5_evaluation" / "2025_v5_four_panel_comparison.png"
THRESHOLD = 0.50

def main():
    print("Creating V5 vs Complete RGI Visual Comparison...")

    with rasterio.open(IMAGE) as src:
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)
        valid_band = src.read(4)
        valid = (
            np.isfinite(red)
            & np.isfinite(green)
            & np.isfinite(blue)
            & (valid_band > 0)
        )

    probability = np.load(PROBABILITY).astype(np.float32)

    with rasterio.open(RGI_MASK) as src:
        rgi = src.read(1)

    if probability.shape != valid.shape or rgi.shape != valid.shape:
        raise ValueError("Image, probability and RGI mask dimensions do not match")

    # Build RGB image using the same percentile normalization as inference.
    rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)

    for index, band in enumerate([red, green, blue]):
        pixels = band[valid]
        if pixels.size == 0:
            continue
        p2, p98 = np.percentile(pixels, [2, 98])
        if p98 > p2:
            normalized = np.clip((band - p2) / (p98 - p2), 0, 1)
            rgb[:, :, index] = (normalized * 255).astype(np.uint8)

    rgb[~valid] = 0

    # Binary masks.
    reference = (rgi == 1) & valid
    prediction = (probability >= THRESHOLD) & valid

    # Comparison categories:
    # 0 = background, 1 = RGI only, 2 = V5 only, 3 = overlap.
    comparison = np.zeros(valid.shape, dtype=np.uint8)
    comparison[reference & ~prediction] = 1
    comparison[prediction & ~reference] = 2
    comparison[reference & prediction] = 3

    cmap = ListedColormap([
        (0, 0, 0, 0),
        (0.1, 0.9, 0.2, 0.75),
        (1.0, 0.1, 0.1, 0.75),
        (1.0, 0.9, 0.0, 0.85)
    ])

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(
        "Global Glacier Monitoring - V5 Model Evaluation (2025)",
        fontsize=18,
        fontweight="bold"
    )

    # Panel 1: Original satellite image.
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. Original Sentinel-2 Image")

    # Panel 2: Complete RGI reference.
    axes[0, 1].imshow(rgb)
    axes[0, 1].imshow(
        reference,
        cmap=ListedColormap(["none", "#00e600"]),
        alpha=0.65
    )
    axes[0, 1].set_title("2. Complete RGI Reference (Green)")

    # Panel 3: V5 prediction.
    axes[1, 0].imshow(rgb)
    axes[1, 0].imshow(
        prediction,
        cmap=ListedColormap(["none", "#ff2222"]),
        alpha=0.65
    )
    axes[1, 0].set_title(f"3. V5 Prediction (Red, Threshold={THRESHOLD})")

    # Panel 4: Difference and overlap.
    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(comparison, cmap=cmap, vmin=0, vmax=3)
    axes[1, 1].set_title("4. RGI vs V5 Comparison")

    legend_items = [
        plt.Line2D([0], [0], color="#00e600", lw=6, label="RGI only - Missed by V5"),
        plt.Line2D([0], [0], color="#ff2222", lw=6, label="V5 only - Outside RGI"),
        plt.Line2D([0], [0], color="#ffe600", lw=6, label="Overlap - Correct detection")
    ]

    axes[1, 1].legend(
        handles=legend_items,
        loc="lower right",
        fontsize=9,
        framealpha=0.9
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nComparison image saved:")
    print(OUTPUT)
    print("RGI glacier pixels:", int(reference.sum()))
    print("V5 predicted pixels:", int(prediction.sum()))
    print("Overlap pixels:", int((reference & prediction).sum()))
    print("\nCompleted successfully.")

if __name__ == "__main__":
    main()
