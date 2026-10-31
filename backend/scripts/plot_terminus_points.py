from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
    / "rgi2000-v7.0-g-15-00467"
    / "predictions"
)

OUTPUT = ROOT / "terminus_points_diagnostic.png"

YEARS = [2023, 2024, 2025, 2026]

# Terminus coordinates from the latest analysis
TERMINUS_POINTS = {
    2023: (313821.81, 3411299.23),
    2024: (314055.83, 3411785.27),
    2025: (314217.84, 3412283.32),
    2026: (314235.84, 3412247.31),
}


def main():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12),
    )

    axes = axes.ravel()

    for ax, year in zip(axes, YEARS):

        path = (
            ROOT
            / f"rgi2000-v7.0-g-15-00467_{year}_rgi_mask.tif"
        )

        with rasterio.open(path) as src:

            mask = src.read(1) > 0
            transform = src.transform

            left = transform.c
            right = left + transform.a * mask.shape[1]

            top = transform.f
            bottom = top + transform.e * mask.shape[0]

            ax.imshow(
                mask,
                extent=[
                    left,
                    right,
                    bottom,
                    top,
                ],
                origin="upper",
            )

        easting, northing = TERMINUS_POINTS[year]

        ax.scatter(
            easting,
            northing,
            s=100,
            marker="x",
            linewidths=3,
            label=f"Detected terminus {year}",
        )

        ax.set_title(
            f"Gangotri — {year}"
        )

        ax.set_xlabel(
            "Easting (m)"
        )

        ax.set_ylabel(
            "Northing (m)"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

    fig.suptitle(
        "Gangotri AI Terminus Position Diagnostic",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()