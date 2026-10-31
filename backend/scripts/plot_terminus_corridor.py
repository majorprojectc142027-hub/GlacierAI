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

OUTPUT = ROOT / "terminus_corridor_diagnostic.png"


# Approximate region containing the downstream
# candidate components found in our diagnostics.
E_MIN = 307500
E_MAX = 311500
N_MIN = 3414500
N_MAX = 3420000


YEARS = [2023, 2024, 2025, 2026]


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

        with rasterio.open(path) as ds:

            window = rasterio.windows.from_bounds(
                E_MIN,
                N_MIN,
                E_MAX,
                N_MAX,
                transform=ds.transform,
            )

            mask = ds.read(
                1,
                window=window,
            ) > 0

            transform = ds.window_transform(
                window
            )

        ax.imshow(
            mask,
            extent=[
                transform.c,
                transform.c
                + transform.a * mask.shape[1],
                transform.f
                + transform.e * mask.shape[0],
                transform.f,
            ],
            origin="upper",
        )

        ax.set_xlim(
            E_MIN,
            E_MAX,
        )

        ax.set_ylim(
            N_MIN,
            N_MAX,
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

        ax.grid(
            alpha=0.25
        )

    fig.suptitle(
        "Gangotri Historical Terminus Corridor Diagnostic",
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