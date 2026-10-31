from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import label


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "historical"
    / "rgi2000-v7.0-g-15-00467"
    / "predictions"
)

FLOW_AZIMUTH_DEG = 315.0


def inspect():

    direction = np.array([
        np.sin(np.deg2rad(FLOW_AZIMUTH_DEG)),
        np.cos(np.deg2rad(FLOW_AZIMUTH_DEG)),
    ])

    files = sorted(
        ROOT.glob("*_rgi_mask.tif")
    )

    for path in files:

        year = path.name.split("_")[-2]

        with rasterio.open(path) as ds:

            mask = ds.read(1) > 0
            transform = ds.transform

            labels, count = label(
                mask,
                structure=np.ones(
                    (3, 3),
                    dtype=np.uint8,
                ),
            )

            component_sizes = np.bincount(
                labels.ravel()
            )

            component_ids = np.argsort(
                component_sizes[1:]
            )[::-1][:10] + 1

            print()
            print("=" * 90)
            print(f"YEAR: {year}")
            print(
                f"Components: {count}"
            )
            print(
                "Top 10 components:"
            )
            print(
                "ID | PIXELS | FLOW_MIN | FLOW_MAX | "
                "CENTER_E | CENTER_N"
            )
            print("-" * 90)

            for component_id in component_ids:

                rows, cols = np.where(
                    labels == component_id
                )

                if len(rows) == 0:
                    continue

                xs, ys = rasterio.transform.xy(
                    transform,
                    rows,
                    cols,
                    offset="center",
                )

                coords = np.column_stack([
                    np.asarray(xs),
                    np.asarray(ys),
                ])

                flow_position = (
                    coords @ direction
                )

                center = np.mean(
                    coords,
                    axis=0,
                )

                print(
                    f"{component_id:2d} | "
                    f"{len(rows):6d} | "
                    f"{flow_position.min():9.1f} | "
                    f"{flow_position.max():9.1f} | "
                    f"{center[0]:9.1f} | "
                    f"{center[1]:9.1f}"
                )


if __name__ == "__main__":
    inspect()