from pathlib import Path

import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]

historical_image = (
    ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467/"
    "satellite/rgi2000-v7.0-g-15-00467_2026.tif"
)

mask_2025 = (
    ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467/"
    "predictions/rgi2000-v7.0-g-15-00467_2025_rgi_mask.tif"
)

mask_2026 = (
    ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467/"
    "predictions/rgi2000-v7.0-g-15-00467_2026_rgi_mask.tif"
)

spectral_image = (
    ROOT / "datasets/live/"
    "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
)


def show_metadata(path):
    with rasterio.open(path) as src:
        print(f"\n--- {path.name} ---")
        print("Bounds:", src.bounds)
        print("CRS:", src.crs)
        print("Band descriptions:", src.descriptions)
        print("Dataset tags:", src.tags())
        for i in range(1, src.count + 1):
            print(f"Band {i} tags:", src.tags(i))


def main():
    for path in (historical_image, mask_2025, mask_2026, spectral_image):
        if not path.exists():
            raise FileNotFoundError(path)

    with rasterio.open(historical_image) as hist, \
         rasterio.open(mask_2025) as src25, \
         rasterio.open(mask_2026) as src26, \
         rasterio.open(spectral_image) as spec:

        print("Historical grid:", hist.width, "x", hist.height)
        print("Spectral grid:", spec.width, "x", spec.height)

        left = max(hist.bounds.left, spec.bounds.left)
        bottom = max(hist.bounds.bottom, spec.bounds.bottom)
        right = min(hist.bounds.right, spec.bounds.right)
        top = min(hist.bounds.top, spec.bounds.top)

        if left >= right or bottom >= top:
            print("\nNo spatial overlap found.")
            return

        print("\nOverlap bounds:")
        print("Left:", left, "Bottom:", bottom)
        print("Right:", right, "Top:", top)

        # Align both annual masks to the spectral grid.
        m25 = np.zeros((spec.height, spec.width), dtype=np.uint8)
        m26 = np.zeros((spec.height, spec.width), dtype=np.uint8)

        reproject(
            source=rasterio.band(src25, 1),
            destination=m25,
            src_transform=src25.transform,
            src_crs=src25.crs,
            dst_transform=spec.transform,
            dst_crs=spec.crs,
            resampling=Resampling.nearest,
        )

        reproject(
            source=rasterio.band(src26, 1),
            destination=m26,
            src_transform=src26.transform,
            src_crs=src26.crs,
            dst_transform=spec.transform,
            dst_crs=spec.crs,
            resampling=Resampling.nearest,
        )

        # Restrict comparison to the actual geographic overlap.
        overlap = (
            (spec.transform * (0, 0))[0] == float("inf")
        )  # placeholder overwritten below

        rows, cols = np.indices((spec.height, spec.width))
        xs = spec.transform.c + (cols + 0.5) * spec.transform.a
        ys = spec.transform.f + (rows + 0.5) * spec.transform.e

        overlap = (
            (xs >= left) & (xs < right)
            & (ys > bottom) & (ys <= top)
        )

        shared = (m25 > 0) & (m26 > 0) & overlap
        only_2026 = (m26 > 0) & (m25 == 0) & overlap
        only_2025 = (m25 > 0) & (m26 == 0) & overlap

        print("\nMask counts on spectral grid within overlap:")
        print("Overlap pixels:", int(overlap.sum()))
        print("Shared 2025/2026:", int(shared.sum()))
        print("2026-only:", int(only_2026.sum()))
        print("2025-only:", int(only_2025.sum()))

        print("\nSpectral reflectance statistics (stored values):")
        for band, name in enumerate(("B08_NIR", "B11_SWIR1", "B12_SWIR2"), start=1):
            values = spec.read(band).astype(np.float32)
            valid = overlap & np.isfinite(values) & (values > 0)

            print(f"\n{name}:")
            for label, region in (
                ("Shared", shared),
                ("2026-only", only_2026),
                ("2025-only", only_2025),
            ):
                sample = values[valid & region]
                if sample.size:
                    print(
                        f"  {label}: n={sample.size}, "
                        f"median={np.median(sample):.1f}, "
                        f"p25={np.percentile(sample, 25):.1f}, "
                        f"p75={np.percentile(sample, 75):.1f}"
                    )
                else:
                    print(f"  {label}: no valid samples")

    show_metadata(spectral_image)


if __name__ == "__main__":
    main()
