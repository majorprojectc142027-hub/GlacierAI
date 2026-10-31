
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sep_path = Path(
    "datasets/historical/satellite/gangotri_2025-09-22.tif"
)
oct_path = Path(
    "datasets/live/rgi2000_v7_0_g_15_00467_debris_rgb.tif"
)

with rasterio.open(sep_path) as sep, rasterio.open(oct_path) as octo:
    print("CRS equal:", sep.crs == octo.crs)
    print("Band counts:", sep.count, octo.count)

    sep_aligned = np.zeros(
        (3, octo.height, octo.width), dtype=np.float32
    )

    for i in range(1, 4):
        reproject(
            source=rasterio.band(sep, i),
            destination=sep_aligned[i - 1],
            src_transform=sep.transform,
            src_crs=sep.crs,
            dst_transform=octo.transform,
            dst_crs=octo.crs,
            resampling=Resampling.nearest,
        )

    oct_rgb = octo.read().astype(np.float32)

    # Exclude areas outside the September raster footprint.
    footprint = np.zeros(
        (octo.height, octo.width), dtype=np.uint8
    )
    reproject(
        source=np.ones((sep.height, sep.width), dtype=np.uint8),
        destination=footprint,
        src_transform=sep.transform,
        src_crs=sep.crs,
        dst_transform=octo.transform,
        dst_crs=octo.crs,
        resampling=Resampling.nearest,
    )

    valid = footprint > 0

    print("Common-footprint pixels:", int(valid.sum()))

    for i, band_name in enumerate(("B04", "B03", "B02")):
        a = sep_aligned[i][valid]
        b = oct_rgb[i][valid]

        print(f"\n{band_name}")
        print("September percentiles:", np.percentile(a, [5, 25, 50, 75, 95]))
        print("October percentiles  :", np.percentile(b, [5, 25, 50, 75, 95]))
        print("Median absolute difference:", float(np.median(np.abs(a - b))))
        print("Spearman/Pearson-style correlation:", float(np.corrcoef(a, b)[0, 1]))
