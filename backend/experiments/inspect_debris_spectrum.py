
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PRED = LIVE / "predictions"
gid = "rgi2000_v7_0_g_15_00467"

spectral_path = LIVE / f"{gid}_debris_spectral.tif"
glacier_path = PRED / f"{gid}_debris_glacier_mask.tif"
debris_path = PRED / f"{gid}_debris_affected_mask.tif"

lat = 30.8760783581705
lon = 79.0053687387015
radius = 64

with rasterio.open(spectral_path) as src:
    x, y = transform("EPSG:4326", src.crs, [lon], [lat])
    row, col = src.index(x[0], y[0])
    bands = src.read()
    names = [src.descriptions[i] or f"Band {i+1}"
             for i in range(src.count)]
    window = (
        slice(max(0, row-radius), min(src.height, row+radius+1)),
        slice(max(0, col-radius), min(src.width, col+radius+1))
    )
    print("Band names:", names)
    print("Sample pixel:", row, col)

with rasterio.open(glacier_path) as src:
    glacier = src.read(1) > 0

with rasterio.open(debris_path) as src:
    debris = src.read(1) > 0

r0, r1 = window[0].start, window[0].stop
c0, c1 = window[1].start, window[1].stop

g = glacier[r0:r1, c0:c1]
d = debris[r0:r1, c0:c1]
print("Glacier pixels in crop:", int(g.sum()))
print("Debris pixels in crop:", int(d.sum()))

def describe(label, mask):
    print(f"\n{label} — {int(mask.sum())} pixels")
    if not mask.any():
        print("No pixels to inspect.")
        return
    for i, name in enumerate(names):
        values = bands[i, r0:r1, c0:c1][mask]
        values = values[np.isfinite(values)]
        if values.size:
            print(
                f"{name}: median={np.median(values):.4f}, "
                f"min={values.min():.4f}, max={values.max():.4f}"
            )

describe("Predicted debris", d)
describe("Other predicted glacier pixels", g & ~d)
