
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(r"D:\GlacierAI")
SATELLITE = ROOT / "datasets" / "live"
PREDICTIONS = ROOT / "datasets" / "live" / "predictions"

PREFIX = "rgi2000_v7.0_g_15_00467_tile_"

for tile in ("001", "002", "003"):
    print("\n" + "=" * 60)
    print("TILE", tile)

    satellite_path = SATELLITE / f"{PREFIX}{tile}.tif"
    prediction_path = PREDICTIONS / f"{PREFIX}{tile}_glacier_mask.tif"
    reference_path = PREDICTIONS / f"{PREFIX}{tile}_rgi_mask.png"

    with rasterio.open(satellite_path) as satellite:
        print("Satellite shape:", satellite.shape)
        print("Satellite CRS:", satellite.crs)
        print("Satellite transform:", satellite.transform)
        print("Satellite bounds:", satellite.bounds)

    with rasterio.open(prediction_path) as prediction:
        print("Prediction shape:", prediction.shape)
        print("Prediction CRS:", prediction.crs)
        print("Prediction transform:", prediction.transform)
        print("Prediction bounds:", prediction.bounds)

    reference = np.array(Image.open(reference_path).convert("L")) > 0
    print("RGI PNG shape:", reference.shape)
    print("RGI foreground pixels:", int(reference.sum()))

print("\nGrid inspection complete.")
