import rasterio
import numpy as np

image_path = r"D:\GlacierAI\datasets\live\rgi2000_v7.0_g_15_00467_tile_001.tif"
mask_path = r"D:\GlacierAI\datasets\masks\gangotri_mask.tif"

with rasterio.open(image_path) as src:
    print("SATELLITE IMAGE")
    print("Shape:", src.height, src.width)
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Bounds:", src.bounds)

with rasterio.open(mask_path) as src:
    mask = src.read(1)
    print("\nTRAINING MASK")
    print("Shape:", src.height, src.width)
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Bounds:", src.bounds)
    print("Values:", np.unique(mask))
    print("Glacier pixels:", np.count_nonzero(mask == 1))