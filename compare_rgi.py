import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import rasterize

rgi_path = "datasets/processed/global_rgi.gpkg"
image_path = "datasets/live/rgi2000_v7_0_g_15_00467_latest.tif"
mask_path = "datasets/live/predictions/test_00467_glacier_mask.tif"

glacier_id = "RGI2000-v7.0-G-15-00467"

print("Loading RGI boundary...")
gdf = gpd.read_file(rgi_path)

selected = gdf[gdf["rgi_id"] == glacier_id]

if len(selected) != 1:
    raise ValueError(f"Expected exactly one RGI polygon, found {len(selected)}")

geometry = selected.geometry.iloc[0]

with rasterio.open(image_path) as src:
    image_crs = src.crs
    transform = src.transform
    shape = (src.height, src.width)

    if src.crs is None:
        raise ValueError("Satellite image has no CRS")

    selected = selected.to_crs(image_crs)
    geometry = selected.geometry.iloc[0]

    rgi_mask = rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

with rasterio.open(mask_path) as src:
    prediction = src.read(1)
    nodata = src.nodata

if prediction.shape != shape:
    raise ValueError("Prediction and satellite image dimensions differ")

valid = np.ones(shape, dtype=bool)

if nodata is not None:
    valid &= prediction != nodata

valid &= np.isin(prediction, [0, 1])

predicted = (prediction == 1) & valid
reference = rgi_mask & valid

intersection = np.count_nonzero(predicted & reference)
predicted_inside = intersection
predicted_outside = np.count_nonzero(predicted & ~reference & valid)
reference_pixels = np.count_nonzero(reference)
predicted_pixels = np.count_nonzero(predicted)

precision = intersection / predicted_pixels if predicted_pixels else 0
recall = intersection / reference_pixels if reference_pixels else 0
iou = intersection / (predicted_pixels + reference_pixels - intersection) if (predicted_pixels + reference_pixels - intersection) else 0

print("\n========== RGI COMPARISON ==========")
print("Glacier ID:", glacier_id)
print("RGI polygon pixels:", np.count_nonzero(rgi_mask))
print("Valid RGI pixels:", reference_pixels)
print("AI predicted pixels:", predicted_pixels)
print("AI pixels inside RGI:", predicted_inside)
print("AI pixels outside RGI:", predicted_outside)
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"IoU:       {iou:.4f} ({iou*100:.2f}%)")
print("===================================")
