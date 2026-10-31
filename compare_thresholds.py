import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

prob_path = "datasets/live/predictions/test_00467_probability.npy"
image_path = "datasets/live/rgi2000_v7_0_g_15_00467_latest.tif"
rgi_path = "datasets/processed/global_rgi.gpkg"

glacier_id = "RGI2000-v7.0-G-15-00467"

prob = np.load(prob_path)

gdf = gpd.read_file(rgi_path)
gdf = gdf[gdf.rgi_id == glacier_id]

if len(gdf) != 1:
    raise ValueError("Expected exactly one RGI polygon")

with rasterio.open(image_path) as src:
    gdf = gdf.to_crs(src.crs)
    ref = rasterize(
        [(gdf.geometry.iloc[0], 1)],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

if prob.shape != ref.shape:
    raise ValueError("Probability and RGI dimensions differ")

valid = np.isfinite(prob)
ref = ref & valid

print("\nTHRESHOLD COMPARISON")
print("-" * 75)
print(f"{'Threshold':<12}{'Precision':<14}{'Recall':<14}{'IoU':<14}{'Predicted':<12}")

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    pred = (prob >= threshold) & valid

    tp = np.count_nonzero(pred & ref)
    fp = np.count_nonzero(pred & ~ref)
    fn = np.count_nonzero(~pred & ref)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0

    print(f"{threshold:<12.1f}{precision:<14.4f}{recall:<14.4f}{iou:<14.4f}{pred.sum():<12}")

print("-" * 75)
