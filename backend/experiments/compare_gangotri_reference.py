
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "datasets/live/predictions/rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
REF = ROOT / "datasets/live/predictions/rgi2000_v7_0_g_15_00467_rgi_reference_mask.tif"

with rasterio.open(PRED) as src:
    pred = src.read(1) > 0
    pred_transform = src.transform
    pred_crs = src.crs

with rasterio.open(REF) as src:
    ref = src.read(1) > 0
    ref_transform = src.transform
    ref_crs = src.crs

print("Prediction shape:", pred.shape)
print("Reference shape:", ref.shape)
print("Same shape:", pred.shape == ref.shape)
print("Same CRS:", pred_crs == ref_crs)
print("Same transform:", pred_transform == ref_transform)

if pred.shape != ref.shape:
    raise ValueError("Raster shapes differ; comparison stopped.")

intersection = np.count_nonzero(pred & ref)
union = np.count_nonzero(pred | ref)
pred_count = np.count_nonzero(pred)
ref_count = np.count_nonzero(ref)

iou = intersection / union if union else 0.0
dice = 2 * intersection / (pred_count + ref_count) if pred_count + ref_count else 0.0

print("\n--- Gangotri Mask Comparison ---")
print("U-Net pixels:", pred_count)
print("RGI reference pixels:", ref_count)
print("Intersection pixels:", intersection)
print("Union pixels:", union)
print(f"IoU: {iou:.4f}")
print(f"Dice: {dice:.4f}")
print(f"U-Net pixels outside RGI: {np.count_nonzero(pred & ~ref)}")
print(f"RGI pixels outside U-Net: {np.count_nonzero(ref & ~pred)}")
