from pathlib import Path
import json
import cv2
import numpy as np
import rasterio

root = Path(".")
pred_dir = root / "datasets" / "live" / "predictions"

prob_path = pred_dir / "rgi2000_v7_0_g_15_00467_debris_probability.npy"
mask_path = pred_dir / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"

# Locate the RGI mask on the prediction's exact raster grid.
rgi_candidates = [
    root / "datasets" / "live" / "rgi2000_v7_0_g_15_00467_rgi_mask.tif",
    root / "datasets" / "live" / "rgi2000_v7_0_g_15_00467_debris_rgi_mask.tif",
    root / "datasets" / "masks" / "gangotri_mask.tif",
]

print("Probability file exists:", prob_path.exists())
print("Final mask exists:", mask_path.exists())

rgi_path = next((p for p in rgi_candidates if p.exists() and p.suffix.lower() == ".tif"), None)
print("RGI candidates:")
for p in rgi_candidates:
    print(" ", p, "exists:", p.exists())

if not prob_path.exists() or not mask_path.exists():
    raise FileNotFoundError("Probability map or final mask was not found.")

prob = np.load(prob_path)
with rasterio.open(mask_path) as src:
    final_mask = src.read(1)
    transform = src.transform
    crs = src.crs
    pixel_area = abs(src.transform.a * src.transform.e)
    print("\nRaster shape:", final_mask.shape)
    print("CRS:", crs)
    print("Pixel area (m²):", pixel_area)

if prob.shape != final_mask.shape:
    raise ValueError(f"Shape mismatch: probability={prob.shape}, mask={final_mask.shape}")

print("\nPROBABILITY COUNTS")
for threshold in (0.20, 0.30, 0.50, 0.70, 0.90):
    print(f">= {threshold:.2f}: {int((prob >= threshold).sum()):,} pixels")

print("\nPROBABILITY COUNTS INSIDE FINAL MASK GRID")
if rgi_path:
    with rasterio.open(rgi_path) as src:
        rgi = src.read(1)
        same_grid = (
            src.shape == final_mask.shape
            and src.transform == transform
            and src.crs == crs
        )
        print("RGI raster:", rgi_path)
        print("Exact grid match:", same_grid)
        if same_grid:
            inside = rgi > 0
            for threshold in (0.20, 0.30, 0.50, 0.70, 0.90):
                print(f"RGI and >= {threshold:.2f}: {int((inside & (prob >= threshold)).sum()):,}")
            print("RGI pixels:", int(inside.sum()))
            print("Final mask outside RGI:", int(((final_mask > 0) & ~inside).sum()))
        else:
            print("Skipping RGI comparisons because grids differ.")
else:
    print("No candidate RGI GeoTIFF found; RGI comparisons skipped.")

print("\nFINAL MASK")
pixels = int((final_mask > 0).sum())
print("Foreground pixels:", f"{pixels:,}")
print("Area (km²):", round(pixels * pixel_area / 1_000_000, 6))

count, labels, stats, _ = cv2.connectedComponentsWithStats(
    (final_mask > 0).astype(np.uint8), connectivity=8
)
areas = stats[1:, cv2.CC_STAT_AREA]
print("Connected components:", len(areas))
if len(areas):
    print("Largest component sizes (pixels):", sorted(map(int, areas), reverse=True)[:10])

print("\nDiagnostic complete. No files were modified.")
