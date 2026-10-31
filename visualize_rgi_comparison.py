import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.features import rasterize
from PIL import Image

IMAGE = "datasets/live/rgi2000_v7_0_g_15_00467_latest.tif"
RGI_FILE = "datasets/processed/global_rgi.gpkg"
MASK = "datasets/live/predictions/test_00467_glacier_mask.tif"
RGB = "datasets/live/predictions/test_00467_full_rgb.png"

GLACIER_ID = "RGI2000-v7.0-G-15-00467"
OUTPUT = "datasets/live/predictions/test_00467_rgi_comparison.png"

print("Loading satellite RGB...")
rgb = np.array(Image.open(RGB).convert("RGB"))

print("Loading RGI boundary...")
gdf = gpd.read_file(RGI_FILE)
gdf = gdf[gdf["rgi_id"] == GLACIER_ID]

if len(gdf) != 1:
    raise ValueError(f"Expected one glacier polygon, found {len(gdf)}")

with rasterio.open(IMAGE) as src:
    gdf = gdf.to_crs(src.crs)

    reference = rasterize(
        [(gdf.geometry.iloc[0], 1)],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

with rasterio.open(MASK) as src:
    prediction_data = src.read(1)
    nodata = src.nodata

if rgb.shape[:2] != reference.shape:
    raise ValueError("RGB and RGI dimensions do not match")

valid = np.isin(prediction_data, [0, 1])

if nodata is not None:
    valid &= prediction_data != nodata

prediction = (prediction_data == 1) & valid
reference = reference & valid

# Comparison categories
true_positive = prediction & reference
false_positive = prediction & ~reference & valid
false_negative = ~prediction & reference & valid

tp = np.count_nonzero(true_positive)
fp = np.count_nonzero(false_positive)
fn = np.count_nonzero(false_negative)

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
iou = tp / (tp + fp + fn) if tp + fp + fn else 0

print("\nComparison metrics:")
print("True positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"IoU: {iou*100:.2f}%")

# Build comparison image
comparison = np.zeros((*reference.shape, 3), dtype=np.uint8)

comparison[true_positive] = [0, 200, 0]       # Green
comparison[false_positive] = [255, 50, 50]    # Red
comparison[false_negative] = [50, 100, 255]  # Blue

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

axes[0, 0].imshow(rgb)
axes[0, 0].set_title("Original Satellite Image")

axes[0, 1].imshow(rgb)
axes[0, 1].imshow(
    np.ma.masked_where(~reference, reference),
    cmap="winter",
    alpha=0.65
)
axes[0, 1].set_title("RGI Reference Boundary")

axes[1, 0].imshow(rgb)
axes[1, 0].imshow(
    np.ma.masked_where(~prediction, prediction),
    cmap="autumn",
    alpha=0.65
)
axes[1, 0].set_title("U-Net V3 Prediction")

axes[1, 1].imshow(comparison)
axes[1, 1].set_title(
    f"Comparison | Precision {precision*100:.1f}% | "
    f"Recall {recall*100:.1f}% | IoU {iou*100:.1f}%"
)

for ax in axes.flat:
    ax.axis("off")

fig.suptitle(
    "GlacierAI - RGI vs U-Net V3 | RGI 15-00467",
    fontsize=16
)

fig.text(
    0.5, 0.025,
    "Green = Correct overlap | Red = AI outside RGI | Blue = RGI missed by AI",
    ha="center",
    fontsize=12
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(OUTPUT, dpi=180, bbox_inches="tight")
plt.close()

print("\nSaved comparison:", OUTPUT)
