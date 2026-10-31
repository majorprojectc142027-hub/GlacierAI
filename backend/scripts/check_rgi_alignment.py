from pathlib import Path

import cv2
import numpy as np
import rasterio

root = Path(__file__).resolve().parents[2]
image_path = root / "datasets/live/khumbu_fixed.tif"
mask_path = root / "datasets/live/predictions/khumbu_rgi_mask.png"
output_path = root / "datasets/live/predictions/khumbu_rgi_boundary_check.png"

with rasterio.open(image_path) as src:
    rgb = np.stack(
        [src.read(i).astype(np.float32) for i in (1, 2, 3)],
        axis=-1,
    )
    rgb = np.clip(rgb / 10000.0, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

rgi = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if rgi is None:
    raise FileNotFoundError(
        f"RGI mask not found: {mask_path}\n"
        "Run python -m backend.scripts.predict_geotiff first."
    )

if rgi.shape != rgb.shape[:2]:
    raise ValueError(
        f"Image shape {rgb.shape[:2]} differs from RGI mask {rgi.shape}"
    )

# Convert RGB to BGR for OpenCV output.
overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# Draw the RGI polygon boundary in red.
contours, _ = cv2.findContours(
    (rgi > 0).astype(np.uint8),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)
cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

if not cv2.imwrite(str(output_path), overlay):
    raise RuntimeError("Could not save diagnostic image.")

print("Saved:", output_path)
print("Image dimensions:", rgb.shape[1], "x", rgb.shape[0])
print("RGI mask pixels:", int((rgi > 0).sum()))
