from pathlib import Path

import cv2
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "datasets" / "live" / "khumbu_fixed.tif"
OUT_DIR = ROOT / "datasets" / "live" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with rasterio.open(IMAGE_PATH) as src:
    raw = np.stack(
        [src.read(i).astype(np.float32) for i in (1, 2, 3)],
        axis=-1,
    )
    profile = src.profile.copy()

# Identify invalid pixels using the same fallback as the current pipeline.
valid = (
    np.isfinite(raw).all(axis=2)
    & ~np.all(raw == 0, axis=2)
)

# Approximate the current inference normalization.
rgb = np.clip(raw / 10000.0, 0.0, 1.0)
rgb8 = (rgb * 255).astype(np.uint8)

# Preliminary cloud candidates: bright and relatively neutral RGB pixels.
# These can ALSO include glacier ice and snow, so do not treat this as
# a validated cloud mask or automatically remove these pixels.
brightness = rgb.mean(axis=2)
spread = rgb.max(axis=2) - rgb.min(axis=2)

candidate = (
    valid
    & (brightness >= 0.55)
    & (spread <= 0.18)
)

# Save candidate mask: white = potential cloud/snow/bright surface.
mask_path = OUT_DIR / "khumbu_cloud_candidates.png"
cv2.imwrite(
    str(mask_path),
    candidate.astype(np.uint8) * 255,
)

# Save an overlay: cyan marks candidate pixels.
# OpenCV uses BGR channel order.
overlay = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
overlay[candidate] = (
    255,
    255,
    0,
)
overlay[~valid] = (0, 0, 0)

overlay_path = OUT_DIR / "khumbu_cloud_candidate_overlay.png"
cv2.imwrite(str(overlay_path), overlay)

# Save a simple brightness diagnostic.
brightness_path = OUT_DIR / "khumbu_brightness.png"
cv2.imwrite(
    str(brightness_path),
    (brightness * 255).clip(0, 255).astype(np.uint8),
)

valid_count = int(valid.sum())
candidate_count = int(candidate.sum())

print("Cloud-candidate diagnostic complete.")
print("Image dimensions:", rgb8.shape[1], "x", rgb8.shape[0])
print("Valid pixels:", valid_count)
print("Potential cloud/bright-surface pixels:", candidate_count)
print(
    "Candidate percentage of valid pixels:",
    round(100 * candidate_count / max(valid_count, 1), 2),
)
print("Saved:", mask_path)
print("Saved:", overlay_path)
print("Saved:", brightness_path)
print("WARNING: candidates are not confirmed clouds.")
