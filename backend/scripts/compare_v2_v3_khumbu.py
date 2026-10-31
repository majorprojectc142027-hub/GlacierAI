from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

from backend.models.unet import UNet

ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "datasets" / "live" / "khumbu_fixed.tif"
CHECKPOINTS = {
    "v2": ROOT / "checkpoints" / "best_unet_v2.pth",
    "v3": ROOT / "checkpoints" / "best_unet_v3.pth",
}
OUT = ROOT / "datasets" / "live" / "predictions" / "v2_v3_comparison"
OUT.mkdir(parents=True, exist_ok=True)

PATCH = 256
STRIDE = 128
THRESHOLDS = [0.20, 0.50]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))

def load_model(path):
    model = UNet().to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded {path.name}: "
        f"epoch={checkpoint.get('epoch')}, "
        f"val_loss={checkpoint.get('val_loss')}, "
        f"val_dice={checkpoint.get('val_dice')}, "
        f"val_iou={checkpoint.get('val_iou')}"
    )
    return model

def positions(length):
    if length <= PATCH:
        return [0]
    starts = list(range(0, length - PATCH + 1, STRIDE))
    if starts[-1] != length - PATCH:
        starts.append(length - PATCH)
    return starts

def predict(model, image, valid):
    height, width, _ = image.shape
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    with torch.inference_mode():
        for y in positions(height):
            for x in positions(width):
                patch = image[y:y + PATCH, x:x + PATCH]
                ph, pw = patch.shape[:2]

                if ph != PATCH or pw != PATCH:
                    padded = np.zeros((PATCH, PATCH, 3), dtype=np.float32)
                    padded[:ph, :pw] = patch
                    patch = padded

                tensor = (
                    torch.from_numpy(patch.copy())
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                probability = torch.sigmoid(model(tensor))[0, 0]
                probability = probability.cpu().numpy()[:ph, :pw]

                total[y:y + ph, x:x + pw] += probability
                count[y:y + ph, x:x + pw] += 1

    result = total / np.maximum(count, 1)
    result[~valid] = 0
    return result

print("Device:", DEVICE)
print("Reading:", IMAGE_PATH)

with rasterio.open(IMAGE_PATH) as src:
    if src.count < 3:
        raise ValueError("Expected at least three RGB bands.")

    raw = np.stack(
        [src.read(i).astype(np.float32) for i in (1, 2, 3)],
        axis=-1,
    )
    profile = src.profile.copy()

# Match predict_geotiff.py's normalization:
# reflectance / 10000, clip to [0, 1], convert to uint8,
# then divide by 255 for the model input.
valid = np.any(raw != 0, axis=2)
image = (
    np.clip(raw / 10000.0, 0.0, 1.0) * 255.0
).astype(np.uint8).astype(np.float32) / 255.0

print("Image dimensions:", image.shape[1], "x", image.shape[0])
print("Valid pixels:", int(valid.sum()), "/", valid.size)

probabilities = {}

for version, checkpoint_path in CHECKPOINTS.items():
    model = load_model(checkpoint_path)
    prob = predict(model, image, valid)
    probabilities[version] = prob

    heatmap = (prob * 255).astype(np.uint8)
    cv2.imwrite(str(OUT / f"khumbu_{version}_probability.png"), heatmap)

    for threshold in THRESHOLDS:
        mask = ((prob >= threshold) & valid).astype(np.uint8) * 255
        filename = OUT / f"khumbu_{version}_threshold_{threshold:.2f}.png"
        cv2.imwrite(str(filename), mask)

        print(
            f"{version.upper()} threshold={threshold:.2f}: "
            f"{int((mask > 0).sum())} glacier candidate pixels "
            f"({100 * (mask > 0).mean():.2f}% of full image)"
        )

# Red = V2 only, green = V3 only, yellow = both models.
v2mask = probabilities["v2"] >= 0.20
v3mask = probabilities["v3"] >= 0.20

comparison = np.zeros((*valid.shape, 3), dtype=np.uint8)
comparison[v2mask & ~v3mask & valid] = (0, 0, 255)
comparison[v3mask & ~v2mask & valid] = (0, 255, 0)
comparison[v2mask & v3mask & valid] = (0, 255, 255)

cv2.imwrite(str(OUT / "khumbu_v2_v3_difference.png"), comparison)

# Probability maps shown side-by-side for visual inspection.
v2gray = (probabilities["v2"] * 255).astype(np.uint8)
v3gray = (probabilities["v3"] * 255).astype(np.uint8)
side_by_side = np.concatenate([v2gray, v3gray], axis=1)
cv2.imwrite(str(OUT / "khumbu_v2_v3_probabilities_side_by_side.png"),
            side_by_side)

print("\nComparison complete.")
print("Results folder:", OUT)
print("Difference map: red=V2 only, green=V3 only, yellow=both")
#'@ | Set-Content -Encoding UTF8 backend\scripts\compare_v2_v3_khumbu.py
