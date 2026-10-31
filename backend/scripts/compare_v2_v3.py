
import torch
import rasterio
import numpy as np
from pathlib import Path

from backend.models.unet import UNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = (
    PROJECT_ROOT / "datasets" / "live"
    / "rgi2000_v7_0_g_15_08331_latest.tif"
)

CHECKPOINTS = {
    "V2": PROJECT_ROOT / "checkpoints" / "best_unet_v2.pth",
    "V3": PROJECT_ROOT / "checkpoints" / "best_unet_v3.pth",
    "V5": PROJECT_ROOT / "checkpoints" / "best_unet_v5.pth",
}

PATCH_SIZE = 256
THRESHOLD = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_band(band):
    band = band.astype(np.float32)
    valid = np.isfinite(band)

    if not valid.any():
        raise ValueError("Band contains no finite pixels.")

    low, high = np.percentile(band[valid], [2, 98])

    result = np.zeros(band.shape, dtype=np.float32)

    if high > low:
        result[valid] = np.clip(
            (band[valid] - low) / (high - low),
            0.0,
            1.0,
        )

    return result


def load_model(path):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = UNet().to(DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    return model


if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

with rasterio.open(IMAGE_PATH) as src:
    if src.count < 3:
        raise ValueError(f"Expected 3 RGB bands, found {src.count}")

    red = src.read(1).astype(np.float32)
    green = src.read(2).astype(np.float32)
    blue = src.read(3).astype(np.float32)

    print("Image:", IMAGE_PATH.name)
    print("Dimensions:", src.width, "x", src.height)
    print("CRS:", src.crs)
    print("Band descriptions:", src.descriptions)

if min(red.shape) < PATCH_SIZE:
    raise ValueError("Image is smaller than the selected patch.")

# Normalize each band using the same p2/p98 approach
# used by predict_geotiff.py.
rgb = np.stack(
    [
        normalize_band(red),
        normalize_band(green),
        normalize_band(blue),
    ],
    axis=-1,
)

patch = rgb[:PATCH_SIZE, :PATCH_SIZE, :]

tensor = torch.from_numpy(
    patch.transpose(2, 0, 1).copy()
).unsqueeze(0).to(DEVICE)

print("\nDevice:", DEVICE)
print("Patch shape:", tuple(tensor.shape))
print("Input range:", tensor.min().item(), tensor.max().item())

results = {}

for name, path in CHECKPOINTS.items():
    if not path.exists():
        print(f"\nSkipping {name}: checkpoint not found: {path.name}")
        continue

    model = load_model(path)

    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0, 0]
        probabilities = probabilities.cpu().numpy()

    mask = probabilities >= THRESHOLD

    results[name] = (probabilities, mask)

    print(f"\n--- {name} ---")
    print("Checkpoint:", path.name)
    print("Probability min:", float(probabilities.min()))
    print("Probability max:", float(probabilities.max()))
    print("Probability mean:", float(probabilities.mean()))
    print("Pixels above threshold:", int(mask.sum()))
    print("Patch coverage (%):", round(float(mask.mean() * 100), 2))

print("\n--- Pairwise prediction differences ---")

names = list(results)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, mask_a = results[names[i]]
        b, mask_b = results[names[j]]

        print(f"\n{names[i]} vs {names[j]}")
        print("Mean absolute probability difference:",
              float(np.abs(a - b).mean()))
        print("Different mask pixels:",
              int(np.count_nonzero(mask_a != mask_b)))
