
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from backend.models.unet import UNet

ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "datasets" / "live" / "rgi2000_v7_0_g_15_08331_latest.tif"
OUTPUT_PATH = ROOT / "datasets" / "live" / "predictions" / "v2_v3_v5_comparison.png"

CHECKPOINTS = {
    "V2": ROOT / "checkpoints" / "best_unet_v2.pth",
    "V3": ROOT / "checkpoints" / "best_unet_v3.pth",
    "V5": ROOT / "checkpoints" / "best_unet_v5.pth",
}

PATCH_SIZE = 256
THRESHOLD = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_band(band):
    band = band.astype(np.float32)
    valid = np.isfinite(band)
    low, high = np.percentile(band[valid], [2, 98])

    result = np.zeros_like(band, dtype=np.float32)
    if high > low:
        result[valid] = np.clip(
            (band[valid] - low) / (high - low), 0, 1
        )
    return result


with rasterio.open(IMAGE_PATH) as src:
    rgb_bands = [
        src.read(i).astype(np.float32)[:PATCH_SIZE, :PATCH_SIZE]
        for i in (1, 2, 3)
    ]

rgb = np.stack([normalize_band(b) for b in rgb_bands], axis=-1)

tensor = torch.from_numpy(
    rgb.transpose(2, 0, 1).copy()
).unsqueeze(0).to(DEVICE)

predictions = {}

for name, path in CHECKPOINTS.items():
    checkpoint = torch.load(
        path, map_location=DEVICE, weights_only=False
    )
    model = UNet().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    predictions[name] = probability

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(rgb)
axes[0, 0].set_title("Original RGB patch")

for col, name in enumerate(("V2", "V3", "V5"), start=1):
    probability = predictions[name]
    mask = probability >= THRESHOLD

    axes[0, col].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(
        f"{name} mask ({mask.sum()} pixels)"
    )

axes[1, 0].imshow(rgb)
axes[1, 0].set_title("Original RGB patch")

for col, name in enumerate(("V2", "V3", "V5"), start=1):
    im = axes[1, col].imshow(
        predictions[name], cmap="viridis", vmin=0, vmax=1
    )
    axes[1, col].set_title(f"{name} probability")
    fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

for ax in axes.flat:
    ax.axis("off")

fig.suptitle(
    "Glacier model comparison — same patch and threshold 0.50"
)
fig.tight_layout()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved comparison image:", OUTPUT_PATH)
