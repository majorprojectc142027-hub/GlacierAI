
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "datasets/live/predictions"

prob_path = PRED / "rgi2000-v7.0-g-15-00467_2026_probability.npy"
mask_path = PRED / "rgi2000-v7.0-g-15-00467_2026_glacier_mask.tif"

if not prob_path.exists():
    raise FileNotFoundError(prob_path)

if not mask_path.exists():
    raise FileNotFoundError(mask_path)

prob = np.load(prob_path)

with rasterio.open(mask_path) as src:
    mask = src.read(1) > 0
    transform = src.transform
    crs = src.crs
    pixel_area = abs(transform.a * transform.e)

print("\n--- PROBABILITY MAP DIAGNOSTICS ---")
print("Probability shape:", prob.shape)
print("Mask shape:", mask.shape)
print("Probability dtype:", prob.dtype)
print("Probability min:", float(np.nanmin(prob)))
print("Probability max:", float(np.nanmax(prob)))
print("Probability mean:", float(np.nanmean(prob)))
print("Probability percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {float(np.nanpercentile(prob, p)):.4f}")

if prob.shape != mask.shape:
    raise ValueError(
        "Probability map and saved mask have different dimensions. "
        "Do not compare them until their grids are verified."
    )

print("Mask pixels:", int(mask.sum()))

fig, axes = plt.subplots(1, 3, figsize=(17, 6))

axes[0].imshow(prob, cmap="magma", vmin=0, vmax=1)
axes[0].set_title("Saved model probabilities")
plt.colorbar(
    axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04
)

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Saved binary mask")

axes[2].imshow(prob, cmap="magma", vmin=0, vmax=1)
axes[2].contour(mask, levels=[0.5], colors=["cyan"], linewidths=0.6)
axes[2].set_title("Probability + mask boundary")

for ax in axes:
    ax.axis("off")

fig.tight_layout()

output = PRED / "gangotri_probability_diagnostic.png"
fig.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)

print("Saved:", output)
