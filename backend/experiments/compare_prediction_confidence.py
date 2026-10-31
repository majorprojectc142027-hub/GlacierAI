
import json
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "datasets/historical/rgi2000-v7.0-g-15-00467"
LIVE = ROOT / "datasets/live/predictions"

JSON_PATH = BASE / "historical_results.json"
PROB_PATH = LIVE / "rgi2000-v7.0-g-15-00467_2026_probability.npy"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

observations = {int(o["year"]): o for o in data["observations"]}


def load_mask(year):
    path = resolve(observations[year]["mask_path"])

    with rasterio.open(path) as src:
        mask = src.read(1) > 0
        grid = (src.crs, src.transform, src.width, src.height)

    return mask, grid


mask25, grid25 = load_mask(2025)
mask26, grid26 = load_mask(2026)

if grid25 != grid26:
    raise ValueError("Historical masks have different grids.")

if not PROB_PATH.exists():
    raise FileNotFoundError(PROB_PATH)

prob = np.load(PROB_PATH)

if prob.shape != mask26.shape:
    raise ValueError(
        "Probability map and historical mask dimensions differ."
    )

shared = mask25 & mask26
only25 = mask25 & ~mask26
only26 = mask26 & ~mask25
neither = ~mask25 & ~mask26

groups = {
    "Shared 2025 + 2026": shared,
    "2026 only": only26,
    "2025 only": only25,
    "Neither year": neither,
}

print("\n--- PROBABILITY BY HISTORICAL MASK GROUP ---")
print("Note: confidence is model probability, not verified accuracy.")
print(f"{'Group':<24}{'Pixels':>12}{'Mean':>12}{'Median':>12}"
      f"{'P90':>12}{'P95':>12}")

for name, region in groups.items():
    values = prob[region]
    values = values[np.isfinite(values)]

    if values.size == 0:
        print(f"{name:<24}{0:>12}")
        continue

    print(
        f"{name:<24}{values.size:>12,}"
        f"{values.mean():>12.4f}"
        f"{np.median(values):>12.4f}"
        f"{np.percentile(values, 90):>12.4f}"
        f"{np.percentile(values, 95):>12.4f}"
    )

# Compare probability distributions, keeping the historical masks unchanged.
fig, ax = plt.subplots(figsize=(10, 6))

for name in ("Shared 2025 + 2026", "2026 only", "2025 only"):
    values = prob[groups[name]]
    values = values[np.isfinite(values)]

    if values.size:
        ax.hist(
            values,
            bins=np.linspace(0, 1, 51),
            density=True,
            histtype="step",
            linewidth=1.8,
            label=name,
        )

ax.set_xlabel("U-Net probability")
ax.set_ylabel("Density")
ax.set_title("Gangotri: Probability by annual mask group")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()

output = LIVE / "gangotri_confidence_comparison.png"
fig.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)

print("\nSaved:", output)
print("No masks, thresholds, checkpoints, or historical results were changed.")
