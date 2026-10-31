
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT / "datasets" / "live" / "predictions"
    / "debris_validation_samples.csv"
)

df = pd.read_csv(CSV_PATH)
df = df[df["group"] == "additional_experimental"].copy()

coords = df[["row", "col"]].to_numpy(dtype=float)

print("Experimental samples:", len(coords))

if len(coords) < 2:
    print("Not enough samples to compare.")
    raise SystemExit

# Pairwise pixel distances.
distances = np.sqrt(
    ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
)

# Ignore each point's distance to itself.
np.fill_diagonal(distances, np.inf)

nearest = distances.min(axis=1)

print("Unique coordinates:", len(np.unique(coords, axis=0)))
print("Median nearest-neighbour distance:",
      round(float(np.median(nearest)), 1), "pixels")
print("Samples within 20 pixels of another sample:",
      int((nearest < 20).sum()))
print("Samples within 50 pixels of another sample:",
      int((nearest < 50).sum()))

print("\nNearest-neighbour distances:")
print(np.round(nearest, 1))
