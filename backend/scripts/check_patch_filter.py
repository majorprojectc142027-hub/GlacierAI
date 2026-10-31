
from pathlib import Path

import numpy as np
import rasterio

PATCH_SIZE = 256
STRIDE = 128
THRESHOLD = 0.005

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK_PATH = PROJECT_ROOT / "datasets" / "masks" / "gangotri_mask.tif"

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)

height, width = mask.shape

total = 0
zero_glacier = 0
some_below_threshold = 0
at_or_above_threshold = 0

for y in range(0, height - PATCH_SIZE + 1, STRIDE):
    for x in range(0, width - PATCH_SIZE + 1, STRIDE):
        patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

        ratio = np.count_nonzero(patch) / patch.size
        total += 1

        if ratio == 0:
            zero_glacier += 1
        elif ratio < THRESHOLD:
            some_below_threshold += 1
        else:
            at_or_above_threshold += 1

print("\n======= PATCH FILTER DIAGNOSTIC =======")
print("Total candidate patches:", total)
print("No glacier pixels:", zero_glacier)
print("Some glacier, below 2%:", some_below_threshold)
print("At least 2% glacier:", at_or_above_threshold)
print("Check:", zero_glacier + some_below_threshold
      + at_or_above_threshold == total)
