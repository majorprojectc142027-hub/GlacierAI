
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(r"D:\GlacierAI")
FOLDER = ROOT / "datasets" / "live" / "predictions"

PREFIX = "rgi2000_v7.0_g_15_00467_tile_"


def inspect_mask(path):
    image = np.array(Image.open(path).convert("L"))
    values, counts = np.unique(image, return_counts=True)

    print("File:", path.name)
    print("Shape:", image.shape)
    print("Pixel values/counts:", dict(zip(values.tolist(), counts.tolist())))
    print("Nonzero pixels:", int(np.count_nonzero(image)))
    print()


def main():
    for tile in ("001", "002", "003"):
        print("=" * 50)
        print("TILE", tile)

        rgi_path = FOLDER / f"{PREFIX}{tile}_rgi_mask.png"
        full_path = FOLDER / f"{PREFIX}{tile}_full_mask.png"

        if not rgi_path.exists() or not full_path.exists():
            print("Missing RGI or full mask image.")
            continue

        inspect_mask(rgi_path)
        inspect_mask(full_path)

        rgi = np.array(Image.open(rgi_path).convert("L")) > 0
        full = np.array(Image.open(full_path).convert("L")) > 0

        if rgi.shape != full.shape:
            print("Image dimensions differ:", rgi.shape, full.shape)
            continue

        intersection = int(np.count_nonzero(rgi & full))
        union = int(np.count_nonzero(rgi | full))
        iou = intersection / union if union else 0.0

        print("Overlap pixels:", intersection)
        print(f"IoU between these two PNGs: {iou:.4f} ({iou * 100:.2f}%)")


if __name__ == "__main__":
    main()
