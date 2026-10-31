
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

SAFE_DIR = (
    Path.home()
    / "Downloads"
    / "S2A_MSIL2A_20250927T053241_N0511_R105_T44SKA_20250927T084914.SAFE"
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "datasets" / "live"
OUTPUT_PATH = OUTPUT_DIR / "gangotri_2025_rgb.png"

PREVIEW_SIZE = 1024


def find_band(name):
    matches = list(SAFE_DIR.rglob(f"*_{name}_10m.jp2"))
    if not matches:
        raise FileNotFoundError(f"Cannot find band {name}")
    return matches[0]


def read_band(name):
    path = find_band(name)
    with rasterio.open(path) as src:
        print(f"Reading {name} at reduced resolution...", flush=True)
        band = src.read(
            1,
            out_shape=(PREVIEW_SIZE, PREVIEW_SIZE),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    return band


def scale_band(band):
    valid = band[np.isfinite(band) & (band > 0)]
    if valid.size == 0:
        raise ValueError("Band contains no valid positive pixels.")

    low, high = np.percentile(valid, [2, 98])
    if high <= low:
        raise ValueError("Band has insufficient contrast.")

    scaled = (band - low) / (high - low)
    return np.clip(scaled, 0, 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    red = read_band("B04")
    green = read_band("B03")
    blue = read_band("B02")

    print("Combining RGB bands...", flush=True)

    rgb = np.stack(
        [scale_band(red), scale_band(green), scale_band(blue)],
        axis=-1,
    )

    image = (rgb * 255).round().astype(np.uint8)
    Image.fromarray(image).save(OUTPUT_PATH)

    print(f"Saved image: {OUTPUT_PATH}", flush=True)
    print(f"Dimensions: {PREVIEW_SIZE} x {PREVIEW_SIZE}")


if __name__ == "__main__":
    main()
