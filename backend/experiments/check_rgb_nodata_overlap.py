from pathlib import Path
import numpy as np
import rasterio

root = Path("datasets/historical/rgi2000-v7.0-g-15-00467")
sat_dir = root / "satellite"
mask_dir = root / "predictions"

for year in (2025, 2026):
    image_path = sat_dir / f"rgi2000-v7.0-g-15-00467_{year}.tif"
    mask_path = mask_dir / f"rgi2000-v7.0-g-15-00467_{year}_rgi_mask.tif"

    print(f"\n{'=' * 55}\nYEAR {year}\n{'=' * 55}")

    with rasterio.open(image_path) as img, rasterio.open(mask_path) as msk:
        image = img.read()
        mask = msk.read(1) > 0

        # A pixel is considered zero-filled if every RGB band is zero.
        all_zero = np.all(image == 0, axis=0)
        any_zero = np.any(image == 0, axis=0)
        valid = ~all_zero

        print("Image pixels:", image.shape[1] * image.shape[2])
        print("All-band-zero pixels:", int(all_zero.sum()))
        print("All-band-zero (%):", round(100 * all_zero.mean(), 2))
        print("At least one zero band (%):", round(100 * any_zero.mean(), 2))
        print("Nonzero RGB pixels:", int(valid.sum()))
        print("Predicted glacier pixels:", int(mask.sum()))

        print("Glacier pixels over all-band-zero image:",
              int((mask & all_zero).sum()))
        print("Glacier pixels over nonzero image:",
              int((mask & valid).sum()))

        print("Mask pixels over zero image (%):",
              round(100 * (mask & all_zero).sum() / max(mask.sum(), 1), 2))

        for i, name in enumerate(("Band 1 (B04 Red)", "Band 2 (B03 Green)", "Band 3 (B02 Blue)")):
            print(f"{name} zero pixels: {int((image[i] == 0).sum())}")
