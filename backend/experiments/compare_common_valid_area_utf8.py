from pathlib import Path
import numpy as np
import rasterio

root = Path("datasets/historical/rgi2000-v7.0-g-15-00467")
sat = root / "satellite"
pred = root / "predictions"

def load(year):
    stem = f"rgi2000-v7.0-g-15-00467_{year}"
    with rasterio.open(sat / f"{stem}.tif") as ds:
        image = ds.read()
        transform = ds.transform
        crs = ds.crs

    with rasterio.open(pred / f"{stem}_rgi_mask.tif") as ds:
        mask = ds.read(1) > 0

    valid = np.any(image != 0, axis=0)
    return image, valid, mask, transform, crs

img25, valid25, mask25, transform25, crs25 = load(2025)
img26, valid26, mask26, transform26, crs26 = load(2026)

assert img25.shape == img26.shape
assert transform25 == transform26
assert crs25 == crs26

common = valid25 & valid26

pixel_area_km2 = abs(
    transform25.a * transform25.e -
    transform25.b * transform25.d
) / 1_000_000

def report(label, region):
    n = int(region.sum())
    a25 = int((mask25 & region).sum())
    a26 = int((mask26 & region).sum())

    print(f"\n{label}")
    print("Valid comparison pixels:", n)
    print("2025 predicted pixels:", a25)
    print("2026 predicted pixels:", a26)
    print("2025 area (km²):", round(a25 * pixel_area_km2, 3))
    print("2026 area (km²):", round(a26 * pixel_area_km2, 3))
    print("2025 predicted (% of region):",
          round(100 * a25 / max(n, 1), 3))
    print("2026 predicted (% of region):",
          round(100 * a26 / max(n, 1), 3))

print("Pixel area (km²):", pixel_area_km2)
print("2025 valid pixels:", int(valid25.sum()))
print("2026 valid pixels:", int(valid26.sum()))
print("Common valid pixels:", int(common.sum()))
print("Valid only in 2026:", int((valid26 & ~valid25).sum()))

report("FULL IMAGE GRID", np.ones(valid25.shape, dtype=bool))
report("COMMON VALID IMAGE AREA", common)

print("\nMask agreement within common valid area:")
union = (mask25 | mask26) & common
intersection = mask25 & mask26 & common
print("Intersection pixels:", int(intersection.sum()))
print("Union pixels:", int(union.sum()))
print("IoU:", round(
    int(intersection.sum()) / max(int(union.sum()), 1), 4
))

