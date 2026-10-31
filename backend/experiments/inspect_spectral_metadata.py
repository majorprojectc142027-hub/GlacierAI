
from pathlib import Path
import rasterio

spectral_path = Path(
    "datasets/live/rgi2000_v7_0_g_15_00467_debris_spectral.tif"
)

historical_root = Path(
    "datasets/historical/rgi2000-v7.0-g-15-00467"
)

image25_path = historical_root / "satellite" / (
    "rgi2000-v7.0-g-15-00467_2025.tif"
)
image26_path = historical_root / "satellite" / (
    "rgi2000-v7.0-g-15-00467_2026.tif"
)

def describe(path):
    print("\n" + "=" * 65)
    print("FILE:", path)

    if not path.exists():
        print("ERROR: File not found")
        return None

    with rasterio.open(path) as ds:
        print("Shape:", ds.height, "x", ds.width)
        print("Bands:", ds.count)
        print("CRS:", ds.crs)
        print("Transform:", ds.transform)
        print("Bounds:", ds.bounds)
        print("Resolution:", ds.res)
        print("Dtype:", ds.dtypes)
        print("Nodata:", ds.nodata)
        print("Band descriptions:", ds.descriptions)
        print("Dataset tags:", ds.tags())

        for i in range(1, ds.count + 1):
            print(f"Band {i} tags:", ds.tags(i))
            arr = ds.read(i, masked=True)
            print(
                f"Band {i} min/max:",
                float(arr.min()),
                float(arr.max())
            )

        return {
            "crs": ds.crs,
            "bounds": ds.bounds,
            "transform": ds.transform,
            "width": ds.width,
            "height": ds.height,
        }

spectral = describe(spectral_path)
image25 = describe(image25_path)
image26 = describe(image26_path)

if spectral and image25 and image26:
    print("\n" + "=" * 65)
    print("SPATIAL OVERLAP CHECK")

    for name, image in [("2025", image25), ("2026", image26)]:
        if spectral["crs"] != image["crs"]:
            print(
                f"{name}: CRS differs; transform to a common CRS "
                "before calculating overlap."
            )
            continue

        a = spectral["bounds"]
        b = image["bounds"]

        left = max(a.left, b.left)
        right = min(a.right, b.right)
        bottom = max(a.bottom, b.bottom)
        top = min(a.top, b.top)

        if left < right and bottom < top:
            print(
                f"{name} overlap bounds:",
                (left, bottom, right, top)
            )
            print(
                f"{name} overlap dimensions (map units):",
                right - left, "x", top - bottom
            )
        else:
            print(f"{name}: No spatial overlap.")
