from pathlib import Path
import rasterio

ROOT = Path("datasets")

SEARCH_TERMS = ("2026", "gangotri", "rgi2000-v7.0-g-15-00467")

def inspect_raster(path):
    print("\n" + "=" * 80)
    print(f"FILE: {path}")

    try:
        with rasterio.open(path) as src:
            print(f"Dimensions : {src.width} x {src.height}")
            print(f"Bands      : {src.count}")
            print(f"CRS        : {src.crs}")
            print(f"Data type  : {src.dtypes}")
            print(f"Resolution : {src.res}")
            print(f"Bounds     : {src.bounds}")

            for band in range(1, src.count + 1):
                print(
                    f"Band {band}: "
                    f"description={src.descriptions[band - 1]}, "
                    f"color_interpretation="
                    f"{src.colorinterp[band - 1]}"
                )

            # Show useful metadata tags without dumping everything.
            tags = src.tags()
            useful = {
                key: value
                for key, value in tags.items()
                if any(word in key.lower()
                       for word in ("band", "sensor", "product", "date"))
            }
            if useful:
                print("Useful tags:", useful)

    except Exception as exc:
        print(f"Could not inspect file: {exc}")


def main():
    if not ROOT.exists():
        print(f"Dataset folder not found: {ROOT.resolve()}")
        return

    candidates = set()

    # Find relevant GeoTIFFs in the datasets directory.
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for path in ROOT.rglob(pattern):
            if any(term in str(path).lower()
                   for term in SEARCH_TERMS):
                candidates.add(path)

    if not candidates:
        print("No matching GeoTIFFs found.")
        return

    print(f"Found {len(candidates)} matching GeoTIFF files.")

    # Inspect likely source images before masks and predictions.
    def priority(path):
        name = path.name.lower()
        if "satellite" in str(path).lower():
            return 0
        if "rgb" in name or "image" in name:
            return 1
        if "probability" in name or "mask" in name:
            return 3
        return 2

    for path in sorted(candidates, key=lambda p: (priority(p), str(p))):
        inspect_raster(path)


if __name__ == "__main__":
    main()
