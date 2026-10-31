from pathlib import Path

import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = PROJECT_ROOT / "datasets" / "live"

BAND_NAMES = [
    "B08_NIR",
    "B11_SWIR1",
    "B12_SWIR2",
]

SCALE = 0.0001


def main():
    files = sorted(LIVE_DIR.glob("*debris_spectral.tif"))

    if not files:
        print("No spectral TIFF files found.")
        return

    print(f"Found {len(files)} spectral TIFF files.")

    for path in files:
        print(f"\nProcessing: {path.name}")

        try:
            with rasterio.open(path, "r+") as dataset:
                if dataset.count != 3:
                    print(
                        f"SKIPPED: expected 3 bands, "
                        f"found {dataset.count}"
                    )
                    continue

                # Add band descriptions.
                for index, name in enumerate(BAND_NAMES, start=1):
                    dataset.set_band_description(index, name)

                # Stored values = reflectance * 10000.
                dataset.scales = (SCALE, SCALE, SCALE)
                dataset.offsets = (0.0, 0.0, 0.0)

                dataset.update_tags(
                    REFLECTANCE_SCALE="10000",
                    REFLECTANCE_UNITS="unitless",
                    BAND_ORDER="B08,B11,B12",
                )

            print("Metadata updated successfully.")

        except Exception as error:
            print(f"FAILED: {error}")

    print("\nMetadata repair completed.")


if __name__ == "__main__":
    main()
