from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPECTRAL_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "live"
    / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
)

REFERENCE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "live"
    / "predictions"
    / "rgi2000_v7_0_g_15_00467_rgi_reference_mask.tif"
)

UNET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "live"
    / "predictions"
    / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
)

BAND_NAMES = ["B08_NIR", "B11_SWIR1", "B12_SWIR2"]


def main():
    for path in [SPECTRAL_PATH, REFERENCE_PATH, UNET_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with rasterio.open(SPECTRAL_PATH) as src:
        raw = src.read().astype(np.float32)
        profile = src.profile.copy()

    with rasterio.open(REFERENCE_PATH) as src:
        reference = src.read(1) > 0

    with rasterio.open(UNET_PATH) as src:
        unet = src.read(1) > 0

    if raw.shape[1:] != reference.shape or raw.shape[1:] != unet.shape:
        raise ValueError("Spectral raster and masks have different shapes.")

    if reference.shape != unet.shape:
        raise ValueError("Reference and U-Net masks are not aligned.")

    print("=" * 65)
    print("GLACIERAI - SPECTRAL REFLECTANCE DIAGNOSTICS")
    print("=" * 65)

    print("Raster dtype:", profile["dtype"])
    print("Band count:", raw.shape[0])
    print("CRS:", profile["crs"])
    print("Resolution:", profile["transform"].a, profile["transform"].e)
    print("Shape:", raw.shape[1:])

    regions = {
        "U-Net + RGI overlap": reference & unet,
        "RGI-only": reference & ~unet,
        "Outside RGI": ~reference,
    }

    for name, region in regions.items():
        print(f"\n--- {name} ---")
        print("Pixels:", int(region.sum()))

        for i, band_name in enumerate(BAND_NAMES):
            values = raw[i][region]

            if values.size == 0:
                continue

            reflectance = values / 10000.0

            print(
                f"{band_name}: "
                f"min={reflectance.min():.4f}, "
                f"median={np.median(reflectance):.4f}, "
                f"p95={np.percentile(reflectance, 95):.4f}, "
                f"max={reflectance.max():.4f}, "
                f">1.0={int((reflectance > 1.0).sum())}"
            )

    print("\n--- Data-quality checks ---")

    finite = np.isfinite(raw)
    nonzero = raw > 0
    above_scale = raw > 10000

    print("Non-finite values:", int((~finite).sum()))
    print("Zero values by band:", [int((raw[i] == 0).sum()) for i in range(3)])
    print(
        "Values above 10000 by band:",
        [int(above_scale[i].sum()) for i in range(3)],
    )
    print(
        "Pixels with all bands nonzero:",
        int(np.all(nonzero, axis=0).sum()),
    )

    # These are diagnostics, not automatic failure criteria.
    # Reflectance above 1.0 must be investigated in context.
    print(
        "\nNote: values above 1.0 are flagged for investigation; "
        "they are not automatically treated as invalid."
    )


if __name__ == "__main__":
    main()
