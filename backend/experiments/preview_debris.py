
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling


# Project paths
ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "datasets" / "live"
PREDICTIONS = LIVE / "predictions"

RGB_PATH = LIVE / "rgi2000_v7_0_g_15_00467_debris_rgb.tif"

GLACIER_MASK_PATH = (
    PREDICTIONS
    / "rgi2000_v7_0_g_15_00467_debris_glacier_mask.tif"
)

DEBRIS_MASK_PATH = (
    PREDICTIONS
    / "rgi2000_v7_0_g_15_00467_debris_affected_mask.tif"
)

SPECTRAL_PATH = (
    LIVE / "rgi2000_v7_0_g_15_00467_debris_spectral.tif"
)

OUTPUT_PATH = PREDICTIONS / "gangotri_debris_preview.png"
ZOOM_PATH = PREDICTIONS / "gangotri_debris_zoom.png"


def read_mask(path, reference):
    """Read a mask and align it to the RGB grid for visualization."""
    with rasterio.open(path) as src:
        mask = src.read(1)

        same_grid = (
            src.width == reference.width
            and src.height == reference.height
            and src.crs == reference.crs
            and src.transform == reference.transform
        )

        print(f"\n{path.name}")
        print(f"  Shape: {mask.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Transform: {src.transform}")
        print(f"  Same grid as RGB: {same_grid}")

        if same_grid:
            return mask > 0

        if src.crs is None or reference.crs is None:
            raise ValueError(
                f"Cannot align {path.name}: missing CRS information."
            )

        aligned = np.zeros(
            (reference.height, reference.width),
            dtype=np.uint8,
        )

        reproject(
            source=mask,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            resampling=Resampling.nearest,
        )

        print("  Reprojected for visualization only.")
        return aligned > 0


def make_overlay(ax, rgb, glacier_mask, debris_mask, title):
    """Draw RGB with glacier and debris overlays."""
    ax.imshow(rgb)

    glacier_overlay = np.zeros(
        (*glacier_mask.shape, 4),
        dtype=np.float32,
    )
    glacier_overlay[glacier_mask] = [0.0, 0.9, 1.0, 0.22]
    ax.imshow(glacier_overlay)

    debris_overlay = np.zeros(
        (*debris_mask.shape, 4),
        dtype=np.float32,
    )
    debris_overlay[debris_mask] = [1.0, 0.1, 0.1, 0.95]
    ax.imshow(debris_overlay)

    ax.set_title(title)
    ax.axis("off")


def create_display_rgb(rgb):
    """Contrast-stretch RGB bands for visualization only."""
    display_rgb = np.zeros_like(rgb, dtype=np.float32)

    for band in range(3):
        values = rgb[:, :, band]
        valid = values[np.isfinite(values)]

        if valid.size == 0:
            continue

        low, high = np.percentile(valid, [2, 98])

        if high > low:
            display_rgb[:, :, band] = np.clip(
                (values - low) / (high - low),
                0,
                1,
            )

    return display_rgb


def main():
    # Verify that all required files exist.
    required_files = (
        RGB_PATH,
        GLACIER_MASK_PATH,
        DEBRIS_MASK_PATH,
        SPECTRAL_PATH,
    )

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    with rasterio.open(RGB_PATH) as rgb_src:
        if rgb_src.count < 3:
            raise ValueError(
                "RGB GeoTIFF must contain at least 3 bands."
            )

        print("\n========== RGB IMAGE ==========")
        print(f"Dimensions: {rgb_src.width} x {rgb_src.height}")
        print(f"CRS: {rgb_src.crs}")
        print(f"Transform: {rgb_src.transform}")
        print(f"Resolution: {rgb_src.res}")

        # RGB download band order: B04, B03, B02.
        rgb = np.moveaxis(
            rgb_src.read([1, 2, 3]),
            0,
            -1,
        ).astype(np.float32)

        glacier_mask = read_mask(
            GLACIER_MASK_PATH,
            rgb_src,
        )

        debris_mask = read_mask(
            DEBRIS_MASK_PATH,
            rgb_src,
        )

        # Verify spectral raster grid alignment.
        with rasterio.open(SPECTRAL_PATH) as spectral_src:
            same_spectral_grid = (
                spectral_src.width == rgb_src.width
                and spectral_src.height == rgb_src.height
                and spectral_src.crs == rgb_src.crs
                and spectral_src.transform == rgb_src.transform
            )

            print("\n========== SPECTRAL IMAGE ==========")
            print(
                f"Dimensions: {spectral_src.width} "
                f"x {spectral_src.height}"
            )
            print(f"CRS: {spectral_src.crs}")
            print(f"Transform: {spectral_src.transform}")
            print(f"Same grid as RGB: {same_spectral_grid}")

        # Mask statistics.
        glacier_pixels = int(glacier_mask.sum())
        debris_pixels = int(debris_mask.sum())
        outside_pixels = int(
            (debris_mask & ~glacier_mask).sum()
        )

        print("\n========== MASK STATISTICS ==========")
        print(f"Glacier pixels: {glacier_pixels}")
        print(f"Debris pixels: {debris_pixels}")
        print(f"Debris pixels outside glacier mask: {outside_pixels}")

        if glacier_pixels:
            percentage = debris_pixels / glacier_pixels * 100
            print(f"Debris candidate percentage: {percentage:.4f}%")

        # Prepare RGB display.
        display_rgb = create_display_rgb(rgb)

        # --------------------------------------------------
        # 1. FULL-AREA PREVIEW
        # --------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        axes[0].imshow(display_rgb)
        axes[0].set_title("Gangotri Glacier - Sentinel-2 RGB")
        axes[0].axis("off")

        make_overlay(
            axes[1],
            display_rgb,
            glacier_mask,
            debris_mask,
            "Glacier mask (cyan) + debris candidates (red)",
        )

        fig.tight_layout()
        fig.savefig(
            OUTPUT_PATH,
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"\nFull preview saved to:\n{OUTPUT_PATH}")

        # --------------------------------------------------
        # 2. ZOOMED DEBRIS PREVIEW
        # --------------------------------------------------
        ys, xs = np.where(debris_mask)

        if len(xs) == 0:
            print("No debris candidate pixels found.")
            return

        # Bounding box around all debris candidates.
        margin = 100

        x0 = max(0, int(xs.min()) - margin)
        x1 = min(rgb_src.width, int(xs.max()) + margin + 1)
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(rgb_src.height, int(ys.max()) + margin + 1)

        rgb_crop = display_rgb[y0:y1, x0:x1]
        glacier_crop = glacier_mask[y0:y1, x0:x1]
        debris_crop = debris_mask[y0:y1, x0:x1]

        fig, ax = plt.subplots(figsize=(10, 10))

        make_overlay(
            ax,
            rgb_crop,
            glacier_crop,
            debris_crop,
            f"Zoomed debris candidates ({debris_pixels} pixels)",
        )

        fig.tight_layout()
        fig.savefig(
            ZOOM_PATH,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Zoomed preview saved to:\n{ZOOM_PATH}")
        print("\nPreview generation completed successfully.")


if __name__ == "__main__":
    main()
