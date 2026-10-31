from pathlib import Path
import os
import json

import cv2
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.crs import CRS

from backend.models.unet import UNet


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_unet_v5.pth"
LIVE_DIR = PROJECT_ROOT / "datasets" / "live"
INPUT_PATTERN = "khumbu_download_test.tif"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "live" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 4 if not torch.cuda.is_available() else 8

# These settings need validation against labelled glacier data.
THRESHOLD = 0.50
MIN_COMPONENT_PIXELS = 4
MORPH_CLOSE_KERNEL_SIZE = 3
MORPH_CLOSE_ITERATIONS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 4))


# ============================================================
# MODEL
# ============================================================
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}"
        )

    model = UNet().to(DEVICE)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Support either a checkpoint dictionary or a raw state_dict.
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.eval()

    epoch = (
        checkpoint.get("epoch", "unknown")
        if isinstance(checkpoint, dict)
        else "unknown"
    )

    print(f"Loaded checkpoint: {MODEL_PATH.name} (epoch={epoch})")
    print(f"Device: {DEVICE}")

    return model


# ============================================================
# INPUT NORMALIZATION
# ============================================================
def normalize_band(
    band: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    Normalize one satellite band using the 2nd and 98th
    percentiles calculated only from valid satellite pixels.
    """
    band = band.astype(np.float32)
    valid_mask = valid_mask.astype(bool)

    if band.shape != valid_mask.shape:
        raise ValueError("Band and valid_mask shapes do not match")

    valid = np.isfinite(band) & valid_mask

    if not np.any(valid):
        return np.zeros(band.shape, dtype=np.uint8)

    p2 = np.percentile(band[valid], 2)
    p98 = np.percentile(band[valid], 98)

    if p98 <= p2:
        return np.zeros(band.shape, dtype=np.uint8)

    normalized = np.zeros(band.shape, dtype=np.float32)

    normalized[valid] = (
        (band[valid] - p2) / (p98 - p2)
    )

    normalized = np.clip(normalized, 0.0, 1.0)

    return (normalized * 255).astype(np.uint8)

def build_rgb(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    Normalize RGB satellite bands using only valid pixels
    and combine them into an RGB image.
    """

    red_norm = normalize_band(red, valid_mask)
    green_norm = normalize_band(green, valid_mask)
    blue_norm = normalize_band(blue, valid_mask)

    rgb = np.stack(
        [red_norm, green_norm, blue_norm],
        axis=-1
    )

    return rgb.astype(np.uint8)
# ============================================================
# RGI REFERENCE MASK (DIAGNOSTIC ONLY)
# ============================================================
def create_rgi_mask(geometry, shape, transform, raster_crs):
    """
    Rasterize the RGI inventory polygon on the satellite image grid.

    This mask is for comparison only. It does not clip the
    model prediction to the historical inventory boundary.
    """
    if geometry is None:
        return None

    if raster_crs is None:
        raise ValueError(
            "GeoTIFF CRS is missing; cannot align RGI geometry."
        )

    geom = transform_geom(
        "EPSG:4326",
        raster_crs,
        geometry,
        precision=6
    )

    mask = rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False
    )

    # Fallback for very small polygons that cover no pixel centres.
    if int(mask.sum()) == 0:
        mask = rasterize(
            [(geom, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )

    return mask


# ============================================================
# TILED INFERENCE WITH OVERLAP BLENDING
# ============================================================
def infer_probabilities(model, rgb_uint8):
    """
    Run overlapping 256x256 patches through the model.
    Blend overlapping predictions to reduce patch-edge seams.
    """
    height, width = rgb_uint8.shape[:2]

    pad_h = max(0, PATCH_SIZE - height)
    pad_w = max(0, PATCH_SIZE - width)

    if height + pad_h > PATCH_SIZE:
        pad_h = (
            STRIDE
            - ((height + pad_h - PATCH_SIZE) % STRIDE)
        ) % STRIDE

    if width + pad_w > PATCH_SIZE:
        pad_w = (
            STRIDE
            - ((width + pad_w - PATCH_SIZE) % STRIDE)
        ) % STRIDE

    padded = cv2.copyMakeBorder(
        rgb_uint8,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT
    )

    padded_h, padded_w = padded.shape[:2]

    ys = list(
        range(0, padded_h - PATCH_SIZE + 1, STRIDE)
    )
    xs = list(
        range(0, padded_w - PATCH_SIZE + 1, STRIDE)
    )

    # Centre-weighted blending window.
    g = cv2.getGaussianKernel(
        PATCH_SIZE,
        PATCH_SIZE / 6.0
    )

    weights = (g @ g.T).astype(np.float32)
    weights /= weights.max()
    weights = np.maximum(weights, 1e-3)

    probability_sum = np.zeros(
        (padded_h, padded_w),
        dtype=np.float32
    )
    weight_sum = np.zeros(
        (padded_h, padded_w),
        dtype=np.float32
    )

    batch_patches = []
    batch_locations = []

    total = len(ys) * len(xs)
    processed = 0

    def run_batch():
        nonlocal processed

        if not batch_patches:
            return

        arr = (
            np.stack(batch_patches).astype(np.float32)
            / 255.0
        )

        tensor = (
            torch.from_numpy(arr)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(DEVICE)
        )

        with torch.inference_mode():
            logits = model(tensor)

            probs = (
                torch.sigmoid(logits)
                .squeeze(1)
                .detach()
                .cpu()
                .numpy()
            )

        for prob, (y, x) in zip(
            probs,
            batch_locations
        ):
            probability_sum[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += prob * weights

            weight_sum[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += weights

        processed += len(batch_locations)

        print(f"Processed {processed}/{total} patches")

        batch_patches.clear()
        batch_locations.clear()

    for y in ys:
        for x in xs:
            patch = padded[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            batch_patches.append(patch)
            batch_locations.append((y, x))

            if len(batch_patches) >= BATCH_SIZE:
                run_batch()

    run_batch()

    weight_sum = np.maximum(weight_sum, 1e-8)

    probability = probability_sum / weight_sum

    return probability[:height, :width]
# ============================================================
# POST-PROCESSING AND AREA
# ============================================================
def clean_mask(binary_mask, valid_mask):
    """Remove tiny components and clean small gaps in the mask."""
    mask = binary_mask.astype(np.uint8)

    kernel_size = max(1, MORPH_CLOSE_KERNEL_SIZE)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MORPH_CLOSE_ITERATIONS
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area >= MIN_COMPONENT_PIXELS:
            cleaned[labels == component_id] = 1

    cleaned[~valid_mask] = 0

    return cleaned


# ============================================================
# CRS AND PIXEL AREA VALIDATION
def get_pixel_area_m2(transform, raster_crs):
    """Calculate pixel area in square metres for a projected raster."""

    if raster_crs is None:
        raise ValueError(
            "GeoTIFF CRS is missing. Cannot calculate area safely."
        )

    crs = CRS.from_user_input(raster_crs)

    if not crs.is_projected:
        raise ValueError(
            f"Expected a projected CRS, but received {crs}."
        )

    # Compatible with Rasterio CRS versions that expose linear_units.
    units = getattr(crs, "linear_units", None)

    if units is None:
        raise ValueError(
            f"Cannot verify the CRS units for {crs}. "
            "Please check the raster CRS before calculating area."
        )

    if str(units).strip().lower() not in {
        "metre", "meter", "metres", "meters", "m"
    }:
        raise ValueError(
            f"CRS must use metres, but {crs} reports units {units!r}."
        )

    pixel_area_m2 = abs(
        transform.a * transform.e
        - transform.b * transform.d
    )

    if not np.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError(f"Invalid pixel area: {pixel_area_m2}")

    return float(pixel_area_m2)

# ============================================================
# SAVE OUTPUTS
# ============================================================
def save_outputs(
    stem,
    rgb,
    probability,
    mask,
    valid_mask,
    profile,
    rgi_mask=None
):
    probability_path = OUTPUT_DIR / f"{stem}_probability.npy"
    mask_tif_path = OUTPUT_DIR / f"{stem}_glacier_mask.tif"
    mask_png_path = OUTPUT_DIR / f"{stem}_full_mask.png"
    rgb_png_path = OUTPUT_DIR / f"{stem}_full_rgb.png"
    heatmap_path = OUTPUT_DIR / f"{stem}_probability_heatmap.png"
    overlay_path = OUTPUT_DIR / f"{stem}_full_overlay.png"
    probability_overlay_path = (
        OUTPUT_DIR / f"{stem}_probability_overlay.png"
    )
    rgi_png_path = OUTPUT_DIR / f"{stem}_rgi_mask.png"

    np.save(
        probability_path,
        probability.astype(np.float32)
    )

    # Save invalid pixels as 255 in the GeoTIFF mask.
    out_profile = profile.copy()

    out_profile.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="lzw"
    )

    geotiff_mask = mask.astype(np.uint8)
    geotiff_mask[~valid_mask] = 255

    with rasterio.open(mask_tif_path, "w", **out_profile) as dst:
        dst.write(geotiff_mask, 1)

    cv2.imwrite(
        str(mask_png_path),
        mask.astype(np.uint8) * 255
    )

    cv2.imwrite(
        str(rgb_png_path),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    )

    # Probability heatmap.
    prob_u8 = np.round(
        np.clip(probability, 0, 1) * 255
    ).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(
        prob_u8,
        cv2.COLORMAP_JET
    )

    heatmap_bgr[~valid_mask] = (0, 0, 0)

    cv2.imwrite(
        str(heatmap_path),
        heatmap_bgr
    )

    # Probability heatmap blended with satellite imagery.
    probability_overlay = cv2.addWeighted(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        0.55,
        heatmap_bgr,
        0.45,
        0
    )

    probability_overlay[~valid_mask] = (0, 0, 0)

    cv2.imwrite(
        str(probability_overlay_path),
        probability_overlay
    )

    # Red overlay on predicted glacier pixels.
    overlay = rgb.copy()
    glacier_pixels = mask == 1

    overlay[glacier_pixels] = (
        0.55 * overlay[glacier_pixels]
        + 0.45 * np.array([255, 0, 0])
    ).astype(np.uint8)

    overlay[~valid_mask] = 0

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    if rgi_mask is not None:
        cv2.imwrite(
            str(rgi_png_path),
            rgi_mask * 255
        )

    return {
        "mask_path": str(mask_tif_path),
        "mask_png_path": str(mask_png_path),
        "rgb_path": str(rgb_png_path),
        "probability_path": str(probability_path),
        "probability_heatmap_path": str(heatmap_path),
        "probability_overlay_path": str(probability_overlay_path),
        "overlay_path": str(overlay_path),
        "rgi_mask_png_path": (
            str(rgi_png_path)
            if rgi_mask is not None
            else None
        )
    }
# ============================================================
# MAIN GEO TIFF PREDICTION
# ============================================================
def predict_geotiff(
    image_path,
    output_stem=None,
    rgi_geometry=None,
    model=None
):
    image_path = Path(image_path)
    if output_stem is None:
        output_stem = image_path.stem.replace("_latest", "")

    if not image_path.exists():
        raise FileNotFoundError(
            f"GeoTIFF not found: {image_path}"
        )

    # --------------------------------------------------------
    # READ SATELLITE IMAGE
    # --------------------------------------------------------
    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise ValueError(
                f"Expected at least 3 bands; found {src.count}."
            )

        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

        height = src.height
        width = src.width

        # Check that the CRS and area units are valid.
        pixel_area_m2 = get_pixel_area_m2(
            transform,
            crs
        )

        # Determine which pixels contain valid satellite data.
        if src.count >= 4:
            # This assumes band 4 is a data-validity mask.
            # Verify this against the input TIFF's band metadata.
            validity_band = src.read(4)

            valid_mask = (
                np.isfinite(red)
                & np.isfinite(green)
                & np.isfinite(blue)
                & np.isfinite(validity_band)
                & (validity_band > 0)
            )
        else:
            valid_mask = (
                np.isfinite(red)
                & np.isfinite(green)
                & np.isfinite(blue)
                & ~(
                    (red == 0)
                    & (green == 0)
                    & (blue == 0)
                )
            )

    if not np.any(valid_mask):
        raise ValueError(
            "No valid satellite pixels were found."
        )

    print(f"Image: {width} x {height}; CRS={crs}")
    print(
        f"Valid pixels: {int(valid_mask.sum()):,}"
        f"/{valid_mask.size:,}"
    )

    # --------------------------------------------------------
    # CREATE RGI REFERENCE MASK
    # --------------------------------------------------------
    rgi_mask = (
        create_rgi_mask(
            rgi_geometry,
            (height, width),
            transform,
            crs
        )
        if rgi_geometry is not None
        else None
    )

    # --------------------------------------------------------
    # PREPARE RGB INPUT
    # --------------------------------------------------------
    rgb = build_rgb(red, green, blue, valid_mask)

    # Avoid using invalid image pixels as model input.
    rgb[~valid_mask] = 0

    # --------------------------------------------------------
    # RUN U-NET INFERENCE
    # --------------------------------------------------------
    if model is None:
        model = load_model()

    print(
        f"Running overlapping patch inference "
        f"(patch={PATCH_SIZE}, stride={STRIDE})..."
    )

    probability = infer_probabilities(model, rgb)

    # Ensure invalid pixels do not enter the prediction.
    probability[~valid_mask] = 0.0

    print("\nProbability statistics (valid pixels):")

    valid_probs = probability[valid_mask]

    for threshold in (
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    ):
        count = int(
            np.count_nonzero(valid_probs >= threshold)
        )

        print(
            f"  >= {threshold:.1f}: {count:,} pixels"
        )

    print(f"  Mean probability: {float(valid_probs.mean()):.4f}")

    # --------------------------------------------------------
    # CREATE AND CLEAN PREDICTION MASK
    # --------------------------------------------------------
    raw_mask = (
        (probability >= THRESHOLD)
        & valid_mask
    )

    mask = clean_mask(
        raw_mask,
        valid_mask
    )

    # --------------------------------------------------------
    # CALCULATE GLACIER AREA
    # --------------------------------------------------------
    glacier_pixels = int(np.count_nonzero(mask == 1))

    area_m2 = glacier_pixels * pixel_area_m2
    area_km2 = area_m2 / 1_000_000.0

    valid_pixels = int(valid_mask.sum())

    coverage_percent = (
        100.0 * glacier_pixels / valid_pixels
        if valid_pixels
        else 0.0
    )

    # --------------------------------------------------------
    # RGI COMPARISON DIAGNOSTICS
    # --------------------------------------------------------
    rgi_diagnostics = None

    if rgi_mask is not None:
        valid_rgi = (
            (rgi_mask == 1)
            & valid_mask
        )

        inside = probability[valid_rgi]

        rgi_diagnostics = {
            "rgi_valid_pixels": int(valid_rgi.sum()),

            "predicted_pixels_inside_rgi": int(
                np.count_nonzero(mask[valid_rgi])
            ),

            "mean_probability_inside_rgi": (
                float(inside.mean())
                if inside.size
                else None
            ),

            "predicted_pixels_outside_rgi": int(
                np.count_nonzero(
                    mask[(rgi_mask == 0) & valid_mask]
                )
            )
        }

        print(
            "\nRGI is used as a comparison reference, "
            "NOT as a prediction clip."
        )

        print(
            "Valid RGI pixels: "
            f"{rgi_diagnostics['rgi_valid_pixels']:,}"
        )

        print(
            "Predicted pixels inside RGI: "
            f"{rgi_diagnostics['predicted_pixels_inside_rgi']:,}"
        )

        print(
            "Predicted pixels outside RGI: "
            f"{rgi_diagnostics['predicted_pixels_outside_rgi']:,}"
        )

    # --------------------------------------------------------
    # SAVE IMAGES AND GEOTIFF MASK
    # --------------------------------------------------------
    outputs = save_outputs(
        output_stem,
        rgb,
        probability,
        mask,
        valid_mask,
        profile,
        rgi_mask
    )

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------
    result = {
        "model": "U-Net V5",
        "checkpoint": MODEL_PATH.name,
        "threshold": THRESHOLD,
        "image_path": str(image_path),

        "width": width,
        "height": height,
        "crs": str(crs),

        "pixel_area_m2": float(pixel_area_m2),

        "valid_image_pixels": valid_pixels,
        "invalid_image_pixels": int(
            valid_mask.size - valid_pixels
        ),

        "glacier_pixels": glacier_pixels,
        "coverage_percent_of_valid_image": round(
            coverage_percent, 3
        ),

        "area_m2": round(area_m2, 2),
        "area_km2": round(area_km2, 6),

        "rgi_diagnostics": rgi_diagnostics,

        **outputs
    }

    json_path = (
        OUTPUT_DIR
        / f"{output_stem}_analysis.json"
    )

    json_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    result["analysis_json_path"] = str(json_path)

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------
    print("\n========== GLACIER ANALYSIS ==========")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Glacier pixels: {glacier_pixels:,}")
    print(f"Pixel area: {pixel_area_m2:.3f} m²")

    print(
        f"Estimated segmented area: {area_km2:.4f} km²"
    )

    print(
        f"Coverage of valid image: {coverage_percent:.2f}%"
    )

    print(f"Saved mask: {outputs['mask_path']}")
    print(f"Saved overlay: {outputs['overlay_path']}")
    print(f"Saved diagnostics: {json_path}")

    print(
        "NOTE: This is a model-estimated area, "
        "not a validated inventory measurement."
    )

    return result


# ============================================================
# SCRIPT ENTRY POINT: AUTOMATIC LIVE-IMAGE DISCOVERY
# ============================================================
if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model checkpoint not found: {MODEL_PATH}")

    if not LIVE_DIR.exists():
        raise SystemExit(f"Live dataset directory not found: {LIVE_DIR}")

    input_paths = sorted(
        path for path in LIVE_DIR.glob(INPUT_PATTERN)
        if path.is_file()
    )

    if not input_paths:
        raise SystemExit(
            f"No input GeoTIFFs matching {INPUT_PATTERN!r} found in {LIVE_DIR}"
        )

    print(f"Found {len(input_paths)} live GeoTIFF file(s).")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Loading model once for all images...")
    model = load_model()

    results = []
    failures = []

    for index, image_path in enumerate(input_paths, start=1):
        output_stem = image_path.stem.removesuffix("_latest")
        print("\\n" + "=" * 72)
        print(f"[{index}/{len(input_paths)}] Processing: {image_path.name}")
        print("=" * 72)

        try:
            # Do not use the former fixed Khumbu coordinates here.
            # Without a verified ID-to-geometry lookup, RGI comparison is
            # omitted rather than comparing against the wrong glacier.
            result = predict_geotiff(
                image_path=image_path,
                output_stem=output_stem,
                rgi_geometry=None,
                model=model,
            )
            results.append({
                "input": str(image_path),
                "status": "success",
                "analysis_json_path": result.get("analysis_json_path"),
                "area_km2": result.get("area_km2"),
            })
        except Exception as exc:
            failures.append({"input": str(image_path), "error": str(exc)})
            print(f"ERROR processing {image_path.name}: {exc}")

    batch_summary = {
        "model": "U-Net V5",
        "checkpoint": MODEL_PATH.name,
        "input_directory": str(LIVE_DIR),
        "pattern": INPUT_PATTERN,
        "processed_count": len(results),
        "failed_count": len(failures),
        "results": results,
        "failures": failures,
    }
    summary_path = OUTPUT_DIR / "batch_analysis_summary.json"
    summary_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")

    print("\\n" + "=" * 72)
    print("BATCH ANALYSIS COMPLETE")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failures)}")
    print(f"Batch summary: {summary_path}")
    print("=" * 72)
