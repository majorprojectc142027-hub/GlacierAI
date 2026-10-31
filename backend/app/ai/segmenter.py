
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.models.unet import UNet


# ==========================================================
# PATHS AND CONFIGURATION
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_DIR / "checkpoints" / "best_unet_v5.pth"

OUTPUT_DIR = PROJECT_DIR / "backend" / "outputs" / "predictions"

INPUT_SIZE = 256

# Provisional threshold for investigation only.
# Do not treat this as a validated production threshold.
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# LOAD MODEL ONCE
# ==========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

model = UNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" not in checkpoint:
    raise KeyError(
        "Checkpoint does not contain 'model_state_dict'."
    )

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"GlacierAI model loaded: {MODEL_PATH}")
print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Device: {DEVICE}")
print(f"Prediction output directory: {OUTPUT_DIR}")


# ==========================================================
# SEGMENT IMAGE
# ==========================================================

def segment_image(image_path: str):
    """
    Run U-Net glacier segmentation.

    Saves:
      - *_mask.png: binary prediction at 256 x 256
      - *_probability.png: probability heatmap at 256 x 256
      - *_overlay.png: predictions over the original image

    Coverage is the percentage of positive pixels in the
    256 x 256 model prediction. It is not physical glacier area.
    """

    image_path = Path(image_path)

    # ------------------------------------------------------
    # 1. Read image
    # ------------------------------------------------------

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    original_height, original_width = image_rgb.shape[:2]

    # ------------------------------------------------------
    # 2. Preprocess image
    # ------------------------------------------------------

    resized = cv2.resize(
        image_rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    normalized = resized.astype(np.float32) / 255.0

    tensor = (
        torch.from_numpy(normalized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # ------------------------------------------------------
    # 3. Run model prediction
    # ------------------------------------------------------

    with torch.inference_mode():
        logits = model(tensor)
        probability = (
            torch.sigmoid(logits)[0, 0]
            .cpu()
            .numpy()
        )

    if not np.isfinite(probability).all():
        raise ValueError(
            "Model produced non-finite probabilities."
        )

    # ------------------------------------------------------
    # 4. Probability diagnostics
    # ------------------------------------------------------

    print("\n--- GlacierAI Prediction Diagnostics ---")
    print("Input:", image_path.name)
    print("Original image shape:", image_rgb.shape)
    print("Model input shape:", resized.shape)
    print("Probability min:", float(probability.min()))
    print("Probability max:", float(probability.max()))
    print("Probability mean:", float(probability.mean()))

    for threshold in [0.01, 0.02, 0.05, 0.1, 0.3, 0.5]:
        count = int(
            (probability >= threshold).sum()
        )
        print(f"Pixels >= {threshold}: {count}")

    print("Threshold used for final mask:", THRESHOLD)

    # ------------------------------------------------------
    # 5. Create binary prediction mask
    # ------------------------------------------------------

    mask = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    glacier_pixels = int(mask.sum())
    total_pixels = int(mask.size)

    glacier_coverage = (
        100.0 * glacier_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    # ------------------------------------------------------
    # 6. Prepare output paths
    # ------------------------------------------------------

    stem = image_path.stem

    mask_path = OUTPUT_DIR / f"{stem}_mask.png"
    probability_path = OUTPUT_DIR / f"{stem}_probability.png"
    overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"

    # ------------------------------------------------------
    # 7. Save binary mask
    # ------------------------------------------------------

    mask_image = (
        mask * 255
    ).astype(np.uint8)

    if not cv2.imwrite(str(mask_path), mask_image):
        raise OSError(
            f"Failed to save mask: {mask_path}"
        )

    # ------------------------------------------------------
    # 8. Save probability heatmap
    # ------------------------------------------------------
    # Scale probabilities for visualization only.
    # This does not change the prediction probabilities.

    probability_uint8 = np.clip(
        probability * 255.0,
        0,
        255
    ).astype(np.uint8)

    probability_heatmap = cv2.applyColorMap(
        probability_uint8,
        cv2.COLORMAP_JET
    )

    if not cv2.imwrite(
        str(probability_path),
        probability_heatmap
    ):
        raise OSError(
            f"Failed to save probability map: {probability_path}"
        )

    # ------------------------------------------------------
    # 9. Overlay predicted regions on original image
    # ------------------------------------------------------

    display_mask = cv2.resize(
        mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    overlay_rgb = image_rgb.copy()

    red = np.zeros_like(overlay_rgb)
    red[:, :, 0] = 255

    overlay_rgb[display_mask] = (
        0.55 * overlay_rgb[display_mask]
        + 0.45 * red[display_mask]
    ).clip(0, 255).astype(np.uint8)

    overlay_bgr = cv2.cvtColor(
        overlay_rgb,
        cv2.COLOR_RGB2BGR
    )

    if not cv2.imwrite(
        str(overlay_path),
        overlay_bgr
    ):
        raise OSError(
            f"Failed to save overlay: {overlay_path}"
        )

    # ------------------------------------------------------
    # 10. Print final result
    # ------------------------------------------------------

    print("Predicted glacier pixels:", glacier_pixels)
    print(
        "Predicted coverage:",
        round(glacier_coverage, 2),
        "%"
    )
    print("Saved mask:", mask_path)
    print("Saved probability map:", probability_path)
    print("Saved overlay:", overlay_path)
    print("----------------------------------------\n")

    # ------------------------------------------------------
    # 11. Return results to the API
    # ------------------------------------------------------

    return {
        "mask": mask,
        "probability": probability,
        "glacier_pixels": glacier_pixels,
        "total_pixels": total_pixels,
        "glacier_percentage": glacier_coverage,
        "mask_path": str(mask_path),
        "probability_path": str(probability_path),
        "overlay_path": str(overlay_path),
    }
