
from pathlib import Path
import random
import shutil

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "datasets" / "training" / "images"
MASK_DIR = PROJECT_ROOT / "datasets" / "training" / "masks"

OUTPUT = PROJECT_ROOT / "datasets" / "training"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ==========================================================
# Validate source directories
# ==========================================================

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Training image directory not found: {IMAGE_DIR}"
    )

if not MASK_DIR.exists():
    raise FileNotFoundError(
        f"Training mask directory not found: {MASK_DIR}"
    )

image_files = sorted(IMAGE_DIR.glob("*.png"))

if not image_files:
    raise ValueError(
        f"No PNG image patches found in {IMAGE_DIR}"
    )

# Ensure every image has a matching mask.
missing_masks = [
    img.name
    for img in image_files
    if not (MASK_DIR / img.name).exists()
]

if missing_masks:
    raise ValueError(
        f"{len(missing_masks)} image patches have no matching masks. "
        f"Examples: {missing_masks[:5]}"
    )

# ==========================================================
# Create clean split folders
# ==========================================================

for split in ["train", "val", "test"]:
    for kind in ["images", "masks"]:
        folder = OUTPUT / split / kind
        folder.mkdir(parents=True, exist_ok=True)

        # Remove old PNG patches to prevent stale files.
        for old_file in folder.glob("*.png"):
            old_file.unlink()

# ==========================================================
# Split patches
# ==========================================================

print(f"Found {len(image_files)} image patches")

random.shuffle(image_files)

total = len(image_files)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

# ==========================================================
# Copy images and matching masks
# ==========================================================

def copy_split(files, split):
    image_output = OUTPUT / split / "images"
    mask_output = OUTPUT / split / "masks"

    for img in files:
        mask = MASK_DIR / img.name

        shutil.copy2(
            img,
            image_output / img.name
        )

        shutil.copy2(
            mask,
            mask_output / img.name
        )


# ==========================================================
# Generate splits
# ==========================================================

copy_split(train_files, "train")
copy_split(val_files, "val")
copy_split(test_files, "test")

# ==========================================================
# Report and verify counts
# ==========================================================

print("\n==========================")
print("Dataset Split Complete")
print("==========================")

for split, expected in [
    ("train", len(train_files)),
    ("val", len(val_files)),
    ("test", len(test_files)),
]:
    image_count = len(
        list((OUTPUT / split / "images").glob("*.png"))
    )
    mask_count = len(
        list((OUTPUT / split / "masks").glob("*.png"))
    )

    print(
        f"{split.capitalize():5} : "
        f"{image_count} images, {mask_count} masks"
    )

    if image_count != expected or mask_count != expected:
        raise RuntimeError(
            f"Count mismatch detected in {split} split."
        )

print("\nTotal patches:", total)
print("Split ratios :", TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
print("All image-mask counts verified.")
