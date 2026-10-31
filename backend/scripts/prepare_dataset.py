
from pathlib import Path
import random
import numpy as np
import rasterio
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLACIER_ID = "rgi2000-v7.0-g-15-00467"
SOURCE_DIR = PROJECT_ROOT / "datasets" / "historical" / GLACIER_ID
SAT_DIR = SOURCE_DIR / "satellite"
MASK_DIR = SOURCE_DIR / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "training_v2" / "rgi_patches"

PATCH_SIZE = 256
STRIDE = 128
BLOCK_SIZE = 512
MIN_GLACIER_RATIO = 0.005
MIN_VALID_RATIO = 0.95
NEGATIVE_TO_POSITIVE_RATIO = 1.0
RANDOM_SEED = 42
YEARS = [2023, 2024, 2025]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def normalize_band(band, valid):
    band = band.astype(np.float32)
    valid = valid.astype(bool) & np.isfinite(band)
    if not np.any(valid):
        raise ValueError("No valid pixels available for normalization.")
    p2, p98 = np.percentile(band[valid], [2, 98])
    if p98 <= p2:
        raise ValueError("Invalid percentile range during normalization.")
    result = np.zeros(band.shape, dtype=np.uint8)
    result[valid] = (
        np.clip((band[valid] - p2) / (p98 - p2), 0, 1) * 255
    ).astype(np.uint8)
    return result

def load_year(year):
    image_path = SAT_DIR / f"{GLACIER_ID}_{year}.tif"
    mask_path = MASK_DIR / f"{GLACIER_ID}_{year}_rgi_mask.tif"
    if not image_path.exists():
        raise FileNotFoundError(f"Satellite image missing: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Reference mask missing: {mask_path}")

    with rasterio.open(image_path) as src:
        if src.count < 4:
            raise ValueError(f"{image_path.name} must contain 4 bands.")
        if src.crs is None:
            raise ValueError(f"CRS missing from {image_path.name}")
        red, green, blue, data_mask = [
            src.read(i).astype(np.float32) for i in range(1, 5)
        ]
        image_shape = (src.height, src.width)
        image_transform = src.transform
        image_crs = src.crs
        image_profile = src.profile.copy()

    with rasterio.open(mask_path) as src:
        labels = src.read(1)
        if (src.height, src.width) != image_shape:
            raise ValueError(f"Dimensions differ for year {year}.")
        if src.transform != image_transform:
            raise ValueError(f"Transforms differ for year {year}.")
        if src.crs != image_crs:
            raise ValueError(f"CRS differs for year {year}.")
        if src.nodata is not None and src.nodata != 255:
            print(f"Warning: mask nodata for {year} is {src.nodata}")

    # Band 4 is the Sentinel Hub dataMask: 10000 = valid.
    satellite_valid = (
        np.isfinite(red) & np.isfinite(green) &
        np.isfinite(blue) & (data_mask == 10000)
    )
    label_valid = np.isin(labels, [0, 1])
    valid = satellite_valid & label_valid

    if not np.any(valid):
        raise ValueError(f"No common valid pixels for {year}.")

    rgb = np.stack([
        normalize_band(red, satellite_valid),
        normalize_band(green, satellite_valid),
        normalize_band(blue, satellite_valid)
    ], axis=-1)
    rgb[~satellite_valid] = 0

    labels = labels.astype(np.uint8)
    labels[~valid] = 0

    print(
        f"{year}: valid={int(valid.sum()):,}, "
        f"glacier={int(((labels == 1) & valid).sum()):,}, "
        f"background={int(((labels == 0) & valid).sum()):,}"
    )
    return rgb, labels, valid, image_profile

def get_candidates(year_data):
    candidates = []
    for year, (rgb, labels, valid, _) in year_data.items():
        height, width = labels.shape
        for y in range(0, height - PATCH_SIZE + 1, STRIDE):
            for x in range(0, width - PATCH_SIZE + 1, STRIDE):
                # Keep each patch fully inside one spatial block.
                block_y = y // BLOCK_SIZE
                block_x = x // BLOCK_SIZE
                block_bottom = min((block_y + 1) * BLOCK_SIZE, height)
                block_right = min((block_x + 1) * BLOCK_SIZE, width)
                if y + PATCH_SIZE > block_bottom or x + PATCH_SIZE > block_right:
                    continue

                valid_patch = valid[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                valid_count = int(valid_patch.sum())
                if valid_count < PATCH_SIZE * PATCH_SIZE * MIN_VALID_RATIO:
                    continue

                mask_patch = labels[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                glacier_count = int(((mask_patch == 1) & valid_patch).sum())
                glacier_ratio = glacier_count / valid_count
                is_positive = glacier_ratio >= MIN_GLACIER_RATIO
                candidates.append({
                    "year": year, "y": y, "x": x,
                    "block": (block_y, block_x),
                    "positive": is_positive
                })
    return candidates

def assign_blocks(candidates):
    blocks = {}
    for item in candidates:
        stats = blocks.setdefault(item["block"], {"positive": 0, "total": 0})
        stats["total"] += 1
        stats["positive"] += int(item["positive"])

    if len(blocks) < 3:
        raise ValueError(
            f"Only {len(blocks)} usable spatial blocks found. "
            "At least 3 are needed for spatial train/val/test splits. "
            "Review patch and block settings before proceeding."
        )

    # Greedily distribute glacier-containing blocks across splits.
    targets = {"train": 0.70, "val": 0.15, "test": 0.15}
    totals = sum(v["positive"] for v in blocks.values())
    if totals == 0:
        raise ValueError("No glacier-positive patches were found.")

    ordered = sorted(
        blocks.items(),
        key=lambda item: (item[1]["positive"], item[1]["total"]),
        reverse=True
    )
    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    # Ensure each split receives a block before balancing the remainder.
    initial = ["train", "val", "test"]
    for index, (block, stats) in enumerate(ordered):
        if index < 3:
            split = initial[index]
        else:
            split = max(
                targets,
                key=lambda name: targets[name] * totals - counts[name]
            )
        assigned[split].append(block)
        counts[split] += stats["positive"]

    for split in ("train", "val", "test"):
        if counts[split] == 0:
            raise ValueError(
                f"The {split} split has no positive glacier patches. "
                "Spatially separated reference data is insufficient for "
                "a reliable three-way split. Do not train yet."
            )
    return {
        block: split
        for split, block_list in assigned.items()
        for block in block_list
    }

def main():
    year_data = {year: load_year(year) for year in YEARS}
    candidates = get_candidates(year_data)
    if not candidates:
        raise ValueError("No usable patches found.")

    block_assignment = assign_blocks(candidates)
    split_items = {"train": [], "val": [], "test": []}
    for item in candidates:
        split = block_assignment[item["block"]]
        split_items[split].append(item)

    for split, items in split_items.items():
        positives = [item for item in items if item["positive"]]
        negatives = [item for item in items if not item["positive"]]
        random.shuffle(negatives)
        negative_limit = int(len(positives) * NEGATIVE_TO_POSITIVE_RATIO)
        selected = positives + negatives[:negative_limit]
        random.shuffle(selected)

        for folder in ("images", "masks", "valid"):
            (OUTPUT_DIR / split / folder).mkdir(parents=True, exist_ok=True)

        saved_positive = 0
        saved_negative = 0

        for index, item in enumerate(selected):
            year = item["year"]
            y, x = item["y"], item["x"]
            rgb, labels, valid, _ = year_data[year]
            sl = np.s_[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            image_patch = rgb[sl]
            mask_patch = labels[sl].copy()
            valid_patch = valid[sl]

            filename = f"{year}_y{y}_x{x}_{index:05d}.png"
            Image.fromarray(image_patch, mode="RGB").save(
                OUTPUT_DIR / split / "images" / filename
            )

            # Glacier=255, background=0; invalid pixels are ignored
            # using the separate validity image.
            saved_mask = (mask_patch == 1).astype(np.uint8) * 255
            saved_mask[~valid_patch] = 0
            Image.fromarray(saved_mask, mode="L").save(
                OUTPUT_DIR / split / "masks" / filename
            )
            Image.fromarray(valid_patch.astype(np.uint8) * 255, mode="L").save(
                OUTPUT_DIR / split / "valid" / filename
            )

            if item["positive"]:
                saved_positive += 1
            else:
                saved_negative += 1

        print(
            f"{split}: {saved_positive} positive, "
            f"{saved_negative} negative, total={len(selected)}"
        )

    print("\nDataset preparation complete.")
    print(f"Output: {OUTPUT_DIR}")
    print("Reference labels: RGI inventory masks for 2023-2025.")
    print("Invalid pixels are excluded using separate valid masks.")

if __name__ == "__main__":
    main()
