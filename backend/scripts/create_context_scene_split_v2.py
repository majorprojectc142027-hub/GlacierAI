from pathlib import Path
import csv
import shutil


# ============================================================
# GLACIERAI - CREATE CONTEXT DATASET USING EXACT V2 SCENE SPLIT
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

SOURCE_CONTEXT_DIR = (
    PROJECT_DIR
    / "datasets"
    / "classification_context"
)

V2_MANIFEST = (
    PROJECT_DIR
    / "datasets"
    / "classification_scene_split_v2"
    / "scene_split_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "datasets"
    / "classification_context_scene_split_v2"
)

CLASSES = [
    "glacieret_snowfield",
    "ice_cap",
    "mountain_glacier",
]


# ============================================================
# VALIDATION
# ============================================================

if not SOURCE_CONTEXT_DIR.exists():
    raise FileNotFoundError(
        f"Context dataset not found:\n{SOURCE_CONTEXT_DIR}"
    )

if not V2_MANIFEST.exists():
    raise FileNotFoundError(
        f"V2 manifest not found:\n{V2_MANIFEST}"
    )


print()
print("=" * 70)
print("GLACIERAI - CONTEXT DATASET V2 SCENE SPLIT")
print("=" * 70)

print()
print("Source context dataset:")
print(SOURCE_CONTEXT_DIR)

print()
print("V2 scene manifest:")
print(V2_MANIFEST)

print()
print("Output:")
print(OUTPUT_DIR)


# ============================================================
# READ V2 MANIFEST
# ============================================================

with open(
    V2_MANIFEST,
    "r",
    encoding="utf-8",
    newline="",
) as file:

    rows = list(csv.DictReader(file))


required_columns = {
    "rgi_id",
    "class_name",
    "scene_id",
    "split",
    "image_path",
}

missing_columns = required_columns - set(rows[0].keys())

if missing_columns:
    raise RuntimeError(
        f"Missing manifest columns: {sorted(missing_columns)}"
    )


print()
print("V2 manifest rows:", len(rows))


# ============================================================
# PREPARE OUTPUT
# ============================================================

if OUTPUT_DIR.exists():
    print()
    print("Removing previous V2 context split...")
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (
            OUTPUT_DIR
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# COPY USING EXACT V2 SPLIT
# ============================================================

copied = 0
missing = []

scene_sets = {
    "train": set(),
    "val": set(),
    "test": set(),
}

class_counts = {
    "train": {name: 0 for name in CLASSES},
    "val": {name: 0 for name in CLASSES},
    "test": {name: 0 for name in CLASSES},
}


for row in rows:

    rgi_id = row["rgi_id"].strip()
    class_name = row["class_name"].strip()
    scene_id = row["scene_id"].strip()
    split = row["split"].strip()

    if class_name not in CLASSES:
        raise RuntimeError(
            f"Unexpected class: {class_name}"
        )

    if split not in {"train", "val", "test"}:
        raise RuntimeError(
            f"Unexpected split: {split}"
        )

    filename = f"{rgi_id.lower()}.jpg"

    # Context dataset uses the original train/val/test layout.
    # Locate the image by class and original split.
    candidates = []

    for original_split in ["train", "val", "test"]:
        candidates.append(
            SOURCE_CONTEXT_DIR
            / original_split
            / class_name
            / filename
        )

    source_path = next(
        (
            path
            for path in candidates
            if path.exists()
        ),
        None,
    )

    if source_path is None:
        missing.append(
            {
                "rgi_id": rgi_id,
                "class_name": class_name,
                "scene_id": scene_id,
                "expected_filename": filename,
            }
        )
        continue

    destination = (
        OUTPUT_DIR
        / split
        / class_name
        / filename
    )

    shutil.copy2(
        source_path,
        destination,
    )

    copied += 1
    class_counts[split][class_name] += 1
    scene_sets[split].add(scene_id)


# ============================================================
# REPORT MISSING FILES
# ============================================================

if missing:

    print()
    print("=" * 70)
    print("MISSING CONTEXT IMAGES")
    print("=" * 70)

    for item in missing:
        print(
            f"{item['rgi_id']} | "
            f"{item['class_name']} | "
            f"{item['scene_id']} | "
            f"{item['expected_filename']}"
        )

    raise RuntimeError(
        f"\nMissing {len(missing)} context images. "
        "Dataset was not accepted as complete."
    )


# ============================================================
# VERIFY SCENE OVERLAP
# ============================================================

print()
print("=" * 70)
print("CONTEXT V2 CLASS COUNTS")
print("=" * 70)

for split in ["train", "val", "test"]:

    print()
    print(split.upper())

    total = 0

    for class_name in CLASSES:

        count = class_counts[split][class_name]
        total += count

        print(
            f"  {class_name:20s}: {count}"
        )

    print(
        f"  {'TOTAL':20s}: {total}"
    )


print()
print("=" * 70)
print("SCENE OVERLAP VERIFICATION")
print("=" * 70)

train_scenes = scene_sets["train"]
val_scenes = scene_sets["val"]
test_scenes = scene_sets["test"]

print(
    "train vs val:",
    len(train_scenes & val_scenes),
    "overlapping scenes"
)

print(
    "train vs test:",
    len(train_scenes & test_scenes),
    "overlapping scenes"
)

print(
    "val vs test:",
    len(val_scenes & test_scenes),
    "overlapping scenes"
)

if (
    train_scenes & val_scenes
    or train_scenes & test_scenes
    or val_scenes & test_scenes
):
    raise RuntimeError(
        "Scene overlap detected."
    )

print()
print("Scene overlap check: PASSED")


# ============================================================
# WRITE MANIFEST
# ============================================================

output_manifest = (
    OUTPUT_DIR
    / "context_scene_split_manifest.csv"
)

with open(
    output_manifest,
    "w",
    encoding="utf-8",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "rgi_id",
            "class_name",
            "scene_id",
            "split",
            "source_path",
            "output_path",
        ],
    )

    writer.writeheader()

    for row in rows:

        rgi_id = row["rgi_id"].strip()
        class_name = row["class_name"].strip()
        scene_id = row["scene_id"].strip()
        split = row["split"].strip()

        filename = f"{rgi_id.lower()}.jpg"

        source_path = None

        for original_split in ["train", "val", "test"]:
            candidate = (
                SOURCE_CONTEXT_DIR
                / original_split
                / class_name
                / filename
            )

            if candidate.exists():
                source_path = candidate
                break

        output_path = (
            OUTPUT_DIR
            / split
            / class_name
            / filename
        )

        writer.writerow(
            {
                "rgi_id": rgi_id,
                "class_name": class_name,
                "scene_id": scene_id,
                "split": split,
                "source_path": (
                    str(source_path)
                    if source_path
                    else ""
                ),
                "output_path": str(output_path),
            }
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("CONTEXT V2 DATASET CREATED")
print("=" * 70)

print()
print("Total images:", copied)

print("Train scenes:", len(train_scenes))
print("Val scenes  :", len(val_scenes))
print("Test scenes :", len(test_scenes))

print()
print("Output:")
print(OUTPUT_DIR)

print()
print("Manifest:")
print(output_manifest)

print()
print("DONE.")
