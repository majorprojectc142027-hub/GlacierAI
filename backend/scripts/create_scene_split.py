from pathlib import Path
import random
import shutil

import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "classification"
)

MANIFEST_PATH = (
    SOURCE_ROOT
    / "prepared_dataset_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "classification_scene_split"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42

SPLITS = [
    "train",
    "val",
    "test",
]

SPLIT_TARGETS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

CLASS_NAMES = [
    "glacieret_snowfield",
    "ice_cap",
    "mountain_glacier",
]


# ==========================================================
# LOAD MANIFEST
# ==========================================================

df = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "rgi_id",
    "class_name",
    "split",
    "scene_id",
    "scene_datetime",
    "cloud_cover",
    "image_path",
    "image_size",
    "glacier_pixels",
    "glacier_fraction",
    "status",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Manifest missing columns: {sorted(missing)}"
    )


df = df[
    df["status"].astype(str).str.lower() == "success"
].copy()

df = df[
    df["class_name"].isin(CLASS_NAMES)
].copy()


# ==========================================================
# NORMALIZE IMAGE PATHS
# ==========================================================

def resolve_source_path(value):
    value = str(value).replace("\\", "/")

    path = Path(value)

    if path.is_absolute():
        return path

    # Manifest paths are relative to project root.
    candidate = PROJECT_ROOT / path

    if candidate.exists():
        return candidate

    # Fallback: relative to classification directory.
    candidate = SOURCE_ROOT / path

    if candidate.exists():
        return candidate

    return PROJECT_ROOT / path


df["source_path"] = df["image_path"].apply(
    resolve_source_path
)


missing_files = df[
    ~df["source_path"].apply(Path.exists)
]

if len(missing_files) > 0:

    print("ERROR: Missing source images:")

    for path in missing_files["source_path"]:
        print(path)

    raise FileNotFoundError(
        f"{len(missing_files)} source images are missing."
    )


# ==========================================================
# CHECK UNIQUE RGI
# ==========================================================

if df["rgi_id"].duplicated().any():

    duplicates = df[
        df["rgi_id"].duplicated(keep=False)
    ]

    print(duplicates)

    raise ValueError(
        "Duplicate RGI IDs detected."
    )


# ==========================================================
# SCENE INFORMATION
# ==========================================================

scene_groups = (
    df.groupby("scene_id")
)

scene_ids = list(
    scene_groups.groups.keys()
)

print("=" * 70)
print("SCENE-LEVEL CLASSIFICATION SPLIT")
print("=" * 70)

print(
    f"Total labeled images : {len(df)}"
)

print(
    f"Unique scenes        : {len(scene_ids)}"
)

print(
    "\nOriginal class distribution:"
)

print(
    df["class_name"].value_counts()
)


# ==========================================================
# BUILD SCENE TABLE
# ==========================================================

scene_rows = []

for scene_id, group in scene_groups:

    counts = {
        class_name: int(
            (group["class_name"] == class_name).sum()
        )
        for class_name in CLASS_NAMES
    }

    scene_rows.append({
        "scene_id": scene_id,
        "total": len(group),
        **counts,
    })


scene_df = pd.DataFrame(scene_rows)


# ==========================================================
# TARGET COUNTS
# ==========================================================

total_images = len(df)

target_images = {
    split: total_images * ratio
    for split, ratio in SPLIT_TARGETS.items()
}

class_totals = {
    class_name: int(
        (df["class_name"] == class_name).sum()
    )
    for class_name in CLASS_NAMES
}

target_class_images = {
    split: {
        class_name:
        class_totals[class_name] * SPLIT_TARGETS[split]
        for class_name in CLASS_NAMES
    }
    for split in SPLITS
}


# ==========================================================
# RANDOM SEARCH FOR GOOD GROUP SPLIT
# ==========================================================

def evaluate_assignment(assignment):

    split_counts = {
        split: {
            class_name: 0
            for class_name in CLASS_NAMES
        }
        for split in SPLITS
    }

    total_counts = {
        split: 0
        for split in SPLITS
    }

    scene_counts = {
        split: 0
        for split in SPLITS
    }

    for _, row in scene_df.iterrows():

        split = assignment[row["scene_id"]]

        scene_counts[split] += 1
        total_counts[split] += row["total"]

        for class_name in CLASS_NAMES:

            split_counts[split][class_name] += (
                row[class_name]
            )

    score = 0.0

    # ------------------------------------------------------
    # Overall image-count error
    # ------------------------------------------------------

    for split in SPLITS:

        score += (
            abs(
                total_counts[split]
                - target_images[split]
            )
            * 2.0
        )

    # ------------------------------------------------------
    # Per-class distribution error
    # ------------------------------------------------------

    for split in SPLITS:

        for class_name in CLASS_NAMES:

            score += (
                abs(
                    split_counts[split][class_name]
                    - target_class_images[
                        split
                    ][class_name]
                )
                * 3.0
            )

    # ------------------------------------------------------
    # Require every class in validation/test
    # ------------------------------------------------------

    for split in ["val", "test"]:

        for class_name in CLASS_NAMES:

            if split_counts[split][class_name] == 0:
                score += 10000

    # ------------------------------------------------------
    # Require at least one scene in each split
    # ------------------------------------------------------

    for split in SPLITS:

        if scene_counts[split] == 0:
            score += 10000

    return score, split_counts, total_counts, scene_counts


best = None

random.seed(SEED)

SEARCH_ITERATIONS = 50000

for _ in range(SEARCH_ITERATIONS):

    shuffled = scene_ids.copy()

    random.shuffle(shuffled)

    assignment = {}

    for scene_id in shuffled:

        r = random.random()

        if r < 0.70:
            split = "train"

        elif r < 0.85:
            split = "val"

        else:
            split = "test"

        assignment[scene_id] = split

    result = evaluate_assignment(
        assignment
    )

    score = result[0]

    if best is None or score < best[0]:

        best = (
            score,
            assignment.copy(),
            result[1],
            result[2],
            result[3],
        )


# ==========================================================
# BEST RESULT
# ==========================================================

score = best[0]
assignment = best[1]
split_class_counts = best[2]
split_total_counts = best[3]
split_scene_counts = best[4]


print("\nBest split score:", score)

print("\nScene counts:")

for split in SPLITS:

    print(
        f"  {split:5} : "
        f"{split_scene_counts[split]} scenes"
    )


print("\nImage counts:")

for split in SPLITS:

    print(
        f"  {split:5} : "
        f"{split_total_counts[split]} images"
    )


print("\nClass counts:")

for split in SPLITS:

    print(
        f"\n{split.upper()}"
    )

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name:22} "
            f"{split_class_counts[split][class_name]}"
        )


# ==========================================================
# VERIFY SCENE DISJOINTNESS
# ==========================================================

scene_sets = {
    split: {
        scene_id
        for scene_id, assigned_split
        in assignment.items()
        if assigned_split == split
    }
    for split in SPLITS
}


for i, split_a in enumerate(SPLITS):

    for split_b in SPLITS[i + 1:]:

        overlap = (
            scene_sets[split_a]
            & scene_sets[split_b]
        )

        if overlap:

            raise RuntimeError(
                f"Scene leakage detected between "
                f"{split_a} and {split_b}: "
                f"{overlap}"
            )


print(
    "\nScene overlap check: PASSED"
)


# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

if OUTPUT_ROOT.exists():

    shutil.rmtree(
        OUTPUT_ROOT
    )


for split in SPLITS:

    for class_name in CLASS_NAMES:

        (
            OUTPUT_ROOT
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ==========================================================
# COPY IMAGES
# ==========================================================

output_rows = []

for _, row in df.iterrows():

    scene_id = row["scene_id"]

    split = assignment[
        scene_id
    ]

    class_name = row["class_name"]

    source_path = Path(
        row["source_path"]
    )

    destination = (
        OUTPUT_ROOT
        / split
        / class_name
        / source_path.name
    )

    shutil.copy2(
        source_path,
        destination
    )

    new_row = row.drop(
        labels=["source_path"]
    ).to_dict()

    new_row["split"] = split

    new_row["image_path"] = str(
        destination.relative_to(
            PROJECT_ROOT
        )
    ).replace("\\", "/")

    output_rows.append(
        new_row
    )


# ==========================================================
# SAVE NEW MANIFEST
# ==========================================================

new_manifest = pd.DataFrame(
    output_rows
)

manifest_output = (
    OUTPUT_ROOT
    / "scene_split_manifest.csv"
)

new_manifest.to_csv(
    manifest_output,
    index=False
)


# ==========================================================
# FINAL VERIFICATION
# ==========================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    f"Output: {OUTPUT_ROOT}"
)

print(
    f"Manifest: {manifest_output}"
)

print(
    "\nImages by split:"
)

print(
    new_manifest.groupby(
        ["split", "class_name"]
    ).size()
)

print(
    "\nUnique scenes by split:"
)

print(
    new_manifest.groupby(
        "split"
    )["scene_id"].nunique()
)

print(
    "\nFinal scene overlap verification:"
)

for split_a in SPLITS:

    for split_b in SPLITS:

        if split_a >= split_b:
            continue

        a = set(
            new_manifest[
                new_manifest["split"] == split_a
            ]["scene_id"]
        )

        b = set(
            new_manifest[
                new_manifest["split"] == split_b
            ]["scene_id"]
        )

        print(
            f"{split_a} vs {split_b}: "
            f"{len(a & b)} overlapping scenes"
        )

print("\nDONE.")