from pathlib import Path
import random
import shutil

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_DIR
    / "datasets"
    / "classification"
    / "prepared_dataset_manifest.csv"
)

SOURCE_ROOT = (
    PROJECT_DIR
    / "datasets"
    / "classification"
)

OUTPUT_ROOT = (
    PROJECT_DIR
    / "datasets"
    / "classification_scene_split_v2"
)

SEED = 42

TRAIN_TARGET = 0.70
VAL_TARGET = 0.15
TEST_TARGET = 0.15


VALID_CLASSES = {
    "glacieret_snowfield",
    "ice_cap",
    "mountain_glacier",
}


# ============================================================
# LOAD MANIFEST
# ============================================================

print()
print("=" * 70)
print("GLACIERAI - SCENE-LEVEL SPLIT V2")
print("=" * 70)

df = pd.read_csv(MANIFEST_PATH)

df = df[
    df["status"].astype(str).str.lower() == "success"
].copy()

df = df[
    df["class_name"].isin(VALID_CLASSES)
].copy()


print()
print("Total labeled images:", len(df))
print("Unique scenes:", df["scene_id"].nunique())


# ============================================================
# VALIDATE
# ============================================================

if df["rgi_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate RGI IDs found."
    )

if df["scene_id"].isna().any():
    raise RuntimeError(
        "Missing scene IDs found."
    )


# ============================================================
# RESOLVE IMAGE PATH
# ============================================================

def resolve_image_path(row):

    raw = str(row["image_path"])

    candidate = Path(raw)

    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidates = [
        PROJECT_DIR / raw,
        SOURCE_ROOT / raw,
        SOURCE_ROOT / candidate.name,
    ]

    for path in candidates:

        if path.exists():
            return path

    # Search by filename as final fallback
    matches = list(
        SOURCE_ROOT.rglob(candidate.name)
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Image not found: {raw}"
    )


df["resolved_path"] = df.apply(
    resolve_image_path,
    axis=1,
)


# ============================================================
# SCENE GROUPS
# ============================================================

scene_groups = []

for scene_id, group in df.groupby(
    "scene_id"
):

    counts = (
        group["class_name"]
        .value_counts()
        .to_dict()
    )

    scene_groups.append({
        "scene_id": scene_id,
        "count": len(group),
        "glacieret": counts.get(
            "glacieret_snowfield",
            0
        ),
        "ice_cap": counts.get(
            "ice_cap",
            0
        ),
        "mountain": counts.get(
            "mountain_glacier",
            0
        ),
    })


# ============================================================
# TARGETS
# ============================================================

total_images = len(df)

target_sizes = {
    "train":
        total_images * TRAIN_TARGET,

    "val":
        total_images * VAL_TARGET,

    "test":
        total_images * TEST_TARGET,
}


class_totals = {
    "glacieret":
        int(
            (df["class_name"]
             == "glacieret_snowfield").sum()
        ),

    "ice_cap":
        int(
            (df["class_name"]
             == "ice_cap").sum()
        ),

    "mountain":
        int(
            (df["class_name"]
             == "mountain_glacier").sum()
        ),
}


target_class = {
    key: {
        "train":
            value * TRAIN_TARGET,

        "val":
            value * VAL_TARGET,

        "test":
            value * TEST_TARGET,
    }

    for key, value in class_totals.items()
}


# ============================================================
# SCORING
# ============================================================

def evaluate_assignment(
    assignment
):

    counts = {
        split: {
            "images": 0,
            "glacieret": 0,
            "ice_cap": 0,
            "mountain": 0,
            "scenes": 0,
        }

        for split in [
            "train",
            "val",
            "test",
        ]
    }


    for scene in scene_groups:

        split = assignment[
            scene["scene_id"]
        ]

        counts[split]["images"] += (
            scene["count"]
        )

        counts[split]["glacieret"] += (
            scene["glacieret"]
        )

        counts[split]["ice_cap"] += (
            scene["ice_cap"]
        )

        counts[split]["mountain"] += (
            scene["mountain"]
        )

        counts[split]["scenes"] += 1


    score = 0.0


    # --------------------------------------------------------
    # Image-count balance
    # --------------------------------------------------------

    for split in counts:

        expected = target_sizes[
            split
        ]

        actual = counts[
            split
        ]["images"]

        score += (
            abs(actual - expected)
            * 2.0
        )


    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    for class_name in [
        "glacieret",
        "ice_cap",
        "mountain",
    ]:

        for split in counts:

            actual = counts[
                split
            ][class_name]

            expected = target_class[
                class_name
            ][split]

            score += (
                abs(actual - expected)
                * 4.0
            )


    # --------------------------------------------------------
    # Require all classes in val/test
    # --------------------------------------------------------

    for split in [
        "val",
        "test",
    ]:

        for class_name in [
            "glacieret",
            "ice_cap",
            "mountain",
        ]:

            if (
                counts[split][class_name]
                == 0
            ):

                score += 1000


    # --------------------------------------------------------
    # Avoid oversized test scene dominance
    # --------------------------------------------------------

    test_scene_sizes = []

    for scene in scene_groups:

        if (
            assignment[
                scene["scene_id"]
            ]
            == "test"
        ):

            test_scene_sizes.append(
                scene["count"]
            )


    if test_scene_sizes:

        largest_test_scene = max(
            test_scene_sizes
        )

        # Penalize if one scene contributes
        # more than 40% of the test set.
        test_images = counts[
            "test"
        ]["images"]

        if test_images > 0:

            fraction = (
                largest_test_scene
                / test_images
            )

            if fraction > 0.40:

                score += (
                    fraction
                    * 100
                )


    return score, counts


# ============================================================
# RANDOM SEARCH
# ============================================================

random.seed(SEED)

best_score = float("inf")
best_assignment = None
best_counts = None

scene_ids = [
    scene["scene_id"]
    for scene in scene_groups
]


print()
print("Searching for balanced scene assignment...")


for attempt in range(100000):

    shuffled = scene_ids.copy()

    random.shuffle(
        shuffled
    )


    # Randomly choose scene counts,
    # then evaluate actual image/class balance.

    assignment = {}

    # Random proportions with approximately
    # 70 / 15 / 15 scene distribution.
    n = len(shuffled)

    train_end = round(
        n * 0.70
    )

    val_end = train_end + round(
        n * 0.15
    )


    for scene_id in shuffled[
        :train_end
    ]:

        assignment[
            scene_id
        ] = "train"


    for scene_id in shuffled[
        train_end:val_end
    ]:

        assignment[
            scene_id
        ] = "val"


    for scene_id in shuffled[
        val_end:
    ]:

        assignment[
            scene_id
        ] = "test"


    score, counts = evaluate_assignment(
        assignment
    )


    if score < best_score:

        best_score = score

        best_assignment = (
            assignment.copy()
        )

        best_counts = counts


print()
print(
    f"Best split score: "
    f"{best_score:.4f}"
)


# ============================================================
# FINAL COUNTS
# ============================================================

print()
print("Scene counts:")

for split in [
    "train",
    "val",
    "test",
]:

    print(
        f"  {split:5s}: "
        f"{best_counts[split]['scenes']} scenes"
    )


print()
print("Image counts:")

for split in [
    "train",
    "val",
    "test",
]:

    print(
        f"  {split:5s}: "
        f"{best_counts[split]['images']} images"
    )


print()
print("Class counts:")

for split in [
    "train",
    "val",
    "test",
]:

    print()
    print(
        split.upper()
    )

    print(
        "  glacieret_snowfield :",
        best_counts[split][
            "glacieret"
        ]
    )

    print(
        "  ice_cap            :",
        best_counts[split][
            "ice_cap"
        ]
    )

    print(
        "  mountain_glacier   :",
        best_counts[split][
            "mountain"
        ]
    )


# ============================================================
# REMOVE OLD V2 OUTPUT
# ============================================================

if OUTPUT_ROOT.exists():

    print()
    print(
        "Removing previous V2 dataset..."
    )

    shutil.rmtree(
        OUTPUT_ROOT
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in [
    "train",
    "val",
    "test",
]:

    for class_name in sorted(
        VALID_CLASSES
    ):

        (
            OUTPUT_ROOT
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# COPY IMAGES
# ============================================================

output_rows = []


for _, row in df.iterrows():

    scene_id = row["scene_id"]

    split = best_assignment[
        scene_id
    ]

    class_name = row[
        "class_name"
    ]

    source = Path(
        row["resolved_path"]
    )

    destination = (
        OUTPUT_ROOT
        / split
        / class_name
        / source.name
    )


    shutil.copy2(
        source,
        destination,
    )


    output_rows.append({

        "rgi_id":
            row["rgi_id"],

        "class_name":
            class_name,

        "scene_id":
            scene_id,

        "split":
            split,

        "image_path":
            str(
                destination.relative_to(
                    PROJECT_DIR
                )
            ),
    })


# ============================================================
# SAVE MANIFEST
# ============================================================

output_df = pd.DataFrame(
    output_rows
)

manifest_output = (
    OUTPUT_ROOT
    / "scene_split_manifest.csv"
)

output_df.to_csv(
    manifest_output,
    index=False,
)


# ============================================================
# VERIFY SCENE OVERLAP
# ============================================================

print()
print("=" * 70)
print("SCENE OVERLAP VERIFICATION")
print("=" * 70)


scene_sets = {
    split: set(
        output_df[
            output_df["split"] == split
        ]["scene_id"]
    )

    for split in [
        "train",
        "val",
        "test",
    ]
}


overlap_found = False


for a, b in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]:

    overlap = (
        scene_sets[a]
        & scene_sets[b]
    )

    print(
        f"{a} vs {b}: "
        f"{len(overlap)} overlapping scenes"
    )

    if overlap:
        overlap_found = True


if overlap_found:

    raise RuntimeError(
        "Scene overlap detected."
    )


print()
print("Scene overlap check: PASSED")


# ============================================================
# TEST SCENE DOMINANCE
# ============================================================

print()
print("Test scene distribution:")

test_scene_counts = (
    output_df[
        output_df["split"] == "test"
    ]
    .groupby("scene_id")
    .size()
    .sort_values(
        ascending=False
    )
)

print(
    test_scene_counts.to_string()
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("SCENE-LEVEL V2 DATASET CREATED")
print("=" * 70)

print()
print("Output:")
print(OUTPUT_ROOT)

print()
print("Manifest:")
print(manifest_output)

print()
print("Total images:")
print(len(output_df))

print()
print("Unique scenes:")
print(
    output_df["scene_id"].nunique()
)

print()
print("DONE.")