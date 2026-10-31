from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

# IMPORTANT:
# This is the leakage-free scene-level dataset created by
# create_scene_split.py.
DATASET_DIR = PROJECT_DIR / "datasets" / "classification_scene_split"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

CHECKPOINT_DIR = PROJECT_DIR / "backend" / "checkpoints"

# Keep the original classifier untouched.
# This experiment gets its own checkpoint.
MODEL_PATH = CHECKPOINT_DIR / "glacier_classifier_scene_v1.pth"
METRICS_PATH = CHECKPOINT_DIR / "glacier_classifier_scene_v1_metrics.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

PATIENCE = 7

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DIRECTORIES
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0),
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# DATASET VALIDATION
# ============================================================

print()
print("=" * 70)
print("GlacierAI - Scene-Level Glacier Type CNN Classifier")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Dataset directory:")
print(DATASET_DIR)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# Verify scene-level dataset exists
# ------------------------------------------------------------

required_directories = [
    DATASET_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
]

for directory in required_directories:
    if not directory.exists():
        raise FileNotFoundError(
            f"\nRequired directory does not exist:\n{directory}\n"
            f"\nRun create_scene_split.py first."
        )


# ============================================================
# DATASETS
# ============================================================

print()
print("Loading scene-level datasets...")

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=eval_transform,
)

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=eval_transform,
)


class_names = train_dataset.classes


print()
print("Classes:")

for index, name in enumerate(class_names):
    print(f"  {index}: {name}")


print()
print("Dataset sizes:")
print("  Train      :", len(train_dataset))
print("  Validation :", len(val_dataset))
print("  Test       :", len(test_dataset))


# ============================================================
# VERIFY CLASS CONSISTENCY
# ============================================================

if train_dataset.classes != val_dataset.classes:
    raise RuntimeError(
        "Training and validation classes do not match."
    )

if train_dataset.classes != test_dataset.classes:
    raise RuntimeError(
        "Training and test classes do not match."
    )


# ============================================================
# VERIFY EXPECTED SCENE-LEVEL DATASET
# ============================================================

expected_classes = [
    "glacieret_snowfield",
    "ice_cap",
    "mountain_glacier",
]

if class_names != expected_classes:
    raise RuntimeError(
        "\nUnexpected class structure.\n"
        f"Expected: {expected_classes}\n"
        f"Found:    {class_names}"
    )


# ============================================================
# CLASS COUNTS
# ============================================================

train_targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    train_targets,
    minlength=len(class_names),
)


print()
print("Training class distribution:")

for index, name in enumerate(class_names):
    print(
        f"  {name}: "
        f"{class_counts[index]} images"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_samples = len(train_dataset)
num_classes = len(class_names)

if np.any(class_counts == 0):
    raise RuntimeError(
        "At least one class has zero training samples."
    )

class_weights = (
    total_samples
    / (
        num_classes
        * class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)


print()
print("Class weights:")

for index, name in enumerate(class_names):
    print(
        f"  {name}: "
        f"{class_weights[index].item():.4f}"
    )


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading ResNet18...")

try:

    weights = ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    print(
        "Pretrained ImageNet weights loaded."
    )

except Exception as error:

    print()
    print(
        "WARNING: Could not load pretrained weights."
    )

    print(
        "Reason:",
        error,
    )

    print(
        "Using ResNet18 without pretrained weights."
    )

    model = models.resnet18(
        weights=None
    )


# ------------------------------------------------------------
# Replace final classification layer
# ------------------------------------------------------------

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes,
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    predictions,
    num_classes,
):

    confusion = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    for target, prediction in zip(
        targets,
        predictions,
    ):

        confusion[
            target,
            prediction
        ] += 1


    total = confusion.sum()

    correct = np.trace(
        confusion
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )


    precision_values = []
    recall_values = []
    f1_values = []


    for class_index in range(
        num_classes
    ):

        true_positive = confusion[
            class_index,
            class_index
        ]

        false_positive = (
            confusion[:, class_index].sum()
            - true_positive
        )

        false_negative = (
            confusion[class_index, :].sum()
            - true_positive
        )


        precision = (
            true_positive
            / (
                true_positive
                + false_positive
            )
            if (
                true_positive
                + false_positive
            ) > 0
            else 0.0
        )


        recall = (
            true_positive
            / (
                true_positive
                + false_negative
            )
            if (
                true_positive
                + false_negative
            ) > 0
            else 0.0
        )


        if precision + recall > 0:

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0


        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        f1_values.append(
            f1
        )


    macro_precision = float(
        np.mean(
            precision_values
        )
    )

    macro_recall = float(
        np.mean(
            recall_values
        )
    )

    macro_f1 = float(
        np.mean(
            f1_values
        )
    )


    return {
        "accuracy": float(
            accuracy
        ),

        "precision": macro_precision,

        "recall": macro_recall,

        "macro_f1": macro_f1,

        "confusion_matrix":
            confusion.tolist(),

        "per_class": {
            class_names[index]: {
                "precision": float(
                    precision_values[index]
                ),

                "recall": float(
                    recall_values[index]
                ),

                "f1": float(
                    f1_values[index]
                ),
            }

            for index in range(
                num_classes
            )
        },
    }


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []


    for images, targets in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            targets
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1,
        )


        all_targets.extend(
            targets.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )


    epoch_loss = (
        running_loss
        / len(train_dataset)
    )


    metrics = calculate_metrics(
        all_targets,
        all_predictions,
        num_classes,
    )


    return epoch_loss, metrics


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate():

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []


    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True,
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                targets
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1,
            )


            all_targets.extend(
                targets.cpu()
                .numpy()
                .tolist()
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )


    epoch_loss = (
        running_loss
        / len(val_dataset)
    )


    metrics = calculate_metrics(
        all_targets,
        all_predictions,
        num_classes,
    )


    return epoch_loss, metrics


# ============================================================
# TEST FUNCTION
# ============================================================

def test_model():

    model.eval()

    all_targets = []
    all_predictions = []


    with torch.no_grad():

        for images, targets in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )


            outputs = model(
                images
            )


            predictions = torch.argmax(
                outputs,
                dim=1,
            )


            all_targets.extend(
                targets.numpy()
                .tolist()
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )


    return calculate_metrics(
        all_targets,
        all_predictions,
        num_classes,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("=" * 70)
print("STARTING SCENE-LEVEL TRAINING")
print("=" * 70)

print()
print("Training images      :", len(train_dataset))
print("Validation images    :", len(val_dataset))
print("Test images          :", len(test_dataset))
print("Epochs               :", NUM_EPOCHS)
print("Batch size           :", BATCH_SIZE)
print("Learning rate        :", LEARNING_RATE)
print("Weight decay         :", WEIGHT_DECAY)
print("Random seed          :", SEED)

print()
print("Checkpoint:")
print(MODEL_PATH)


best_val_f1 = -1.0

epochs_without_improvement = 0


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    train_loss, train_metrics = (
        train_one_epoch()
    )

    val_loss, val_metrics = (
        evaluate()
    )


    scheduler.step(
        val_metrics["macro_f1"]
    )


    current_lr = optimizer.param_groups[
        0
    ]["lr"]


    print()
    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS}"
    )

    print(
        f"  Learning Rate : "
        f"{current_lr:.7f}"
    )

    print(
        f"  Train Loss    : "
        f"{train_loss:.4f}"
    )

    print(
        f"  Train Accuracy: "
        f"{train_metrics['accuracy']:.4f}"
    )

    print(
        f"  Val Loss      : "
        f"{val_loss:.4f}"
    )

    print(
        f"  Val Accuracy  : "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"  Val Macro F1  : "
        f"{val_metrics['macro_f1']:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if (
        val_metrics["macro_f1"]
        > best_val_f1
    ):

        best_val_f1 = (
            val_metrics["macro_f1"]
        )

        epochs_without_improvement = 0


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "class_names":
                class_names,

            "image_size":
                IMAGE_SIZE,

            "best_val_macro_f1":
                best_val_f1,

            "normalization": {

                "mean": [
                    0.485,
                    0.456,
                    0.406,
                ],

                "std": [
                    0.229,
                    0.224,
                    0.225,
                ],
            },

            "model":
                "ResNet18",

            "dataset":
                "classification_scene_split",

            "scene_level_split":
                True,

            "seed":
                SEED,
        }


        torch.save(
            checkpoint,
            MODEL_PATH,
        )


        print(
            "  ✓ Best scene-level model saved."
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST SCENE-LEVEL MODEL")
print("=" * 70)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# FINAL TEST
# ============================================================

test_metrics = test_model()


print()
print("=" * 70)
print("FINAL SCENE-LEVEL TEST RESULTS")
print("=" * 70)

print()

print(
    f"Test Accuracy : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test Precision: "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Test Recall   : "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"Test Macro F1 : "
    f"{test_metrics['macro_f1']:.4f}"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print("Per-class results:")

for class_name, values in (
    test_metrics["per_class"]
    .items()
):

    print()
    print(
        f"  {class_name}"
    )

    print(
        f"    Precision: "
        f"{values['precision']:.4f}"
    )

    print(
        f"    Recall   : "
        f"{values['recall']:.4f}"
    )

    print(
        f"    F1       : "
        f"{values['f1']:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix:")

print(
    "Rows = actual class"
)

print(
    "Columns = predicted class"
)

print()

print(
    "             "
    + " ".join(
        f"{name[:12]:>14}"
        for name in class_names
    )
)


for index, row in enumerate(
    test_metrics["confusion_matrix"]
):

    print(
        f"{class_names[index][:12]:>12} "
        + " ".join(
            f"{value:>14}"
            for value in row
        )
    )


# ============================================================
# SAVE METRICS
# ============================================================

results = {

    "experiment":
        "scene_level_classification_v1",

    "classes":
        class_names,

    "dataset": {

        "name":
            "classification_scene_split",

        "scene_level_split":
            True,

        "train":
            len(train_dataset),

        "validation":
            len(val_dataset),

        "test":
            len(test_dataset),
    },

    "split":

        {
            "train_scenes":
                30,

            "validation_scenes":
                7,

            "test_scenes":
                5,

            "scene_overlap":
                False,
        },

    "best_validation_macro_f1":
        float(best_val_f1),

    "test_metrics":
        test_metrics,

    "model":
        "ResNet18",

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_configured":
        NUM_EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "device":
        str(DEVICE),

    "seed":
        SEED,

    "checkpoint":
        str(MODEL_PATH),
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=4,
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("SCENE-LEVEL CNN TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

print()

print(
    "Metrics saved to:"
)

print(
    METRICS_PATH
)

print()

print(
    "Scene-level classification experiment completed."
)

print(
    "The original glacier_classifier_v3.pth was not modified."
)

print()