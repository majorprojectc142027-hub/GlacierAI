from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets" / "classification" / "test"

MODEL_PATH = PROJECT_ROOT / "checkpoints" / "glacier_classifier_v3.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "glacieret_snowfield",
    "ice_cap",
    "mountain_glacier",
]


TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=TRANSFORM,
)

print("=" * 70)
print("GLACIER CLASSIFIER V3 - TEST EVALUATION")
print("=" * 70)

print("Device:", DEVICE)
print("Test images:", len(dataset))
print("Classes:", dataset.classes)


# ==========================================================
# MODEL
# ==========================================================

model = resnet18(weights=None)

in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.35),
    nn.Linear(
        in_features,
        len(CLASS_NAMES),
    ),
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

print("\nCheckpoint version:", checkpoint.get("version"))
print("Checkpoint classes:", checkpoint.get("classes"))
print("Checkpoint image size:", checkpoint.get("img_size"))

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)


# ==========================================================
# PREDICTIONS
# ==========================================================

all_targets = []
all_predictions = []

image_results = []


with torch.no_grad():

    for images, targets in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        for i in range(len(images)):

            actual_index = int(targets[i].item())

            predicted_index = int(
                predictions[i].item()
            )

            confidence = float(
                probabilities[i][predicted_index].item()
            )

            image_path = dataset.samples[
                len(image_results)
            ][0]

            image_results.append({
                "filename": Path(image_path).name,
                "actual": CLASS_NAMES[actual_index],
                "predicted": CLASS_NAMES[predicted_index],
                "confidence": confidence,
                "correct": actual_index == predicted_index,
            })

        all_targets.extend(
            targets.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )


# ==========================================================
# IMAGE-LEVEL RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("IMAGE-LEVEL PREDICTIONS")
print("=" * 70)

for result in image_results:

    status = "CORRECT" if result["correct"] else "WRONG"

    print(
        f"{status:7} | "
        f"Actual: {result['actual']:22} | "
        f"Predicted: {result['predicted']:22} | "
        f"Confidence: {result['confidence'] * 100:6.2f}% | "
        f"{result['filename']}"
    )


# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(
    all_targets,
    all_predictions,
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_targets,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
)

macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
)


print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Accuracy        : {accuracy:.4f}")
print(f"Macro Precision : {macro_precision:.4f}")
print(f"Macro Recall    : {macro_recall:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")


print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    print(f"\n{class_name}")
    print(f"  Precision : {precision[i]:.4f}")
    print(f"  Recall    : {recall[i]:.4f}")
    print(f"  F1 Score  : {f1[i]:.4f}")
    print(f"  Support   : {support[i]}")


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("Rows = Actual")
print("Columns = Predicted\n")

print(
    f"{'':24}"
    f"{CLASS_NAMES[0]:>20}"
    f"{CLASS_NAMES[1]:>20}"
    f"{CLASS_NAMES[2]:>20}"
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:24}"
        f"{cm[i,0]:>20}"
        f"{cm[i,1]:>20}"
        f"{cm[i,2]:>20}"
    )


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

print("=" * 70)