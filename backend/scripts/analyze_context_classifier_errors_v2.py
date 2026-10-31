from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image, ImageDraw


# ============================================================
# GLACIERAI - CONTEXT V2 CLASSIFIER ERROR ANALYSIS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_DIR
    / "datasets"
    / "classification_context_scene_split_v2"
)

TEST_DIR = DATASET_DIR / "test"

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "backend"
    / "checkpoints"
    / "glacier_classifier_context_v2.pth"
)

OUTPUT_DIR = (
    DATASET_DIR
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
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
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=transform,
)

class_names = dataset.classes


print()
print("=" * 70)
print("GLACIERAI - CONTEXT V2 CLASSIFIER ERROR ANALYSIS")
print("=" * 70)

print()
print("Device:", DEVICE)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Checkpoint:")
print(CHECKPOINT_PATH)

print()
print("Classes:")

for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print()
print("Test images:", len(dataset))


# ============================================================
# MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

errors = []
correct = []

with torch.no_grad():

    for index in range(len(dataset)):

        image_tensor, target = dataset[index]

        image_tensor = image_tensor.unsqueeze(0).to(
            DEVICE
        )

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        prediction = int(
            torch.argmax(
                probabilities
            ).item()
        )

        confidence = float(
            probabilities[prediction].item()
        )

        target_name = class_names[target]
        prediction_name = class_names[prediction]

        image_path, _ = dataset.samples[index]

        result = {
            "path": image_path,
            "actual": target_name,
            "predicted": prediction_name,
            "confidence": confidence,
            "probabilities": {
                class_names[i]:
                    float(probabilities[i].item())
                for i in range(len(class_names))
            },
        }

        if prediction != target:
            errors.append(result)
        else:
            correct.append(result)


# ============================================================
# PRINT ERRORS
# ============================================================

print()
print("=" * 70)
print("MISCLASSIFIED CONTEXT V2 TEST IMAGES")
print("=" * 70)

print()
print("Total errors:", len(errors))
print("Correct:", len(correct))
print("Total:", len(dataset))

for i, item in enumerate(errors, start=1):

    print()
    print(f"ERROR {i}")
    print("-" * 60)

    print(
        "File      :",
        Path(item["path"]).name
    )

    print(
        "Actual    :",
        item["actual"]
    )

    print(
        "Predicted :",
        item["predicted"]
    )

    print(
        "Confidence:",
        f"{item['confidence'] * 100:.2f}%"
    )

    print("Probabilities:")

    for class_name, probability in (
        item["probabilities"].items()
    ):

        print(
            f"  {class_name:25s}"
            f"{probability * 100:7.2f}%"
        )


# ============================================================
# CONTACT SHEET
# ============================================================

print()
print("=" * 70)
print("CREATING CONTEXT V2 ERROR CONTACT SHEET")
print("=" * 70)

THUMBNAIL_SIZE = 300
CELL_WIDTH = 340
CELL_HEIGHT = 390
columns = 3

rows = max(
    (len(errors) + columns - 1) // columns,
    1,
)

sheet = Image.new(
    "RGB",
    (
        columns * CELL_WIDTH,
        rows * CELL_HEIGHT,
    ),
    "white",
)

draw = ImageDraw.Draw(sheet)


for index, item in enumerate(errors):

    row = index // columns
    col = index % columns

    x = col * CELL_WIDTH
    y = row * CELL_HEIGHT

    image = Image.open(
        item["path"]
    ).convert("RGB")

    image.thumbnail(
        (
            THUMBNAIL_SIZE,
            THUMBNAIL_SIZE,
        )
    )

    image_x = (
        x
        + (CELL_WIDTH - image.width) // 2
    )

    image_y = y + 10

    sheet.paste(
        image,
        (
            image_x,
            image_y,
        ),
    )

    text_y = y + 320

    draw.text(
        (
            x + 10,
            text_y,
        ),
        f"Actual: {item['actual']}",
        fill="black",
    )

    draw.text(
        (
            x + 10,
            text_y + 20,
        ),
        f"Predicted: {item['predicted']}",
        fill="black",
    )

    draw.text(
        (
            x + 10,
            text_y + 40,
        ),
        f"Confidence: "
        f"{item['confidence'] * 100:.1f}%",
        fill="black",
    )


CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "context_classifier_errors_v2.png"
)

sheet.save(
    CONTACT_SHEET_PATH
)


# ============================================================
# TEXT REPORT
# ============================================================

REPORT_PATH = (
    OUTPUT_DIR
    / "context_classifier_errors_v2.txt"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "GlacierAI Context V2 Classifier "
        "Error Analysis\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"Dataset: {DATASET_DIR}\n"
    )

    file.write(
        f"Checkpoint: {CHECKPOINT_PATH}\n"
    )

    file.write(
        f"Total test images: {len(dataset)}\n"
    )

    file.write(
        f"Correct: {len(correct)}\n"
    )

    file.write(
        f"Total errors: {len(errors)}\n\n"
    )

    for i, item in enumerate(
        errors,
        start=1
    ):

        file.write(
            f"ERROR {i}\n"
        )

        file.write(
            f"File: {Path(item['path']).name}\n"
        )

        file.write(
            f"Actual: {item['actual']}\n"
        )

        file.write(
            f"Predicted: {item['predicted']}\n"
        )

        file.write(
            f"Confidence: "
            f"{item['confidence'] * 100:.2f}%\n"
        )

        file.write(
            "Probabilities:\n"
        )

        for class_name, probability in (
            item["probabilities"].items()
        ):

            file.write(
                f"  {class_name}: "
                f"{probability * 100:.2f}%\n"
            )

        file.write("\n")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("CONTEXT V2 ERROR ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Misclassified images:", len(errors))
print("Correct images:", len(correct))

print()
print("Contact sheet:")
print(CONTACT_SHEET_PATH)

print()
print("Text report:")
print(REPORT_PATH)

print()
print("Done.")
