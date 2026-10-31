from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "checkpoints"
    / "glacier_classifier_scene_v1.pth"
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


CLASS_NAMES = [
    "glacieret_snowfield",
    "ice_cap",
    "mountain_glacier",
]


DISPLAY_NAMES = {
    "glacieret_snowfield": "Glacieret / Snowfield",
    "ice_cap": "Ice Cap",
    "mountain_glacier": "Mountain Glacier",
}


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])






def build_model():
    model = resnet18(weights=None)
    in_features = model.fc.in_features

    # Match the existing scene_v1 checkpoint architecture.
    model.fc = nn.Linear(
        in_features,
        len(CLASS_NAMES),
    )

    return model


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ResNet18 V3 classifier checkpoint not found:\n{MODEL_PATH}"
    )


_checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model = build_model()

if "model_state_dict" in _checkpoint:
    model.load_state_dict(
        _checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(_checkpoint)

model = model.to(DEVICE)

model.eval()


def classify_glacier(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    tensor = TRANSFORM(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index].item()
    )

    probability_dict = {}

    for index, class_name in enumerate(CLASS_NAMES):

        probability_dict[class_name] = round(
            float(probabilities[index].item()),
            4,
        )

    if confidence >= 0.80:
        confidence_level = "High"

    elif confidence >= 0.60:
        confidence_level = "Moderate"

    else:
        confidence_level = "Low"

    return {
        "class": predicted_class,

        "display_name": DISPLAY_NAMES[
            predicted_class
        ],

        "confidence": round(
            confidence,
            4,
        ),

        "confidence_percent": round(
            confidence * 100,
            2,
        ),

        "confidence_level": confidence_level,

        "probabilities": probability_dict,

        "source": "ResNet18 V3",
    }