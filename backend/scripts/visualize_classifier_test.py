from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "classification"
    / "test"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "classification"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CLASSES = [
    "glacieret_snowfield",
    "ice_cap",
    "mountain_glacier",
]

IMAGE_SIZE = 224
LABEL_HEIGHT = 45
COLUMNS = 5


for class_name in CLASSES:

    class_dir = TEST_DIR / class_name

    images = sorted(
        class_dir.glob("*.jpg")
    )

    if not images:
        images = sorted(
            class_dir.glob("*.png")
        )

    if not images:
        print(
            f"No images found for {class_name}"
        )
        continue

    rows = (
        len(images) + COLUMNS - 1
    ) // COLUMNS

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * IMAGE_SIZE,
            rows * (IMAGE_SIZE + LABEL_HEIGHT),
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(images):

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        row = index // COLUMNS
        col = index % COLUMNS

        x = col * IMAGE_SIZE
        y = row * (
            IMAGE_SIZE + LABEL_HEIGHT
        )

        sheet.paste(
            image,
            (x, y),
        )

        label = image_path.stem

        draw.text(
            (
                x + 3,
                y + IMAGE_SIZE + 3,
            ),
            label,
            fill="black",
        )

    output_path = (
        OUTPUT_DIR
        / f"test_{class_name}_contact_sheet.png"
    )

    sheet.save(output_path)

    print(
        f"Saved: {output_path}"
    )