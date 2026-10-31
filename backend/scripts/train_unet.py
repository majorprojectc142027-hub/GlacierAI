
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.models.unet import UNet
from backend.utils.dataset import GlacierDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "training_v2" / "rgi_patches"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
THRESHOLD = 0.5

if DEVICE.type == "cpu":
    torch.set_num_threads(4)

def create_dataset(split):
    base = DATA_DIR / split
    return GlacierDataset(
        base / "images",
        base / "masks",
        base / "valid"
    )

def create_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

def masked_bce_loss(logits, masks, valid):
    pixel_loss = F.binary_cross_entropy_with_logits(
        logits, masks, reduction="none"
    )
    valid_count = valid.sum()
    if valid_count.item() == 0:
        raise ValueError("Batch contains no valid pixels.")
    return (pixel_loss * valid).sum() / valid_count

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    intersection = 0
    predicted_total = 0
    actual_total = 0

    with torch.no_grad():
        for images, masks, valid in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            valid = valid.to(DEVICE)

            logits = model(images)
            loss = masked_bce_loss(logits, masks, valid)
            total_loss += loss.item()
            total_batches += 1

            predictions = (torch.sigmoid(logits) >= THRESHOLD) & (valid > 0.5)
            actual = (masks >= 0.5) & (valid > 0.5)

            intersection += (predictions & actual).sum().item()
            predicted_total += predictions.sum().item()
            actual_total += actual.sum().item()

    if total_batches == 0:
        raise ValueError("Evaluation loader is empty.")

    dice = 2 * intersection / max(predicted_total + actual_total, 1)
    iou = intersection / max(
        predicted_total + actual_total - intersection, 1
    )
    return total_loss / total_batches, dice, iou

def main():
    print(f"Device: {DEVICE}")
    train_dataset = create_dataset("train")
    val_dataset = create_dataset("val")
    test_dataset = create_dataset("test")

    if not len(train_dataset) or not len(val_dataset) or not len(test_dataset):
        raise ValueError("Train, validation and test datasets must all contain patches.")

    train_loader = create_loader(train_dataset, shuffle=True)
    val_loader = create_loader(val_dataset)
    test_loader = create_loader(test_dataset)

    print(f"Training patches: {len(train_dataset)}")
    print(f"Validation patches: {len(val_dataset)}")
    print(f"Test patches: {len(test_dataset)}")

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_unet_rgi_v1.pth"

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, masks, valid in progress:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            valid = valid.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = masked_bce_loss(logits, masks, valid)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        val_loss, val_dice, val_iou = evaluate(model, val_loader)

        print("\n" + "=" * 48)
        print(f"Epoch: {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation Dice: {val_dice:.4f}")
        print(f"Validation IoU: {val_iou:.4f}")
        print("=" * 48)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "threshold": THRESHOLD,
                "architecture": "backend.models.unet.UNet",
                "training_reference": "RGI v7.0 masks, 2023-2025",
                "input_bands": ["B04", "B03", "B02"],
                "normalization": "per-band p2-p98 over valid satellite pixels"
            }, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

    # Evaluate the selected best checkpoint on the held-out test set.
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_dice, test_iou = evaluate(model, test_loader)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU: {test_iou:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Scores measure agreement with RGI reference masks.")

if __name__ == "__main__":
    main()
