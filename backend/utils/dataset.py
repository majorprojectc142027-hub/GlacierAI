
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class GlacierDataset(Dataset):
    def __init__(self, image_dir, mask_dir, valid_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.valid_dir = Path(valid_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image folder missing: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask folder missing: {self.mask_dir}")
        if not self.valid_dir.is_dir():
            raise FileNotFoundError(f"Validity folder missing: {self.valid_dir}")
        self.images = sorted(self.image_dir.glob("*.png"))
        for image_path in self.images:
            for folder in (self.mask_dir, self.valid_dir):
                if not (folder / image_path.name).is_file():
                    raise FileNotFoundError(
                        f"Matching file missing: {folder / image_path.name}"
                    )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.mask_dir / image_path.name
        valid_path = self.valid_dir / image_path.name

        image = np.asarray(
            Image.open(image_path).convert("RGB"), dtype=np.float32
        ) / 255.0
        mask = np.asarray(
            Image.open(mask_path).convert("L"), dtype=np.uint8
        )
        valid = np.asarray(
            Image.open(valid_path).convert("L"), dtype=np.uint8
        )

        if image.shape[:2] != mask.shape or mask.shape != valid.shape:
            raise ValueError(f"Image, mask and valid dimensions differ: {image_path.name}")

        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
        mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        valid = torch.from_numpy((valid > 127).astype(np.float32)).unsqueeze(0)

        return image, mask, valid
