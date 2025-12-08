import argparse
import csv
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from loguru import logger
from tqdm import tqdm

from dataset import build_transforms
from utils import build_model


IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _is_image(fname: str) -> bool:
    return fname.lower().endswith(IMG_EXTS)


class InferenceDataset(Dataset):
    def __init__(self, root: str, transform: transforms.Compose):
        self.root = root
        self.transform = transform
        self.samples: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fname in sorted(filenames):
                if _is_image(fname):
                    self.samples.append(os.path.join(dirpath, fname))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        img = self.transform(img)
        filename = os.path.basename(path)
        return img, filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake inference to result.csv")
    parser.add_argument("--data_root", type=str, default="dataset", help="Root folder containing test split")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to trained checkpoint")
    parser.add_argument("--output", type=str, default="result.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model builder to use",
    )
    return parser.parse_args()


@torch.no_grad()
def run_inference():
    args = parse_args()
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    _, eval_tf = build_transforms(
        img_size=args.img_size,
    )
    test_dir = os.path.join(args.data_root, "test")
    dataset = InferenceDataset(test_dir, transform=eval_tf)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model, num_classes=2, map_location=device).to(device)
    if os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model.eval()

    rows: List[Tuple[str, int]] = []
    for images, filenames in tqdm(loader, desc="Inference", leave=False):
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        rows.extend(zip(filenames, preds))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        for fname, label in rows:
            writer.writerow([fname, label])
    logger.info(f"Wrote predictions to {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    run_inference()

