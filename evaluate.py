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


class TestDataset(Dataset):
    def __init__(self, root: str, transform: transforms.Compose, labels: List[int]):
        self.root = root
        self.transform = transform
        self.labels = labels
        self.samples = self._gather_image_paths(root)

    def _gather_image_paths(self, root: str) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for index, fname in enumerate(os.listdir(root)):
            if not _is_image(fname):
                continue
            samples.append((os.path.join(root, fname), self.labels[index]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        img = self.transform(img)
        return img, label


def load_labels(csv_path: str) -> List[int]:
    labels: List[int] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
    logger.info(f"Loaded {len(labels)} labels from {csv_path}")
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake evaluation on test set with labels")
    parser.add_argument("--data_root", type=str, default="dataset", help="Root folder containing test split")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to trained checkpoint")
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="dataset/label_test.csv",
        help="Path to CSV file with ground-truth labels for the test set",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        type=str,
        default="ferretnet",
        help="Model builder to use",
    )
    return parser.parse_args()


@torch.no_grad()
def run_evaluation():
    args = parse_args()
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    labels = load_labels(args.labels_csv)

    _, eval_tf, _ = build_transforms(
        img_size=args.img_size,
    )
    test_dir = os.path.join(args.data_root, "test")
    dataset = TestDataset(test_dir, transform=eval_tf, labels=labels)
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

    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, targets in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().tolist()

        batch_targets = targets.tolist()

        all_targets.extend(batch_targets)
        all_preds.extend(preds)

    num_samples = len(all_targets)
    num_correct = sum(int(t == p) for t, p in zip(all_targets, all_preds))
    accuracy = num_correct / num_samples if num_samples > 0 else 0.0

    logger.info(f"Evaluated on {num_samples} images")
    logger.info(f"Accuracy: {accuracy * 100:.2f}% ({num_correct}/{num_samples})")


if __name__ == "__main__":
    run_evaluation()

