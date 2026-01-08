import argparse
import csv
import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from loguru import logger
from tqdm import tqdm

from dataset import build_transforms
from utils import build_model, is_image, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_desc: Optional[str] = None,
) -> Tuple[float, float]:
    model.eval()
    running_loss, running_acc, total = 0.0, 0.0, 0
    for images, labels in tqdm(loader, desc=epoch_desc, leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size
        running_acc += accuracy(logits, labels) * batch_size

    return running_loss / total, running_acc / total


class TestDataset(Dataset):
    def __init__(self, root: str, transform: transforms.Compose, label_path: Optional[str] = None):
        self.root = root
        self.transform = transform
        self.labels = self.load_labels(label_path)
        self.samples = self._gather_image_paths(root)

    def load_labels(self, label_path: Optional[str] = None) -> List[int]:
        if label_path is None:
            label_path = os.path.join(os.path.dirname(self.root), "label_test.csv")
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        labels: List[int] = []
        with open(label_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["label"]))
        return labels

    def _gather_image_paths(self, root: str) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for index, fname in enumerate(sorted(os.listdir(root))):
            if not is_image(fname):
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
        return img, torch.tensor(label, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake evaluation on test set with labels")
    parser.add_argument("--data_root", type=str, default="dataset", help="Root folder containing test split")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to trained checkpoint")
    parser.add_argument(
        "--label_path",
        type=str,
        default=None,
        help="Path to label file for the test set",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        type=str,
        default="asrnet",
        help="Model builder to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    _, eval_tf = build_transforms(
        img_size=args.img_size,
    )
    test_dir = os.path.join(args.data_root, "test")
    dataset = TestDataset(test_dir, transform=eval_tf, label_path=args.label_path)
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
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion, device, "Evaluating")
    logger.info(f"Evaluated on {len(loader.dataset)} images")
    logger.info(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

