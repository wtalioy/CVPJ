import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class DataConfig:
    data_root: str = "dataset"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 16
    val_split: float = 0.2  # fraction of train set used for validation
    seed: int = 42



def _is_image(fname: str) -> bool:
    return fname.lower().endswith(IMG_EXTS)


def _gather_image_paths(root: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    for cls_name, label in (("0_real", 0), ("1_fake", 1)):
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if _is_image(fname):
                samples.append((os.path.join(cls_dir, fname), label))
    return samples


class DeepFakeDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        self.root = root
        self.transform = transform
        self.samples = _gather_image_paths(root)
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root}. Expected 0_real/ and 1_fake/ subfolders.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class RandomJPEG:
    def __init__(self, quality=95, interval=1, p=0.1):
        if isinstance(quality, tuple):
            self.quality = [i for i in range(quality[0], quality[1]) if i % interval == 0]
        else:
            self.quality = quality
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            if isinstance(self.quality, list):
                quality = int(torch.tensor(self.quality).flatten()[torch.randint(len(self.quality), (1,)).item()].item())
            else:
                quality = self.quality
            buffer = torch.jit.annotate(torch.Tensor, None)
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            from PIL import Image as _Image
            img = _Image.open(buffer)
        return img


class RandomGaussianBlur:
    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=1.0):
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return self.blur(img)
        return img


class RandomMask(object):
    def __init__(self, ratio=0.5, patch_size=16, p=0.5):
        if isinstance(ratio, float):
            self.fixed_ratio = True
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("Ratio must be a float or a tuple of two floats.")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() > self.p:
            return tensor

        _, h, w = tensor.shape
        mask = torch.ones((h, w), dtype=torch.float32)

        if self.fixed_ratio:
            ratio = self.ratio[0]
        else:
            ratio = torch.empty(1).uniform_(self.ratio[0], self.ratio[1]).item()

        num_masks = int((h * w * ratio) / (self.patch_size ** 2))
        selected_positions = set()
        while len(selected_positions) < num_masks:
            top = torch.randint(0, (h // self.patch_size), (1,)).item() * self.patch_size
            left = torch.randint(0, (w // self.patch_size), (1,)).item() * self.patch_size
            selected_positions.add((top, left))

        for (top, left) in selected_positions:
            mask[top:top + self.patch_size, left:left + self.patch_size] = 0

        return tensor * mask.expand_as(tensor)


def build_transforms(
    img_size: int = 224,
) -> Tuple[Callable, Callable]:
    transform_train = transforms.Compose([
        transforms.RandomCrop([img_size, img_size], pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),
    ])

    transform_eval = transforms.Compose([
        transforms.CenterCrop([img_size, img_size]),
        transforms.ToTensor(),
    ])
    return transform_train, transform_eval


def create_dataloaders(
    cfg: DataConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_dir = os.path.join(cfg.data_root, "train")

    train_tf, eval_tf = build_transforms(
        img_size=cfg.img_size,
    )

    base_dataset = DeepFakeDataset(train_dir, transform=None)

    if 0.0 < cfg.val_split < 1.0:
        n_total = len(base_dataset)
        n_val = int(round(cfg.val_split * n_total))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(cfg.seed)
        train_subset, val_subset = random_split(base_dataset, [n_train, n_val], generator=generator)
        train_ds = _SubsetWithTransform(train_subset, train_tf)
        val_ds = _SubsetWithTransform(val_subset, eval_tf)
    else:
        train_ds = _SubsetWithTransform(base_dataset, train_tf)
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


class _SubsetWithTransform(Dataset):
    def __init__(self, subset: torch.utils.data.Subset, transform: Callable):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        img, label = self.subset[index]
        img = self.transform(img) if self.transform else img
        return img, label


if __name__ == "__main__":
    cfg = DataConfig()
    train_loader, val_loader = create_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")

