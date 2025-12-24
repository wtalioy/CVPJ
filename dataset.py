import os
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from loguru import logger


IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
def _is_image(fname: str) -> bool:
    return fname.lower().endswith(IMG_EXTS)

@dataclass
class DataConfig:
    data_root: str = "dataset"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 16
    seed: int = 42


class LocalDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        self.transform = transform
        self.samples = self._gather_image_paths(root)
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

    def _gather_image_paths(self, root: str) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for cls_name, label in (("0_real", 0), ("1_fake", 1)):
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if _is_image(fname):
                    samples.append((os.path.join(cls_dir, fname), label))
        return samples


class GenImageDataset(Dataset):
    def __init__(self, base_dir: str, dataset_names: List[str], split: str = "train",
                 transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for name in dataset_names:
            root = os.path.join(base_dir, name, split)
            if not os.path.isdir(root):
                logger.warning(f"Split '{split}' not found for {name} at {root}; skipping.")
                continue

            class_map = {"nature": 0, "ai": 1}
            for cls_name, label in class_map.items():
                cls_dir = os.path.join(root, cls_name)
                for fname in sorted(os.listdir(cls_dir)):
                    if _is_image(fname):
                        self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples collected from datasets {dataset_names} under {base_dir} (split={split}).")

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
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        RandomMask(ratio=(0.00, 0.50), patch_size=16, p=0.5),
    ])

    transform_eval = transforms.Compose([
        transforms.CenterCrop([img_size, img_size]),
        transforms.ToTensor(),
    ])
    return transform_train, transform_eval


def create_dataloaders(
    cfg: DataConfig,
) -> Tuple[DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(img_size=cfg.img_size)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = _create_local_dataloader(cfg, train_tf, generator)
    val_loader = _create_genimage_dataloader(cfg, eval_tf, generator)
    return train_loader, val_loader


def _create_local_dataloader(
    cfg: DataConfig,
    train_tf: Callable,
    generator: torch.Generator,
) -> DataLoader:
    train_dir = os.path.join(cfg.data_root, "train")
    train_ds = LocalDataset(train_dir, transform=train_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )

    return train_loader


def _create_genimage_dataloader(
    cfg: DataConfig,
    eval_tf: Callable,
    generator: torch.Generator,
) -> DataLoader:
    os.environ["KAGGLEHUB_CACHE"] = cfg.data_root
    import kagglehub
    path = kagglehub.dataset_download('yangsangtai/tiny-genimage')
    
    val_dataset_names = [
        "imagenet_ai_0424_wukong",
        "imagenet_glide",
    ]
    
    logger.info(f"Loading validation datasets from {path}...")
    val_ds = GenImageDataset(
        path,
        val_dataset_names,
        split="val",
        transform=eval_tf
    )
    logger.info(f"Loaded {len(val_ds)} validation samples")
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        generator=generator,
    )
    
    return val_loader

