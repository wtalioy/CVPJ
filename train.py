import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import DataConfig, create_dataloaders
from utils import build_model, accuracy
from evaluate import evaluate


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    running_loss, running_acc, total = 0.0, 0.0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, labels in progress:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size
        running_acc += accuracy(logits, labels) * batch_size
        progress.set_postfix(loss=loss.item())

    return running_loss / total, running_acc / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake detection training")
    parser.add_argument("--data_root", type=str, default="dataset", help="Root folder containing train/test splits")
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--betas", type=float, nargs="+", default=(0.937, 0.999))
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01, help="Eta min ratio for cosine annealing")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log dir root")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        type=str,
        default="ferretnet",
        help="Model builder to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    cfg = DataConfig(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader, val_loader = create_dataloaders(cfg)

    model = build_model(args.model, num_classes=2, map_location=device).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * args.min_lr_ratio)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.log_dir, f"run-{timestamp}")
    writer = SummaryWriter(log_dir=output_dir)
    logger.info(f"TensorBoard logging to: {output_dir}")

    ckpt_path = os.path.join(output_dir, f"checkpoint_{args.model}.pth")
    best_val_acc: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, f"Epoch {epoch} [val]")
            logger.info(f"Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)

            if best_val_acc is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "val_acc": val_acc,
                        "config": vars(args),
                    },
                    ckpt_path,
                )
                logger.info(f"Saved checkpoint to {ckpt_path} (val_acc={val_acc:.4f})")

        scheduler.step()

    writer.close()


if __name__ == "__main__":
    main()

