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
from utils import build_model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_desc: str,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake detection training")
    parser.add_argument("--data_root", type=str, default="dataset", help="Root folder containing train/test splits")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ruined_weight", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--betas", type=float, nargs="+", default=(0.937, 0.999))
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Eta min for cosine annealing")
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
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_loader, val_clean_loader, val_ruined_loader = create_dataloaders(cfg)

    model = build_model(args.model, num_classes=2, map_location=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.log_dir, f"run-{timestamp}")
    writer = SummaryWriter(log_dir=output_dir)
    logger.info(f"TensorBoard logging to: {output_dir}")

    ckpt_path = os.path.join(output_dir, f"checkpoint_{args.model}.pth")
    best_score: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        clean_loss, clean_acc = evaluate(
            model, val_clean_loader, criterion, device, f"Epoch {epoch} [val_clean]"
        )
        writer.add_scalar("val_clean/loss", clean_loss, epoch)
        writer.add_scalar("val_clean/acc", clean_acc, epoch)

        ruined_loss, ruined_acc = evaluate(
            model, val_ruined_loader, criterion, device, f"Epoch {epoch} [val_ruined]"
        )
        writer.add_scalar("val_ruined/loss", ruined_loss, epoch)
        writer.add_scalar("val_ruined/acc", ruined_acc, epoch)

        score = (1 - args.ruined_weight) * clean_acc + args.ruined_weight * ruined_acc

        logger.info(
            f"Epoch {epoch}: "
            f"clean_loss={clean_loss:.4f} clean_acc={clean_acc:.4f} "
            f"ruined_loss={ruined_loss:.4f} ruined_acc={ruined_acc:.4f} "
            f"score={score:.4f}"
        )
        writer.add_scalar("val/score", score, epoch)

        if best_score is None or score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "score": score,
                    "val_acc_clean": clean_acc,
                    "val_acc_ruined": ruined_acc,
                    "config": vars(args),
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint to {ckpt_path} (score={score:.4f})")

        scheduler.step()

    writer.close()


if __name__ == "__main__":
    main()

