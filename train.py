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

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer, 
                    scheduler: optim.lr_scheduler._LRScheduler, device: torch.device):
    """加载检查点并恢复训练状态"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型状态
    model_state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(model_state, strict=False)
    
    # 加载优化器状态
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info("Optimizer state restored")
    
    # 加载调度器状态
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        logger.info("Scheduler state restored")
    
    # 获取已训练的epoch数和最佳验证准确率
    start_epoch = checkpoint.get("epoch", 0) + 1  # 从下一个epoch开始
    best_val_acc = checkpoint.get("val_acc", 0.0)
    
    # 获取保存的配置
    saved_config = checkpoint.get("config", {})
    logger.info(f"Checkpoint info: epoch={start_epoch-1}, val_acc={best_val_acc:.4f}")
    
    return start_epoch, best_val_acc, saved_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deepfake detection training")
    parser.add_argument("--data_root", type=str, default="dataset", help="Root folder containing train/test splits")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--betas", type=float, nargs="+", default=(0.937, 0.999))
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Eta min for cosine annealing")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log dir root")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        type=str,
        default="artifactnet",
        help="Model builder to use",
    )
    # 新增参数：恢复训练
    parser.add_argument("--resume", type=str, default=None, 
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--continue_log_dir", action="store_true", default=True,
                       help="Continue logging to the same directory as the checkpoint")
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
        val_split=args.val_split,
        seed=args.seed,
    )
    train_loader, val_loader = create_dataloaders(cfg)

    model = build_model(args.model, num_classes=2, map_location=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

     # 恢复训练相关变量
    start_epoch = 1
    best_val_acc = None
    saved_config = {}

    # 如果指定了恢复训练的检查点
    if args.resume:
        try:
            start_epoch, best_val_acc, saved_config = load_checkpoint(
                args.resume, model, optimizer, scheduler, device
            )
            logger.info(f"Resuming training from epoch {start_epoch}")
            
            # 如果继续使用相同的日志目录
            if args.continue_log_dir and os.path.dirname(args.resume):
                output_dir = os.path.dirname(args.resume)
                logger.info(f"Continuing logging to existing directory: {output_dir}")
            else:
                # 创建新的日志目录，但标明是继续训练
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                output_dir = os.path.join(args.log_dir, f"run-{timestamp}-resume")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = os.path.join(args.log_dir, f"run-{timestamp}")
    else:
        # 正常训练，创建新的日志目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.log_dir, f"run-{timestamp}")

    writer = SummaryWriter(log_dir=output_dir)
    logger.info(f"TensorBoard logging to: {output_dir}")

    ckpt_path = os.path.join(output_dir, f"checkpoint_{args.model}.pth")

    for epoch in range(start_epoch, args.epochs + 1):
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

