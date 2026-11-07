"""Train the baseline U-Net on CAMUS ED/ES frames (2CH + 4CH, one model).

Loss: cross-entropy + soft Dice (foreground classes). Optimizer: AdamW with
linear warmup then cosine decay. Model selection: best mean validation LV Dice.
Writes config.json (incl. hardware/software info for the write-up), log.csv,
best.pt and last.pt to --out-dir.
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import LV_LABEL, N_CLASSES
from dataset import IGNORE, CamusFrames
from unet import UNet, count_params


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    random.seed(s + worker_id)
    np.random.seed((s + worker_id) % 2**32)


def soft_dice_loss(logits, target, eps=1e-5):
    """Soft Dice over foreground classes, restricted to the valid field of view."""
    valid = (target != IGNORE).unsqueeze(1).float()
    probs = logits.softmax(1) * valid
    onehot = F.one_hot(target.masked_fill(target == IGNORE, 0), N_CLASSES)
    onehot = onehot.permute(0, 3, 1, 2).float() * valid
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice[1:].mean()  # skip background


def seg_loss(logits, target):
    return F.cross_entropy(logits, target, ignore_index=IGNORE) + soft_dice_loss(logits, target)


def hard_dice_per_image(pred, target, label):
    valid = target != IGNORE
    p, t = (pred == label) & valid, target == label
    inter = (p & t).sum((1, 2)).float()
    denom = p.sum((1, 2)).float() + t.sum((1, 2)).float()
    return torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom))


def hardware_info():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count_visible": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu_name=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 1),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return info


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    dices = {c: [] for c in range(1, N_CLASSES)}
    total_loss, n = 0.0, 0
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(img)
        logits = logits.float()
        loss = seg_loss(logits, mask)
        total_loss += loss.item() * img.size(0)
        n += img.size(0)
        pred = logits.argmax(1)
        for c in dices:
            dices[c].append(hard_dice_per_image(pred, mask, c).cpu())
    return total_loss / n, {c: torch.cat(v).mean().item() for c, v in dices.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="splits/splits.json")
    ap.add_argument("--out-dir", default="runs/unet_baseline")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30,
                    help="early stop after this many epochs without val LV Dice improvement")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and not args.no_amp) else None

    splits = json.loads(Path(args.splits).read_text())
    t0 = time.time()
    train_ds = CamusFrames(args.data_root, splits["train"], args.size, augment=True)
    val_ds = CamusFrames(args.data_root, splits["val"], args.size, augment=False)
    print(f"loaded {len(train_ds)} train / {len(val_ds)} val frames in {time.time() - t0:.0f}s")

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.num_workers, worker_init_fn=worker_init,
                          generator=g, pin_memory=device.type == "cuda",
                          persistent_workers=args.num_workers > 0)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    model = UNet(n_classes=N_CLASSES, base=args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        prog = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    config = {
        "args": vars(args),
        "splits_method": splits.get("method"),
        "n_patients": {k: len(splits[k]) for k in ("train", "val", "test")},
        "n_frames": {"train": len(train_ds), "val": len(val_ds)},
        "model": {"arch": "UNet", "depth": 5, "base_channels": args.base_ch,
                  "n_classes": N_CLASSES, "n_params": count_params(model)},
        "loss": "cross-entropy + soft Dice (foreground classes, equal weight), both restricted to the valid field of view",
        "optimizer": f"AdamW(lr={args.lr}, weight_decay={args.weight_decay}, betas=(0.9, 0.999))",
        "schedule": f"linear warmup {args.warmup_epochs} epochs, then cosine decay to 0",
        "mixed_precision": str(amp_dtype) if amp_dtype else "off",
        "augmentation": "rotation ±10°, scale 0.9–1.1, translation ±5%, gamma 0.7–1.5, "
                        "contrast ×0.85–1.15, brightness ±0.1, Gaussian noise σ≤0.05; no flips",
        "preprocessing": f"intensities / 255; isotropic physical-aspect-preserving letterbox into {args.size}x{args.size} (images bilinear, masks nearest); per-image z-score over the valid field of view",
        "hardware": hardware_info(),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config["hardware"], indent=2))
    print(f"model params: {config['model']['n_params']:,}")

    log_f = open(out / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "lr", "train_loss", "val_loss", "val_dice_lv", "val_dice_myo",
                  "val_dice_la", "epoch_sec"])

    best, best_epoch, train_start = -1.0, -1, time.time()
    for epoch in range(args.epochs):
        model.train()
        t_ep, run_loss, n = time.time(), 0.0, 0
        for img, mask, _ in train_dl:
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(img)
            logits = logits.float()
            loss = seg_loss(logits, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * img.size(0)
            n += img.size(0)
        lr = opt.param_groups[0]["lr"]
        sched.step()

        val_loss, vd = evaluate(model, val_dl, device, amp_dtype)
        dt = time.time() - t_ep
        log.writerow([epoch, f"{lr:.3e}", f"{run_loss / n:.4f}", f"{val_loss:.4f}",
                      f"{vd[1]:.4f}", f"{vd[2]:.4f}", f"{vd[3]:.4f}", f"{dt:.1f}"])
        log_f.flush()
        print(f"ep {epoch:3d} lr {lr:.2e} train {run_loss / n:.4f} val {val_loss:.4f} "
              f"dice LV {vd[1]:.4f} myo {vd[2]:.4f} LA {vd[3]:.4f} ({dt:.0f}s)", flush=True)

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_dice_lv": vd[LV_LABEL],
                "size": args.size, "base_ch": args.base_ch}
        torch.save(ckpt, out / "last.pt")
        if vd[LV_LABEL] > best:
            best, best_epoch = vd[LV_LABEL], epoch
            torch.save(ckpt, out / "best.pt")
        elif epoch - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    log_f.close()
    config["result"] = {"best_epoch": best_epoch, "best_val_dice_lv": best,
                        "epochs_run": epoch + 1,
                        "train_time_min": round((time.time() - train_start) / 60, 1)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"best val LV Dice {best:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
