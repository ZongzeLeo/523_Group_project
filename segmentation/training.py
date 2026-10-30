"""Train only on the fixed training split, select using validation Dice only."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from data.download_camus import atomic_json
from data.prepare import write_csv
from .io import PreparedFrames, sha256
from .model import UNet
from data.protocol import identity


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resident_dataset(dataset, device):
    samples = [dataset[index] for index in range(len(dataset))]
    image = torch.from_numpy(np.stack([row["image"] for row in samples])).to(device)
    mask = torch.from_numpy(np.stack([row["mask"] for row in samples])[:, None]).to(device)
    valid = torch.from_numpy(np.stack([row["valid_region"] for row in samples])[:, None]).to(device)
    return image, mask, valid


def augment(image, mask, valid, config):
    n, device = len(image), image.device
    uniform = lambda low, high: low + (high-low)*torch.rand(n, device=device)
    angle = uniform(-config["rotation_degrees"], config["rotation_degrees"]) * (np.pi/180)
    scale = uniform(config["scale_min"], config["scale_max"])
    flip = torch.where(torch.rand(n, device=device) < config["horizontal_flip_probability"], -1.0, 1.0)
    theta = torch.zeros((n, 2, 3), device=device)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(angle)*scale*flip, -torch.sin(angle)*scale
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(angle)*scale*flip, torch.cos(angle)*scale
    theta[:, :, 2] = (torch.rand(n, 2, device=device)*2-1)*(2*config["translation_fraction"])
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = F.grid_sample(mask.float(), grid, mode="nearest", align_corners=False)
    valid = F.grid_sample(valid.float(), grid, mode="nearest", align_corners=False)
    gamma = uniform(config["gamma_min"], config["gamma_max"])[:, None, None, None]
    image = (image.clamp(0, 1).pow(gamma) + torch.randn_like(image)*config["noise_std"]).clamp(0, 1)*valid
    return image, mask, valid


def segmentation_loss(logits, mask, valid):
    logits, mask, valid = logits.float(), mask.float(), valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, mask, reduction="none")*valid).sum()/valid.sum().clamp_min(1)
    probability, target = logits.sigmoid()*valid, mask*valid
    dims = (1, 2, 3)
    dice = (2*(probability*target).sum(dims)+1e-6)/(probability.sum(dims)+target.sum(dims)+1e-6)
    return .5*bce + .5*(1-dice.mean())


@torch.inference_mode()
def validate(model, tensors, batch_size, amp):
    model.eval()
    images, masks, valid = tensors
    losses, scores = [], []
    for start in range(0, len(images), batch_size):
        x, y, v = (value[start:start+batch_size] for value in tensors)
        with torch.amp.autocast(device_type=x.device.type, enabled=amp):
            logits = model(x.contiguous(memory_format=torch.channels_last))
            loss = segmentation_loss(logits, y, v)
        prediction, target = (logits.float() > 0)*v.bool(), y.bool()*v.bool()
        dims = (1, 2, 3)
        dice = (2*(prediction & target).sum(dims)+1e-6)/(prediction.sum(dims)+target.sum(dims)+1e-6)
        losses.append((float(loss), len(x)))
        scores.extend(dice.cpu().tolist())
    return sum(loss*n for loss,n in losses)/len(images), float(np.mean(scores))


def train(prepared, output, config, device="cuda", smoke_steps=0):
    prepared, output = Path(prepared), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output is nonempty; use a new named run")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; use the documented CUDA interpreter")
    seed_everything(config["seed"])
    training, validation = PreparedFrames(prepared, "train"), PreparedFrames(prepared, "validation")
    if set(training.patient_metadata) & set(validation.patient_metadata):
        raise ValueError("Training/validation patient overlap")
    (output / "weights").mkdir(parents=True)
    atomic_json(output / "config.json", config)
    source_files = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("io.py"), Path("data/transforms.py")]
    metadata = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "prepared_metadata_sha256": sha256(prepared / "run_metadata.json"),
                "data_identity": identity(prepared),
                "code_sha256": {path.as_posix() if not path.is_absolute() else "segmentation/"+path.name: sha256(path) for path in source_files},
                "train_patients": sorted(training.patient_metadata), "validation_patients": sorted(validation.patient_metadata),
                "train_frames": len(training), "validation_frames": len(validation), "test_frames_used": 0,
                "model_input": "grayscale image only; no masks, EF, patient ID, view, or phase as features",
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda},
                "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "smoke_steps": smoke_steps}
    atomic_json(output / "run_metadata.json", metadata)
    print(f"Loading {len(training)} train and {len(validation)} validation frames onto {device}", flush=True)
    train_tensors, val_tensors = resident_dataset(training, device), resident_dataset(validation, device)
    model = UNet(config["base_channels"], config.get("normalization", "group")).to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=config["lr_factor"],
                                                        patience=config["lr_patience"], min_lr=config["min_learning_rate"])
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best, best_epoch, stale = [], -1.0, 0, 0
    started = time.perf_counter()
    for epoch in range(1, (1 if smoke_steps else config["max_epochs"])+1):
        model.train()
        order = torch.randperm(len(training), device=device)
        running_loss, seen = 0.0, 0
        epoch_started = time.perf_counter()
        for step, start in enumerate(range(0, len(training), config["batch_size"])):
            if smoke_steps and step >= smoke_steps:
                break
            indices = order[start:start+config["batch_size"]]
            x, y, v = augment(*(value[indices] for value in train_tensors), config["augmentation"])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=amp):
                logits = model(x.contiguous(memory_format=torch.channels_last))
                loss = segmentation_loss(logits, y, v)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())*len(x)
            seen += len(x)
        val_loss, val_dice = validate(model, val_tensors, config["batch_size"], amp)
        improved = val_dice > best
        if improved:
            best, best_epoch, stale = val_dice, epoch, 0
            torch.save({"model": model.state_dict(), "base_channels": config["base_channels"],
                        "epoch": epoch, "validation_dice": best, "config": config}, output / "weights/best.pt")
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": running_loss/seen, "validation_loss": val_loss,
               "validation_dice": val_dice, "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter()-epoch_started, "selected_checkpoint": improved}
        history.append(row)
        write_csv(output / "history.csv", history)
        print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} val_dice={val_dice:.5f} best={best:.5f}@{best_epoch} seconds={row['seconds']:.1f}", flush=True)
        scheduler.step(val_dice)
        if stale >= config["early_stopping_patience"]:
            break
    metadata.update(status="smoke_only" if smoke_steps else "complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                    epochs_completed=len(history), best_epoch=best_epoch, best_validation_dice=best,
                    elapsed_seconds=time.perf_counter()-started, parameter_count=sum(p.numel() for p in model.parameters()),
                    checkpoint_sha256=sha256(output / "weights/best.pt"),
                    artifact_sha256={name: sha256(output / name) for name in ("config.json", "history.csv")})
    atomic_json(output / "run_metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("outputs/training"))
    parser.add_argument("--config", type=Path, default=Path("segmentation/configs/groupnorm.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    train(args.prepared, args.output, json.loads(args.config.read_text()), args.device, args.smoke_steps)
