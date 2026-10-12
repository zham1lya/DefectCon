"""Training loop (Section 4.6, Algorithm 1).

Encoder, projector and classifier are trained jointly with
``contrastive_weight * SupCon(both views) + classification_weight * Focal(view 1)``,
Adam with linear warm-up and cosine decay, AMP and gradient clipping, for a
fixed number of optimizer steps. There is no early stopping and no checkpoint
selection: the final model is kept. Validation is used for monitoring only.

Seeding is done in one place, ``seed_everything``. cuDNN is set to
deterministic mode; GPU kernel selection and AMP can still cause small
run-to-run differences on different hardware or driver versions.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from . import paths
from .data import EvalDataset, TwoViewDataset, load_split, split_signature
from .losses import FocalLoss, SupConLoss
from .model import DefectCon
from .protocol import RunConfig, Schedule, protocol
from .sampler import QuotaBatchSampler, sample_groups
from .transforms import eval_transform, train_transform

log = logging.getLogger("defectcon")

CURVE_COLUMNS = ["epoch", "val_ap", "val_ap_smooth5", "val_auroc", "train_ap",
                 "train_defect_recall", "train_loss_con", "train_loss_ce", "epoch_seconds"]


def deterministic_cudnn() -> None:
    """Select deterministic cuDNN algorithms (training and scoring)."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA); make cuDNN deterministic."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    deterministic_cudnn()


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def validate(model: DefectCon, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """AP and AUROC of the defect score on a validation loader."""
    model.eval()
    logits, labels = [], []
    for images, y in loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(y)
    y = torch.cat(labels).numpy()
    p = F.softmax(torch.cat(logits), dim=1)[:, 1].numpy()
    return {"val_ap": float(average_precision_score(y, p)), "val_auroc": float(roc_auc_score(y, p))}


def train(cfg: RunConfig, split_dir: Path, device: torch.device,
          on_epoch: Optional[Callable[[int, Dict[str, Any]], None]] = None,
          num_workers: Optional[int] = None):
    """Train one model on ``split_dir``.

    ``on_epoch(epoch_index, record)`` is called after every epoch with the
    validation AP/AUROC, the smoothed validation AP and the training losses.
    Returns ``(model, history, info)``.
    """
    p = protocol()
    hp = cfg.hparams
    workers = p["num_workers"] if num_workers is None else num_workers

    seed_everything(cfg.seed)
    train_lists, val_lists = load_split(split_dir)
    schedule = cfg.schedule(n_normals=len(train_lists["normal"]))
    info = {"split": os.path.relpath(split_dir, paths.SPLITS_ROOT),
            "split_sig_train": split_signature(train_lists["normal"], train_lists["defect"]),
            "split_sig_val": split_signature(val_lists["normal"], val_lists["defect"]),
            "n_train": {k: len(v) for k, v in train_lists.items()},
            "n_val": {k: len(v) for k, v in val_lists.items()},
            "schedule": schedule.__dict__ | {"total_steps": schedule.total_steps}}
    log.info("split %s | train %s | val %s | schedule %s", split_dir, info["n_train"],
             info["n_val"], info["schedule"])

    train_ds = TwoViewDataset(train_lists["normal"], train_lists["defect"], train_transform())
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    sampler = QuotaBatchSampler(sample_groups(train_ds.samples, schedule.quotas),
                                schedule.quotas, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=workers,
                              pin_memory=True, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(EvalDataset(val_lists["normal"], val_lists["defect"], eval_transform()),
                            batch_size=p["val_batch_size"], shuffle=False, num_workers=workers)

    model = DefectCon.from_arch(cfg.architecture).to(device)
    supcon = SupConLoss(temperature=hp["temperature"]).to(device)
    focal = FocalLoss(gamma=hp["focal_gamma"])
    optimizer = torch.optim.Adam([{"params": list(model.parameters()), "lr": hp["lr"]}],
                                 weight_decay=p["weight_decay"])

    epochs, warmup = schedule.epochs, schedule.warmup_epochs

    def lr_factor(e: int) -> float:
        if e < warmup:
            return float(e + 1) / float(max(1, warmup))
        progress = (e - warmup) / float(max(1, epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_amp = bool(p["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    w_con, w_cls = hp["contrastive_weight"], p["classification_weight"]

    history: List[Dict[str, Any]] = []
    val_aps: List[float] = []
    window = p["smoothing_window"]
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        sum_con = sum_ce = 0.0
        train_probs, train_targets = [], []
        for v1, v2, labels in train_loader:
            x = torch.cat([v1, v2], dim=0).to(device)
            y_con = torch.cat([labels, labels], dim=0).to(device)
            y_cls = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                z, logits = model.forward_train(x, n_cls=labels.size(0))
                loss_con = supcon(z, y_con)
                loss_ce = focal(logits, y_cls)
                loss = w_con * loss_con + w_cls * loss_ce
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=p["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            sum_con += loss_con.item()
            sum_ce += loss_ce.item()
            with torch.no_grad():
                train_probs.append(F.softmax(logits.float(), dim=1)[:, 1].cpu())
                train_targets.append(y_cls.cpu())
        n_batches = max(1, len(train_loader))
        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()

        tp = torch.cat(train_probs).numpy()
        ty = torch.cat(train_targets).numpy()
        record: Dict[str, Any] = {"epoch": epoch + 1}
        record.update(validate(model, val_loader, device))
        val_aps.append(record["val_ap"])
        record["val_ap_smooth5"] = sum(val_aps[-window:]) / len(val_aps[-window:])
        record["train_ap"] = float(average_precision_score(ty, tp))
        record["train_defect_recall"] = float(((tp >= 0.5) & (ty == 1)).sum() / max(1, int((ty == 1).sum())))
        record["train_loss_con"] = sum_con / n_batches
        record["train_loss_ce"] = sum_ce / n_batches
        record["epoch_seconds"] = round(time.time() - t0, 1)
        history.append(record)
        log.info("epoch %d/%d | lr %.2e | con %.4f | ce %.4f | val AP %.4f | val AUROC %.4f",
                 epoch + 1, epochs, lr_used, record["train_loss_con"], record["train_loss_ce"],
                 record["val_ap"], record["val_auroc"])
        if on_epoch is not None:
            on_epoch(epoch, record)
    return model, history, info


def write_curve(path: Path, history: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_COLUMNS)
        w.writeheader()
        w.writerows({**r, "val_ap_smooth5": round(r["val_ap_smooth5"], 6)} for r in history)


def save_checkpoint(path: Path, model: DefectCon, cfg: RunConfig, schedule: Dict[str, Any],
                    epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "arch": cfg.architecture,
                "hparams": cfg.hparams, "condition": cfg.condition, "seed": cfg.seed,
                "schedule": schedule, "epoch": epoch}, path)
