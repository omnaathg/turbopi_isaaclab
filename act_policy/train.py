"""Train the language-conditioned ACT + CVAE policy."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import DEFAULT_ACT_DATA_ROOT, DEFAULT_ACTION_CHUNK_SIZE, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH
from .dataset import build_cached_datasets, build_datasets, has_cached_datasets
from .model import ACTCVAEConfig, build_model, save_checkpoint


def resolve_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base_dir / f"{candidate.name}_{suffix:02d}"
    candidate.mkdir(parents=True)
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TurboPi mountain ACT + language + CVAE policy")
    parser.add_argument("--episodes-dir", default=DEFAULT_ACT_DATA_ROOT)
    parser.add_argument("--cache-dir", default=None, help="Optional predecoded tensor cache directory with train.pt/val.pt.")
    parser.add_argument(
        "--cache-mode",
        choices=("auto", "require", "off"),
        default="auto",
        help="Use cached tensors when available. `require` fails if the cache is missing.",
    )
    parser.add_argument("--run-dir", default="runs/mountain_act_cvae")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_ACTION_CHUNK_SIZE)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_WIDTH,
                        help="Square image size (width=height). Must be a multiple of 16.")
    parser.add_argument("--frames-per-episode", type=int, default=0,
                        help="Uniformly subsample each episode to this many frames (0 = use all frames).")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
    kl_weight: float,
) -> tuple[torch.Tensor, float, float]:
    recon = criterion(output["action"], target)
    mu = output["mu"]
    logvar = output["logvar"]
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + kl_weight * kl
    return loss, float(recon.detach().item()), float(kl.detach().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    kl_weight: float,
    desc: str,
    show_progress: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_examples = 0
    abs_error = torch.zeros(4, dtype=torch.float64)
    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if show_progress else loader
    for batch in iterator:
        images = batch["image"].to(device)
        task_ids = batch["task_index"].to(device)
        targets = batch["action_chunk"].to(device)
        with torch.set_grad_enabled(training):
            output = model(images, task_ids, targets)
            loss, recon, kl = compute_loss(output, targets, criterion, kl_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_recon += recon * batch_size
        total_kl += kl * batch_size
        total_examples += batch_size
        abs_error += torch.abs(output["action"].detach() - targets).sum(dim=(0, 1)).double().cpu()
    denom = max(1, total_examples)
    mae_denom = max(1, total_examples * DEFAULT_ACTION_CHUNK_SIZE)
    return {
        "loss": total_loss / denom,
        "recon": total_recon / denom,
        "kl": total_kl / denom,
        "mae_vx": float(abs_error[0].item() / mae_denom),
        "mae_vy": float(abs_error[1].item() / mae_denom),
        "mae_wz": float(abs_error[2].item() / mae_denom),
        "mae_stop": float(abs_error[3].item() / mae_denom),
    }


def _save_loss_plot(run_dir: Path, history: list[dict[str, float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h.get("val_loss", float("nan")) for h in history]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, train_losses, label="train loss", color="tab:blue")
        has_val = any(not math.isnan(v) for v in val_losses)
        if has_val:
            ax.plot(epochs, val_losses, label="val loss", color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("ACT Training / Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "loss_curve.png", dpi=100)
        plt.close(fig)
    except Exception:
        pass


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = resolve_run_dir(Path(args.run_dir))
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    image_size = (args.image_size, args.image_size)
    chunk_size = args.chunk_size

    use_cache = False
    if args.cache_dir and args.cache_mode != "off":
        use_cache = has_cached_datasets(args.cache_dir)
        if args.cache_mode == "require" and not use_cache:
            raise RuntimeError(f"Required ACT tensor cache is missing or incomplete: {args.cache_dir}")
    if use_cache:
        train_ds, val_ds, task_names = build_cached_datasets(args.cache_dir, augment=not args.no_augment)
        dataset_source = str(args.cache_dir)
        print(f"[train] Using cached tensor dataset: {args.cache_dir}")
    else:
        train_ds, val_ds, task_names = build_datasets(
            args.episodes_dir,
            image_size=image_size,
            chunk_size=chunk_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            augment=not args.no_augment,
            max_frames_per_episode=args.frames_per_episode,
        )
        dataset_source = str(args.episodes_dir)
    if len(train_ds) == 0:
        raise RuntimeError(f"No ACT episodes found under {args.episodes_dir}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if len(val_ds) else None

    config = ACTCVAEConfig(task_vocab_size=len(task_names), chunk_size=chunk_size, image_width=image_size[0], image_height=image_size[1])
    model = build_model(config).to(device)
    print(f"[train] Model parameters: {model.parameter_count():,}")
    print(f"[train] Tasks: {task_names}")
    print(f"[train] Run dir: {run_dir}")
    criterion = nn.SmoothL1Loss(beta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_metric = math.inf
    best_epoch = 0
    interrupted = False
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                kl_weight=args.kl_weight,
                desc=f"epoch {epoch:03d} train",
                show_progress=not args.no_progress,
            )
            val_metrics = (
                run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    optimizer=None,
                    kl_weight=args.kl_weight,
                    desc=f"epoch {epoch:03d} val",
                    show_progress=not args.no_progress,
                )
                if val_loader is not None
                else {"loss": math.nan}
            )
            metric = val_metrics["loss"] if not math.isnan(val_metrics["loss"]) else train_metrics["loss"]
            is_best = metric < best_metric
            if is_best:
                best_metric = metric
                best_epoch = epoch
            metrics = {
                "epoch": float(epoch),
                **{f"train_{key}": float(value) for key, value in train_metrics.items()},
                **{f"val_{key}": float(value) for key, value in val_metrics.items()},
            }
            history.append(metrics)
            extra = {"task_names": task_names, "dataset_root": dataset_source, "run_dir": str(run_dir)}
            save_checkpoint(checkpoint_dir / "last.pt", model, epoch=epoch, metrics=metrics, extra=extra)
            if is_best:
                save_checkpoint(checkpoint_dir / "best.pt", model, epoch=epoch, metrics=metrics, extra=extra)
            (run_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "device": str(device),
                        "task_names": task_names,
                        "dataset_root": dataset_source,
                        "cache_dir": str(args.cache_dir) if use_cache else None,
                        "model_config": asdict(config),
                        "parameter_count": model.parameter_count(),
                        "history": history,
                        "best_epoch": best_epoch,
                        "best_metric": float(best_metric),
                        "interrupted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            val_str = f"{val_metrics['loss']:.4f}" if not math.isnan(val_metrics.get("loss", math.nan)) else "n/a"
            print(
                f"[train] epoch {epoch:03d}/{args.epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_str}  "
                f"recon={train_metrics['recon']:.4f}  kl={train_metrics['kl']:.4f}  "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            _save_loss_plot(run_dir, history)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[train] Interrupted; last/best checkpoints already saved.", flush=True)
    finally:
        summary_path = run_dir / "training_summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            data["interrupted"] = interrupted
            summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[train] Saved checkpoints to {checkpoint_dir}")


if __name__ == "__main__":
    main()
