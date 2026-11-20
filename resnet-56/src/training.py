"""SGD training and checkpoint helpers for the CIFAR-10 experiment."""

from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import torch
from torch import nn

from .model import ResNet56


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    milestones: tuple[int, ...] = (32_000, 48_000)
    total_steps: int = 64_000


def make_optimizer(model, config):
    return torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def run_epoch(model, batches, device, optimizer=None, scheduler=None):
    """Run one pass over the given batches and return loss and accuracy (%)."""
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Encountered a non-finite loss.")
            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            count = labels.size(0)
            loss_sum += loss.detach() * count
            correct += (logits.argmax(dim=1) == labels).sum()
            total += count

    if total == 0:
        raise ValueError("The loader yielded no examples.")
    return loss_sum.item() / total, 100 * correct.item() / total


def _save_checkpoint(payload, path):
    temporary_path = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_model(data, device, checkpoint_dir, config=TrainingConfig(), *, resume=False):
    """Train to the configured optimizer step, saving after each validation pass."""
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best.pth"
    if not resume and (latest_path.exists() or best_path.exists()):
        raise FileExistsError(f"A run already exists in {checkpoint_dir}. Use resume=True.")

    torch.manual_seed(data.seed)
    model = ResNet56().to(device)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.milestones, gamma=0.1)
    train_loader = data.make_train_loader()
    if not len(train_loader):
        raise ValueError("The training loader must contain at least one batch.")
    settings = {
        **asdict(config),
        "seed": data.seed,
        "batch_size": train_loader.batch_size,
        "num_workers": train_loader.num_workers,
        "train_examples": len(train_loader.dataset),
        "val_examples": len(data.val_loader.dataset),
    }

    step = 0
    history = []
    best_val_accuracy = -1.0
    if resume:
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        if latest["settings"] != settings or not torch.equal(latest["mean_image"], data.mean_image):
            raise ValueError("The training settings or data mean differ from the saved run.")
        model.load_state_dict(latest["model_state_dict"])
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        scheduler.load_state_dict(latest["scheduler_state_dict"])
        step = latest["step"]
        history = latest["history"]
        best_val_accuracy = latest["best_val_accuracy"]
        train_loader.generator.set_state(latest["loader_rng_state"])
        torch.set_rng_state(latest["torch_rng_state"])
        if device.type == "cuda" and latest["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(latest["cuda_rng_state"], device=device)

    while step < config.total_steps:
        epoch = len(history) + 1
        batches = min(len(train_loader), config.total_steps - step)
        train_loss, train_accuracy = run_epoch(
            model, islice(train_loader, batches), device, optimizer, scheduler,
        )
        step += batches
        val_loss, val_accuracy = run_epoch(model, data.val_loader, device)
        history.append({
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
        })

        improved = val_accuracy > best_val_accuracy
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "settings": settings,
            "mean_image": data.mean_image,
            "step": step,
            "best_val_accuracy": best_val_accuracy,
            "loader_rng_state": train_loader.generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        if improved:
            _save_checkpoint(checkpoint, best_path)
        _save_checkpoint(checkpoint, latest_path)
        print(
            f"Epoch {epoch:3d} | step {step:5d}/{config.total_steps} | "
            f"train {train_loss:.4f}, {train_accuracy:.2f}% | "
            f"val {val_loss:.4f}, {val_accuracy:.2f}% | "
            f"next LR {optimizer.param_groups[0]['lr']:.4g}",
            flush=True,
        )

    return history, best_path


def load_checkpoint(path, device):
    """Load an evaluation model and the data and run metadata it was saved with."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ResNet56().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), checkpoint
