"""Training, checkpointing and evaluation helpers used in notebooks 05 and 06."""

import random
import time

import numpy as np
import torch
import torch.nn as nn


EPOCHS = 20
LEARNING_RATE = 1e-3
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(net, loader, device, optimizer=None, max_grad_norm=MAX_GRAD_NORM):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates.

    Returns (sample-weighted mean loss, accuracy in %).
    """
    criterion = nn.CrossEntropyLoss()
    training = optimizer is not None
    net.train(training)

    running_loss = 0.0
    correct, total = 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = net(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss: stop and inspect training")

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    max_norm=max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()

            count = labels.size(0)
            running_loss += loss.item() * count
            correct += (logits.argmax(1) == labels).sum().item()
            total += count

    return running_loss / total, 100.0 * correct / total


def train_model(
    model_cls,
    model_config,
    make_train_loader,
    val_loader,
    device,
    checkpoint_dir,
    run_name,
    seed=42,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    min_lr=MIN_LR,
    weight_decay=WEIGHT_DECAY,
):
    """Train with AdamW + cosine annealing, selecting the best epoch on validation accuracy.

    Saves `<run_name>_latest.pth` after every epoch and `<run_name>_best.pth`
    whenever validation accuracy improves. Rerunning overwrites both files.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / f"{run_name}_latest.pth"
    best_path = checkpoint_dir / f"{run_name}_best.pth"

    set_seed(seed)

    net = model_cls(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr,
    )
    train_loader = make_train_loader(seed=seed)

    history = {"train_loss": [], "train_accuracy": [],
               "val_loss": [], "val_accuracy": [], "lr": []}
    best_val_acc, best_epoch = float("-inf"), None
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        lr_used = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(net, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(net, val_loader, device)
        scheduler.step()

        for key, value in {"train_loss": tr_loss, "train_accuracy": tr_acc,
                           "val_loss": va_loss, "val_accuracy": va_acc,
                           "lr": lr_used}.items():
            history[key].append(value)

        payload = {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "model_config": model_config,
            "seed": seed,
            "run_name": run_name,
        }
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save(payload, best_path)
        torch.save(payload, latest_path)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc:.2f}% | "
              f"LR {lr_used:.6f} | best {best_val_acc:.2f}% (epoch {best_epoch})",
              flush=True)

    print(f"Completed in {(time.perf_counter() - start_time) / 60:.1f} min")
    print("Best validation accuracy:", f"{best_val_acc:.2f}%", "at epoch", best_epoch)
    print("Best checkpoint:", best_path)

    return history, best_path


def load_checkpoint(model_cls, path, device):
    """Rebuild a model from a checkpoint saved by `train_model`, in eval mode.

    Returns (model, checkpoint dict).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = model_cls(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def predict(net, loader, device):
    """Return (predicted labels, true labels) as CPU tensors."""
    net.eval()
    predictions, targets = [], []

    for images, labels in loader:
        predictions.append(net(images.to(device)).argmax(dim=1).cpu())
        targets.append(labels)

    return torch.cat(predictions), torch.cat(targets)


def per_class_accuracy(predictions, targets, num_classes=10):
    """Accuracy in % for each class."""
    correct = torch.bincount(targets[predictions == targets], minlength=num_classes)
    total = torch.bincount(targets, minlength=num_classes)

    return (100.0 * correct / total).tolist()
