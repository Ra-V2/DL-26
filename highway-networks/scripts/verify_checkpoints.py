"""Check that the final checkpoints load into src/improved_model.py and
reproduce the validation results saved during training.

Run from the project root:
    python scripts/verify_checkpoints.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_cifar10
from src.improved_model import ImprovedHighwayCNN, ImprovedStandardCNN, count_parameters
from src.training import load_checkpoint, run_epoch


CHECKPOINTS = {
    "Highway CNN": (
        ImprovedHighwayCNN,
        PROJECT_ROOT / "checkpoints" / "improved_highway" / "improved_highway_v1_best.pth",
        870250,
    ),
    "Standard CNN": (
        ImprovedStandardCNN,
        PROJECT_ROOT / "checkpoints" / "final_comparison" / "standard_multistage_v1_best.pth",
        482730,
    ),
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_cifar10(PROJECT_ROOT / "data", device)
    all_ok = True

    for name, (model_cls, path, expected_params) in CHECKPOINTS.items():
        if not path.exists():
            print(f"{name}: MISSING {path}")
            all_ok = False
            continue

        # load_state_dict is strict, so any renamed or missing module fails here
        model, checkpoint = load_checkpoint(model_cls, path, device)
        epoch = checkpoint["epoch"]
        saved_acc = checkpoint["history"]["val_accuracy"][epoch - 1]
        saved_loss = checkpoint["history"]["val_loss"][epoch - 1]

        val_loss, val_acc = run_epoch(model, data.val_loader, device)
        params = count_parameters(model)

        ok = (
            params == expected_params
            and abs(val_acc - saved_acc) < 0.05
            and abs(val_loss - saved_loss) < 1e-3
        )
        all_ok &= ok

        print(f"{name}: {'OK' if ok else 'MISMATCH'} | epoch {epoch} | {params:,} params | "
              f"val acc {val_acc:.2f}% (saved {saved_acc:.2f}%) | "
              f"val loss {val_loss:.4f} (saved {saved_loss:.4f})")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
