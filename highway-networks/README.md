# Highway Networks on CIFAR-10

A PyTorch implementation of convolutional Highway Networks ([Srivastava et al., 2015](https://arxiv.org/abs/1505.00387)) and a series of experiments on CIFAR-10.

## Research question

Do Highway gates, with output

$$y = H(x)\,T(x) + x\,(1 - T(x)),$$

make CNNs easier to train or more accurate than plain convolutional blocks? We asked this in two settings:

1. **Trainability:** deep, constant-width networks with 4–100 blocks.
2. **Accuracy:** a stronger multi-stage CNN in which only the internal blocks differ.

This is our own implementation, inspired by the paper. It is not an exact reproduction.

## Project structure

```
highway-networks/
├── src/
│   ├── highway.py          # ConvHighwayLayer (notebooks 03–04)
│   ├── model.py            # Constant-width HighwayCNN / StandardCNN (notebooks 03–04)
│   ├── improved_model.py   # Final multi-stage ImprovedHighwayCNN / ImprovedStandardCNN
│   ├── data.py             # CIFAR-10 45k/5k/10k split, transforms, loaders
│   └── training.py         # run_epoch, train_model, load_checkpoint, evaluation helpers
├── notebooks/
│   ├── 01_test_env.ipynb
│   ├── 02_paper_exploration.ipynb
│   ├── 03_cifar10_training.ipynb
│   ├── 04_depth_experiments.ipynb
│   ├── 05_improved_highway_cifar10.ipynb
│   └── 06_final_model_comparison.ipynb
├── scripts/
│   └── verify_checkpoints.py
└── docs/
    └── experiment_summary.md
```

Notebooks 02–04 are kept as historical experiments, and their code is unchanged. Notebooks 05 and 06 import the final models and training code from `src/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # or the CPU build
pip install -r requirements.txt
```

The experiments were run with PyTorch 2.14.1 + CUDA 12.6. CIFAR-10 downloads to `data/` automatically on first use.

## Running the final models

The trained weights in `checkpoints/` are **not in the repository** because they are too large. The saved notebook outputs show the results without rerunning anything.

**With the checkpoints:**

```powershell
python scripts/verify_checkpoints.py
```

This loads both final models and checks that they reproduce the saved validation accuracy and loss. In the notebooks, run every cell except the training cells: section 7 in notebook 05 and section 8 in notebook 06.

**Without the checkpoints:** run the training cell in notebook 05 (about 18 minutes on a GPU), then the one in notebook 06 (about 9 minutes). Both use seed 42 and write to `checkpoints/`.

From your own code:

```python
from src.improved_model import ImprovedHighwayCNN
from src.training import load_checkpoint

model, checkpoint = load_checkpoint(
    ImprovedHighwayCNN,
    "checkpoints/improved_highway/improved_highway_v1_best.pth",
    device="cpu",
)
```

## Main results

Data split: 45,000 training / 5,000 validation images (seed 42) and the 10,000 CIFAR-10 test images. Checkpoints were selected on validation accuracy.

**Depth experiments (notebook 04)**: final validation accuracy:

| Blocks | Standard CNN | Highway CNN |
| ------ | -----------: | ----------: |
| 4      |       56.68% |      57.34% |
| 10     |       61.18% |      63.92% |
| 20     |       46.82% |      63.44% |
| 50     |       10.34% |  **66.06%** |
| 100    |        9.86% |       9.86% |

**Final multi-stage models (notebooks 05–06)**

| Metric              | Highway CNN | Standard CNN |
| ------------------- | ----------: | -----------: |
| Parameters          |     870,250 |      482,730 |
| Training accuracy   |      93.30% |       91.39% |
| Validation accuracy |  **88.80%** |       88.42% |
| Validation loss     |      0.3456 |       0.3515 |
| Test accuracy       |  **88.07%** |            – |
| Training time       |    18.5 min |      9.0 min |

**Conclusions**

- Highway gates made a big difference to **trainability**: at 50 blocks the Highway CNN reached 66.06%, while the Standard CNN failed to learn (10.34%). At 100 blocks, both were unstable.
- In a well-designed multi-stage CNN, the gates gave only a **small accuracy gain** (+0.38 percentage points). That gain cost 80% more parameters and about twice the training time.
- Limitations: each comparison used a single seed, and the models have different parameter counts. More details are in [`docs/experiment_summary.md`](docs/experiment_summary.md).
