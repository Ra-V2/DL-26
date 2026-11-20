# DL-26

A collection of deep learning projects. Each project lives in its own folder with its own code, notebooks and environment.

## Projects

| Project | Description | Paper |
| --- | --- | --- |
| [highway-networks](highway-networks/) | PyTorch implementation of Highway Networks, with CIFAR-10 experiments comparing highway and standard CNNs at different depths. | [Highway Networks](https://arxiv.org/pdf/1505.00387) |
| [resnet-56](resnet-56/) | PyTorch implementation of the original ResNet-56 architecture for CIFAR-10. | [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385) |

More projects will be added here.

## Setup

Each project has its own `requirements.txt`. The PowerShell commands below
set up a project with the CUDA 12.6 build of PyTorch.

```powershell
cd <project>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Datasets (`data/`) and model checkpoints (`checkpoints/`) are not tracked in git.

## AI assistance

AI assistants (ChatGPT and Claude) were used during this work. They helped with things like explaining concepts, writing and debugging code, and editing documentation.
