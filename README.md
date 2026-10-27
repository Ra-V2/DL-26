# DL

A collection of deep learning projects. Each project lives in its own folder with its own code, notebooks and environment.

## Projects

| Project | Description |
| --- | --- |
| [highway-networks](highway-networks/) | Implementation of Highway Networks in PyTorch, with experiments on CIFAR-10 comparing highway and standard CNNs at different depths. |

More projects will be added here.

## Setup

Each project has its own `requirements.txt`. To set up a project:

```powershell
cd <project>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Datasets (`data/`) and model checkpoints (`checkpoints/`) are not tracked in git.

## AI assistance

AI assistants (ChatGPT and Claude) were used during this work. They helped with things like explaining concepts, writing and debugging code, and editing documentation.
