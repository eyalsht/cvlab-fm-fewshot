# cvlab-fm-fewshot

An experimental study of flow-matching blocks as layers inside discriminative
networks, on few-shot image classification over frozen encoder features.

Work in progress. Usage instructions land with the first runnable experiment.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
uv run ruff check src/
```

On the CPU-only development machine `uv sync` resolves the CPU torch wheel;
on a Linux CUDA machine the same lockfile resolves the CUDA build. No flags
needed on either.

## Datasets

Two datasets, both downloading automatically through torchvision on first use
into `data/raw/`:

- **DTD** (Describable Textures, 47 classes), official partition 1
- **FGVC-Aircraft** (100 classes), `variant` annotation level

All classes and the official train/val/test splits are used. Train and
validation are never merged, and the test split is read only for final
evaluation.
