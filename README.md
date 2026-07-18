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

MNIST and CIFAR-10 download automatically through torchvision on first use
into `data/raw/`.

### Mini-ImageNet acquisition

Mini-ImageNet (Vinyals et al. 2016, splits by Ravi and Larochelle 2017) has
no official host and must be downloaded manually under research terms. The
loader expects one directory per class containing that class's images:

```
data/raw/mini_imagenet/
  train/  64 class directories, e.g. n01532829/
  val/    16 class directories
  test/   20 class directories
```

Class directories follow the standard Ravi-Larochelle split. Any image
format PIL reads is accepted; images are converted to RGB at load time.
File names within a class are read in sorted order, so the on-disk layout
fully determines item order.

After placing the data, verify with:

```
uv run pytest tests/unit/test_datasets.py -m slow --no-cov
```

If the directory is missing the loader raises an error pointing back to
this section instead of a traceback from deeper in the stack.
