"""Data loaders for the experiments.

We hand-roll small loaders to avoid pulling in torchvision:

* Toy regression: in-memory synthetic dataset.
* MNIST: downloaded once from a mirror if not cached, otherwise loaded
  from the IDX files in ``data/MNIST/raw/``.
* CIFAR-10: downloaded once from a mirror if not cached, otherwise
  loaded from the pickled batches in ``data/cifar-10-batches-py/``.

All loaders return ``torch.utils.data.Dataset`` objects so they slot
into a normal ``DataLoader``.
"""
