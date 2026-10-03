"""Hand-rolled CIFAR-10 loader (no torchvision dependency).

Downloads the CIFAR-10 python version from a public mirror if not in
``data/cifar-10-batches-py/``, then unpickles the five training
batches and the test batch into a single ``TensorDataset``.

Returns train and test ``Dataset`` objects suitable for a normal
``DataLoader``. Images are returned as ``(B, 3, 32, 32)`` float32 in
[0, 1].
"""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


_MIRRORS = [
    # University of Toronto CS mirror.
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    # Sometimes faster alternative.
    "https://ossci-datasets.s3.amazonaws.com/cifar-10-python.tar.gz",
]


def _download_if_missing(root: Path) -> None:
    target_dir = root / "cifar-10-batches-py"
    if target_dir.exists():
        return
    last_err = None
    for url in _MIRRORS:
        dst = root / "cifar-10-python.tar.gz"
        try:
            print(f"[cifar10] downloading {url} -> {dst}")
            urllib.request.urlretrieve(url, dst)
            print(f"[cifar10] extracting {dst}")
            with tarfile.open(dst, "r:gz") as tar:
                tar.extractall(path=root)
            os.remove(dst)
            return
        except Exception as e:  # pragma: no cover
            print(f"[cifar10] mirror {url} failed: {e}")
            last_err = e
    raise RuntimeError(f"could not download CIFAR-10: {last_err}")


def _load_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"]  # (N, 3072)
    y = np.array(d[b"labels"], dtype=np.int64)
    return x, y


def make_cifar10(root: str = "data", normalize: bool = True
                 ) -> Tuple[Dataset, Dataset]:
    root_p = Path(root)
    _download_if_missing(root_p)
    base = root_p / "cifar-10-batches-py"
    # Train: data_batch_1..5
    tr_xs, tr_ys = [], []
    for i in range(1, 6):
        x, y = _load_batch(base / f"data_batch_{i}")
        tr_xs.append(x)
        tr_ys.append(y)
    tr_x = np.concatenate(tr_xs, axis=0)
    tr_y = np.concatenate(tr_ys, axis=0)
    te_x, te_y = _load_batch(base / "test_batch")
    # Reshape to (N, 3, 32, 32). CIFAR stores channel-last.
    tr_x = tr_x.reshape(-1, 3, 32, 32).astype(np.float32)
    te_x = te_x.reshape(-1, 3, 32, 32).astype(np.float32)
    if normalize:
        tr_x /= 255.0
        te_x /= 255.0
    train_ds = TensorDataset(torch.from_numpy(tr_x), torch.from_numpy(tr_y))
    test_ds = TensorDataset(torch.from_numpy(te_x), torch.from_numpy(te_y))
    return train_ds, test_ds
