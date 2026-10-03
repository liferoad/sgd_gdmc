"""Hand-rolled MNIST loader (no torchvision dependency).

Downloads the four IDX files from a public mirror if they are not in
``data/MNIST/raw/``, then parses them into torch tensors.

Returns train and test ``Dataset`` objects suitable for a normal
``DataLoader``.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


_MIRRORS = [
    # Original Yann LeCun mirror; can be slow. Kept as a last-resort fallback.
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    # Alternative mirrors that are usually faster.
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic for image file: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, h, w)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic for label file: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def _download_if_missing(root: Path) -> None:
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for fn in _FILES:
        if (raw / fn).exists():
            continue
        last_err = None
        for mirror in _MIRRORS:
            url = mirror + fn
            dst = raw / fn
            try:
                print(f"[mnist] downloading {url} -> {dst}")
                urllib.request.urlretrieve(url, dst)
                last_err = None
                break
            except Exception as e:  # pragma: no cover - network failures
                print(f"[mnist] mirror {mirror} failed: {e}")
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"could not download {fn}: {last_err}")


def _load(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _download_if_missing(root)
    raw = root / "MNIST" / "raw"
    tr_x = _read_idx_images(raw / _FILES[0])
    tr_y = _read_idx_labels(raw / _FILES[1])
    te_x = _read_idx_images(raw / _FILES[2])
    te_y = _read_idx_labels(raw / _FILES[3])
    return tr_x, tr_y, te_x, te_y


def make_mnist(root: str = "data", normalize: bool = True
               ) -> Tuple[Dataset, Dataset]:
    """Return ``(train_dataset, test_dataset)`` as ``TensorDataset``."""
    root_p = Path(root)
    tr_x, tr_y, te_x, te_y = _load(root_p)
    if normalize:
        tr_x = tr_x.astype(np.float32) / 255.0
        te_x = te_x.astype(np.float32) / 255.0
    # Flatten for the MLP case; for the CNN we keep the (B, 1, 28, 28) shape.
    # We expose a 4-D representation and let the model reshape if needed.
    tr_x = tr_x[:, None, :, :]  # (N, 1, 28, 28)
    te_x = te_x[:, None, :, :]
    train_ds = TensorDataset(torch.from_numpy(tr_x), torch.from_numpy(tr_y.astype(np.int64)))
    test_ds = TensorDataset(torch.from_numpy(te_x), torch.from_numpy(te_y.astype(np.int64)))
    return train_ds, test_ds
