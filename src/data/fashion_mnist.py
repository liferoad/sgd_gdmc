"""Fashion-MNIST loader.

Same IDX file format as MNIST (28x28 grayscale, 10 classes), so we reuse
the readers from src.data.mnist and only change the mirror and the
cache directory. Fashion-MNIST is a drop-in replacement for MNIST that
is substantially harder (it is clothing categories rather than digits),
which makes it a useful second use case for the GDMC comparison.

Downloads to data/FashionMNIST/raw/ on first use.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from .mnist import _read_idx_images, _read_idx_labels

# Multiple mirrors: the https S3 endpoint is unreachable from some
# networks (connection reset), the http S3 endpoint and the GitHub raw
# / Google storage mirrors work.
_MIRRORS = (
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
)
_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def _download_if_missing(root: Path) -> None:
    raw = root / "FashionMNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for fn in _FILES:
        if (raw / fn).exists():
            continue
        dst = raw / fn
        last_err = None
        for mirror in _MIRRORS:
            url = mirror + fn
            try:
                print(f"[fashion-mnist] downloading {url} -> {dst}")
                urllib.request.urlretrieve(url, dst)
                last_err = None
                break
            except Exception as e:  # pragma: no cover - network
                print(f"[fashion-mnist] mirror failed: {e}")
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"could not download {fn}: {last_err}")


def make_fashion_mnist(root: str = "data", normalize: bool = True
                       ) -> Tuple[Dataset, Dataset]:
    """Return (train_dataset, test_dataset) as TensorDataset.

    Images are (N, 1, 28, 28) float32.
    """
    root_p = Path(root)
    _download_if_missing(root_p)
    raw = root_p / "FashionMNIST" / "raw"
    tr_x = _read_idx_images(raw / _FILES[0])
    tr_y = _read_idx_labels(raw / _FILES[1])
    te_x = _read_idx_images(raw / _FILES[2])
    te_y = _read_idx_labels(raw / _FILES[3])
    if normalize:
        tr_x = tr_x.astype(np.float32) / 255.0
        te_x = te_x.astype(np.float32) / 255.0
    tr_x = tr_x[:, None, :, :]
    te_x = te_x[:, None, :, :]
    train_ds = TensorDataset(torch.from_numpy(tr_x),
                             torch.from_numpy(tr_y.astype(np.int64)))
    test_ds = TensorDataset(torch.from_numpy(te_x),
                            torch.from_numpy(te_y.astype(np.int64)))
    return train_ds, test_ds
