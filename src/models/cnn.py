"""Small CNN used for MNIST and CIFAR-10."""

from __future__ import annotations
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """A small VGG-ish CNN with 3 conv blocks + 2 FC layers.

    For MNIST (1 input channel) and CIFAR-10 (3 input channels).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 base_width: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = base_width, base_width * 2, base_width * 4
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # -> (B, c3, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
