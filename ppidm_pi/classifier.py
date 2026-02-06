"""1-D ResNet-18 downstream classifier for ECG electrolyte classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1dClassifier(nn.Module):
    """Basic 1-D residual block (two conv layers + skip)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x))


class ResNet1d(nn.Module):
    """1-D ResNet-18 for ECG electrolyte-level classification.

    Args:
        in_channels: Number of ECG leads (default 12).
        num_classes: Number of output classes (default 3 for low/normal/high).
    """

    def __init__(self, in_channels: int = 12, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(
        in_ch: int, out_ch: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [ResBlock1dClassifier(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock1dClassifier(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ECG input ``[B, 12, L]``.

        Returns:
            Logits ``[B, num_classes]``.
        """
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)
