import math

import torch
import torch.nn as nn

from src.utils.training_utils import build_classification_head


class ClassicalBaselineCNN(nn.Module):
    """Classical counterpart of DAQCNN with the same head.

    A bias-free Conv2d (kernel_size, stride) with out_channels filters replaces
    the quantum layer: trainable, or random and frozen with
    fixed_random_filters=True. raw_features=True feeds the image straight to
    the head (the "Raw pixels" baseline).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 3,
        out_channels: int = 180,
        dropout: float = 0.1,
        activation: str = "relu",
        head_hidden_channels: int = 64,
        fixed_random_filters: bool = False,
        raw_features: bool = False,
    ):
        super().__init__()

        self.raw_features = raw_features
        head_in = out_channels if not raw_features else in_channels

        if not raw_features:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, bias=False)
            if fixed_random_filters:
                # re-initialised after the default init: this draw order is part
                # of the published runs' random state
                nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
                for param in self.conv.parameters():
                    param.requires_grad = False

        self.head = build_classification_head(
            head_in, num_classes, dropout, activation, hidden_channels=head_hidden_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.raw_features:
            x = self.conv(x).float()
        target_device = next(self.head.parameters()).device
        if x.device != target_device:
            x = x.to(target_device)
        return self.head(x)
