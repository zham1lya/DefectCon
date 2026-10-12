"""DefectCon network (Section 4.2).

ResNet-18 encoder on 1-channel input with optional dilation of layer3/layer4,
global max (or average) pooling, a projector for the contrastive branch and an
MLP classifier on the pooled 512-d encoder feature.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

ENCODER_DIM = 512


def _dilate_layer(layer: nn.Sequential, dilation: int) -> None:
    """Turn a stride-2 ResNet stage into a stride-1 dilated stage.

    Only stride, padding and dilation change, so pretrained weights still load.
    """
    for block in layer:
        if block.conv1.stride != (1, 1):
            block.conv1.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        for conv in (block.conv1, block.conv2):
            if conv.kernel_size == (3, 3):
                conv.dilation = (dilation, dilation)
                conv.padding = (dilation, dilation)


class Encoder(nn.Module):
    """ImageNet-initialised ResNet-18 with a 1-channel stem.

    ``dilate='layer34'`` dilates layer3 (d=2) and layer4 (d=4), giving a 40x40
    final map at 320x320 input; ``'none'`` keeps the standard 10x10 map.
    """

    def __init__(self, dilate: str = "layer34"):
        super().__init__()
        if dilate not in ("none", "layer34"):
            raise ValueError(f"dilate must be 'none' or 'layer34', got {dilate!r}")
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if dilate == "layer34":
            _dilate_layer(resnet.layer3, dilation=2)
            _dilate_layer(resnet.layer4, dilation=4)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class ConvProjector(nn.Module):
    """1x1-conv projector applied to the feature map before pooling."""

    def __init__(self, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(ENCODER_DIM, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(256, proj_dim, 1),
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.proj(fmap)


class MLPProjector(nn.Module):
    """MLP projector applied to the pooled 512-d vector (ablation cell J1)."""

    def __init__(self, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ENCODER_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.proj(v)


class DefectCon(nn.Module):
    """Encoder, pooling, projector and classifier, trained jointly."""

    def __init__(self, dilate: str = "layer34", pooling: str = "max", proj_type: str = "conv",
                 proj_dim: int = 128, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.encoder = Encoder(dilate=dilate)
        if pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"pooling must be 'max' or 'avg', got {pooling!r}")
        if proj_type not in ("conv", "mlp"):
            raise ValueError(f"proj_type must be 'conv' or 'mlp', got {proj_type!r}")
        self.proj_type = proj_type
        self.projector = ConvProjector(proj_dim) if proj_type == "conv" else MLPProjector(proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(ENCODER_DIM, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    @classmethod
    def from_arch(cls, arch: Dict[str, Any]) -> "DefectCon":
        return cls(dilate=arch["dilate"], pooling=arch["pooling"], proj_type=arch["proj_type"],
                   proj_dim=arch["proj_dim"], hidden_dim=arch["hidden_dim"])

    def _project(self, fmap: torch.Tensor) -> torch.Tensor:
        if self.proj_type == "conv":
            return self.pool(self.projector(fmap)).flatten(1)
        return self.projector(self.pool(fmap).flatten(1))

    def forward_train(self, x: torch.Tensor, n_cls: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """One encoder pass over both views (2B images).

        Returns the L2-normalised projections of all 2B images (contrastive
        branch) and the logits of the first ``n_cls`` images (view 1).
        """
        fmap = self.encoder(x)
        z = F.normalize(self._project(fmap), dim=1)
        logits = self.classifier(self.pool(fmap).flatten(1)[:n_cls])
        return z, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits; the defect score is ``softmax(logits)[:, 1]``."""
        return self.classifier(self.pool(self.encoder(x)).flatten(1))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised pooled 512-d encoder feature (Appendix D)."""
        return F.normalize(self.pool(self.encoder(x)).flatten(1), dim=1)
