import torch
import torch.nn as nn
from torchvision import models


class VisionEncoderV2(nn.Module):
    def __init__(self, feature_dim: int = 512, dropout: float = 0.0,
                 pretrained: bool = True):
        super().__init__()
        self.feature_dim = feature_dim

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            with torch.no_grad():
                new_conv1.weight[:] = old_conv1.weight.repeat(1, 2, 1, 1) / 2.0
        backbone.conv1 = new_conv1

        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.proj = nn.Linear(512, feature_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["img_pair"]
        feat = self.backbone(x)
        feat = self.proj(feat)
        return self.dropout(feat)
