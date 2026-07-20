import torch
import torch.nn as nn


class InertialEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.proj = nn.Linear(128, feature_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["imu"]
        x = x.transpose(1, 2)
        feat = self.conv(x)
        feat = self.proj(feat)
        return self.dropout(feat)
