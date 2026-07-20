import torch
import torch.nn as nn

from encoder_vision_v2 import VisionEncoderV2
from encoder_inertial import InertialEncoder


class FusionEncoderV2(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        vision_feat_dim: int = 512,
        inertial_feat_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        vision_pretrained: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.vision = VisionEncoderV2(
            feature_dim=vision_feat_dim, pretrained=vision_pretrained, dropout=0.0
        )
        self.inertial = InertialEncoder(
            feature_dim=inertial_feat_dim, dropout=0.0
        )

        v_out = self.vision.feature_dim
        i_out = self.inertial.feature_dim
        concat_dim = v_out + i_out

        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, batch: dict) -> torch.Tensor:
        v_feat = self.vision(batch)
        i_feat = self.inertial(batch)
        concat = torch.cat([v_feat, i_feat], dim=-1)
        fused = self.fusion(concat)
        return self.dropout(fused)
