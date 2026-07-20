import torch
import torch.nn as nn
from encoder_vision import VisionEncoder
from encoder_inertial import InertialEncoder

class FusionEncoder(nn.Module):

    def __init__(self, feature_dim: int=256, vision_feat_dim: int=256, inertial_feat_dim: int=128, hidden_dim: int=256, dropout: float=0.0, vision_encoder: nn.Module=None, inertial_encoder: nn.Module=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.vision = vision_encoder if vision_encoder is not None else VisionEncoder(feature_dim=vision_feat_dim)
        self.inertial = inertial_encoder if inertial_encoder is not None else InertialEncoder(feature_dim=inertial_feat_dim)
        v_out = self.vision.feature_dim
        i_out = self.inertial.feature_dim
        concat_dim = v_out + i_out
        self.fusion = nn.Sequential(nn.Linear(concat_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, feature_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, batch: dict) -> torch.Tensor:
        v_feat = self.vision(batch)
        i_feat = self.inertial(batch)
        concat = torch.cat([v_feat, i_feat], dim=-1)
        fused = self.fusion(concat)
        return self.dropout(fused)
