import torch
import torch.nn as nn

class VIOModelV4(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.pose_head = nn.Linear(feature_dim, 7)
        self.homo_head = nn.Linear(feature_dim, 8)

    def forward(self, batch: dict) -> dict:
        feat = self.encoder(batch)

        pose_out = self.pose_head(feat)
        delta_t = pose_out[..., :3]
        q_raw = pose_out[..., 3:]
        q_norm = q_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta_q = q_raw / q_norm

        h4pt = self.homo_head(feat)

        return {"delta_t": delta_t, "delta_q": delta_q, "h4pt": h4pt}
