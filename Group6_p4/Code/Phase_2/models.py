import torch
import torch.nn as nn

class VIOModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.head = nn.Linear(feature_dim, 7)

    def forward(self, batch: dict) -> dict:
        feat = self.encoder(batch)
        out = self.head(feat)
        delta_t = out[..., :3]
        q_raw = out[..., 3:]
        q_norm = q_raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta_q = q_raw / q_norm
        return {"delta_t": delta_t, "delta_q": delta_q}
