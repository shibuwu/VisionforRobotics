import torch
import torch.nn as nn

def quaternion_loss_with_sign_flip(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    diff_pos = (q_pred - q_gt).pow(2).sum(dim=-1)
    diff_neg = (q_pred + q_gt).pow(2).sum(dim=-1)
    return torch.minimum(diff_pos, diff_neg)

class PoseLoss(nn.Module):
    def __init__(self, lambda_rot: float = 10.0):
        super().__init__()
        self.lambda_rot = lambda_rot

    def forward(self, pred: dict, target: dict) -> tuple[torch.Tensor, dict]:
        delta_t_pred = pred["delta_t"]
        delta_q_pred = pred["delta_q"]
        delta_t_gt = target["delta_t"]
        delta_q_gt = target["delta_q"]
        t_err = (delta_t_pred - delta_t_gt).pow(2).sum(dim=-1)
        loss_t = t_err.mean()
        r_err = quaternion_loss_with_sign_flip(delta_q_pred, delta_q_gt)
        loss_r = r_err.mean()
        total = loss_t + self.lambda_rot * loss_r
        components = {
            "loss_total": total.item(),
            "loss_t": loss_t.item(),
            "loss_r": loss_r.item(),
        }
        return total, components
