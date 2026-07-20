import torch
import torch.nn as nn
from losses import PoseLoss

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_rot: float = 10.0,
        lambda_homo: float = 1.0,
        h4pt_scale: float = 400.0,
    ):
        super().__init__()
        self.pose_loss = PoseLoss(lambda_rot=lambda_rot)
        self.lambda_homo = lambda_homo
        self.h4pt_scale = h4pt_scale

    def forward(self, pred: dict, target: dict) -> tuple[torch.Tensor, dict]:
        loss_pose, pose_components = self.pose_loss(pred, target)

        h4pt_pred = pred["h4pt"] / self.h4pt_scale
        h4pt_gt   = target["h4pt_gt"] / self.h4pt_scale

        h_err = (h4pt_pred - h4pt_gt).pow(2).sum(dim=-1)
        loss_homo = h_err.mean()

        total = loss_pose + self.lambda_homo * loss_homo

        components = {
            "loss_total": total.item(),
            "loss_t":     pose_components["loss_t"],
            "loss_r":     pose_components["loss_r"],
            "loss_pose":  pose_components["loss_total"],
            "loss_homo":  loss_homo.item(),
        }
        return total, components
