import torch, torch.nn as nn, torch.nn.functional as F

class DistillL2Loss(nn.Module):
    """Simple L2 distillation:  ‖f_student − f_teacher‖²."""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.w = weight

    def forward(self,
                stu_feat: torch.Tensor,   # f_ViT   (B,D)
                tea_feat: torch.Tensor):  # f_nomic (B,D)
        return self.w * F.mse_loss(
            F.normalize(stu_feat, p=2, dim=1),
            F.normalize(tea_feat, p=2, dim=1)
        )
