import torch, torch.nn as nn, torch.nn.functional as F
import math

class VisionLanguageInfoNCELoss(nn.Module):
    r"""
    Vision–Language InfoNCE
      L = -log  exp( cos(f_v , f_t) / τ )  /  Σ_j exp( cos(f_v , f_t(j)) / τ )
    """
    def __init__(self, tau: float = 0.08):
        super().__init__()
        self.tau = tau

    def forward(self, img_feat: torch.Tensor,
                txt_feat: torch.Tensor) -> torch.Tensor:
        img_feat = F.normalize(img_feat, p=2, dim=1)          # [B,D]
        txt_feat = F.normalize(txt_feat, p=2, dim=1)          # [B,D]
        logits   = img_feat @ txt_feat.T / self.tau           # similarity / τ
        targets  = torch.arange(img_feat.size(0), device=img_feat.device)
        loss     = F.cross_entropy(logits, targets)           # InfoNCE
        return loss
