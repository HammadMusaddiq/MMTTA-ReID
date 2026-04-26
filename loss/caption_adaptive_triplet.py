import torch
import torch.nn as nn
import torch.nn.functional as F


# class CaptionAdaptiveTripletLoss(nn.Module):
#     r"""
#     Adaptive-margin batch-hard triplet loss
#       m_ij = α · ( 1 − cos(ft_i , ft_j) )
#       L    = Σ_i [ m_i + hardest_pos - hardest_neg ]_+

#     Args
#     ----
#     alpha (float) : scaling factor for margin (α).
#     """
#     def __init__(self, alpha: float = 1.0):
#         super().__init__()
#         self.alpha = alpha

#     @staticmethod
#     def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
#         """cosine similarity matrix -> shape (B,B)"""
#         x = F.normalize(x, p=2, dim=1)
#         return x @ x.t()

#     def forward(
#         self,
#         img_feat: torch.Tensor,      # B×D   – visual CLS after adapter
#         txt_feat: torch.Tensor,      # B×D   – cached caption embedding
#         pid: torch.Tensor            # B     – ground-truth person IDs
#     ) -> torch.Tensor:
#         # ---- adaptive margin ----------------------------------------
#         with torch.no_grad():
#             cos_tt = self.pairwise_cosine(txt_feat)       # (B,B)
#             margin  = self.alpha * (1.0 - cos_tt)         # m_ij

#         # ---- batch-hard anchors -------------------------------------
#         dist_mat = 1 - self.pairwise_cosine(img_feat)     # cosine distance
#         eye = torch.eye(dist_mat.size(0), device=img_feat.device) * 1e12
#         dist_mat = dist_mat + eye                        # avoid self-pick

#         pos_mask = pid.unsqueeze(0) == pid.unsqueeze(1)  # same ID
#         neg_mask = ~pos_mask

#         hardest_pos = (dist_mat * pos_mask.float()).max(1)[0]
#         hardest_neg = (dist_mat + 1e12 * (~neg_mask)).min(1)[0]

#         # use per-sample margin m_i = margin[i, anchor_pos] (take any pos j)
#         first_pos_idx = pos_mask.float().argmax(dim=1)
#         adaptive_m = margin[torch.arange(len(pid)), first_pos_idx]

#         loss = F.relu(hardest_pos - hardest_neg + adaptive_m).mean()
#         return loss
class CaptionAdaptiveTripletLoss(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def _cosine_mat(x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        return x @ x.t()                       # (B, B)

    def forward(self,
                img_feat: torch.Tensor,        # B × D
                txt_feat: torch.Tensor,        # B × D
                pid:      torch.Tensor         # B
               ) -> torch.Tensor:

        # ---------- adaptive margin (text side) --------------------
        with torch.no_grad():
            cos_tt  = self._cosine_mat(txt_feat)          # (B,B)
            margin  = self.alpha * (1.0 - cos_tt)         # m_ij

        # ---------- distances on the image side -------------------
        dist_mat = 1.0 - self._cosine_mat(img_feat)       # cosine distance
        B = dist_mat.size(0)
        dist_mat = dist_mat + torch.eye(B, device=dist_mat.device) * 1e9  # mask self-pairs

        # ---------- masks for positives / negatives ---------------
        pos_mask = pid.view(1, -1) == pid.view(-1, 1)     # (B,B) True for same-ID
        pos_mask.fill_diagonal_(False)                    # ⚠️ exclude self-pairs
        neg_mask = ~pos_mask                              # different IDs

        # ---------- batch-hard selection --------------------------
        hardest_pos = (dist_mat * pos_mask).max(dim=1)[0]
        hardest_neg = (dist_mat + 1e9 * pos_mask).min(dim=1)[0]

        # choose one positive index per anchor to pick its margin
        first_pos = pos_mask.float().argmax(dim=1)
        adaptive_m = margin[torch.arange(B, device=pid.device), first_pos]

        # ---------- final loss ------------------------------------
        loss = F.relu(hardest_pos - hardest_neg + adaptive_m).mean()
        return loss