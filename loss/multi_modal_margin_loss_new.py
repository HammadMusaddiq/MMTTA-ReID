import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable


class multiModalMarginLossNew(nn.Module):
    def __init__(self, margin=3, dist_type='l2'):
        super(multiModalMarginLossNew, self).__init__()
        self.dist_type = dist_type
        self.margin = margin
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='mean') #sum
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, feat3, label1):
      """
      Args:
          feat{1,2,3}:  (B, D) tensors for RGB / IR / TI
          label1:       (B,)    tensor of person-IDs
      Returns:
          scalar loss averaged over the identities in this batch
      """
      label_num = len(label1.unique())
      sample_num = len(label1) // label_num
      feature_dimension = feat1.size(1)

      # split the batch by identity
      feat1_chunks = feat1.chunk(label_num, 0)
      feat2_chunks = feat2.chunk(label_num, 0)
      feat3_chunks = feat3.chunk(label_num, 0)

      id_losses = []                                         # ← collect per-ID max

      for i in range(label_num):
          # centres of one identity in each modality
          c1 = torch.mean(feat1_chunks[i], dim=0)
          c2 = torch.mean(feat2_chunks[i], dim=0)
          c3 = torch.mean(feat3_chunks[i], dim=0)

          if self.dist_type in ('l2', 'l1'):
              # margin penalty for this ID: take the worst pair
              id_max = max(
                  abs(self.margin - self.dist(c1, c2)),
                  abs(self.margin - self.dist(c2, c3)),
                  abs(self.margin - self.dist(c1, c3)),
              )
              id_losses.append(id_max)

      # final loss = mean over identities  → O(1) scale
      dist = torch.stack(id_losses).mean()
      dist = dist / feature_dimension
      return dist
